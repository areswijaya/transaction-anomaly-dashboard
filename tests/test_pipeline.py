"""Pemeriksaan sederhana: file hasil notebook ada, saling cocok, dan masuk akal.

Jalankan dari folder proyek:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import app

BASE_DIR = Path(__file__).resolve().parents[1]
CLUSTERING_FEATURES = [
    "TransactionAmount", "CustomerAge", "TransactionDuration", "AccountBalance", "Balance_Ratio",
]
REQUIRED_COLUMNS = CLUSTERING_FEATURES + [
    "LoginAttempts", "Amount_Per_Second", "Cluster", "Segment_Name",
    "High_Login_Risk", "Anomaly_Flag", "Anomaly_Raw", "Anomaly_Score",
    "CustomerOccupation", "Channel", "TransactionType",
]


def _relative_luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(c1, c2):
    l1, l2 = sorted([_relative_luminance(c1), _relative_luminance(c2)], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(BASE_DIR / "data" / "data_clustering_clean.csv")
        cls.scaler = joblib.load(BASE_DIR / "models" / "scaler_clustering.joblib")
        cls.kmeans = joblib.load(BASE_DIR / "models" / "kmeans_model.joblib")
        cls.anomaly = joblib.load(BASE_DIR / "models" / "anomaly_model.joblib")
        cls.segment_map = app.get_segment_map(cls.df)

    def predict(self, amount, balance, duration, age, login):
        return app.predict_transaction(
            amount, balance, duration, age, login,
            self.scaler, self.kmeans, self.anomaly, self.segment_map,
        )

    def preset(self, name):
        p = app.PRESET_SCENARIOS[name]
        return self.predict(p["amount"], p["balance"], p["duration"], p["age"], p["login"])

    # --- data dan model saling cocok ---

    def test_columns_exist(self):
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        self.assertEqual(missing, [])

    def test_no_missing_values(self):
        self.assertEqual(int(self.df[REQUIRED_COLUMNS].isnull().sum().sum()), 0)

    def test_login_rule_matches_data(self):
        expected = self.df["LoginAttempts"] >= 3
        self.assertTrue((self.df["High_Login_Risk"] == expected).all())

    def test_kmeans_matches_saved_clusters(self):
        scaled = pd.DataFrame(self.scaler.transform(self.df[CLUSTERING_FEATURES]),
                              columns=CLUSTERING_FEATURES)
        self.assertTrue((self.kmeans.predict(scaled) == self.df["Cluster"]).all())

    def test_each_cluster_has_one_name(self):
        names_per_cluster = self.df.groupby("Cluster")["Segment_Name"].nunique()
        self.assertTrue((names_per_cluster == 1).all())

    def test_segment_names_match_app(self):
        self.assertEqual(set(self.df["Segment_Name"].unique()), set(app.SEGMENT_COLORS))
        self.assertEqual(set(app.SEGMENT_ACTIONS), set(app.SEGMENT_COLORS))

    def test_anomaly_score_range(self):
        self.assertTrue(self.df["Anomaly_Score"].between(0, 100).all())

    def test_anomaly_rate_close_to_contamination(self):
        rate = self.df["Anomaly_Flag"].mean()
        target = self.anomaly["contamination"]
        self.assertLess(abs(rate - target), 0.005)

    def test_anomaly_model_matches_saved_flags(self):
        scaled = self.anomaly["scaler"].transform(self.df[self.anomaly["features"]])
        flags = self.anomaly["model"].predict(scaled) == -1
        self.assertTrue((flags == self.df["Anomaly_Flag"].to_numpy()).all())

    def test_train_scores_match_saved_raw_scores(self):
        train_scores = np.asarray(self.anomaly["train_scores"])
        self.assertTrue((np.diff(train_scores) >= 0).all())
        self.assertTrue(np.allclose(np.sort(self.df["Anomaly_Raw"].to_numpy()), train_scores))

    def test_percentile_threshold_separates_flags(self):
        thresh = app.anomaly_threshold(self.anomaly["train_scores"])
        self.assertTrue(((self.df["Anomaly_Score"] > thresh) == self.df["Anomaly_Flag"]).all())
        self.assertAlmostEqual(thresh, 100 * (1 - self.anomaly["contamination"]), delta=0.5)

    # --- predict_transaction ---

    def test_predict_reproduces_training_rows(self):
        for _, r in self.df.sample(200, random_state=0).iterrows():
            out = self.predict(r["TransactionAmount"], r["AccountBalance"], r["TransactionDuration"],
                               r["CustomerAge"], r["LoginAttempts"])
            self.assertEqual(out["segment"], r["Segment_Name"])
            self.assertEqual(out["is_anomaly"], bool(r["Anomaly_Flag"]))
            self.assertAlmostEqual(out["anomaly_score"], r["Anomaly_Score"], delta=0.01)

    def test_zero_balance_is_not_treated_as_normal(self):
        out = self.predict(100.0, 0.0, 90, 30, 2)
        self.assertEqual(out["balance_ratio"], 100.0 / app.MIN_BALANCE)
        self.assertTrue(out["is_anomaly"])
        self.assertFalse(out["is_high_login_risk"])
        self.assertTrue(0.0 <= out["anomaly_score"] <= 100.0)
        self.assertTrue(0.0 <= out["threshold_score"] <= 100.0)

    def test_login_rule_edge(self):
        self.assertTrue(self.predict(100.0, 5000.0, 90, 30, 3)["is_high_login_risk"])
        self.assertFalse(self.predict(100.0, 5000.0, 90, 30, 2)["is_high_login_risk"])

    def test_preset_scenarios(self):
        young = self.preset("Nasabah muda, normal")
        self.assertEqual(young["segment"], app.SEG_YOUNG)
        self.assertFalse(young["is_anomaly"])

        older = self.preset("Nasabah mapan, normal")
        self.assertEqual(older["segment"], app.SEG_OLDER)
        self.assertFalse(older["is_anomaly"])

        extreme = self.preset("Nominal besar, saldo kecil")
        self.assertTrue(extreme["is_anomaly"])

        login = self.preset("Login berulang")
        self.assertTrue(login["is_high_login_risk"])
        self.assertFalse(login["is_anomaly"])

    def test_out_of_range_detection(self):
        lo, hi = app.get_feature_bounds(self.df)
        inside = {c: float((lo[c] + hi[c]) / 2) for c in CLUSTERING_FEATURES}
        self.assertEqual(app.out_of_range(inside, lo, hi), [])
        outside = dict(inside, TransactionAmount=float(hi["TransactionAmount"]) * 10)
        self.assertEqual(app.out_of_range(outside, lo, hi), [app.format_feature_label("TransactionAmount")])

    # --- repo hygiene ---

    def test_no_stale_artifacts(self):
        for stale in [
            BASE_DIR / "models" / "anomaly_bundle.joblib",
            BASE_DIR / "notebooks" / "classification.ipynb",
            BASE_DIR / "models" / "rf_classifier_pipeline.joblib",
        ]:
            self.assertFalse(stale.exists(), f"Stale artifact exists: {stale}")

    def test_theme_config(self):
        config_text = (BASE_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn('primaryColor = "#1D4ED8"', config_text)

    def test_no_em_or_en_dash(self):
        text_files = [
            BASE_DIR / "app.py",
            BASE_DIR / "README.md",
            BASE_DIR / "notebooks" / "clustering.ipynb",
            BASE_DIR / "tests" / "test_pipeline.py",
        ]
        for file_path in text_files:
            content = file_path.read_text(encoding="utf-8")
            self.assertNotIn("\u2014", content, f"Em dash found in {file_path.name}")
            self.assertNotIn("\u2013", content, f"En dash found in {file_path.name}")

    def test_app_style_markers(self):
        app_text = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        for slop_marker in ["linear-gradient", "hero-badge", "CREATED BY"]:
            self.assertNotIn(slop_marker, app_text)
        self.assertIn("padding-top: 3.75rem", app_text)

    def test_notebook_is_clean(self):
        nb = json.loads((BASE_DIR / "notebooks" / "clustering.ipynb").read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [], "Notebook masih berisi output, bersihkan sebelum commit")

        nb_sources = "\n".join(
            "".join(cell.get("source", [])) for cell in nb["cells"] if cell["cell_type"] == "code"
        )
        self.assertIn("ROOT_DIR", nb_sources)
        self.assertIn("raw.xlsx", nb_sources)

    def test_feature_labels(self):
        self.assertTrue(set(CLUSTERING_FEATURES).issubset(app.FEATURE_LABELS))
        self.assertNotEqual(app.format_feature_label("TransactionAmount"), "TransactionAmount")

    def test_devcontainer_is_valid_json(self):
        devcontainer_raw = (BASE_DIR / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")
        self.assertNotIn("\u2705", devcontainer_raw)
        self.assertIsInstance(json.loads(devcontainer_raw), dict)

    def test_palette_contrast(self):
        text_pairs = [
            ("#F8FAFC", "#0F172A"),
            ("#E2E8F0", "#0F172A"),
            ("#E2E8F0", "#1E293B"),
            ("#CBD5E1", "#1E293B"),
            ("#F8FAFC", "#1E293B"),
            ("#0F172A", "#F8FAFC"),
            ("#0F172A", "#FFFFFF"),
            ("#334155", "#FFFFFF"),
            ("#475569", "#FFFFFF"),
            ("#1D4ED8", "#FFFFFF"),
            ("#1E3A8A", "#DBEAFE"),
            ("#0F172A", "#BFDBFE"),
            ("#B45309", "#FFFFFF"),
            ("#B91C1C", "#FFFFFF"),
        ]
        for fg, bg in text_pairs:
            self.assertGreaterEqual(_contrast_ratio(fg, bg), 4.5, f"Contrast failed for {fg} on {bg}")

        self.assertGreaterEqual(_contrast_ratio("#64748B", "#FFFFFF"), 3.0)


if __name__ == "__main__":
    unittest.main()

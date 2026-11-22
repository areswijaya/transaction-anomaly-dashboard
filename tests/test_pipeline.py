"""Pemeriksaan sederhana: file hasil notebook ada, saling cocok, dan masuk akal.

Jalankan dari folder proyek:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

import app

BASE_DIR = Path(__file__).resolve().parents[1]
CLUSTERING_FEATURES = [
    "TransactionAmount", "CustomerAge", "TransactionDuration", "AccountBalance", "Balance_Ratio",
]
REQUIRED_COLUMNS = CLUSTERING_FEATURES + [
    "LoginAttempts", "Amount_Per_Second", "Cluster", "Segment_Name",
    "High_Login_Risk", "Anomaly_Flag", "Anomaly_Score",
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

    def test_anomaly_score_range(self):
        self.assertTrue(self.df["Anomaly_Score"].between(0, 100).all())

    def test_anomaly_rate_close_to_contamination(self):
        rate = self.df["Anomaly_Flag"].mean()
        target = self.anomaly["contamination"]
        self.assertLess(abs(rate - target), 0.02)

    def test_anomaly_model_matches_saved_flags(self):
        scaled = self.anomaly["scaler"].transform(self.df[self.anomaly["features"]])
        flags = self.anomaly["model"].predict(scaled) == -1
        self.assertTrue((flags == self.df["Anomaly_Flag"].to_numpy()).all())

    def test_predict_transaction_rules_and_edge_cases(self):
        segment_map = app.get_segment_map(self.df)

        zero_bal = app.predict_transaction(
            100.0, 0.0, 90, 30, 2, self.scaler, self.kmeans, self.anomaly, segment_map
        )
        self.assertEqual(zero_bal["balance_ratio"], 0.0)
        self.assertFalse(zero_bal["is_high_login_risk"])
        self.assertTrue(0.0 <= zero_bal["anomaly_score"] <= 100.0)
        self.assertTrue(0.0 <= zero_bal["threshold_score"] <= 100.0)

        login_edge = app.predict_transaction(
            100.0, 5000.0, 90, 30, 3, self.scaler, self.kmeans, self.anomaly, segment_map
        )
        self.assertTrue(login_edge["is_high_login_risk"])

        extreme = app.predict_transaction(
            1450.0, 800.0, 35, 40, 1, self.scaler, self.kmeans, self.anomaly, segment_map
        )
        self.assertTrue(extreme["is_anomaly"])

        conservative = app.predict_transaction(
            250.0, 7500.0, 115, 56, 1, self.scaler, self.kmeans, self.anomaly, segment_map
        )
        self.assertEqual(conservative["segment"], "Nasabah Mapan Konservatif")

    def test_writing_and_artifacts(self):
        for stale in [
            BASE_DIR / "models" / "anomaly_bundle.joblib",
            BASE_DIR / "notebooks" / "classification.ipynb",
            BASE_DIR / "models" / "rf_classifier_pipeline.joblib",
        ]:
            self.assertFalse(stale.exists(), f"Stale artifact exists: {stale}")

        config_text = (BASE_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn('primaryColor = "#1D4ED8"', config_text)

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

        app_text = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        for slop_marker in ["linear-gradient", "hero-badge", "CREATED BY"]:
            self.assertNotIn(slop_marker, app_text)
        self.assertIn("padding-top: 3.75rem", app_text)

        nb = json.loads((BASE_DIR / "notebooks" / "clustering.ipynb").read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [])

    def test_palette_contrast(self):
        text_pairs = [
            ("#F8FAFC", "#0F172A"),
            ("#E2E8F0", "#0F172A"),
            ("#CBD5E1", "#1E293B"),
            ("#F8FAFC", "#1E293B"),
            ("#0F172A", "#F8FAFC"),
            ("#475569", "#FFFFFF"),
            ("#1D4ED8", "#FFFFFF"),
            ("#B45309", "#FFFFFF"),
            ("#B91C1C", "#FFFFFF"),
        ]
        for fg, bg in text_pairs:
            self.assertGreaterEqual(_contrast_ratio(fg, bg), 4.5, f"Contrast failed for {fg} on {bg}")

        self.assertGreaterEqual(_contrast_ratio("#64748B", "#FFFFFF"), 3.0)


if __name__ == "__main__":
    unittest.main()

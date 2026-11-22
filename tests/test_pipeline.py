import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

import app

ROOT_DIR = Path(__file__).resolve().parent.parent


def _relative_luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(c1, c2):
    l1, l2 = sorted([_relative_luminance(c1), _relative_luminance(c2)], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


class TestTransactionAnomalyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ROOT_DIR / "data" / "data_clustering_clean.csv")
        cls.scaler_km = joblib.load(ROOT_DIR / "models" / "scaler_clustering.joblib")
        cls.kmeans = joblib.load(ROOT_DIR / "models" / "kmeans_model.joblib")
        cls.anomaly_bundle = joblib.load(ROOT_DIR / "models" / "anomaly_bundle.joblib")

    def test_clean_dataset_integrity(self):
        self.assertEqual(len(self.df), 2516)
        self.assertEqual(set(self.df["Cluster"].unique()), {0, 1})
        required_cols = [
            "Days_Since_Prev_Tx",
            "Amount_Per_Second",
            "Anomaly_Score",
            "Anomaly_Votes",
            "High_Login_Risk",
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)
        self.assertEqual(int(self.df["High_Login_Risk"].sum()), 93)
        self.assertEqual(int((self.df["Anomaly_Votes"] >= 2).sum()), 70)
        self.assertTrue(((self.df["Anomaly_Score"] >= 0) & (self.df["Anomaly_Score"] <= 100)).all())

    def test_models_match_features_and_csv(self):
        self.assertEqual(list(self.scaler_km.feature_names_in_), app.CLUSTERING_FEATURES)
        self.assertEqual(self.kmeans.n_clusters, 2)
        self.assertEqual(self.anomaly_bundle["features"], app.ANOMALY_FEATURES)
        self.assertEqual(self.anomaly_bundle["pca_n_components"], 6)

        km_scaled = pd.DataFrame(
            self.scaler_km.transform(self.df[app.CLUSTERING_FEATURES]),
            columns=app.CLUSTERING_FEATURES,
        )
        self.assertTrue((self.kmeans.predict(km_scaled) == self.df["Cluster"].to_numpy()).all())

    def test_predict_transaction_normal_and_anomaly(self):
        normal = app.predict_transaction(
            tx_amount=250.0,
            acc_balance=7500.0,
            tx_duration=115,
            cust_age=56,
            login_attempts=1,
            days_since_prev_tx=490.0,
            scaler_km=self.scaler_km,
            kmeans=self.kmeans,
            anomaly_bundle=self.anomaly_bundle,
        )
        self.assertEqual(normal["cluster"], 1)
        self.assertFalse(normal["is_high_login_risk"])
        self.assertEqual(normal["anomaly_votes"], 0)
        self.assertTrue(0.0 <= normal["anomaly_score"] <= 100.0)

        extreme = app.predict_transaction(
            tx_amount=1450.0,
            acc_balance=800.0,
            tx_duration=35,
            cust_age=40,
            login_attempts=4,
            days_since_prev_tx=320.0,
            scaler_km=self.scaler_km,
            kmeans=self.kmeans,
            anomaly_bundle=self.anomaly_bundle,
        )
        self.assertTrue(extreme["is_high_login_risk"])
        self.assertGreaterEqual(extreme["anomaly_votes"], 2)
        self.assertGreater(extreme["anomaly_score"], normal["anomaly_score"])

    def test_predict_transaction_zero_balance_edge_case(self):
        result = app.predict_transaction(
            tx_amount=100.0,
            acc_balance=0.0,
            tx_duration=0,
            cust_age=30,
            login_attempts=2,
            days_since_prev_tx=450.0,
            scaler_km=self.scaler_km,
            kmeans=self.kmeans,
            anomaly_bundle=self.anomaly_bundle,
        )
        self.assertEqual(result["balance_ratio"], 0.0)
        self.assertEqual(result["amount_per_second"], 100.0)
        self.assertFalse(result["is_high_login_risk"])

    def test_build_summary_table_normal_and_empty(self):
        df = app.load_data()
        summary = app.build_summary_table(df)
        self.assertEqual(len(summary), 2)
        self.assertEqual(int(summary["Jumlah_Transaksi"].sum()), 2516)

        empty_summary = app.build_summary_table(df.iloc[0:0])
        self.assertEqual(len(empty_summary), 2)
        self.assertTrue((empty_summary["Persentase (%)"] == 0.0).all())

    def test_writing_consistency_antislop_and_contrast(self):
        self.assertFalse((ROOT_DIR / "notebooks" / "classification.ipynb").exists())
        self.assertFalse((ROOT_DIR / "models" / "rf_classifier_pipeline.joblib").exists())

        config_text = (ROOT_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn('primaryColor = "#1D4ED8"', config_text)

        text_files = [
            ROOT_DIR / "app.py",
            ROOT_DIR / "README.md",
            ROOT_DIR / "notebooks" / "clustering.ipynb",
            ROOT_DIR / "tests" / "test_pipeline.py",
        ]
        for file_path in text_files:
            content = file_path.read_text(encoding="utf-8")
            self.assertNotIn("\u2014", content, f"Em dash found in {file_path.name}")
            self.assertNotIn("\u2013", content, f"En dash found in {file_path.name}")

        for file_path in [ROOT_DIR / "app.py", ROOT_DIR / "README.md", ROOT_DIR / "notebooks" / "clustering.ipynb"]:
            content = file_path.read_text(encoding="utf-8")
            self.assertIn("Transaction Anomaly Dashboard", content)
            self.assertNotIn("rf_classifier_pipeline", content)

        app_text = (ROOT_DIR / "app.py").read_text(encoding="utf-8")
        for slop_marker in ["hero-badge", "CREATED BY", "linear-gradient"]:
            self.assertNotIn(slop_marker, app_text)

        nb = json.loads((ROOT_DIR / "notebooks" / "clustering.ipynb").read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [])

        # WCAG AA text (>= 4.5:1) and non-text border (>= 3.0:1) checks
        text_pairs = [
            ("#F8FAFC", "#0F172A"),
            ("#CBD5E1", "#0F172A"),
            ("#0F172A", "#F8FAFC"),
            ("#475569", "#FFFFFF"),
            ("#1D4ED8", "#FFFFFF"),
            ("#B45309", "#FFFFFF"),
            ("#B91C1C", "#FFFFFF"),
            ("#1E3A8A", "#DBEAFE"),
        ]
        for fg, bg in text_pairs:
            self.assertGreaterEqual(_contrast_ratio(fg, bg), 4.5, f"Contrast failed for {fg} on {bg}")

        self.assertGreaterEqual(_contrast_ratio("#64748B", "#FFFFFF"), 3.0)


if __name__ == "__main__":
    unittest.main()

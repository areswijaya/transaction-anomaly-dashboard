import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

import app

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestTransactionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ROOT_DIR / "data" / "data_clustering_clean.csv")
        cls.scaler = joblib.load(ROOT_DIR / "models" / "scaler_clustering.joblib")
        cls.kmeans = joblib.load(ROOT_DIR / "models" / "kmeans_model.joblib")

    def test_clean_dataset_integrity(self):
        self.assertEqual(len(self.df), 2516)
        self.assertEqual(set(self.df["Cluster"].unique()), {0, 1})
        self.assertIn("High_Login_Risk", self.df.columns)
        expected_risk = self.df["LoginAttempts"] >= app.LOGIN_RISK_THRESHOLD
        self.assertTrue((self.df["High_Login_Risk"] == expected_risk).all())
        self.assertEqual(int(self.df["High_Login_Risk"].sum()), 93)

    def test_kmeans_models_and_features_match_csv(self):
        self.assertEqual(list(self.scaler.feature_names_in_), app.CLUSTERING_FEATURES)
        self.assertNotIn("LoginAttempts", app.CLUSTERING_FEATURES)
        self.assertEqual(self.kmeans.n_clusters, 2)

        scaled = pd.DataFrame(
            self.scaler.transform(self.df[app.CLUSTERING_FEATURES]),
            columns=app.CLUSTERING_FEATURES,
        )
        predicted_clusters = self.kmeans.predict(scaled)
        self.assertTrue((predicted_clusters == self.df["Cluster"].to_numpy()).all())

    def test_predict_transaction_segments_and_risk_threshold(self):
        young_active = app.predict_transaction(
            tx_amount=350.0,
            acc_balance=1200.0,
            tx_duration=120,
            cust_age=25,
            login_attempts=1,
            scaler=self.scaler,
            kmeans=self.kmeans,
        )
        self.assertEqual(young_active["cluster"], 0)
        self.assertFalse(young_active["is_high_login_risk"])
        self.assertEqual(set(young_active["centroid_distances"].keys()), set(app.CLUSTER_COLORS.keys()))

        conservative = app.predict_transaction(
            tx_amount=250.0,
            acc_balance=7500.0,
            tx_duration=115,
            cust_age=56,
            login_attempts=2,
            scaler=self.scaler,
            kmeans=self.kmeans,
        )
        self.assertEqual(conservative["cluster"], 1)
        self.assertFalse(conservative["is_high_login_risk"])

        high_risk = app.predict_transaction(
            tx_amount=250.0,
            acc_balance=7500.0,
            tx_duration=115,
            cust_age=56,
            login_attempts=3,
            scaler=self.scaler,
            kmeans=self.kmeans,
        )
        self.assertEqual(high_risk["cluster"], 1)
        self.assertTrue(high_risk["is_high_login_risk"])

    def test_predict_transaction_zero_balance_edge_case(self):
        result = app.predict_transaction(
            tx_amount=100.0,
            acc_balance=0.0,
            tx_duration=90,
            cust_age=30,
            login_attempts=1,
            scaler=self.scaler,
            kmeans=self.kmeans,
        )
        self.assertEqual(result["balance_ratio"], 0.0)
        self.assertIn(result["cluster"], {0, 1})

    def test_build_summary_table_normal_and_empty(self):
        df = app.load_data()
        summary = app.build_summary_table(df)
        self.assertEqual(len(summary), 2)
        self.assertEqual(int(summary["Jumlah_Transaksi"].sum()), 2516)

        empty_summary = app.build_summary_table(df.iloc[0:0])
        self.assertEqual(len(empty_summary), 2)
        self.assertTrue((empty_summary["Persentase (%)"] == 0.0).all())

    def test_circular_artifacts_and_stale_claims_removed(self):
        self.assertFalse((ROOT_DIR / "notebooks" / "classification.ipynb").exists())
        self.assertFalse((ROOT_DIR / "models" / "rf_classifier_pipeline.joblib").exists())

        stale_tokens = ["rf_classifier_pipeline", "classification.ipynb", "97.82", "0.2657"]
        files_to_check = [
            ROOT_DIR / "app.py",
            ROOT_DIR / "README.md",
            ROOT_DIR / "notebooks" / "clustering.ipynb",
        ]
        for file_path in files_to_check:
            content = file_path.read_text(encoding="utf-8")
            for token in stale_tokens:
                self.assertNotIn(token, content, f"Found stale token '{token}' in {file_path.name}")

        nb = json.loads((ROOT_DIR / "notebooks" / "clustering.ipynb").read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()

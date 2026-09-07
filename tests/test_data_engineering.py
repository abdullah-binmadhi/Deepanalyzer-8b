"""Unit tests for DeepAnalyze Automated Data Engineering Engine."""

import unittest
from unittest.mock import patch
import polars as pl
from deepanalyze.data_engineering import (
    apply_quick_features,
    build_engineering_briefing,
    profile_engineering_opportunities,
    stitch_code_with_local_model,
)


class TestDataEngineeringEngine(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            "order_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
            "amount": [100.0, 250.0, 15.0, 8000.0, 320.0],
            "customer_tier": ["Gold", "Silver", "Gold", "Bronze", "Silver"],
            "feedback": ["Fast shipping", "Good quality", "Late arrival", "Broken box", "Exceptional service"]
        })

    def test_profile_engineering_opportunities(self):
        opps = profile_engineering_opportunities(self.df)
        self.assertIn("temporal", opps)
        self.assertIn("numerical", opps)
        self.assertIn("categorical", opps)
        self.assertIn("text", opps)

        # Check temporal detection
        self.assertTrue(any(item["column"] == "order_date" for item in opps["temporal"]))
        # Check numerical detection
        self.assertTrue(any(item["column"] == "amount" for item in opps["numerical"]))
        # Check categorical detection
        self.assertTrue(any(item["column"] == "customer_tier" for item in opps["categorical"]))

    def test_apply_quick_features_polars(self):
        transformed_df, new_cols = apply_quick_features(self.df)
        self.assertGreater(len(new_cols), 0)
        self.assertIn("order_date_weekday", transformed_df.columns)
        self.assertIn("amount_zscore", transformed_df.columns)
        self.assertIn("customer_tier_freq", transformed_df.columns)
        self.assertIn("feedback_char_len", transformed_df.columns)
        self.assertGreater(transformed_df.width, self.df.width)

    def test_build_engineering_briefing(self):
        briefing = build_engineering_briefing(self.df, dataset_name="orders", user_goal="churn prediction")
        self.assertIn("# Feature Engineering & Predictive Enhancement Briefing: orders", briefing)
        self.assertIn("churn prediction", briefing)
        self.assertIn("order_date", briefing)
        self.assertIn("Polars", briefing)

    @patch("deepanalyze.client.request_model_fix")
    def test_stitch_code_with_local_model(self, mock_fix):
        mock_fix.return_value = (
            "Aligned column 'OrderDate' to 'order_date'",
            "df = df.with_columns(pl.col('order_date').alias('dt'))"
        )
        frontier_code = "df['OrderDate'] = pd.to_datetime(df['OrderDate'])"
        success, stitched, diag = stitch_code_with_local_model(frontier_code, self.df)
        self.assertTrue(success)
        self.assertIn("pl.col('order_date')", stitched)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for DeepAnalyze Frontier API Gateway and Pre-Flight DLP Airlock."""

import unittest
from deepanalyze.frontier import detect_available_providers, extract_code_blocks, preflight_dlp_check


class TestFrontierAirlock(unittest.TestCase):

    def test_preflight_dlp_blocks_national_id(self):
        unsafe_prompt = "Please clean this table. User National ID is 1098765432 and name is Omar."
        is_safe, violations = preflight_dlp_check(unsafe_prompt)
        self.assertFalse(is_safe)
        self.assertTrue(any("National ID" in v for v in violations))

    def test_preflight_dlp_blocks_email_and_credit_card(self):
        unsafe_prompt = "User email is test@company.com and card is 4111 2222 3333 4444."
        is_safe, violations = preflight_dlp_check(unsafe_prompt)
        self.assertFalse(is_safe)
        self.assertTrue(any("Email" in v for v in violations))
        self.assertTrue(any("Credit Card" in v for v in violations))

    def test_preflight_dlp_permits_sanitized_prompt(self):
        safe_prompt = """
        # Autonomous Engineering Briefing: Customer_Data
        Columns: [cust_id, age_bracket, region_code, tx_amount_bin]
        Types: {cust_id: Utf8, age_bracket: Categorical, tx_amount_bin: Utf8}
        Sample Row: {"cust_id": "SYN_CUST_101", "age_bracket": "30-39", "region_code": "REG_WEST"}
        Objective: Impute missing values and compute standard snake_case column names.
        """
        is_safe, violations = preflight_dlp_check(safe_prompt)
        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)

    def test_extract_code_blocks(self):
        llm_response = """
        Here is the Polars transformation script:
        ```python
        import polars as pl
        df = df.with_columns(pl.col("age") + 1)
        ```
        Let me know if you need anything else!
        """
        code = extract_code_blocks(llm_response)
        self.assertIn("df = df.with_columns", code)
        self.assertNotIn("Here is the Polars", code)


if __name__ == "__main__":
    unittest.main()

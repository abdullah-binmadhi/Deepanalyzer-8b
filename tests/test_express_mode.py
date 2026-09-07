"""Unit tests for DeepAnalyze Express Clean Mode."""

import unittest
from unittest.mock import patch
import polars as pl
from deepanalyze.wizard import AirGapWizard


class TestExpressMode(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            "age": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            "city": ["Riyadh"] * 5 + ["Jeddah"] * 5,
            "salary": [10000, 15000, 20000, 25000, 30000, 10000, 15000, 20000, 25000, 30000]
        })
        self.wizard = AirGapWizard()

    @patch("deepanalyze.wizard.Prompt.ask")
    @patch("deepanalyze.wizard.read_multiline_input")
    @patch("deepanalyze.wizard.save_prompt_to_disk", return_value="/tmp/test_prompt.md")
    @patch("deepanalyze.wizard.copy_to_clipboard", return_value=True)
    def test_express_clean_mode_flow(
        self, mock_copy, mock_save, mock_multiline, mock_ask
    ):
        prompts_seen = []

        def dynamic_ask(prompt_text, **kwargs):
            p = str(prompt_text).lower()
            prompts_seen.append(p)
            if "will code be provided" in p:
                return "Y"
            if "select delivery" in p:
                return "1"
            if "press enter to audit" in p:
                return ""
            if "satisfied with these transformation results" in p:
                return "Y"
            if "export the final cleaned dataset" in p:
                return "N"
            return "1"

        mock_ask.side_effect = dynamic_ask
        mock_multiline.side_effect = ["import polars as pl\ndf = df.with_columns(pl.col('age') + 1)\n", ""]

        # Run with mode="express"
        res = self.wizard.run(self.df, mode="express")
        self.assertIsNotNone(res)

        # In express mode, manual configuration questions are bypassed, but the encrypted duplicate prompt MUST be offered before Step 9:
        self.assertFalse(any("operating from" in p for p in prompts_seen), "Country question should be skipped in Express mode")
        self.assertFalse(any("governing framework" in p for p in prompts_seen), "Statute question should be skipped in Express mode")
        self.assertFalse(any("dataset structure" in p for p in prompts_seen), "Architecture question should be skipped in Express mode")
        self.assertFalse(any("more columns" in p for p in prompts_seen), "Teaching loop should be skipped in Express mode")
        self.assertFalse(any("modify or add instructions" in p for p in prompts_seen), "Prompt editor should be skipped in Express mode")
        self.assertTrue(any("download encrypted dataset duplicate" in p for p in prompts_seen), "Duplicate prompt must be offered before Step 9 in Express mode")

        # Step 9 satisfaction check must have been reached
        self.assertTrue(any("satisfied with these transformation results" in p for p in prompts_seen))


if __name__ == "__main__":
    unittest.main()

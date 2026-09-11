"""Tests for Wizard universal back navigation and benchmark gating."""

import unittest
from unittest.mock import patch
import polars as pl
from deepanalyze.wizard import AirGapWizard


class TestWizardEnhancements(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "age": [25, 25, 25, 25, 25, 40, 40, 40, 40, 40],
            "city": ["Riyadh"] * 5 + ["Jeddah"] * 5,
            "salary": [10000, 15000, 20000, 25000, 30000, 10000, 15000, 20000, 25000, 30000]
        })
        self.wizard = AirGapWizard()

    @patch("deepanalyze.wizard.Prompt.ask")
    @patch("deepanalyze.wizard.read_multiline_input")
    @patch("deepanalyze.wizard.save_prompt_to_disk", return_value="/tmp/test_prompt.md")
    @patch("deepanalyze.wizard.copy_to_clipboard", return_value=True)
    def test_back_navigation_and_uncleaned_warning(
        self, mock_copy, mock_save, mock_multiline, mock_ask
    ):
        step7_calls = 0
        step7_5_calls = 0
        step9_calls = 0

        def dynamic_ask(prompt_text, **kwargs):
            nonlocal step7_calls, step7_5_calls, step9_calls
            p = str(prompt_text).lower()
            if "select mode" in p or "execution mode" in p:
                return "2"
            if "country" in p or "origin" in p:
                return "5"
            if "statute" in p or "framework" in p:
                return "1"
            if "architecture" in p:
                return "1"
            if "more columns" in p:
                step7_calls += 1
                if step7_calls == 1:
                    return "b"  # test back at step 7
                return "N"      # proceed
            if "business requests" in p or "column extraction" in p:
                step7_5_calls += 1
                if step7_5_calls == 1:
                    return "b"  # test back to step 7
                return "N"      # proceed
            if "modify or add instructions" in p:
                return "N"
            if "select action" in p and "refinement" not in p and "deepanalyze" not in p:
                return "3"
            if "encrypted dataset duplicate" in p:
                return "n"
            if "will code be provided" in p:
                step9_calls += 1
                if step9_calls == 1:
                    return "b"  # test back to duplicate
                return "N"
            if "export the final cleaned dataset" in p:
                return "n"
            return "1"

        mock_ask.side_effect = dynamic_ask
        res = self.wizard.run(self.df)
        self.assertIsNotNone(res)
        self.assertGreaterEqual(step7_calls, 2)
        self.assertGreaterEqual(step7_5_calls, 2)
        self.assertGreaterEqual(step9_calls, 2)

    @patch("deepanalyze.wizard.Prompt.ask")
    @patch("deepanalyze.wizard.read_multiline_input")
    @patch("deepanalyze.wizard.save_prompt_to_disk", return_value="/tmp/test_prompt.md")
    @patch("deepanalyze.wizard.copy_to_clipboard", return_value=True)
    def test_option1_auto_generalize_on_benchmark_fail(
        self, mock_copy, mock_save, mock_multiline, mock_ask
    ):
        df_vuln = pl.DataFrame({
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "city": ["NYC"] * 10,
            "salary": [50000 + i * 1000 for i in range(10)]
        })
        option1_selected = False

        def dynamic_ask(prompt_text, **kwargs):
            nonlocal option1_selected
            p = str(prompt_text).lower()
            if "select mode" in p or "execution mode" in p:
                return "2"
            if "country" in p or "origin" in p:
                return "5"
            if "statute" in p or "framework" in p:
                return "1"
            if "architecture" in p:
                return "1"
            if "more columns" in p:
                return "N"
            if "business requests" in p:
                return "N"
            if "modify or add instructions" in p:
                return "N"
            if "select action" in p:
                if not option1_selected:
                    option1_selected = True
                    return "1"  # Auto-generalize
                return "3"  # Abort if another benchmark fails
            if "encrypted dataset duplicate" in p:
                return "n"
            if "will code be provided" in p:
                return "N"
            if "export the final cleaned dataset" in p:
                return "n"
            return "1"

        mock_ask.side_effect = dynamic_ask
        res = self.wizard.run(df_vuln)
        self.assertIsNotNone(res)
        self.assertTrue(option1_selected)

    @patch("deepanalyze.wizard.Prompt.ask")
    @patch("deepanalyze.wizard.read_multiline_input")
    @patch("deepanalyze.wizard.save_prompt_to_disk", return_value="/tmp/test_prompt.md")
    @patch("deepanalyze.wizard.copy_to_clipboard", return_value=True)
    @patch("deepanalyze.wizard.check_model_health", return_value=True)
    def test_step9_error_prompt_choices(
        self, mock_health, mock_copy, mock_save, mock_multiline, mock_ask
    ):
        deepanalyze_prompt_seen = False

        def dynamic_ask(prompt_text, **kwargs):
            nonlocal deepanalyze_prompt_seen
            p = str(prompt_text).lower()
            if "select mode" in p or "execution mode" in p:
                return "2"
            if "country" in p or "origin" in p:
                return "5"
            if "statute" in p or "framework" in p:
                return "1"
            if "architecture" in p:
                return "1"
            if "more columns" in p:
                return "N"
            if "business requests" in p:
                return "N"
            if "modify or add instructions" in p:
                return "N"
            if "select action" in p and "auto-repair" not in p:
                return "3"
            if "encrypted dataset duplicate" in p:
                return "n"
            if "will code be provided" in p:
                return "y"
            if "select delivery format" in p:
                return "1"
            if "press enter" in p:
                return ""
            if "deepanalyze 8b is online" in p:
                deepanalyze_prompt_seen = True
                return "3"  # Quit action
            if "export the final cleaned dataset" in p:
                return "n"
            return "1"

        mock_ask.side_effect = dynamic_ask
        mock_multiline.side_effect = ["this is invalid python syntax !!!", ""]

        res = self.wizard.run(self.df)
        self.assertIsNotNone(res)
        self.assertTrue(deepanalyze_prompt_seen)


    @patch("deepanalyze.wizard.Prompt.ask")
    @patch("deepanalyze.wizard.read_multiline_input")
    @patch("deepanalyze.wizard.save_prompt_to_disk", return_value="/tmp/test_prompt.md")
    @patch("deepanalyze.wizard.copy_to_clipboard", return_value=True)
    @patch("deepanalyze.wizard.render_three_way_airlock_inspection")
    def test_step9_three_way_inspection_and_satisfaction(
        self, mock_render_three_way, mock_copy, mock_save, mock_multiline, mock_ask
    ):
        satisfied_prompt_seen = False

        def dynamic_ask(prompt_text, **kwargs):
            nonlocal satisfied_prompt_seen
            p = str(prompt_text).lower()
            if "select mode" in p or "execution mode" in p:
                return "2"
            if "country" in p or "origin" in p:
                return "5"
            if "statute" in p or "framework" in p:
                return "1"
            if "architecture" in p:
                return "1"
            if "more columns" in p:
                return "N"
            if "business requests" in p:
                return "N"
            if "modify or add instructions" in p:
                return "N"
            if "select action" in p and "refinement" not in p and "block" not in p and "deepanalyze" not in p:
                return "3"
            if "encrypted dataset duplicate" in p:
                return "n"
            if "will code be provided" in p:
                return "y"
            if "select delivery format" in p:
                return "1"
            if "press enter" in p:
                return ""
            if "satisfied with these transformation results" in p:
                satisfied_prompt_seen = True
                return "Y"
            if "export the final cleaned dataset" in p:
                return "N"
            return "1"

        mock_ask.side_effect = dynamic_ask
        mock_multiline.side_effect = ["import polars as pl\ndf = df.with_columns(pl.col('age') + 1)\n", ""]

        res = self.wizard.run(self.df)
        self.assertIsNotNone(res)
        self.assertTrue(satisfied_prompt_seen)
        self.assertTrue(mock_render_three_way.called)


    @patch("deepanalyze.wizard.Prompt.ask")
    @patch("deepanalyze.wizard.read_multiline_input")
    @patch("deepanalyze.wizard.save_prompt_to_disk", return_value="/tmp/test_prompt.md")
    @patch("deepanalyze.wizard.copy_to_clipboard", return_value=True)
    @patch("deepanalyze.wizard.render_three_way_airlock_inspection")
    def test_step9_dissatisfaction_refinement_flow(
        self, mock_render_three_way, mock_copy, mock_save, mock_multiline, mock_ask
    ):
        sat_count = 0
        refine_prompt_seen = False

        def dynamic_ask(prompt_text, **kwargs):
            nonlocal sat_count, refine_prompt_seen
            p = str(prompt_text).lower()
            if "select mode" in p or "execution mode" in p:
                return "2"
            if "country" in p or "origin" in p:
                return "5"
            if "statute" in p or "framework" in p:
                return "1"
            if "architecture" in p:
                return "1"
            if "more columns" in p:
                return "N"
            if "business requests" in p:
                return "N"
            if "modify or add instructions" in p:
                return "N"
            if "select action" in p and "refinement" not in p and "block" not in p and "deepanalyze" not in p:
                return "3"
            if "encrypted dataset duplicate" in p:
                return "n"
            if "will code be provided" in p:
                return "y"
            if "select delivery format" in p:
                return "1"
            if "press enter" in p:
                return ""
            if "satisfied with these transformation results" in p:
                sat_count += 1
                if sat_count == 1:
                    return "N"
                return "Y"
            if "select refinement action" in p:
                refine_prompt_seen = True
                return "2"  # Paste corrected script manually
            if "export the final cleaned dataset" in p:
                return "N"
            return "1"

        mock_ask.side_effect = dynamic_ask
        mock_multiline.side_effect = [
            "import polars as pl\ndf = df.with_columns(pl.col('age') + 1)\n",
            "import polars as pl\ndf = df.with_columns(pl.col('age') + 5)\n",
            ""
        ]

        res = self.wizard.run(self.df)
        self.assertIsNotNone(res)
        self.assertEqual(sat_count, 2)
        self.assertTrue(refine_prompt_seen)
        self.assertEqual(mock_render_three_way.call_count, 2)


    @patch("platform.system", return_value="Windows")
    @patch("subprocess.Popen")
    def test_copy_to_clipboard_windows_secure_subprocess(self, mock_popen, mock_platform):
        from deepanalyze.wizard import copy_to_clipboard
        mock_proc = unittest.mock.MagicMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        with patch.dict("sys.modules", {"pyperclip": None}):
            res = copy_to_clipboard("test text")
            self.assertTrue(res)
            mock_popen.assert_called_once_with(["clip"], stdin=-1)


if __name__ == "__main__":
    unittest.main()



"""Test Suite for Rich Markup Safety & Syntax Highlighting in DeepAnalyze."""

import io
import unittest
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


class TestRichMarkupSafety(unittest.TestCase):

    def setUp(self):
        self.output = io.StringIO()
        self.console = Console(file=self.output, force_terminal=False, color_system=None)

    def test_regex_bracket_markup_safety_in_code_panel(self):
        r"""Validates that code containing bracket-heavy regex like [/|\-] renders without MarkupError."""
        problematic_code = (
            "import re\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "# Delimited composite string structure decomposition\n"
            "df['Location'] = df['Location'].astype(str).str.split(r'[/|\\-]')\n"
            "df['All'] = df['All'].astype(str).str.split(r'[/|\\-]')\n"
            "pattern = r'[a-zA-Z0-9_+]+[/@#]' \n"
        )
        # Should render cleanly without raising MarkupError
        panel = Panel(
            Syntax(problematic_code, "python", theme="monokai", line_numbers=True),
            title="Incoming Script Preview",
            border_style="cyan"
        )
        self.console.print(panel)
        rendered = self.output.getvalue()
        self.assertIn("Incoming Script Preview", rendered)
        self.assertIn("str.split", rendered)

    def test_diagnosis_and_patched_code_group_safety(self):
        """Validates that diagnosis strings and synthesized code containing tags render safely."""
        diag = r"Error encountered at tag [/|\-] during parsing: unmatched bracket [idx=0]."
        patched_code = "df['clean'] = df['raw'].str.replace(r'[/|\\-]', '_', regex=True)"

        diag_esc = escape(str(diag))
        panel = Panel(
            Group(
                Text.from_markup(f"[bold cyan]Diagnosis:[/bold cyan] {diag_esc}\n\n[bold green]Patched Code Synthesized:[/bold green]"),
                Syntax(patched_code, "python", theme="monokai", line_numbers=True)
            ),
            title="DeepAnalyze 8B Autonomous Diagnosis",
            border_style="cyan"
        )
        self.console.print(panel)
        rendered = self.output.getvalue()
        self.assertIn("Diagnosis", rendered)
        self.assertIn("Patched Code Synthesized", rendered)
        self.assertIn("[/|\\-]", rendered)

    def test_execution_error_panel_escaping(self):
        """Validates that execution errors with tracebacks and regex patterns don't trigger MarkupError."""
        simulated_err = "ValueError: Pattern r'[/|\\-]' failed to compile with tag [/tag]."
        simulated_repair = "Try changing r'[/|\\-]' to r'[\\/\\|\\-]'"

        err_esc = escape(str(simulated_err))
        rep_esc = escape(str(simulated_repair))

        panel = Panel(
            Text.from_markup(
                f"[bold red]Execution Error:[/bold red]\n{err_esc}\n\n"
                f"[bold cyan]Ouroboros Self-Healing Autopsy & Repair Prompt:[/bold cyan]\n"
                f"{rep_esc}"
            ),
            title="Ouroboros Self-Healing Airlock",
            border_style="red"
        )
        self.console.print(panel)
        rendered = self.output.getvalue()
        self.assertIn("Execution Error", rendered)
        self.assertIn("Ouroboros Self-Healing Autopsy", rendered)


if __name__ == "__main__":
    unittest.main()

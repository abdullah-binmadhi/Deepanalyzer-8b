"""Unit tests for DeepAnalyze Interactive Terminal Cockpit."""

import unittest
import polars as pl
from rich.table import Table
from deepanalyze.cockpit import build_kpi_cards, render_column_shift_table, launch_interactive_terminal_cockpit


class TestTerminalCockpit(unittest.TestCase):

    def setUp(self):
        self.raw_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", None],
            "age": [25, None, 45]
        })
        self.clean_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Unknown"],
            "age": [25, 30, 45],
            "age_zscore": [-0.8, -0.2, 1.0]
        })

    def test_build_kpi_cards(self):
        kpi_grid = build_kpi_cards(self.raw_df, self.clean_df)
        self.assertIsInstance(kpi_grid, Table)

    def test_render_column_shift_table(self):
        table = render_column_shift_table(self.raw_df, self.clean_df)
        self.assertIsInstance(table, Table)
        # Should include columns from both raw and clean
        self.assertIn("Column-Level Transformation & Hygiene Shifts", table.title)


if __name__ == "__main__":
    unittest.main()

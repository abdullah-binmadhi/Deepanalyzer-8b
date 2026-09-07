"""Unit tests for DeepAnalyze Textual Interactive Cockpit TUI."""

import asyncio
import unittest
import polars as pl
from deepanalyze.cockpit_tui import DeepAnalyzeCockpitApp


class TestCockpitTUI(unittest.TestCase):

    def setUp(self):
        self.raw_df = pl.DataFrame({
            "Customer": ["Alpha Corp", "Beta LLC", "Gamma Co"],
            "Invoice ID": ["INV-1001", "INV-1002", "INV-1003"],
            "Amount": [12500.0, 4300.5, 980.0]
        })

    def test_app_instantiation(self):
        app = DeepAnalyzeCockpitApp(raw_df=self.raw_df)
        self.assertEqual(app.TITLE, "DeepAnalyze Interactive Cockpit")
        self.assertEqual(app.raw_df.shape, (3, 3))

    def test_app_pilot_render(self):
        async def run_pilot():
            app = DeepAnalyzeCockpitApp(raw_df=self.raw_df)
            async with app.run_test() as pilot:
                self.assertIsNotNone(pilot.app.query_one("#dt-original"))
                self.assertIsNotNone(pilot.app.query_one("#dt-encrypted"))
                self.assertIsNotNone(pilot.app.query_one("#dt-cleaned"))
                self.assertIsNotNone(pilot.app.query_one("#firewall-log"))
                self.assertIsNotNone(pilot.app.query_one("#code-viewer"))
                self.assertIsNotNone(pilot.app.query_one("#brain-visualizer"))
                self.assertIsNotNone(pilot.app.query_one("#ouroboros-log"))
                self.assertIsNotNone(pilot.app.query_one("#ouroboros-input"))
                self.assertIsNotNone(pilot.app.query_one("#spark-kanon"))
                self.assertIsNotNone(pilot.app.query_one("#spark-ldiv"))
                self.assertIsNotNone(pilot.app.query_one("#preflight-table"))

        asyncio.run(run_pilot())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for DeepAnalyze Textual Interactive Cockpit TUI."""

import asyncio
import unittest
import polars as pl
from deepanalyze.cockpit_tui import DeepAnalyzeCockpitApp


class TestCockpitTUI(unittest.TestCase):

    def setUp(self):
        self.raw_df = pl.DataFrame({
            "Customer": ["Alpha Corp", "Beta LLC", "Gamma Co", "Delta Ltd", "Epsilon Inc"],
            "Invoice ID": ["INV-1001", "INV-1002", "INV-1003", "INV-1004", "INV-1005"],
            "Amount": [12500.0, 4300.5, 980.0, 5400.0, 150.0],
            "Region": ["Riyadh", "Jeddah", "Dammam", "Riyadh", "Mecca"],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        })

    def test_app_instantiation(self):
        app = DeepAnalyzeCockpitApp(raw_df=self.raw_df, dataset_name="TestDataset")
        self.assertEqual(app.TITLE, "DEEPANALYZE INTERACTIVE COCKPIT")
        self.assertEqual(app.raw_df.shape, (5, 5))

    def test_app_pilot_navigation_and_pages(self):
        async def run_pilot():
            app = DeepAnalyzeCockpitApp(raw_df=self.raw_df, dataset_name="TestDataset")
            async with app.run_test(size=(160, 50)) as pilot:
                # 1. Check Data Airlock (initial)
                self.assertEqual(app.active_page_id, "page-airlock")
                self.assertIsNotNone(pilot.app.query_one("#dt-original"))
                self.assertIsNotNone(pilot.app.query_one("#dt-encrypted"))
                self.assertIsNotNone(pilot.app.query_one("#dt-cleaned"))
                self.assertIsNotNone(pilot.app.query_one("#dt-benchmarks"))
                self.assertIsNotNone(pilot.app.query_one("#dt-cartography"))

                # 2. Switch to Wizard & Governance
                await pilot.click("#tab-wizard")
                self.assertEqual(app.active_page_id, "page-wizard")
                self.assertIsNotNone(pilot.app.query_one("#wiz-steps-list"))
                self.assertIsNotNone(pilot.app.query_one("#wiz-desc"))

                # 3. Switch to Topology
                await pilot.click("#tab-topology")
                self.assertEqual(app.active_page_id, "page-topology")
                self.assertIsNotNone(pilot.app.query_one("#topo-log"))
                self.assertIsNotNone(pilot.app.query_one("#dt-topology-audit"))

                # 4. Switch to Analytics Studio
                await pilot.click("#tab-studio")
                self.assertEqual(app.active_page_id, "page-studio")
                self.assertIsNotNone(pilot.app.query_one("#chart-display-log"))
                self.assertIsNotNone(pilot.app.query_one("#btn-render-chart"))

                # 5. Check Presets
                await pilot.click("#pre-corr")
                await pilot.click("#pre-pareto")
                await pilot.click("#pre-waterfall")
                await pilot.click("#pre-dist")

                # 6. Test F1 Auto-Remedy All
                await pilot.click("#btn-f1-remedy")
                self.assertTrue(len(app.clean_df) > 0)

        asyncio.run(run_pilot())


if __name__ == "__main__":
    unittest.main()

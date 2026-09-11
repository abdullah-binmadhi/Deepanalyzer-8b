"""Unit tests for DeepAnalyze Textual Interactive Cockpit TUI."""

import asyncio
import unittest
import polars as pl
from textual.widgets import DataTable
from deepanalyze.cockpit_tui import DeepAnalyzeCockpitApp, CloudPromptModal, CleaningRecipeModal


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

    def test_cloud_prompt_modal_no_language_error(self):
        modal = CloudPromptModal(prompt_text="# Markdown briefing text\nSome instructions.")
        self.assertEqual(modal.prompt_text, "# Markdown briefing text\nSome instructions.")

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

                # Check bottom dock container (buttons above footer ribbon)
                dock = pilot.app.query_one("#bottom-dock-container")
                self.assertIsNotNone(dock)
                self.assertIsNotNone(pilot.app.query_one("#export-bar"))
                self.assertIsNotNone(pilot.app.query_one("#btn-f1-remedy"))
                self.assertIsNotNone(pilot.app.query_one("#btn-f2-prompt"))

                # 2. Switch to Wizard & Governance
                await pilot.click("#tab-wizard")
                self.assertEqual(app.active_page_id, "page-wizard")
                self.assertIsNotNone(pilot.app.query_one("#wiz-steps-list"))
                self.assertIsNotNone(pilot.app.query_one("#wiz-desc"))

                # 3. Switch to Topology (full terminal ER schema and data lineage)
                await pilot.click("#tab-topology")
                self.assertEqual(app.active_page_id, "page-topology")
                topo_log = pilot.app.query_one("#topo-log")
                self.assertIsNotNone(topo_log)
                self.assertIsNotNone(pilot.app.query_one("#dt-topology-audit"))

                # 4. Switch to Analytics Studio (Multi-chart dynamic canvas)
                await pilot.click("#tab-studio")
                self.assertEqual(app.active_page_id, "page-studio")
                grid = pilot.app.query_one("#multi-chart-grid")
                self.assertIsNotNone(grid)
                self.assertEqual(len(app.dashboard_charts), 4)

                # 5. Add a 5th chart dynamically
                await pilot.click("#btn-add-chart")
                self.assertEqual(len(app.dashboard_charts), 5)

                # 6. Delete a chart card via dynamic button
                del_btn_id = f"#btn-del-{app.dashboard_charts[0]['id']}"
                await pilot.click(del_btn_id)
                self.assertEqual(len(app.dashboard_charts), 4)

                # 7. Layout mode toggle
                await pilot.click("#btn-layout-1")
                await pilot.pause()
                self.assertTrue("grid-layout-1" in grid.classes)
                await pilot.click("#btn-layout-2")
                await pilot.pause()
                self.assertTrue("grid-layout-4" in grid.classes)

                # 8. Test F1 Auto-Remedy All
                await pilot.click("#btn-f1-remedy")
                self.assertTrue(len(app.clean_df) > 0)

        asyncio.run(run_pilot())

    def test_dynamic_topology_population_multi_sheet(self):
        async def run_multi_sheet():
            multi_sheets = {
                "Invoices": pl.DataFrame({
                    "invoice_id": ["INV-1", "INV-2"],
                    "customer_id": ["C1", "C2"],
                    "total": [100.0, 200.0]
                }),
                "Customers": pl.DataFrame({
                    "customer_id": ["C1", "C2"],
                    "name": ["Acme Corp", "Beta LLC"]
                })
            }
            app = DeepAnalyzeCockpitApp(
                raw_df=multi_sheets["Invoices"],
                multi_sheets=multi_sheets,
                dataset_name="MultiSheetTest"
            )
            async with app.run_test(size=(160, 50)) as pilot:
                await pilot.click("#tab-topology")
                table = pilot.app.query_one("#dt-topology-audit", DataTable)
                # Table rows should reflect actual sheets
                row_names = [table.get_row_at(r)[0] for r in range(table.row_count)]
                self.assertIn("INVOICES", row_names)
                self.assertIn("CUSTOMERS", row_names)
                self.assertNotIn("Orders / Main", row_names)
                self.assertNotIn("Products_Ref", row_names)

        asyncio.run(run_multi_sheet())

    def test_dynamic_topology_population_hierarchical_single_sheet(self):
        async def run_hierarchical():
            erp_df = pl.DataFrame({
                "doc_no": ["IV-100", "IV-100"],
                "doc_date": ["2025-08-01", "2025-08-01"],
                "Sequence": [1000, 2000],
                "Item Amount": [50.0, 75.0]
            })
            app = DeepAnalyzeCockpitApp(raw_df=erp_df, dataset_name="ERP_Test")
            async with app.run_test(size=(160, 50)) as pilot:
                await pilot.click("#tab-topology")
                table = pilot.app.query_one("#dt-topology-audit", DataTable)
                row_names = [table.get_row_at(r)[0] for r in range(table.row_count)]
                self.assertTrue(any("DOCUMENT_MASTER" in r for r in row_names))
                self.assertTrue(any("LINE_ITEMS" in r for r in row_names))
                self.assertNotIn("Orders / Main", row_names)
                self.assertNotIn("Products_Ref", row_names)

        asyncio.run(run_hierarchical())

    def test_cleaning_recipe_modal_standalone(self):
        modal = CleaningRecipeModal(raw_df=self.raw_df, dataset_name="TestDataset")
        self.assertIn("Power Query", modal.pq_recipe)
        self.assertIn("Python", modal.py_recipe)

    def test_app_pilot_recipes_modal(self):
        async def run_recipes_pilot():
            app = DeepAnalyzeCockpitApp(raw_df=self.raw_df, dataset_name="TestDataset")
            async with app.run_test(size=(160, 50)) as pilot:
                # Click F6 recipes button
                await pilot.click("#btn-f6-recipes")
                # Modal should be open
                self.assertIsInstance(app.screen, CleaningRecipeModal)
                # Toggle tabs
                await pilot.click("#btn-tab-py")
                self.assertEqual(app.screen.active_tab, "py")
                await pilot.click("#btn-tab-pq")
                self.assertEqual(app.screen.active_tab, "pq")
                # Close modal
                await pilot.click("#btn-close-recipe-modal")
                self.assertNotIsInstance(app.screen, CleaningRecipeModal)

        asyncio.run(run_recipes_pilot())


if __name__ == "__main__":
    unittest.main()

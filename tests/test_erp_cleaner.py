"""Unit tests for DeepAnalyze Autonomous ERP Cleaner & Guided Recipes."""

import unittest
import pandas as pd
import polars as pl
from deepanalyze.erp_cleaner import (
    detect_ragged_erp,
    flatten_hierarchical_erp,
    generate_powerquery_recipe,
    generate_python_recipe,
)


class TestERPCleaner(unittest.TestCase):

    def setUp(self):
        # Synthetic ragged ERP dataset mimicking West Malayan / Autocount / SAP exports
        self.ragged_data = [
            # Top report noise (Archetype D)
            ["WEST MALAYAN TRADING CORP", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            ["INVOICE LISTING AS AT 31/08/2025", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            ["Doc. No", "Code", "Date", "Description", "Cust Code", "Customer", "Name", None, None, None, "Qty", "UOM", "Price", "Amount", None, "Total"],
            # Invoice 1 (IV-1001) - Archetype A Header
            ["IV-1001", None, "2025-08-01", None, "300-A01", None, "ALPHA SUPERMARKET", None, None, None, None, None, None, None, None, "1,500.00"],
            # Line 1: Sequence 1000 with multi-line wrap (Archetype B)
            ["1000", "500-000", None, "PREMIUM JASMINE RICE 10KG", None, None, None, None, None, None, "10", "BAG", "100.00", "1,000.00", None, None],
            [None, None, None, "BATCH #8891 EXPIRY: 2026-12", None, None, None, None, None, None, None, None, None, None, None, None],
            # Line 2: Sequence 2000
            ["2000", "500-000", None, "ORGANIC COOKING OIL 5L", None, None, None, None, None, None, "5", "BTL", "100.00", "500.00", None, None],
            # Invoice 1 Subtotal / Noise (Archetype D)
            ["Sub Total:", None, None, None, None, None, None, None, None, None, None, None, None, "1,500.00", None, None],
            # Invoice 2 (IV-1002) - Archetype A Header
            ["IV-1002", None, "2025-08-02", None, "300-B02", None, "BETA HYPERMART", None, None, None, None, None, None, None, None, "300.00"],
            # Line 1: Sequence 1000
            ["1000", "500-000", None, "MINERAL WATER 500ML X 24", None, None, None, None, None, None, "20", "CTN", "15.00", "300.00", None, None],
            # Bottom Noise
            ["Grand Total:", None, None, None, None, None, None, None, None, None, None, None, None, "1,800.00", None, None],
            ["Page 1 of 1", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        ]
        self.raw_pdf = pd.DataFrame(self.ragged_data)
        self.raw_pl = pl.from_pandas(self.raw_pdf)

    def test_detect_ragged_erp_positive(self):
        is_ragged, reason = detect_ragged_erp(self.raw_pl)
        self.assertTrue(is_ragged)
        self.assertIn("Hierarchical ERP", reason)

    def test_detect_ragged_erp_negative(self):
        clean_df = pl.DataFrame({
            "order_id": [1, 2, 3, 4],
            "customer": ["A", "B", "C", "D"],
            "amount": [10.0, 20.0, 30.0, 40.0]
        })
        is_ragged, reason = detect_ragged_erp(clean_df)
        self.assertFalse(is_ragged)

    def test_flatten_hierarchical_erp_structure(self):
        clean_df = flatten_hierarchical_erp(self.raw_pl)
        self.assertIsInstance(clean_df, pl.DataFrame)
        
        # Should have exactly 3 line items (2 for IV-1001, 1 for IV-1002)
        self.assertEqual(clean_df.height, 3)

        # Expected canonical columns
        expected_cols = [
            "Sequence", "GL-Code", "Quantity", "UOM", "Unit Price", "Item Amount",
            "doc_no", "doc_date", "customer_code", "customer_name", "invoice_total", "Full_Description"
        ]
        for col in expected_cols:
            self.assertIn(col, clean_df.columns)

        # Check Zero Null guarantee across all columns
        for col in clean_df.columns:
            self.assertEqual(clean_df[col].null_count(), 0, f"Column '{col}' had unexpected nulls")

        # Test Multi-line description wrap concatenation (Archetype B)
        item_1 = clean_df.filter(pl.col("doc_no") == "IV-1001").row(0, named=True)
        self.assertEqual(item_1["doc_no"], "IV-1001")
        self.assertEqual(item_1["customer_code"], "300-A01")
        self.assertEqual(item_1["customer_name"], "ALPHA SUPERMARKET")
        self.assertEqual(item_1["invoice_total"], 1500.0)
        self.assertEqual(item_1["Quantity"], 10.0)
        self.assertEqual(item_1["Unit Price"], 100.0)
        self.assertEqual(item_1["Item Amount"], 1000.0)
        # Description should contain both line 1 and line 2
        self.assertIn("PREMIUM JASMINE RICE 10KG", item_1["Full_Description"])
        self.assertIn("BATCH #8891 EXPIRY: 2026-12", item_1["Full_Description"])

        # Test Second Invoice
        item_3 = clean_df.filter(pl.col("doc_no") == "IV-1002").row(0, named=True)
        self.assertEqual(item_3["doc_no"], "IV-1002")
        self.assertEqual(item_3["customer_name"], "BETA HYPERMART")
        self.assertEqual(item_3["Quantity"], 20.0)
        self.assertEqual(item_3["Unit Price"], 15.0)
        self.assertEqual(item_3["Item Amount"], 300.0)

    def test_generate_powerquery_recipe_content(self):
        recipe = generate_powerquery_recipe(self.raw_pl, dataset_name="SalesLedger")
        self.assertIn("Power Query", recipe)
        self.assertIn("Table.FillDown", recipe)
        # Guarantees caution against Table.Skip
        self.assertIn("Table.Skip", recipe)
        self.assertIn("doc_no", recipe)

    def test_generate_python_recipe_content(self):
        py_recipe = generate_python_recipe(self.raw_pl, dataset_name="SalesLedger")
        self.assertIn("STATE-MACHINE PATTERN", py_recipe.upper())
        self.assertIn("def clean_erp_report", py_recipe)
    def test_flatten_arbitrary_sap_oracle_layout(self):
        # Different 7-column layout with different column names and structure
        sap_data = [
            ['Company Code: 1000', None, None, None, None, None, None],
            ['Invoice No', 'Invoice Date', 'Customer', 'Line No', 'Description', 'Qty', 'Amount'],
            ['INV-99001', '2025-09-01', 'DEUTSCHLAND GMBH', None, None, None, '5,000.00'],
            [None, None, None, '10', 'INDUSTRIAL BEARING 40MM', '50', '2,500.00'],
            [None, None, None, None, 'EXTENDED WARRANTY 12M', None, None],
            [None, None, None, '20', 'HYDRAULIC VALVE SEAL', '100', '2,500.00'],
            ['Grand Total', None, None, None, None, None, '5,000.00']
        ]
        sap_df = pd.DataFrame(sap_data)
        clean_sap = flatten_hierarchical_erp(sap_df)
        self.assertEqual(clean_sap.height, 2)
        row0 = clean_sap.row(0, named=True)
        self.assertEqual(row0["doc_no"], "INV-99001")
        self.assertEqual(row0["customer_name"], "DEUTSCHLAND GMBH")
        self.assertEqual(row0["Sequence"], 10)
        self.assertEqual(row0["Quantity"], 50.0)
        self.assertEqual(row0["Item Amount"], 2500.0)
        # Verify wrap
        self.assertIn("INDUSTRIAL BEARING 40MM", row0["Full_Description"])
        self.assertIn("EXTENDED WARRANTY 12M", row0["Full_Description"])


if __name__ == "__main__":
    unittest.main()

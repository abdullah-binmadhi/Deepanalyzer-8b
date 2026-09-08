"""Tests for Multi-Column Excel Ingestion, Power Query Dual-Track, and ERP Transformation."""

import os
import pytest
import pandas as pd
import polars as pl

from deepanalyze.wizard import ingest_file
from deepanalyze.powerquery import generate_powerquery_m_code, generate_powerquery_step_by_step_guide
from deepanalyze.transformer import clean_unflattened_invoice_erp


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ERP_CANDIDATES = [
    os.path.join(BASE_DIR, "Testing files", "INV LISTING 31082025 copy.xlsx"),
    os.path.join(BASE_DIR, "INV LISTING 31082025 copy.xlsx"),
]
TARGET_ERP_CANDIDATES = [
    os.path.join(BASE_DIR, "Testing files", "INV LISTING 31082025 (Cleaned) copy.xlsx"),
    os.path.join(BASE_DIR, "INV LISTING 31082025 (Cleaned) copy.xlsx"),
]
RAW_ERP = next((p for p in RAW_ERP_CANDIDATES if os.path.exists(p)), RAW_ERP_CANDIDATES[0])
TARGET_ERP = next((p for p in TARGET_ERP_CANDIDATES if os.path.exists(p)), TARGET_ERP_CANDIDATES[0])


def test_excel_ingest_preserves_all_columns():
    """Validates that unflattened ERP spreadsheets retain all 16 columns instead of 3."""
    if not os.path.exists(RAW_ERP):
        pytest.skip("Test ERP spreadsheet not found.")

    df = ingest_file(RAW_ERP)
    assert df.height == 3924
    assert df.width == 16
    assert "0" in df.columns
    assert "15" in df.columns


def test_erp_transformation_exact_fidelity():
    """Validates schema match, zero data loss (retaining IV-11319), and numeric fidelity."""
    if not os.path.exists(RAW_ERP) or not os.path.exists(TARGET_ERP):
        pytest.skip("Test ERP spreadsheets not found.")

    target = pd.read_excel(TARGET_ERP)
    cleaned = clean_unflattened_invoice_erp(RAW_ERP)

    # Validate that early invoice IV-11319 is preserved (yielding 1893 rows vs 1892 in lossy target)
    assert len(cleaned) == 1893
    assert "IV-11319" in cleaned["doc_no"].values
    assert list(cleaned.columns) == list(target.columns)

    # For all invoices present in target, verify exact numeric and text fidelity
    cleaned_subset = cleaned[cleaned["doc_no"].isin(target["doc_no"])].reset_index(drop=True)
    target_subset = target.sort_values(by="invoice_total", ascending=False, kind="mergesort").reset_index(drop=True)
    cleaned_subset = cleaned_subset.sort_values(by="invoice_total", ascending=False, kind="mergesort").reset_index(drop=True)

    for col in ["Quantity", "Unit Price", "Item Amount", "invoice_total"]:
        diff = (cleaned_subset[col] - target_subset[col]).abs().max()
        assert diff < 1e-4, f"Numeric diff in {col}: {diff}"

    assert (cleaned_subset["doc_no"] == target_subset["doc_no"]).all()
    assert (cleaned_subset["customer_code"] == target_subset["customer_code"]).all()
    # Verify that multi-line wrapped descriptions were properly concatenated rather than truncated
    assert (cleaned_subset["Full_Description"].str.len() >= target_subset["Full_Description"].str.len()).all()


def test_powerquery_m_code_generation():
    """Validates that Power Query M-code contains all necessary transformation steps dynamically."""
    m_code = generate_powerquery_m_code("/test/path/erp.xlsx", "Report")
    assert m_code.startswith("let")
    assert 'Excel.Workbook(File.Contents("/test/path/erp.xlsx"), null, true)' in m_code
    assert "Table.SelectRows" in m_code
    assert 'Text.StartsWith([Column1], "IV-")' in m_code
    assert "Table.FillDown" in m_code
    assert "Table.SelectColumns" in m_code
    assert "Table.RenameColumns" in m_code
    assert "invoice_total" in m_code


def test_powerquery_guide_generation():
    """Validates that the step-by-step guide contains both quick and manual UI instructions."""
    guide = generate_powerquery_step_by_step_guide("Test_ERP.xlsx")
    assert "Method 1: The 60-Second Copy-Paste (Recommended)" in guide
    assert "Method 2: Click-by-Click Manual UI Walkthrough" in guide
    assert "Advanced Editor" in guide
    assert "Conditional Column" in guide
    assert "Fill Down" in guide

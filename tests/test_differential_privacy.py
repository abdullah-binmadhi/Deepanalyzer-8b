"""Tests for Differential Privacy Synthetic Mock Generator."""

import polars as pl
import pytest
from deepanalyze.sentinel import generate_synthetic_mock


def test_differential_privacy_mock_generation():
    df = pl.DataFrame({
        "revenue": [1000.0, 2000.0, 1500.0, 1800.0, 1200.0],
        "profit_margin": [0.15, 0.25, 0.18, 0.22, 0.19],
        "patient_age": [35, 45, 52, 61, 28]
    })

    mocks = generate_synthetic_mock(df, n_rows=5)
    assert len(mocks) == 5

    # Verify column presence
    for row in mocks:
        assert "revenue" in row
        assert "profit_margin" in row
        assert "patient_age" in row
        # Must be non-negative
        assert row["revenue"] >= 0
        assert row["patient_age"] >= 18

    # Ensure synthetic values do not exactly copy any proprietary record
    mock_revs = [r["revenue"] for r in mocks]
    orig_revs = df["revenue"].to_list()
    # At least some difference guaranteed by DP noise
    assert mock_revs != orig_revs


def test_structural_erp_mock_generation():
    """Validates dynamic 4-row structural mock generation for hierarchical ERP datasets."""
    from deepanalyze.sentinel import generate_structural_erp_mock

    erp_df = pl.DataFrame({
        "Date": ["Doc. No", "IV-11319", "Seq", "1000", "Wrapped line part 2"],
        " : ": [None, None, "GL Code", "500-000", None],
        "Value": ["Doc. Date", "2025-08-01", None, "14,520.00", None]
    })

    mock_rows = generate_structural_erp_mock(erp_df)
    assert len(mock_rows) == 4

    # Row 0: Master Header (contains doc id and date)
    assert any("DOC" in str(v) or "IV" in str(v) for v in mock_rows[0].values() if v)
    assert any("2026" in str(v) for v in mock_rows[0].values() if v)

    # Row 1: Line Item (contains sequence 1000 and item description)
    assert any(str(v) == "1000" for v in mock_rows[1].values() if v)
    assert any("Primary Line Item Description" in str(v) for v in mock_rows[1].values() if v)

    # Row 2: Wrapped Continuation (description text only, seq is None)
    assert any("wrapped specification details" in str(v) for v in mock_rows[2].values() if v)

    # Row 3: Line Item 2 (contains sequence 2000)
    assert any(str(v) == "2000" for v in mock_rows[3].values() if v)

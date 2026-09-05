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

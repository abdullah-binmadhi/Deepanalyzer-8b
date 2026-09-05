"""Tests for Real-Time Data Diff & Quality Scorecard."""

import pandas as pd
import polars as pl
import pytest
from deepanalyze.scorecard import generate_quality_scorecard, render_quality_scorecard


def test_quality_scorecard_calculation():
    raw_df = pd.DataFrame({
        "Patient Name ": ["Alice", "Bob", "Bob", None],
        "Age": [30, 40, 40, None],
        "Phone Number": ["123", "456", "456", None]
    })
    clean_df = pd.DataFrame({
        "patient_name": ["Alice", "Bob"],
        "age": [30.0, 40.0],
        "phone_number": ["(123) 000-0000", "(456) 000-0000"]
    })

    card = generate_quality_scorecard(raw_df, clean_df)
    assert card.raw_rows == 4
    assert card.clean_rows == 2
    assert card.rows_diff == -2
    assert card.duplicates_removed >= 1
    assert card.clean_null_count == 0
    assert card.null_reduction_pct == 100.0
    assert card.cleanliness_score >= 90
    assert card.standardized_column_names_pct == 100.0


def test_render_quality_scorecard():
    raw_df = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    clean_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    card = generate_quality_scorecard(raw_df, clean_df)
    table = render_quality_scorecard(card)
    assert table is not None
    assert table.title == "Data Transformation & Quality Scorecard"

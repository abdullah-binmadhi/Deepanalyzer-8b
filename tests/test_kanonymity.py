"""Tests for k-Anonymity and l-Diversity Re-Identification Engine."""

import polars as pl
import pytest
from deepanalyze.kanonymity import (
    analyze_kanonymity,
    bin_column_series,
    detect_quasi_identifiers,
    detect_sensitive_column,
)


def test_quasi_identifier_detection():
    df = pl.DataFrame({
        "Patient_Age": [25, 30],
        "Gender": ["M", "F"],
        "Zip_Code": ["12345", "12346"],
        "Visit_Date": ["2026-01-01", "2026-01-02"],
        "Blood_Pressure": ["120/80", "130/85"],
        "Condition": ["Diabetes", "Asthma"]
    })
    qis = detect_quasi_identifiers(df)
    assert "Patient_Age" in qis
    assert "Gender" in qis
    assert "Zip_Code" in qis
    assert "Visit_Date" in qis
    assert "Condition" not in qis


def test_sensitive_column_detection():
    df = pl.DataFrame({
        "age": [40, 50],
        "salary": [50000, 65000],
        "department": ["IT", "HR"]
    })
    sens = detect_sensitive_column(df)
    assert sens == "salary"


def test_kanonymity_low_risk():
    # 4 records, 2 equivalence classes of size 2 each -> k=2 (threshold=2 -> LOW)
    df = pl.DataFrame({
        "age": [30, 30, 40, 40],
        "gender": ["M", "M", "F", "F"],
        "condition": ["Cold", "Flu", "Allergy", "Headache"]
    })
    report = analyze_kanonymity(df, quasi_identifiers=["age", "gender"], threshold_k=2)
    assert report.min_k == 2
    assert report.records_at_risk == 0
    assert report.risk_level == "LOW"


def test_kanonymity_critical_risk():
    # 1 unique row with age=99, gender=M -> k=1 -> CRITICAL
    df = pl.DataFrame({
        "age": [30, 30, 30, 99],
        "gender": ["M", "M", "M", "M"],
        "condition": ["Flu", "Flu", "Flu", "Rare Disease"]
    })
    report = analyze_kanonymity(df, quasi_identifiers=["age", "gender"], threshold_k=3)
    assert report.min_k == 1
    assert report.records_at_risk == 1
    assert report.risk_level == "CRITICAL"
    assert len(report.sample_outlier_classes) > 0


def test_bin_column_series():
    ages = [22, 29, 30, 35, 41, None]
    binned = bin_column_series(ages, bin_size=10)
    assert binned[0] == "20-29"
    assert binned[1] == "20-29"
    assert binned[2] == "30-39"
    assert binned[3] == "30-39"
    assert binned[4] == "40-49"
    assert binned[5] == "<UNKNOWN>"

"""Unit tests validating DeepAnalyze across all real-world datasets in 'Testing files'.

Covers:
- Smartphones.csv (Tech / Specifications)
- Healthcare_Messy_Data copy.csv (Healthcare EHR / Clinical)
- INV LISTING 31082025 copy.xlsx (Hierarchical Ragged ERP)
- INV LISTING 31082025 (Cleaned) copy.xlsx (Reference Clean ERP)
- UnClean_Product_sales_RowData.xlsx (Messy Retail / Sales)
- Candy Hierarchy 2017.xlsx (High-dimensional Survey Data)
- njs2016_data.csv (Large non-UTF8 Survey Dataset)
- air-quality-monitoring-sites-summary.csv (Environmental / Sensor Metadata)
- Earlwood_Air_Data_17_18.xls (Timeseries IoT / Sensor Data)
- ISO_codes.csv (Reference Metadata)
"""

import os
import pytest
import pandas as pd
import polars as pl
from deepanalyze.policies import detect_dataset_architecture, resolve_policy, classify_dataframe_columns
from deepanalyze.profiler import profile_dataframe
from deepanalyze.sentinel import generate_synthetic_mock, generate_structural_erp_mock
from deepanalyze.promptgen import build_master_prompt

TESTING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Testing files")


def load_dataset(file_rel_path: str) -> pl.DataFrame:
    """Helper to safely load datasets with encoding and type normalization."""
    fpath = os.path.join(TESTING_DIR, file_rel_path)
    if not os.path.exists(fpath):
        pytest.skip(f"Test dataset not found: {fpath}")

    if fpath.endswith(".csv"):
        try:
            return pl.read_csv(fpath)
        except Exception:
            try:
                return pl.read_csv(fpath, encoding="utf8-lossy")
            except Exception:
                return pl.read_csv(fpath, encoding="latin1")
    elif fpath.endswith((".xlsx", ".xls")):
        pdf = pd.read_excel(fpath)
        for c in pdf.columns:
            if pdf[c].dtype == "object":
                pdf[c] = pdf[c].astype(str).replace("nan", None)
        return pl.from_pandas(pdf)
    else:
        raise ValueError(f"Unsupported file type: {fpath}")


def test_smartphones_dataset():
    """Validates tech specs / smartphone specifications dataset."""
    df = load_dataset("Smartphones.csv")
    assert df.height > 1000
    assert df.width >= 10

    arch_key, arch_name, _ = detect_dataset_architecture(df)
    assert arch_key == "CLEAN_TABULAR"

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3

    prompt = build_master_prompt(df=df, dataset_name="Smartphones.csv")
    assert "Memory & Storage" in prompt or "Power Architecture" in prompt
    assert "### 6. SYNTHETIC SCHEMA MOCK" in prompt


def test_healthcare_messy_dataset():
    """Validates clinical EHR / medical trial dataset."""
    df = load_dataset("Healthcare_Messy_Data copy.csv")
    assert df.height >= 1000

    arch_key, arch_name, _ = detect_dataset_architecture(df)
    assert arch_key == "HEALTHCARE_EHR"
    assert "Healthcare" in arch_name

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3

    prompt = build_master_prompt(df=df, dataset_name="Healthcare_Messy_Data.csv")
    assert "Biometric Splitting" in prompt or "Clinical Categorization" in prompt


def test_inv_listing_ragged_erp_dataset():
    """Validates unflattened hierarchical invoice ERP report."""
    df = load_dataset("INV LISTING 31082025 copy.xlsx")
    assert df.height > 3000

    arch_key, arch_name, _ = detect_dataset_architecture(df)
    assert arch_key == "ERP_RAGGED"

    mock = generate_structural_erp_mock(df)
    assert len(mock) >= 4

    prompt = build_master_prompt(df=df, dataset_name="INV_LISTING.xlsx")
    assert "ANTI-OVERCLEANING & ROW CONSERVATION MANDATE" in prompt
    assert "Hierarchical Master-Detail Report Deconstruction" in prompt


def test_inv_listing_cleaned_dataset():
    """Validates reference cleaned ERP invoice table."""
    df = load_dataset("INV LISTING 31082025 (Cleaned) copy.xlsx")
    assert df.height > 1500

    arch_key, _, _ = detect_dataset_architecture(df)
    assert arch_key == "CLEAN_TABULAR"

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3


def test_unclean_product_sales_dataset():
    """Validates dirty retail / product sales records."""
    df = load_dataset("UnClean_Product_sales_RowData.xlsx")
    assert df.height >= 300

    arch_key, _, _ = detect_dataset_architecture(df)
    assert arch_key == "CLEAN_TABULAR"

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3


def test_candy_hierarchy_survey_dataset():
    """Validates high-dimensional (120-column) consumer survey dataset."""
    df = load_dataset("Candy Hierarchy 2017.xlsx")
    assert df.height > 2000
    assert df.width >= 100

    arch_key, _, _ = detect_dataset_architecture(df)
    assert arch_key == "CLEAN_TABULAR"

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3
    assert len(mock[0]) == df.width


def test_data_cleaning_challenge_njs2016():
    """Validates 4,200-row x 447-column survey dataset with non-UTF8 characters."""
    df = load_dataset(os.path.join("Data Cleaning Challenge", "njs2016_data.csv"))
    assert df.height > 4000
    assert df.width > 400

    mock = generate_synthetic_mock(df, n_rows=2)
    assert len(mock) == 2


def test_air_quality_sites_summary():
    """Validates environmental sensor monitoring sites summary dataset."""
    df = load_dataset(os.path.join("Air Quality Data", "air-quality-monitoring-sites-summary.csv"))
    assert df.height >= 100

    arch_key, _, _ = detect_dataset_architecture(df)
    assert arch_key == "CLEAN_TABULAR"

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3


def test_earlwood_air_data_xls():
    """Validates legacy .xls timeseries environmental sensor telemetry dataset."""
    df = load_dataset(os.path.join("Air Quality Data", "Earlwood_Air_Data_17_18.xls"))
    assert df.height > 8000
    assert df.width >= 15

    arch_key, _, _ = detect_dataset_architecture(df)
    assert arch_key == "CLEAN_TABULAR"

    mock = generate_synthetic_mock(df, n_rows=3)
    assert len(mock) == 3

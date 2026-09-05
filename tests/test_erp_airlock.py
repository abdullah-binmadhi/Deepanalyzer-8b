"""Unit and Integration Tests for DeepAnalyze v4.0 Zero-Code ERP Airlock."""

import os
import shutil
import tempfile
import polars as pl
import pytest

from deepanalyze.policies import (
    detect_dataset_architecture,
    detect_statute_for_country,
    get_statute_options_for_country,
    resolve_policy,
)
from deepanalyze.sentinel import (
    get_masked_pattern_summary,
    mask_structural_erp,
)
from deepanalyze.vault import (
    detokenize_dataframe,
    flush,
    get_vault_stats,
    learn_custom_pattern,
)
from deepanalyze.firewall import (
    append_code_to_pipeline,
    audit_code,
    create_pipeline_file,
    execute_code_safely,
)


def test_statute_options_and_auto_detect():
    """Validates country-specific statute options and auto-detection fallback."""
    # Saudi Arabia
    sa_opts = get_statute_options_for_country("Saudi Arabia")
    assert "Not Sure (Auto-Detect)" in sa_opts
    assert any("PDPL" in opt for opt in sa_opts)
    sa_detect = detect_statute_for_country("Saudi Arabia")
    assert "PDPL" in sa_detect

    # Poland
    pl_opts = get_statute_options_for_country("Poland")
    assert "Not Sure (Auto-Detect)" in pl_opts
    pl_detect = detect_statute_for_country("Poland")
    assert "GDPR" in pl_detect or "Personal Data" in pl_detect

    # United States
    us_opts = get_statute_options_for_country("United States")
    assert any("HIPAA" in opt for opt in us_opts)
    us_detect = detect_statute_for_country("United States")
    assert "HIPAA" in us_detect or "CCPA" in us_detect


def test_dataset_architecture_detection():
    """Validates auto-detection of ragged ERP vs clean tabular."""
    # Clean tabular
    clean_df = pl.DataFrame({
        "customer_id": [1, 2, 3],
        "sales_amount": [100.5, 200.0, 350.75],
        "region": ["North", "South", "East"]
    })
    key, name, _ = detect_dataset_architecture(clean_df)
    assert key == "CLEAN_TABULAR"

    # Ragged ERP matrix with colon markers and headers
    erp_df = pl.DataFrame({
        "Date": ["Document", "Company", "Doc. No", "IV-11319", "Seq", "1000"],
        " : ": [" : ", " : ", None, None, "GL Code", "500-000"],
        "Value": ["All", "All", "Doc. Date", "2025-08-01", None, None]
    })
    key, name, exp = detect_dataset_architecture(erp_df)
    assert key == "ERP_RAGGED"
    assert "Hierarchical / Ragged ERP" in name


def test_erp_masking_and_pattern_summary():
    """Validates structural geometric masking and pattern extraction."""
    erp_df = pl.DataFrame({
        "Date": ["WEST MALAYAN GROUP SDN BHD (202101004803 (1405102-U))", "Doc. No", "IV-11319", "Seq", "1000"],
        " : ": [None, None, None, "GL Code", "500-000"],
        "Value": [None, "Doc. Date", "2025-08-01 00:00:00", None, "14,520.00"]
    })
    masked = mask_structural_erp(erp_df)

    # Anchors preserved
    assert masked["Date"][1] == "Doc. No"
    assert masked[" : "][3] == "GL Code"
    assert masked["Value"][1] == "Doc. Date"

    # Invoice ID masked
    assert masked["Date"][2] == "XX-99999"

    # GL code masked
    assert masked[" : "][4] == "999-999"

    # Company name masked
    assert "WEST" not in masked["Date"][0]
    assert "XXXX" in masked["Date"][0]

    # Pattern summary extraction
    summary = get_masked_pattern_summary(erp_df, masked)
    assert len(summary) >= 3
    cats = [s["category"] for s in summary]
    assert any("Document" in c or "Codes" in c for c in cats)


def test_dynamic_pattern_teaching_and_detokenization():
    """Validates interactive teaching with custom example values."""
    flush()
    df = pl.DataFrame({
        "meta": ["Account: 500-000", "Account: 500-001", "Seq: 1000", "Seq: 2000"]
    })

    # Teach GL Code
    pat_gl, m_df1 = learn_custom_pattern("GL Code", "500-000", df)
    assert r"\d{3}-\d{3}" in pat_gl
    assert "<GL_CODE_1>" in m_df1["meta"][0]
    assert "<GL_CODE_2>" in m_df1["meta"][1]

    # Teach Sequence
    pat_seq, m_df2 = learn_custom_pattern("Seq", "1000", m_df1)
    assert "<SEQ_1>" in m_df2["meta"][2]

    # Exact detokenization back
    restored = detokenize_dataframe(m_df2)
    assert restored["meta"].to_list() == df["meta"].to_list()


def test_pipeline_file_generation_py_and_ipynb():
    """Validates creation and appending for both .py scripts and .ipynb notebooks."""
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Python script
        py_path = create_pipeline_file(temp_dir, "py")
        assert os.path.exists(py_path)
        assert py_path.endswith(".py")
        append_code_to_pipeline(py_path, "x = 42\ny = x * 2")
        with open(py_path, "r", encoding="utf-8") as f:
            py_content = f.read()
        assert "x = 42" in py_content

        # 2. Jupyter Notebook
        nb_path = create_pipeline_file(temp_dir, "ipynb")
        assert os.path.exists(nb_path)
        assert nb_path.endswith(".ipynb")
        append_code_to_pipeline(nb_path, "import polars as pl\ndf = pl.DataFrame({'a': [1]})")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_content = f.read()
        assert "polars as pl" in nb_content
    finally:
        shutil.rmtree(temp_dir)


def test_real_erp_file_integration():
    """End-to-end verification using the real portfolio ERP spreadsheet."""
    real_path = "/Users/abdullahbinmadhi/Desktop/portfolio projects/INV LISTING 31082025.xlsx"
    if not os.path.exists(real_path):
        pytest.skip("Sample ERP file not found in test environment.")

    # Ingest
    df = pl.read_excel(real_path, engine="openpyxl")
    assert df.height == 2803
    assert df.width == 3

    # Architecture auto-detection
    key, name, _ = detect_dataset_architecture(df)
    assert key == "ERP_RAGGED"

    # Structural masking
    masked_df = mask_structural_erp(df)
    assert masked_df.height == 2803

    # Verify sensitive company name is masked
    raw_company = df["Date"][11]
    masked_company = masked_df["Date"][11]
    assert "WEST MALAYAN" in raw_company
    assert "WEST MALAYAN" not in masked_company
    assert "XXXX" in masked_company

    # Pattern summary
    summary = get_masked_pattern_summary(df, masked_df)
    assert len(summary) >= 4

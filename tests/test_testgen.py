"""Tests for Automated Pytest Pipeline Generator."""

import os
import pandas as pd
import pytest
from deepanalyze.testgen import generate_pipeline_test_code, write_pipeline_test_file


def test_generate_pipeline_test_code():
    clean_df = pd.DataFrame({
        "invoice_id": [1, 2],
        "customer_name": ["Alpha", "Beta"],
        "total_amount": [150.50, 300.00]
    })
    code = generate_pipeline_test_code(
        input_file="/mock/input.csv",
        output_file="/mock/output.csv",
        pipeline_script="/mock/pipeline.py",
        clean_df=clean_df
    )
    assert "def test_source_files_exist():" in code
    assert "def test_pipeline_execution():" in code
    assert "def test_schema_and_column_integrity():" in code
    assert "invoice_id" in code
    assert "customer_name" in code
    assert "total_amount" in code


def test_write_pipeline_test_file(tmp_path):
    out_dir = str(tmp_path)
    file_path = write_pipeline_test_file(
        output_dir=out_dir,
        input_file="/mock/input.csv",
        output_file="/mock/output.csv",
        pipeline_script="/mock/pipeline.py"
    )
    assert os.path.exists(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "test_pipeline_execution" in content

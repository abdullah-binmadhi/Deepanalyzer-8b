"""Unit tests for DeepAnalyze Prompt Generation, Domain Engineering & Refinement Engine."""

import os
import tempfile
import polars as pl
import pytest
from deepanalyze.policies import resolve_policy
from deepanalyze.profiler import profile_dataframe, profile_workbook
from deepanalyze.promptgen import (
    build_master_prompt,
    enrich_prompt_with_local_model,
    infer_domain_feature_engineering,
    save_prompt_to_disk,
)


def test_infer_domain_feature_engineering_tech_specs():
    df = pl.DataFrame({
        "model": ["Phone A", "Phone B"],
        "price": ["₹54,999", "₹19,989"],
        "ram": ["12 GB RAM, 256 GB inbuilt", "6 GB RAM, 128 GB inbuilt"],
        "battery": ["5000 mAh Battery with 100W Fast Charging", "4500 mAh"],
        "camera": ["50 MP Rear, 16 MP Front", "64 MP Rear"],
        "display": ["6.7 inches, 120 Hz", "6.5 inches, 90 Hz"],
    })
    features = infer_domain_feature_engineering(df)
    assert any("Memory & Storage" in f for f in features)
    assert any("Power Architecture" in f for f in features)
    assert any("Optics Specification" in f for f in features)
    assert any("Display Geometry" in f for f in features)
    assert any("Value Ratios" in f for f in features)


def test_infer_domain_feature_engineering_healthcare():
    df = pl.DataFrame({
        "patient_name": ["John Doe", "Jane Smith"],
        "age": [45, 62],
        "blood_pressure": ["120/80", "140/90"],
        "cholesterol": [180.0, 245.0],
        "visit_date": ["2025-01-15", "2025-02-20"],
    })
    features = infer_domain_feature_engineering(df)
    assert any("Biometric Splitting" in f for f in features)
    assert any("Clinical Categorization" in f for f in features)
    assert any("Demographic Cohort" in f for f in features)
    assert any("Temporal Intelligence" in f for f in features)


def test_build_master_prompt_contains_all_sections():
    df = pl.DataFrame({
        "customer_id": ["C101", "C102"],
        "amount": ["$1,000.00", "(250.00)"],
        "tx_date": ["2025-01-01", "15/02/2025"],
    })
    policy = resolve_policy("Saudi Arabia", "PDPL")
    custom_instructions = "Calculate 15% VAT and flag weekend transactions"

    prompt = build_master_prompt(
        df=df,
        policy=policy,
        user_custom_instructions=custom_instructions,
        dataset_name="Sales_Ledger"
    )

    # Verify all enterprise sections exist
    assert "### SYSTEM ROLE & OBJECTIVE" in prompt
    assert "### 1. DATASET GEOMETRY & COMPLIANCE MASKING SPECIFICATION" in prompt
    assert "### 2. STRUCTURAL REPORT TOPOLOGY & FIELD ANOMALIES" in prompt
    assert "### 3. REQUIRED DATA CLEANING & RECONCILIATION LOGIC" in prompt
    assert "### 4. DOMAIN FEATURE ENGINEERING (AUTOMATIC SPEC EXTRACTIONS)" in prompt
    assert "### 5. USER DOMAIN-SPECIFIC BUSINESS LOGIC & CUSTOM SPECIFICATIONS" in prompt
    assert "### 6. SYNTHETIC SCHEMA MOCK (Laplace DP, 0% Real Production Records)" in prompt
    assert "### 7. CODE OUTPUT & SECURITY CONSTRAINTS" in prompt

    # Verify custom user instructions injected
    assert "Calculate 15% VAT and flag weekend transactions" in prompt

    # Verify DP synthetic mock injected
    assert "```json" in prompt

    # Verify AST firewall constraints mentioned
    assert "AST Firewall Sandbox Restrictions" in prompt


def test_save_prompt_to_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_content = "# MASTER CLEANING PROMPT\nClean and normalize data."
        saved_path = save_prompt_to_disk(
            prompt_content,
            dataset_dir=tmpdir,
            dataset_base_name="Customer_Orders"
        )

        assert os.path.isfile(saved_path)
        assert saved_path.endswith("Customer_Orders_cleaning_prompt.md")
        with open(saved_path, "r", encoding="utf-8") as f:
            read_back = f.read()
        assert read_back == prompt_content


def test_enrich_prompt_with_local_model_offline_graceful():
    prompt = "Base prompt text."
    # Using an unreachable port to simulate offline server
    enriched = enrich_prompt_with_local_model(prompt, server_url="http://127.0.0.1:59999")
    assert enriched == prompt

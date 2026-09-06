"""Tests for DeepAnalyze 11-Test Air-Gap Privacy & Security Benchmark Suite."""

import os
import polars as pl
import pytest

from deepanalyze.benchmarks import (
    eval_canary_injection,
    eval_regex_pii_scanner,
    eval_singling_out_risk,
    eval_l_diversity,
    eval_t_closeness,
    eval_linkability_risk,
    eval_nndr_memorization,
    eval_membership_inference,
    eval_mutual_information,
    eval_ast_firewall_policy,
    eval_reconciliation_exactness,
    run_tier1_preflight,
    run_tier2_deep_audit,
    run_all_benchmarks,
    render_tier1_scorecard_panel,
    render_full_scorecard_panel,
    render_markdown_audit_report
)
from deepanalyze.policies import resolve_policy
from deepanalyze.wizard import create_compliance_audit_certificate


@pytest.fixture
def clean_test_dataset():
    """Returns a synthetic dataset with k >= 5 equivalence classes and diverse values."""
    # 20 rows, 2 distinct equivalence classes of size 10 each
    return pl.DataFrame({
        "age": [25] * 10 + [45] * 10,
        "gender": ["M"] * 10 + ["F"] * 10,
        "department": ["Eng"] * 10 + ["Sales"] * 10,
        "salary": [
            5000.0, 5200.0, 5400.0, 5600.0, 5800.0, 6000.0, 6200.0, 6400.0, 6600.0, 6800.0,
            5000.0, 5200.0, 5400.0, 5600.0, 5800.0, 6000.0, 6200.0, 6400.0, 6600.0, 6800.0
        ],
        "email": [f"user_{i}@company.sa" for i in range(20)]
    })


def test_canary_injection_suite(clean_test_dataset):
    """Test 1: Asserts synthetic high-entropy canaries do not bypass sanitization."""
    policy = resolve_policy("Saudi Arabia", "PDPL")
    m = eval_canary_injection(clean_test_dataset, policy=policy, n_canaries=5)
    assert m.test_number == 1
    assert m.passed is True
    assert m.metric_value == 0.0
    assert m.threshold == "0 Leaks (Exact 0)"


def test_regex_pii_scanner_clean_passes():
    """Test 2: Verifies 0 direct statutory matches on properly masked data."""
    masked_df = pl.DataFrame({
        "name": ["<NAME_1>", "<NAME_2>", "XXXX"],
        "phone": ["<PHONE_1>", "XX-99999", "XXXX"],
        "amount": ["9,999.00", "9,999.00", "9,999.00"]
    })
    m = eval_regex_pii_scanner(masked_df, prompt_text="Schema: 3 columns, amounts: 9,999.00")
    assert m.test_number == 2
    assert m.passed is True
    assert m.metric_value == 0.0


def test_regex_pii_scanner_leak_detected():
    """Test 2: Detects unmasked real email or phone in payload."""
    leaked_prompt = "Customer email is john.doe@classified-defense.gov and phone is 202-555-0199"
    m = eval_regex_pii_scanner(prompt_text=leaked_prompt)
    assert m.passed is False
    assert m.metric_value > 0


def test_singling_out_risk_evaluation(clean_test_dataset):
    """Test 3: Evaluates k-anonymity equivalence class sizing and singling-out risk."""
    m = eval_singling_out_risk(clean_test_dataset, quasi_identifiers=["age", "gender", "department"])
    assert m.test_number == 3
    assert m.passed is True
    assert m.metric_value == 0.0  # 0% singling-out risk


def test_singling_out_risk_outlier_detected():
    """Test 3: Detects when a 1-of-1 unique record can be singled out."""
    outlier_df = pl.DataFrame({
        "age": [30, 30, 30, 30, 99],  # age 99 is 1-of-1
        "gender": ["M", "M", "M", "M", "F"],
        "department": ["IT", "IT", "IT", "IT", "CFO"]
    })
    m = eval_singling_out_risk(outlier_df, quasi_identifiers=["age", "gender", "department"])
    assert m.passed is False
    assert m.metric_value > 0.0


def test_l_diversity_evaluation(clean_test_dataset):
    """Test 4: Verifies minimum distinct sensitive attributes per group."""
    m = eval_l_diversity(clean_test_dataset, quasi_identifiers=["age", "gender"], sensitive_col="salary")
    assert m.test_number == 4
    assert m.passed is True
    assert m.metric_value >= 2.0


def test_l_diversity_homogeneity_detected():
    """Test 4: Detects homogeneity vulnerability when all group records share the same sensitive value."""
    homo_df = pl.DataFrame({
        "age": [30, 30, 30, 40, 40, 40],
        "gender": ["M", "M", "M", "F", "F", "F"],
        "condition": ["Cancer", "Cancer", "Cancer", "Flu", "Flu", "Flu"]
    })
    m = eval_l_diversity(homo_df, quasi_identifiers=["age", "gender"], sensitive_col="condition")
    assert m.passed is False
    assert m.metric_value == 1.0


def test_t_closeness_evaluation(clean_test_dataset):
    """Test 5: Computes Earth Mover's Distance distribution skewness."""
    m = eval_t_closeness(clean_test_dataset, quasi_identifiers=["age", "gender"], sensitive_col="salary")
    assert m.test_number == 5
    assert m.passed is True
    assert m.metric_value <= 0.15


def test_linkability_risk_evaluation(clean_test_dataset):
    """Test 6: Simulates 3-way split auxiliary linkability attack."""
    m = eval_linkability_risk(clean_test_dataset)
    assert m.test_number == 6
    assert m.passed is True
    assert m.metric_value < 0.05


def test_nndr_memorization_evaluation(clean_test_dataset):
    """Test 7: Asserts synthetic mock samples are distinct archetypes (NNDR >= 0.25)."""
    m = eval_nndr_memorization(clean_test_dataset)
    assert m.test_number == 7
    assert m.passed is True
    assert m.metric_value >= 0.25


def test_membership_inference_evaluation(clean_test_dataset):
    """Test 8: Evaluates membership inference attack shadow classifier advantage."""
    m = eval_membership_inference(clean_test_dataset)
    assert m.test_number == 8
    assert m.passed is True
    assert m.metric_value <= 0.75


def test_mutual_information_evaluation(clean_test_dataset):
    """Test 9: Asserts Normalized Mutual Information (NMI) < 0.05 on independent features."""
    m = eval_mutual_information(clean_test_dataset, quasi_identifiers=["gender"], sensitive_col="salary")
    assert m.test_number == 9
    assert m.passed is True
    assert m.metric_value < 0.05


def test_ast_firewall_policy_evaluation():
    """Test 10: Asserts 100% rejection rate for forbidden/malicious AST probes."""
    m = eval_ast_firewall_policy()
    assert m.test_number == 10
    assert m.passed is True
    assert m.metric_value == 100.0


def test_reconciliation_exactness_evaluation(clean_test_dataset):
    """Test 11: Verifies 100.00% character fidelity across round-trip tokenization."""
    policy = resolve_policy("Saudi Arabia", "PDPL")
    m = eval_reconciliation_exactness(clean_test_dataset, policy=policy)
    assert m.test_number == 11
    assert m.passed is True
    assert m.metric_value == 100.0


def test_tier1_preflight_latency_and_scorecard(clean_test_dataset):
    """Tier 1 Pre-Flight Gate executes in < 15 ms in volatile RAM (deterministic checks)."""
    policy = resolve_policy("Saudi Arabia", "PDPL")
    t1 = run_tier1_preflight(clean_test_dataset, policy=policy)
    assert t1.tier == 1
    assert len(t1.metrics) == 4  # Tests T1.1, T1.2, T1.3, T1.4
    assert t1.execution_time_ms < 150.0, f"Tier 1 too slow: {t1.execution_time_ms} ms"

    # Verify Rich rendering works without error
    panel = render_tier1_scorecard_panel(t1, dataset_name="test_ledger.xlsx", policy=policy)
    assert panel is not None


def test_tier2_deep_audit_execution(clean_test_dataset):
    """Tier 2 Deep Audit evaluates Tests T2.5, T2.6, T2.7, T2.8, T2.9, T2.10, T2.11."""
    policy = resolve_policy("Saudi Arabia", "PDPL")
    t2 = run_tier2_deep_audit(clean_test_dataset, policy=policy)
    assert t2.tier == 2
    assert len(t2.metrics) == 7  # Tests T2.5, T2.6, T2.7, T2.8, T2.9, T2.10, T2.11
    assert t2.all_passed is True


def test_compliance_audit_markdown_contains_tier1_and_tier2(clean_test_dataset, tmp_path):
    """Verifies that compliance_audit.md automatically contains the 11-test statutory matrix and attestation."""
    policy = resolve_policy("Saudi Arabia", "PDPL")
    cert_path = os.path.join(tmp_path, "compliance_audit.md")

    cert = create_compliance_audit_certificate(
        df_initial=clean_test_dataset,
        df_final=clean_test_dataset,
        policy=policy,
        output_path=cert_path
    )

    assert os.path.exists(cert_path)
    with open(cert_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Assert 11-test statutory matrix and attestation sections are automatically present
    assert "## 4. AIR-GAP PRIVACY & SECURITY BENCHMARK AUDIT (11-TEST SUITE)" in content
    assert "### 4.1 Statutory Compliance & Technical Benchmark Matrix" in content
    assert "### STATUTORY METHODOLOGY ATTESTATION" in content
    assert "Direct identifiers are irreversibly masked or surrogate-tokenized within volatile system memory" in content
    assert "Quasi-identifiers achieve mathematical equivalence class thresholds (k >= 5, l >= 2)" in content

    # Assert all test IDs are present
    for tid in ["T1.1", "T1.2", "T1.3", "T1.4", "T2.5", "T2.6", "T2.7", "T2.8", "T2.9", "T2.10", "T2.11"]:
        assert tid in content, f"Missing test ID {tid} in compliance_audit.md"

    # Assert governing standards & technical clauses are cited
    assert "NIST SP 800-188 §3.2" in content
    assert "Saudi PDPL Art. 29" in content
    assert "HIPAA Safe Harbor" in content
    assert "ISO/IEC 27559:2022" in content
    assert "CWE-94 / OWASP Top 10" in content
    assert "ISO 8000 / BCBS 239" in content

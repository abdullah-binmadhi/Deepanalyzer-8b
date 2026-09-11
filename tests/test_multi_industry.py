"""Unit tests for Multi-Industry Agnostic Data Cleaning & Synthetic Mock Engine.

Validates that DeepAnalyze dynamically handles data from Finance, Healthcare, E-Commerce,
SaaS, Logistics, HR, Manufacturing/IoT, Real Estate, Education, and Marketing.
"""

import math
import polars as pl
import pytest
from deepanalyze.sentinel import generate_synthetic_mock
from deepanalyze.promptgen import infer_domain_feature_engineering, build_master_prompt
from deepanalyze.policies import detect_dataset_architecture, resolve_policy


def test_sentinel_line_450_resilience():
    """Validates that generate_synthetic_mock handles edge case numeric series (NaNs, Infs, Nones, single value, empty)."""
    # All NaNs and Nones
    df_nans = pl.DataFrame({
        "all_nans": [float("nan"), float("nan"), float("nan")],
        "all_nones": [None, None, None],
        "zeroes": [0.0, 0.0, 0.0],
    })
    mock = generate_synthetic_mock(df_nans, n_rows=5)
    assert len(mock) == 5
    assert all("all_nans" in row for row in mock)
    assert all("all_nones" in row for row in mock)

    # Single Row DataFrame
    df_single = pl.DataFrame({
        "single_val": [42.0],
        "single_text": ["only_record"],
    })
    mock_single = generate_synthetic_mock(df_single, n_rows=3)
    assert len(mock_single) == 3

    # Empty DataFrame
    df_empty = pl.DataFrame({
        "num": pl.Series([], dtype=pl.Float64),
        "text": pl.Series([], dtype=pl.String),
    })
    mock_empty = generate_synthetic_mock(df_empty, n_rows=3)
    assert len(mock_empty) == 0


def test_formatted_numeric_string_mocks():
    """Validates that string-encoded numbers (currencies, percentages, units) retain format while perturbing values."""
    df = pl.DataFrame({
        "price_usd": ["$1,250.00", "$980.50", "$3,400.00", "$450.25"],
        "discount_rate": ["15.5%", "20.0%", "5.2%", "0.0%"],
        "battery_cap": ["5000 mAh", "4500 mAh", "6000 mAh", "4800 mAh"],
        "sar_amt": ["SAR 15,000.00", "SAR 22,500.00", "SAR 8,200.00", "SAR 11,000.00"],
    })
    mock = generate_synthetic_mock(df, n_rows=5)
    assert len(mock) == 5

    for row in mock:
        # USD prices should start with $ and have decimals
        assert row["price_usd"].startswith("$")
        assert "." in row["price_usd"]
        # Discount should end with %
        assert row["discount_rate"].endswith("%")
        # Battery should end with mAh
        assert "mAh" in row["battery_cap"]
        # SAR should start with SAR
        assert "SAR" in row["sar_amt"]


def test_multi_industry_mock_synthesizer():
    """Validates realistic synthetic entity generation across all 10 non-ERP industries."""
    df = pl.DataFrame({
        # 1. Finance
        "account_number": ["ACC-1", "ACC-2"],
        "card_type": ["Visa", "Mastercard"],
        "tx_type": ["DEBIT", "CREDIT"],
        "iban": ["SA000", "SA001"],
        # 2. Healthcare
        "patient_mrn": ["MRN-1", "MRN-2"],
        "diagnosis_code": ["E11", "I10"],
        "rx_medication": ["Drug A", "Drug B"],
        # 3. E-Commerce
        "product_sku": ["SKU-1", "SKU-2"],
        "tracking_number": ["TRK-1", "TRK-2"],
        # 4. SaaS
        "user_id": ["U-1", "U-2"],
        "subscription_plan": ["Free", "Pro"],
        "ip_address": ["10.0.0.1", "10.0.0.2"],
        # 5. Logistics
        "vin": ["1HG1", "1HG2"],
        "container_id": ["MSKU1", "MSKU2"],
        # 6. HR
        "employee_id": ["EMP1", "EMP2"],
        "job_title": ["Dev", "PM"],
        # 7. IoT / Manufacturing
        "sensor_id": ["SEN1", "SEN2"],
        "error_code": ["ERR1", "ERR2"],
        # 8. Real Estate
        "property_type": ["Condo", "Apt"],
        "room_type": ["Suite", "Standard"],
        # 9. Education
        "student_id": ["STU1", "STU2"],
        "course_code": ["CS101", "MATH101"],
        # 10. Marketing
        "campaign_name": ["CampA", "CampB"],
        "lead_status": ["New", "Won"],
    })

    mock = generate_synthetic_mock(df, n_rows=4)
    assert len(mock) == 4

    # Check synthetic values are present and not leaked raw data
    assert all("ACCT-" in r["account_number"] for r in mock)
    assert all("MRN-" in r["patient_mrn"] for r in mock)
    assert all("SKU-" in r["product_sku"] for r in mock)
    assert all("usr_" in r["user_id"] for r in mock)
    assert all("EMP-" in r["employee_id"] for r in mock)
    assert all("SENSOR-" in r["sensor_id"] for r in mock)
    assert all("STU-" in r["student_id"] for r in mock)


def test_multi_industry_feature_engineering_deductions():
    """Validates that infer_domain_feature_engineering correctly generates domain-specific rules."""
    # Finance / Lending
    df_fin = pl.DataFrame({
        "borrower_income": [85000.0, 110000.0],
        "total_debt": [25000.0, 30000.0],
        "loan_amount": [200000.0, 350000.0],
        "collateral_value": [300000.0, 450000.0],
        "credit_score": [720, 650],
    })
    feat_fin = infer_domain_feature_engineering(df_fin)
    assert any("Debt-to-Income" in f for f in feat_fin)
    assert any("Loan-to-Value" in f for f in feat_fin)
    assert any("Credit Risk Grading" in f for f in feat_fin)

    # E-Commerce
    df_ecom = pl.DataFrame({
        "product_price": [50.0, 120.0],
        "item_cost": [25.0, 70.0],
        "discount_amount": [5.0, 15.0],
        "order_date": ["2026-01-01", "2026-01-02"],
        "ship_date": ["2026-01-03", "2026-01-05"],
    })
    feat_ecom = infer_domain_feature_engineering(df_ecom)
    assert any("Profit Margin" in f for f in feat_ecom)
    assert any("Discount Sensitivity" in f for f in feat_ecom)
    assert any("Fulfillment Latency" in f for f in feat_ecom)

    # SaaS
    df_saas = pl.DataFrame({
        "mrr": [500.0, 1200.0],
        "session_duration_sec": [180.0, 450.0],
        "event_name": ["click", "export"],
        "user_id": ["u1", "u2"],
    })
    feat_saas = infer_domain_feature_engineering(df_saas)
    assert any("ARR Metric" in f for f in feat_saas)
    assert any("Session Engagement" in f for f in feat_saas)
    assert any("User Activity Frequency" in f for f in feat_saas)

    # Logistics
    df_log = pl.DataFrame({
        "dispatch_date": ["2026-02-01", "2026-02-02"],
        "delivery_date": ["2026-02-04", "2026-02-07"],
        "cargo_weight_kg": [500.0, 1200.0],
        "volume_cbm": [2.5, 6.0],
    })
    feat_log = infer_domain_feature_engineering(df_log)
    assert any("Transit Lead Time" in f for f in feat_log)
    assert any("Freight Density" in f for f in feat_log)

    # HR
    df_hr = pl.DataFrame({
        "hire_date": ["2020-03-15", "2022-07-01"],
        "base_salary": [95000.0, 115000.0],
        "bonus_amt": [10000.0, 15000.0],
    })
    feat_hr = infer_domain_feature_engineering(df_hr)
    assert any("Workforce Tenure" in f for f in feat_hr)
    assert any("Total Compensation" in f for f in feat_hr)

    # IoT
    df_iot = pl.DataFrame({
        "temperature_c": [72.5, 95.0, 71.0, 73.0],
        "kwh_consumption": [120.0, 180.0, 115.0, 130.0],
        "runtime_hours": [8.0, 8.0, 8.0, 8.0],
    })
    feat_iot = infer_domain_feature_engineering(df_iot)
    assert any("Thermal Anomaly" in f for f in feat_iot)
    assert any("Specific Energy Consumption" in f for f in feat_iot)

    # Marketing
    df_mkt = pl.DataFrame({
        "ad_clicks": [450, 1200],
        "impressions": [15000, 40000],
        "ad_spend": [800.0, 2500.0],
        "revenue_attributed": [2400.0, 8900.0],
    })
    feat_mkt = infer_domain_feature_engineering(df_mkt)
    assert any("Click-Through Rate" in f for f in feat_mkt)
    assert any("Return on Ad Spend" in f for f in feat_mkt)


def test_policies_multi_industry_architecture_detection():
    """Validates that detect_dataset_architecture correctly tags multi-industry schemas."""
    # Finance
    df_fin = pl.DataFrame({"loan_id": [1], "borrower_debt": [5000], "interest_rate": [0.05]})
    key, name, desc = detect_dataset_architecture(df_fin)
    assert key == "CLEAN_TABULAR"
    assert "Financial" in name

    # E-Commerce
    df_ecom = pl.DataFrame({"order_id": [1], "sku_code": ["A"], "cart_discount": [5.0]})
    key, name, desc = detect_dataset_architecture(df_ecom)
    assert key == "CLEAN_TABULAR"
    assert "E-Commerce" in name

    # SaaS
    df_saas = pl.DataFrame({"user_id": [1], "subscription_mrr": [99.0], "churn_risk": [0.1]})
    key, name, desc = detect_dataset_architecture(df_saas)
    assert key == "CLEAN_TABULAR"
    assert "SaaS" in name

    # Logistics
    df_log = pl.DataFrame({"shipment_id": [1], "carrier_fleet": ["Truck1"], "transit_time_days": [3]})
    key, name, desc = detect_dataset_architecture(df_log)
    assert key == "CLEAN_TABULAR"
    assert "Logistics" in name

    # HR
    df_hr = pl.DataFrame({"employee_id": [1], "hire_date": ["2022-01-01"], "salary": [75000]})
    key, name, desc = detect_dataset_architecture(df_hr)
    assert key == "CLEAN_TABULAR"
    assert "Workforce" in name

    # IoT
    df_iot = pl.DataFrame({"sensor_id": [1], "voltage": [230.0], "vibration": [0.02]})
    key, name, desc = detect_dataset_architecture(df_iot)
    assert key == "CLEAN_TABULAR"
    assert "IoT" in name

    # Marketing
    df_mkt = pl.DataFrame({"campaign_id": [1], "impressions": [5000], "ad_spend": [120.0]})
    key, name, desc = detect_dataset_architecture(df_mkt)
    assert key == "CLEAN_TABULAR"
    assert "Marketing" in name

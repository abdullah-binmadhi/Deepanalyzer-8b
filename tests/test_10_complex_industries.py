"""Comprehensive 10-Industry Extreme Messy Data Benchmark & 10-SQL Polyglot Suite
Tests that DeepAnalyze handles data structures significantly harder and more complex than INV LISTING 31082025.xlsx across:
1. Healthcare & Clinical Trials (EHR / HL7)
2. Retail & E-Commerce Multi-Currency Omnichannel Ledger
3. Supply Chain, Logistics & Freight Manifests (SAP / Navision)
4. Financial Services & Multi-Branch General Ledger
5. Energy, Oil & Gas Smart Grid Temporal Matrix
6. SaaS & Subscription Churn Cohorts
7. Real Estate & Mortgage Appraisal Registry
8. Telecommunications & IoT Network CDRs (Zero-PII)
9. Manufacturing Six Sigma Defect Logs
10. Cross-Lingual International Customs & HS Tariff Matrix

Also validates 10 advanced DuckDB / Arrow ANSI SQL query engines across all flags.
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import polars as pl
import pytest
import duckdb

from deepanalyze import cleaners
from deepanalyze import privacy_knife
from deepanalyze import statistical_engine
from deepanalyze import forecaster
from deepanalyze import feature_forge
from deepanalyze import drift_sentinel
from deepanalyze import schema_synthesizer
from deepanalyze import causal_engine
from deepanalyze import optimizer
from deepanalyze import pipeline_compiler
from deepanalyze import storyteller
from deepanalyze import synthetic_data
from deepanalyze import debate_router
from deepanalyze.core import (
    _get_deep_workspace_context,
    _format_micro_schema,
    _lint_and_format_code,
    _take_snapshot,
    _restore_snapshot,
    _DF_SNAPSHOT_STACK,
    _DF_SNAPSHOTS,
    deepanalyze
)
from IPython.core.interactiveshell import InteractiveShell


# =============================================================================
# 1. HEALTHCARE & CLINICAL TRIALS (EHR / HL7)
# =============================================================================
def test_industry_01_healthcare_clinical_trials():
    """Healthcare EHR: Nested patient records, vital sub-tables, JSON diagnostics, and causal treatment effect."""
    ip = InteractiveShell.instance()

    raw_ehr = pd.DataFrame({
        "patient_id": [f"PT-{1000+i}" for i in range(100)],
        "age": np.random.randint(25, 80, size=100),
        "treatment_arm": np.random.choice(["Drug_A", "Placebo"], size=100),
        "blood_pressure_change": [f"({abs(x):.1f})" if x < 0 else f"+{x:.1f}" for x in np.random.randn(100) * 12 - 4],
        "biomarker_json": ['{"crp": 3.4, "ldl": 120.5, "status": "ACTIVE"}' if i % 2 == 0 else '{"crp": 1.2, "ldl": 95.0, "status": "COMPLETED"}' for i in range(100)],
        "admission_date": np.random.choice(["2025-01-10", "10/01/2025", "2025.01.10", "9999-12-31"], size=100),
        "discharge_notes": ["  Patient tolerated therapy well \u200b\u200c " if i % 3 == 0 else "Caf\xc3\xa9 diet resumed; normal vitals\ufeff" for i in range(100)]
    })

    # Clean units, currencies & sanitize text
    pldf = pl.from_pandas(raw_ehr)
    cleaned = cleaners.sanitize_unicode_and_mojibake(pldf)
    cleaned = cleaners.normalize_units_and_currencies(cleaned)
    cleaned = cleaners.explode_nested_json(cleaned)
    assert any("crp" in c for c in cleaned.columns)
    assert any("ldl" in c for c in cleaned.columns)

    ip.user_ns["ehr_df"] = cleaned

    # Analytical Execution Flags
    deepanalyze("--stats --target ehr_df")
    deepanalyze("--schema --target ehr_df")

    # SQL Test 1: Window functions for clinical biomarker rank
    deepanalyze('--sql SELECT patient_id, treatment_arm, CAST(biomarker_json_ldl AS DOUBLE) as ldl_val, RANK() OVER (PARTITION BY treatment_arm ORDER BY CAST(biomarker_json_ldl AS DOUBLE) DESC) as ldl_rank FROM ehr_df LIMIT 10 --target ehr_sql_res')
    assert "ehr_sql_res" in ip.user_ns
    assert len(ip.user_ns["ehr_sql_res"]) == 10


# =============================================================================
# 2. RETAIL & E-COMMERCE MULTI-CURRENCY OMNICHANNEL LEDGER
# =============================================================================
def test_industry_02_retail_multicurrency_ledger():
    """Retail: Multi-currency sales, fuzzy category errors, discount parsing, and conformal 14-day forecast."""
    ip = InteractiveShell.instance()

    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    raw_retail = pd.DataFrame({
        "order_date": [d.strftime("%Y-%m-%d") for d in dates],
        "category": np.random.choice(["Electronics", "electrnoics", "Apparel", "Apparel & Acc", "Home Goods", "HomeGoods"], size=60),
        "raw_sales": [f"${x:,.2f}" if i % 3 == 0 else (f"{x*4.5:,.2f} SAR" if i % 3 == 1 else f"RM {x*4.7:,.2f}") for i, x in enumerate(np.random.uniform(500, 5000, size=60))],
        "discount_pct": [f"{np.random.choice([5, 10, 15, 20])}%" for _ in range(60)]
    })

    pldf = pl.from_pandas(raw_retail)
    cleaned = cleaners.fuzzy_harmonize_categories(pldf)
    cleaned = cleaners.normalize_units_and_currencies(cleaned)
    cleaned = cleaners.auto_cast_data_types(cleaned)
    ip.user_ns["retail_df"] = cleaned

    # Analytical Flags
    deepanalyze("--forecast --target retail_df")
    deepanalyze("--winsorize --target retail_df")

    # SQL Test 2: Moving Average and Cumulative Revenue
    deepanalyze("--sql SELECT order_date, raw_sales, AVG(raw_sales) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as rolling_7d_sales, SUM(raw_sales) OVER (ORDER BY order_date) as cumulative_revenue FROM retail_df --target retail_sql_res")
    assert len(ip.user_ns["retail_sql_res"]) == 60


# =============================================================================
# 3. SUPPLY CHAIN, LOGISTICS & FREIGHT MANIFESTS (SAP / NAVISION)
# =============================================================================
def test_industry_03_supply_chain_freight_manifests():
    """Supply Chain: Multi-tier Bills of Lading with wrapped commodity descriptions and PSI drift detection."""
    ip = InteractiveShell.instance()

    manifest_data = [
        ["VESSEL : CMA CGM ALEXANDER", None, None, None, "VOYAGE : 2025-08W"],
        ["PORT OF LOADING : MYPKG", None, None, None, "PORT OF DISCHARGE : SGSIN"],
        ["--------------------------------------------------------------------------------"],
        ["Doc. No", "Doc. Date", "Code", "Name", "Total Amount (RM)"],
        ["BL-8801", "2025-08-01", "SHP-001", "MAERSK LOGISTICS SDN BHD", "15,000.00"],
        ["Seq", "Container", "Description", "Weight (KG)", "UOM", "Rate", "Amount (RM)"],
        [1, "MSKU-992101", "INDUSTRIAL SOLVENTS 200L DRUMS", 12000, "DRM", 1.00, 12000.00],
        [None, None, "- DANGEROUS GOODS CLASS 3 FLAMMABLE BATCH #77A", None, None, None, None],
        [2, "MSKU-992102", "SURFACTANT RAW PALLETS", 3000, "PLT", 1.00, 3000.00],
        [None, None, "  TEMPERATURE CONTROLLED CARGO (15-25C)", None, None, None, None],
        ["Doc. No", "Doc. Date", "Code", "Name", "Total Amount (RM)"],
        ["BL-8802", "2025-08-02", "SHP-002", "EVERGREEN MARINE CORP", "8,500.00"],
        [1, "EGLU-441099", "AUTOMOTIVE SPARE PARTS", 8500, "CTN", 1.00, 8500.00],
        ["Grand Total Amount (RM)", None, None, None, "23,500.00"],
        ["Account Summary", None, None, None, None],
        ["400-000 Freight Revenue: 23,500.00", None, None, None, None]
    ]

    unravelled = cleaners.unravel_hierarchical_erp_report(pd.DataFrame(manifest_data))
    assert unravelled.height == 3
    assert round(float(unravelled["Item Amount"].sum()), 2) == 23500.00
    assert any("CLASS 3 FLAMMABLE" in str(d) for d in unravelled["Full_Description"])

    ip.user_ns["manifest_df"] = unravelled

    # Analytical Flags
    deepanalyze("--engineer --target manifest_df")
    deepanalyze("--drift --target manifest_df")

    # SQL Test 3: Container Freight Density & Aggregate Analytics
    deepanalyze("--sql SELECT customer_name, COUNT(*) as container_count, SUM(\"Item Amount\") as total_freight, AVG(\"Item Amount\") as avg_container_value FROM manifest_df GROUP BY customer_name ORDER BY total_freight DESC --target scm_sql_res")
    assert len(ip.user_ns["scm_sql_res"]) == 2


# =============================================================================
# 4. FINANCIAL SERVICES & BANKING MULTI-BRANCH GENERAL LEDGER
# =============================================================================
def test_industry_04_financial_banking_gl():
    """Banking GL: Parenthetical debits/credits, multi-line memos, European notation, and executive story."""
    ip = InteractiveShell.instance()

    gl_df = pd.DataFrame({
        "account_no": [f"GL-10{i:03d}" for i in range(50)],
        "branch_code": np.random.choice(["BR-KL01", "BR-JB02", "BR-PEN03"], size=50),
        "debit_amount": [f"({x:,.2f})" if i % 2 == 0 else f"{x:,.2f}" for i, x in enumerate(np.random.uniform(1000, 50000, size=50))],
        "credit_amount": [f"{x:,.2f} €" if i % 3 == 0 else f"RM {x:,.2f}" for i, x in enumerate(np.random.uniform(1000, 50000, size=50))],
        "reconciliation_status": np.random.choice(["CLEARED", "PENDING_AUDIT", "DISPUTED"], size=50)
    })

    cleaned = cleaners.normalize_units_and_currencies(gl_df)
    cleaned = cleaners.auto_cast_data_types(cleaned)
    ip.user_ns["gl_df"] = cleaned

    # Analytical Flags
    deepanalyze("--stats --target gl_df")
    deepanalyze("--story --target gl_df")

    # SQL Test 4: Financial Branch Exposure and Risk Categorization
    deepanalyze("--sql SELECT branch_code, reconciliation_status, COUNT(*) as txn_count, SUM(debit_amount) as total_debits, SUM(credit_amount) as total_credits, (SUM(credit_amount) - SUM(debit_amount)) as net_balance FROM gl_df GROUP BY branch_code, reconciliation_status ORDER BY net_balance DESC --target gl_sql_res")
    assert len(ip.user_ns["gl_sql_res"]) > 0


# =============================================================================
# 5. ENERGY, OIL & GAS / SMART GRID TEMPORAL MATRIX
# =============================================================================
def test_industry_05_energy_smart_grid_matrix():
    """Energy Smart Grid: Wide 24-hour horizontal temporal matrix unpivoted to tidy time series with sparklines."""
    ip = InteractiveShell.instance()

    # Create wide temporal matrix (Substation x Hour_00 to Hour_23)
    grid_data = {"substation_id": [f"SUB-NORD-{i:02d}" for i in range(20)], "grid_region": np.random.choice(["Alpha", "Beta", "Gamma"], size=20)}
    for h in range(24):
        grid_data[f"{h:02d}:00"] = [f"{x:.1f} MW" if x > 0 else "0.0 MW (OFFLINE)" for x in np.random.uniform(10, 250, size=20)]

    wide_df = pd.DataFrame(grid_data)
    unpivoted = cleaners.unpivot_temporal_matrix(wide_df)
    assert (unpivoted.height if hasattr(unpivoted, "height") else len(unpivoted)) == 20 * 24
    assert "period" in unpivoted.columns or "time_period" in unpivoted.columns

    cleaned = cleaners.normalize_units_and_currencies(unpivoted)
    cleaned = cleaners.auto_cast_data_types(cleaned)
    ip.user_ns["grid_df"] = cleaned

    # Analytical Flags
    deepanalyze("--spark --target grid_df")
    deepanalyze("--winsorize --target grid_df")

    # SQL Test 5: Peak Load and Hourly Grid Stress Analysis
    deepanalyze("--sql SELECT substation_id, MAX(CAST(value AS DOUBLE)) as peak_load_mw, MIN(CAST(value AS DOUBLE)) as base_load_mw, AVG(CAST(value AS DOUBLE)) as avg_load_mw FROM grid_df GROUP BY substation_id ORDER BY peak_load_mw DESC LIMIT 5 --target grid_sql_res")
    assert len(ip.user_ns["grid_sql_res"]) == 5


# =============================================================================
# 6. SAAS & SUBSCRIPTION CHURN COHORTS
# =============================================================================
def test_industry_06_saas_subscription_cohorts():
    """SaaS Analytics: Cohort churn rates, MRR expansion/contraction, causal debugger, and dialectical debate."""
    ip = InteractiveShell.instance()

    saas_df = pl.DataFrame({
        "tenant_id": [f"ORG_{i:04d}" for i in range(100)],
        "cohort_month": np.random.choice(["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"], size=100),
        "plan_tier": np.random.choice(["Starter", "Professional", "Enterprise"], size=100),
        "mrr_usd": np.random.uniform(50, 5000, size=100),
        "expansion_mrr": np.random.uniform(0, 1200, size=100),
        "churn_flag": np.random.choice([0, 1], p=[0.85, 0.15], size=100),
        "nps_score": np.random.randint(1, 11, size=100)
    })
    ip.user_ns["saas_df"] = saas_df

    # Analytical Flags
    deepanalyze("--why churn_flag --target saas_df")
    deepanalyze("--debate --target saas_df")
    deepanalyze("--next --target saas_df")

    # SQL Test 6: Cohort Retention and Net Revenue Retention (NRR) Matrix
    deepanalyze("--sql SELECT cohort_month, plan_tier, COUNT(*) as total_tenants, SUM(churn_flag) as churned_tenants, (1.0 - (SUM(churn_flag) * 1.0 / COUNT(*))) * 100.0 as retention_rate_pct, SUM(mrr_usd + expansion_mrr) as ending_mrr FROM saas_df GROUP BY cohort_month, plan_tier ORDER BY cohort_month, plan_tier --target saas_sql_res")
    assert len(ip.user_ns["saas_sql_res"]) > 0


# =============================================================================
# 7. REAL ESTATE & MORTGAGE APPRAISAL REGISTRY
# =============================================================================
def test_industry_07_real_estate_mortgage():
    """Real Estate: Property valuation, fuzzy zoning codes, synthetic digital twin clone, and DuckDB DDL."""
    ip = InteractiveShell.instance()

    re_df = pl.DataFrame({
        "parcel_id": [f"PCL-9900{i:02d}" for i in range(80)],
        "zoning_type": np.random.choice(["Residential R-1", "Commercial C-2", "Industrial I-1", "Mixed Use M-X"], size=80),
        "square_feet": np.random.randint(800, 12000, size=80),
        "appraisal_value": np.random.uniform(250000, 2500000, size=80),
        "mortgage_interest_rate": np.random.uniform(3.2, 7.8, size=80),
        "delinquency_status": np.random.choice(["CURRENT", "30_DAYS", "90_DAYS_PLUS"], p=[0.9, 0.07, 0.03], size=80)
    })
    ip.user_ns["re_df"] = re_df

    # Analytical Flags
    deepanalyze("--synthetic --target re_df")
    deepanalyze("--schema --target re_df")
    deepanalyze("--radar --target re_df")

    # SQL Test 7: Price Per Square Foot and Delinquency Risk Clustering
    deepanalyze("--sql SELECT zoning_type, COUNT(*) as property_count, AVG(appraisal_value / square_feet) as avg_price_per_sqft, MAX(appraisal_value) as max_valuation, SUM(CASE WHEN delinquency_status != 'CURRENT' THEN 1 ELSE 0 END) as at_risk_count FROM re_df GROUP BY zoning_type ORDER BY avg_price_per_sqft DESC --target re_sql_res")
    assert len(ip.user_ns["re_sql_res"]) == 4


# =============================================================================
# 8. TELECOMMUNICATIONS & IOT CALL DETAIL RECORDS (ZERO-PII)
# =============================================================================
def test_industry_08_telecom_iot_cdr_zeropii():
    """Telecom: IoT session CDRs, personal identifiers, cryptographic tokenization vault, and LP solver."""
    ip = InteractiveShell.instance()

    tel_records = [
        {
            "subscriber_id": f"SUB-{i:05d}",
            "national_id": f"880{i:02d}12-10-54{i:02d}",
            "msisdn": f"+6012-34567{i:02d}",
            "data_usage_mb": float(np.random.uniform(100, 25000)),
            "call_duration_sec": int(np.random.randint(10, 3600)),
            "signal_rssi_dbm": float(np.random.uniform(-110, -50)),
            "roaming_flag": int(np.random.choice([0, 1], p=[0.8, 0.2]))
        }
        for i in range(50)
    ]
    df_tel = pl.DataFrame(tel_records)

    # Tokenize PII locally
    knife = privacy_knife.DeepAnalyzePrivacyKnife(df_tel, dataset_id="tel_vault")
    masked_tel = knife.tokenize_pii_columns(["national_id", "msisdn"])
    assert "880" not in str(masked_tel["national_id"].to_list())

    ip.user_ns["tel_df"] = masked_tel

    # Analytical Flags
    deepanalyze("--solve --target tel_df")
    deepanalyze("--pipeline --target tel_df")

    # SQL Test 8: Heavy Roaming Data Consuming Subscribers
    deepanalyze("--sql SELECT subscriber_id, data_usage_mb, call_duration_sec, signal_rssi_dbm FROM tel_df WHERE roaming_flag = 1 ORDER BY data_usage_mb DESC LIMIT 10 --target tel_sql_res")
    assert "tel_sql_res" in ip.user_ns


# =============================================================================
# 9. MANUFACTURING QUALITY CONTROL & SIX SIGMA DEFECT LOGS
# =============================================================================
def test_industry_09_manufacturing_six_sigma():
    """Manufacturing: Assembly tolerance deviations, multi-level defect codes, skeptic battery, and HTML report."""
    ip = InteractiveShell.instance()

    mfg_df = pl.DataFrame({
        "batch_lot": [f"LOT-2025-{i:04d}" for i in range(120)],
        "production_line": np.random.choice(["Line_1_SMT", "Line_2_CNC", "Line_3_Assy"], size=120),
        "tolerance_deviation_mm": np.random.normal(0.0, 0.02, size=120),
        "surface_roughness_ra": np.random.uniform(0.2, 1.8, size=120),
        "defect_count": np.random.poisson(0.8, size=120),
        "operator_id": np.random.choice(["OP_ALICE", "OP_BOB", "OP_CHARLIE"], size=120)
    })
    ip.user_ns["mfg_df"] = mfg_df

    # Analytical Flags
    deepanalyze("--falsify --target mfg_df")
    deepanalyze("--report --target mfg_df")
    deepanalyze("--diff-stats --target mfg_df")

    # SQL Test 9: Six Sigma Defect Rates and Tolerance Out-of-Spec (OOS) Outliers
    deepanalyze("--sql SELECT production_line, COUNT(*) as total_inspections, SUM(defect_count) as total_defects, AVG(ABS(tolerance_deviation_mm)) as mean_abs_deviation, SUM(CASE WHEN ABS(tolerance_deviation_mm) > 0.04 THEN 1 ELSE 0 END) as oos_outliers FROM mfg_df GROUP BY production_line ORDER BY total_defects DESC --target mfg_sql_res")
    assert len(ip.user_ns["mfg_sql_res"]) == 3


# =============================================================================
# 10. CROSS-LINGUAL CUSTOMS & HS TARIFF MATRIX
# =============================================================================
def test_industry_10_crosslingual_customs_tariff():
    """Customs & International Trade: Multi-lingual HS codes, currency conversions, semantic cross-join, and DAG."""
    ip = InteractiveShell.instance()

    tariff_df = pl.DataFrame({
        "hs_code": ["8471.30.00", "8517.13.00", "8703.80.00", "2202.99.00", "3004.90.00"],
        "description_en": ["Laptops & Portable Computers", "Smartphones 5G", "Electric Passenger Vehicles", "Non-Alcoholic Beverages", "Medicaments for Therapeutic Use"],
        "description_ar": ["أجهزة كمبيوتر محمولة", "هواتف ذكية", "مركبات ركاب كهربائية", "مشروبات غير كحولية", "أدوية للاستخدام العلاجي"],
        "base_duty_rate": [0.0, 0.05, 0.30, 0.15, 0.0],
        "customs_cif_value": [1200000.0, 3500000.0, 8900000.0, 450000.0, 2100000.0]
    })
    ip.user_ns["tariff_df"] = tariff_df

    # Analytical Flags
    deepanalyze("--distill --target tariff_df")
    deepanalyze("--dag --target tariff_df")

    # SQL Test 10: Total Duty Revenue Calculation and Tariff Bracket Aggregations
    deepanalyze("--sql SELECT hs_code, description_en, customs_cif_value, base_duty_rate, (customs_cif_value * base_duty_rate) as total_duty_payable, CASE WHEN base_duty_rate = 0.0 THEN 'DUTY_FREE' WHEN base_duty_rate <= 0.10 THEN 'LOW_TARIFF' ELSE 'HIGH_TARIFF' END as tariff_bracket FROM tariff_df ORDER BY total_duty_payable DESC --target tariff_sql_res")
    assert len(ip.user_ns["tariff_sql_res"]) == 5

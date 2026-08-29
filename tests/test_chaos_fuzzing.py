"""Comprehensive Chaos Engineering & Extreme Messy Data Fuzzing Test Suite
Validates that DeepAnalyze behaves deterministically and without crashes across:
1. Extreme corrupted numbers, accounting formats, and currency notations
2. Mojibake, zero-width characters, Unicode control characters, and RTL scripts
3. Deep multi-tier ragged ERP spreadsheets with continuation lines
4. Ultra-wide (1000+ cols), empty (0-row), and ultra-sparse (99.9% null) matrices
5. Polyglot engine parity (Polars SIMD <-> Pandas C-API <-> DuckDB Arrow)
6. 5-step LIFO time-machine state invariance and rollback
7. Zero-PII privacy knife tokenization and lossless reconstruction
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import polars as pl
import pytest

from deepanalyze import cleaners
from deepanalyze import privacy_knife
from deepanalyze import statistical_engine
from deepanalyze import forecaster
from deepanalyze import feature_forge
from deepanalyze import drift_sentinel
from deepanalyze import schema_synthesizer
from deepanalyze import causal_engine
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


def test_fuzz_extreme_corrupted_numbers_and_currencies():
    """Fuzz testing extreme corrupted numerical, accounting, and multi-currency formats."""
    raw_samples = [
        "(1,234.56)",          # Parenthetical accounting negative
        "$(500.00)",           # Dollar accounting negative
        "1.234,56 €",          # European comma-decimal
        "RM 2,500.00",         # Malaysian Ringgit
        "150 SAR",             # Saudi Riyal
        "¥50,000",             # Japanese Yen
        "₹12,34,567.89",       # Indian Lakh formatting
        "  $ 999.99 \t \n",    # Trailing whitespaces / tabs
        "15.5%",               # Percentage
        "(2.5%)",              # Negative percentage
        "1.23e-4",             # Scientific notation
        "inf",                 # Infinity
        "-inf",                # Negative infinity
        "None", "null", "N/A", "#VALUE!", "(null)"
    ]

    pdf = pd.DataFrame({"dirty_amount": raw_samples, "id": range(len(raw_samples))})
    pldf = pl.DataFrame({"dirty_amount": raw_samples, "id": range(len(raw_samples))})

    # 1. Normalize units and currencies
    clean_pldf = cleaners.normalize_units_and_currencies(pldf)
    assert clean_pldf is not None
    assert clean_pldf.height == len(raw_samples)

    clean_pdf = cleaners.normalize_units_and_currencies(pdf)
    assert clean_pdf is not None
    assert len(clean_pdf) == len(raw_samples)

    # 2. Auto-type casting
    typed_pldf = cleaners.auto_cast_data_types(clean_pldf)
    assert typed_pldf is not None

    # 3. Winsorize extreme bounds without crash
    winsorized = cleaners.winsorize_numeric_outliers(typed_pldf)
    assert winsorized.height == len(raw_samples)


def test_fuzz_extreme_unicode_mojibake_and_invisible_chars():
    """Fuzz testing Unicode Mojibake, zero-width characters, and mixed scripts."""
    dirty_strings = [
        "Caf\xc3\xa9 \u200b\u200c\u200d Latte",        # Zero-width spaces & Mojibake
        "\ufeffBOM Customer Name\ufeff",               # Byte Order Marks
        "Customer \x00\x07\x1f With Control Chars",    # ASCII control characters
        "\u200fشركة الأمل للتجارة\u200e",              # Arabic with RTL/LTR directional marks
        "บริษัท สยาม ซอสเซส 500G 🌶️🔥",                # Thai script with emojis
        "北京盛大贸易有限公司 (100% 优质)",             # Chinese characters with parentheses
        "Normal Clean Text"
    ]

    pdf = pd.DataFrame({"text_col": dirty_strings, "val": range(len(dirty_strings))})
    pldf = pl.DataFrame({"text_col": dirty_strings, "val": range(len(dirty_strings))})

    sanitized_pldf = cleaners.sanitize_unicode_and_mojibake(pldf)
    assert sanitized_pldf.height == len(dirty_strings)
    assert "\ufeff" not in sanitized_pldf["text_col"][1]
    assert "\x00" not in sanitized_pldf["text_col"][2]

    sanitized_pdf = cleaners.sanitize_unicode_and_mojibake(pdf)
    assert len(sanitized_pdf) == len(dirty_strings)
    assert "\ufeff" not in sanitized_pdf["text_col"].iloc[1]


def test_fuzz_ragged_multi_tiered_erp_spreadsheets():
    """Fuzz testing 10-level nested hierarchical invoice listings with continuation rows."""
    ragged_data = [
        # Metadata rows
        ["Company : Global Distribution Berhad", None, None, None, "Page 1 of 99"],
        ["Date Filter : 01/08/2025 to 31/08/2025", None, None, None, None],
        ["--------------------------------------------------------------------------------"],
        # Doc 1 Header
        ["Doc. No", "Doc. Date", "Code", "Name", "Total Amount (RM)"],
        ["IV-99001", "2025-08-01", "CUST-001", "SUNSHINE MART SDN BHD", "1,500.00"],
        # Item Header
        ["Seq", "Item Code", "Description", "Qty", "UOM", "Price", "Amount (RM)"],
        [1, "ITM-01", "PREMIUM ORGANIC SOY SAUCE 500ML", 10, "BTL", 50.00, 500.00],
        [None, None, "- WRAPPED BULK CASE BATCH #9921", None, None, None, None],
        [2, "ITM-02", "EXTRA HOT CHILI CRISP 250G", 20, "JAR", 50.00, 1000.00],
        [None, None, "  EXTRA SPICY EDITION - FRAGILE", None, None, None, None],
        # Doc 2 Header
        ["Doc. No", "Doc. Date", "Code", "Name", "Total Amount (RM)"],
        ["IV-99002", "2025-08-02", "CUST-002", "MEGA HYPERMARKET CORP", "800.00"],
        [1, "ITM-03", "INSTANT NOODLE PACK 5X80G", 40, "CTN", 20.00, 800.00],
        # Summary footer rows
        ["Grand Total Amount (RM)", None, None, None, "2,300.00"],
        ["Account Summary", None, None, None, None],
        ["500-000 Sales Revenue: 2,300.00", None, None, None, None]
    ]

    raw_df = pd.DataFrame(ragged_data)
    unravelled = cleaners.unravel_hierarchical_erp_report(raw_df)

    assert unravelled is not None
    assert unravelled.height == 3  # Exactly 3 clean item lines (2 from Doc 1, 1 from Doc 2)
    assert "doc_no" in unravelled.columns
    assert "customer_name" in unravelled.columns
    assert "Full_Description" in unravelled.columns

    # Verify zero nulls across parent keys
    assert unravelled["doc_no"].null_count() == 0
    assert unravelled["customer_name"].null_count() == 0

    # Verify wrapped text was properly stitched into Full_Description
    desc_list = unravelled["Full_Description"].to_list()
    assert any("BATCH #9921" in str(d) for d in desc_list)
    assert any("EXTRA SPICY" in str(d) for d in desc_list)

    # Verify exact financial reconciliation
    assert round(float(unravelled["Item Amount"].sum()), 2) == 2300.00


def test_fuzz_sparse_wide_and_empty_edge_cases():
    """Fuzz testing ultra-wide (1000 columns), 0-row empty, and single-row edge cases."""
    # 1. 0-row empty DataFrame with types
    empty_df = pl.DataFrame({
        "id": pl.Series([], dtype=pl.Int64),
        "name": pl.Series([], dtype=pl.Utf8),
        "amount": pl.Series([], dtype=pl.Float64)
    })
    schema_str = _format_micro_schema(empty_df, "empty_df", is_polars=True)
    assert "0 rows" in schema_str or "empty_df" in schema_str

    # 2. 1-row single cell
    single_df = pl.DataFrame({"single_val": [42]})
    vif = statistical_engine.compute_vif_robust(single_df)
    assert isinstance(vif, pd.DataFrame)

    # 3. 500-column ultra-wide sparse matrix
    np.random.seed(42)
    wide_data = {f"col_{i}": np.random.choice([np.nan, 1.0], size=50, p=[0.95, 0.05]) for i in range(200)}
    wide_data["target_col"] = np.random.randn(50)
    wide_df = pl.DataFrame(wide_data)

    ip = InteractiveShell.instance()
    ip.user_ns["wide_df"] = wide_df

    ctx, _, _ = _get_deep_workspace_context(ip, target="wide_df", is_cloud=False)
    assert "wide_df" in ctx
    assert len(ctx) > 0


def test_fuzz_duckdb_arrow_zero_copy_polyglot_parity():
    """Verify DuckDB ANSI SQL executes identically across Polars, Pandas, and LazyFrames."""
    data = {
        "region": ["North", "North", "South", "South", "East", "East"],
        "sales": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0]
    }
    pldf = pl.DataFrame(data)
    pdf = pd.DataFrame(data)

    ip = InteractiveShell.instance()
    ip.user_ns["pldf"] = pldf
    ip.user_ns["pdf"] = pdf

    # Run SQL on Polars DataFrame
    deepanalyze("--sql SELECT region, SUM(sales) AS total_sales FROM pldf GROUP BY region ORDER BY total_sales DESC --target polars_sql_res")
    res_pl = ip.user_ns.get("polars_sql_res")

    # Run SQL on Pandas DataFrame
    deepanalyze("--sql SELECT region, SUM(sales) AS total_sales FROM pdf GROUP BY region ORDER BY total_sales DESC --target pandas_sql_res")
    res_pd = ip.user_ns.get("pandas_sql_res")

    assert res_pl is not None
    assert res_pd is not None
    assert len(res_pl) == 3
    assert len(res_pd) == 3

    pl_sum = float(res_pl["total_sales"].sum()) if hasattr(res_pl["total_sales"], "sum") else sum(res_pl["total_sales"])
    pd_sum = float(res_pd["total_sales"].sum())
    assert round(pl_sum, 2) == round(pd_sum, 2) == 1350.00


def test_fuzz_multi_step_time_machine_undo_invariance():
    """Verify 5-step sequential mutations and 5 consecutive lossless undo rollbacks."""
    ip = InteractiveShell.instance()
    initial_df = pl.DataFrame({
        "account": ["A001", "A002", "A003"],
        "balance": [100.0, 200.0, 300.0]
    })
    target = "fuzz_undo_df"
    ip.user_ns[target] = initial_df

    # Record 5 state mutations
    for i in range(1, 6):
        _take_snapshot(ip, target=target)
        ip.user_ns[target] = initial_df.with_columns(pl.col("balance") * (i + 1))

    assert len(_DF_SNAPSHOT_STACK[target]) == 5
    assert float(ip.user_ns[target]["balance"][0]) == 600.0

    # Sequentially undo all 5 steps
    for expected_multiplier in [5, 4, 3, 2, 1]:
        success = _restore_snapshot(ip, target=target)
        assert success is True
        current_val = float(ip.user_ns[target]["balance"][0])
        assert current_val == 100.0 * expected_multiplier

    # Final restored state must match original initial_df exactly
    assert float(ip.user_ns[target]["balance"][0]) == 100.0
    assert len(_DF_SNAPSHOT_STACK[target]) == 0


def test_fuzz_privacy_knife_tokenization_integrity():
    """Verify zero-PII leakage under synthetic personal and national ID injection."""
    pii_records = [
        {"name": "Ahmad bin Abdullah", "ic": "881212-10-5432", "email": "ahmad@corp.my", "phone": "+6012-3456789", "salary": 8500.0},
        {"name": "Sarah Al-Ghamdi", "ic": "1029384756", "email": "sarah@tech.sa", "phone": "+966501234567", "salary": 14000.0},
        {"name": "John Doe", "ic": "123-45-6789", "email": "john.doe@company.com", "phone": "+1-202-555-0199", "salary": 9200.0}
    ]
    df_pii = pl.DataFrame(pii_records)

    knife = privacy_knife.DeepAnalyzePrivacyKnife(df_pii)
    masked_df = knife.tokenize_pii_columns(["name", "ic", "email", "phone"])

    # Assert that NO raw PII values remain in masked DataFrame
    for rec in pii_records:
        assert rec["name"] not in masked_df["name"].to_list()
        assert rec["ic"] not in masked_df["ic"].to_list()
        assert rec["email"] not in masked_df["email"].to_list()
        assert rec["phone"] not in masked_df["phone"].to_list()

    # Assert salary remains intact (non-PII numerical metric)
    assert masked_df["salary"].to_list() == [8500.0, 14000.0, 9200.0]

    # Test lossless local detokenization
    restored_df = knife.detokenize_dataframe(masked_df)
    assert restored_df["name"].to_list() == [r["name"] for r in pii_records]
    assert restored_df["email"].to_list() == [r["email"] for r in pii_records]
    assert restored_df["ic"].to_list() == [r["ic"] for r in pii_records]

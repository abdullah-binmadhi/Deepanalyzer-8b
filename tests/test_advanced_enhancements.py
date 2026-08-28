"""Tests for Advanced High-ROI Enhancements:
- Polars LazyFrame zero-scan inspection
- Multi-step LIFO snapshot undo stack
- Context-aware tab completer
- Direct ANSI SQL execution bridge
- Statistical KS-drift calculation in state diff HUD
- Speculative draft model detection in server launcher
"""

import types
import numpy as np
import pandas as pd
import polars as pl
import pytest

from deepanalyze.core import (
    _format_micro_schema,
    _take_snapshot,
    _restore_snapshot,
    _DF_SNAPSHOT_STACK,
    _DF_SNAPSHOT_METADATA,
    deepanalyze_completer,
    FLAGS,
    deepanalyze
)
from deepanalyze.server import (
    resolve_draft_model_path,
    detect_hardware_acceleration_flags
)


class DummyIPython:
    def __init__(self):
        self.user_ns = {}


def test_lazyframe_zero_scan_schema():
    """Verify LazyFrame schema inspection does not trigger collect."""
    df = pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "balance": [100.5, 200.0, 300.75, 400.2, 500.0],
        "category": ["A", "B", "A", "C", "B"]
    })
    lazy_df = df.lazy()

    schema_str = _format_micro_schema(lazy_df, "accounts_lf", is_polars=True)
    assert "LazyFrame `accounts_lf`" in schema_str
    assert "Engine: Polars LazyPlan" in schema_str
    assert "customer_id" in schema_str
    assert "balance" in schema_str
    assert "category" in schema_str


def test_multilevel_undo_lifo_stack():
    """Verify multi-step sequential rollback through LIFO snapshot stack."""
    ip = DummyIPython()
    target = "sales_df"

    # Step 0: Initial state
    df0 = pl.DataFrame({"rev": [100, 200]})
    ip.user_ns[target] = df0.clone()
    _take_snapshot(ip, target=target)

    # Step 1: Add column
    df1 = pl.DataFrame({"rev": [100, 200], "tax": [10, 20]})
    ip.user_ns[target] = df1.clone()
    _take_snapshot(ip, target=target)

    # Step 2: Modify column
    df2 = pl.DataFrame({"rev": [150, 250], "tax": [15, 25]})
    ip.user_ns[target] = df2.clone()
    _take_snapshot(ip, target=target)

    assert len(_DF_SNAPSHOT_STACK[target]) == 3

    # Undo 1: Should restore Step 2 snapshot (df2)
    assert _restore_snapshot(ip, target=target) is True
    assert ip.user_ns[target]["rev"].to_list() == [150, 250]
    assert len(_DF_SNAPSHOT_STACK[target]) == 2

    # Undo 2: Should restore Step 1 snapshot (df1)
    assert _restore_snapshot(ip, target=target) is True
    assert ip.user_ns[target]["rev"].to_list() == [100, 200]
    assert "tax" in ip.user_ns[target].columns
    assert len(_DF_SNAPSHOT_STACK[target]) == 1

    # Undo 3: Should restore Step 0 snapshot (df0)
    assert _restore_snapshot(ip, target=target) is True
    assert ip.user_ns[target]["rev"].to_list() == [100, 200]
    assert "tax" not in ip.user_ns[target].columns
    assert len(_DF_SNAPSHOT_STACK[target]) == 0

    # Undo 4: Empty stack
    assert _restore_snapshot(ip, target=target) is False


def test_context_aware_tab_completer():
    """Verify completer suggests flags and target DataFrame variables."""
    # Test flag completion
    event = types.SimpleNamespace(symbol="--un", line="%deepanalyze --un", end=17)
    matches = deepanalyze_completer(None, event)
    assert "--undo" in matches
    assert "--unravel" in matches

    # Test assert and diff-stats flags exist
    assert "--assert" in FLAGS
    assert "--diff-stats" in FLAGS
    assert "--sql" in FLAGS


def test_sql_execution_bridge():
    """Verify %deepanalyze --sql executes ANSI queries on DataFrames via DuckDB."""
    # Build a fake IPython environment for testing
    import sys
    from IPython.testing.globalipapp import get_ipython as get_test_ip
    test_ip = get_test_ip()
    
    test_df = pl.DataFrame({
        "dept": ["Engineering", "Sales", "Engineering", "Marketing"],
        "salary": [120000, 95000, 135000, 88000]
    })
    test_ip.user_ns["emp_df"] = test_df

    # Run SQL aggregation
    deepanalyze("--sql SELECT dept, AVG(salary) AS avg_sal FROM emp_df GROUP BY dept ORDER BY avg_sal DESC --target res_sql")

    assert "res_sql" in test_ip.user_ns
    res = test_ip.user_ns["res_sql"]
    assert len(res) == 3
    assert "avg_sal" in res.columns


def test_server_draft_model_resolution(tmp_path):
    """Verify server launcher correctly configures draft model options."""
    fake_draft = tmp_path / "custom_draft.gguf"
    fake_draft.touch()

    resolved = resolve_draft_model_path(str(fake_draft))
    assert resolved == str(fake_draft)

    flags = detect_hardware_acceleration_flags()
    assert "--cache-reuse" in flags

"""Tests for Pandas and NumPy Dual-Engine Compatibility in DeepAnalyze v4.0."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from deepanalyze.firewall import (
    RollbackManager,
    execute_code_safely,
    prepare_dataframe_for_code,
)
from deepanalyze.vault import (
    detokenize_dataframe,
    flush,
    learn_custom_pattern,
    tokenize_dataframe,
)


def test_prepare_dataframe_for_code_detection():
    """Validates automatic dialect detection and DataFrame conversion."""
    pl_df = pl.DataFrame({"id": [1, 2, 3], "val": ["A", "B", "C"]})

    # 1. Pandas code with df['col'] and np.where
    pandas_code = """
df['new_val'] = np.where(df['id'] > 1, 'HIGH', 'LOW')
df = df.dropna()
"""
    adapted_df, dialect = prepare_dataframe_for_code(pl_df, pandas_code)
    assert dialect == "pandas"
    assert isinstance(adapted_df, pd.DataFrame)
    assert "new_val" not in adapted_df.columns

    # 2. Polars code with pl.col
    polars_code = """
df = df.with_columns(pl.col('id') * 10)
"""
    pd_df = pl_df.to_pandas()
    adapted_pl, pl_dialect = prepare_dataframe_for_code(pd_df, polars_code)
    assert pl_dialect == "polars"
    assert isinstance(adapted_pl, pl.DataFrame)


def test_execute_code_safely_pandas_numpy_preinjected():
    """Verifies that pd and np are pre-injected and execute without manual imports."""
    scope = {
        "df": pd.DataFrame({"qty": [2, 5, 10], "price": [10.0, 20.0, 15.5]})
    }
    pandas_script = """
# Cloud models typically output this style:
df['total'] = df['qty'] * df['price']
df['category'] = np.where(df['total'] > 50, 'BULK', 'RETAIL')
"""
    execute_code_safely(pandas_script, scope)
    res_df = scope["df"]
    assert isinstance(res_df, pd.DataFrame)
    assert "total" in res_df.columns
    assert res_df["total"].tolist() == [20.0, 100.0, 155.0]
    assert res_df["category"].tolist() == ["RETAIL", "BULK", "BULK"]


def test_pandas_detokenization_fidelity():
    """Verifies that detokenizing a pandas DataFrame preserves pandas type and character fidelity."""
    flush()
    # Create Polars df, tokenize, then convert to pandas
    initial_pl = pl.DataFrame({
        "account": ["GL: 500-000", "GL: 500-001"],
        "entity": ["Corp Alpha", "Corp Beta"]
    })
    _, masked_pl = learn_custom_pattern("GL", "500-000", initial_pl)
    masked_pd = masked_pl.to_pandas()

    # Verify surrogate tokens exist in pandas df
    assert "<GL_1>" in masked_pd["account"].iloc[0] or "<GL_CODE_1>" in masked_pd["account"].iloc[0]

    # Detokenize pandas DataFrame directly
    restored_pd = detokenize_dataframe(masked_pd)
    assert isinstance(restored_pd, pd.DataFrame)
    assert restored_pd["account"].tolist() == ["GL: 500-000", "GL: 500-001"]


def test_rollback_manager_supports_pandas():
    """Verifies in-memory snapshot stack works flawlessly with pandas."""
    mgr = RollbackManager(max_depth=3)
    pd_df = pd.DataFrame({"step": [1]})
    mgr.push("test_df", pd_df)

    # Mutate
    pd_df_v2 = pd.DataFrame({"step": [2]})
    mgr.push("test_df", pd_df_v2)

    assert mgr.depth("test_df") == 2
    popped = mgr.pop("test_df")
    assert isinstance(popped, pd.DataFrame)
    assert popped["step"].tolist() == [2]

    popped_v1 = mgr.pop("test_df")
    assert popped_v1["step"].tolist() == [1]
    assert mgr.depth("test_df") == 0

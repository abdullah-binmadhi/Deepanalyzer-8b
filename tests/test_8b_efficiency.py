"""Tests for 8B Local LLM Efficiency Stack:
- Dynamic AST Exemplar Bank
- Surgical Traceback Distillation
- Dynamic Categorical GBNF Enums
- Micro-Schema Compression
- Strict Output Delimiter Grammar (<Execute>...</Execute>)
- Structural Query Caching
"""

import inspect
import pandas as pd
import polars as pl
import pytest

from deepanalyze.core import (
    _retrieve_ast_exemplar,
    _distill_surgical_traceback,
    _build_dynamic_enum_grammar,
    _format_micro_schema,
    _extract_deepanalyze_content,
    _call_llm
)
from deepanalyze.brain import BiomimeticBrain


def test_min_p_sampling_support():
    """Verify _call_llm includes min_p in its signature."""
    sig = inspect.signature(_call_llm)
    assert "min_p" in sig.parameters
    assert sig.parameters["min_p"].default == 0.05


def test_ast_exemplar_retrieval():
    """Verify keyword to canonical Polars idiom matching."""
    res_rolling = _retrieve_ast_exemplar("Calculate 7-day rolling average of sales")
    assert "rolling_mean" in res_rolling
    assert "df.with_columns" in res_rolling

    res_when = _retrieve_ast_exemplar("If revenue > 1000 then High else Low")
    assert "pl.when" in res_when

    res_group = _retrieve_ast_exemplar("Group by region and calculate total sales")
    assert "group_by" in res_group

    res_unpivot = _retrieve_ast_exemplar("Unpivot wide quarterly columns into long format")
    assert "unpivot" in res_unpivot

    res_unknown = _retrieve_ast_exemplar("Do something random")
    assert res_unknown == ""


def test_surgical_traceback_distillation():
    """Verify pruning of stack frames and fuzzy column suggestion."""
    df = pl.DataFrame({
        "gross_revenue": [100.0, 200.0],
        "net_income": [50.0, 90.0]
    })
    
    # Simulate a missing column exception
    try:
        raise KeyError("column 'gross_rev' not found in schema")
    except Exception as exc:
        surgical = _distill_surgical_traceback(exc, "df.select(pl.col('gross_rev') * 2)", df)

    assert "FAILED AT: `gross_rev`" in surgical
    assert "gross_revenue" in surgical
    assert "DIRECTIVE: Fix the column reference" in surgical


def test_dynamic_enum_grammar():
    """Verify categorical value masking on low-cardinality columns."""
    df = pl.DataFrame({
        "status": ["ACTIVE", "PENDING", "ACTIVE", "CANCELLED"],
        "region": ["NA", "EMEA", "APAC", "NA"],
        "id": [f"ID_{i}" for i in range(4)]
    })

    enums = _build_dynamic_enum_grammar(df)
    assert "STRICT CATEGORICAL VALUE ENUMS" in enums
    assert "ACTIVE" in enums
    assert "PENDING" in enums
    assert "CANCELLED" in enums


def test_micro_schema_compression():
    """Verify single-line high-density micro-schema formatting."""
    df = pl.DataFrame({
        "client_name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
        "amount": [1250.50, 4500.00, 300.25],
        "is_active": [True, False, True]
    })

    schema_str = _format_micro_schema(df, "sales_df", is_polars=True)
    assert "[MICRO-SCHEMA CONTRACT (Case-Sensitive)]" in schema_str
    assert "sales_df" in schema_str
    assert "Range: [" in schema_str
    assert "amount" in schema_str
    # Verify concise single-line representation
    lines = [l for l in schema_str.splitlines() if l.strip().startswith("•")]
    assert len(lines) == 3


def test_strict_execute_delimiter_extraction():
    """Verify <Execute>...</Execute> extraction bypasses conversational preambles."""
    raw_response = (
        "Here is the requested Polars query for your task:\n"
        "<Execute>\n"
        "df = df.filter(pl.col('amount') > 1000)\n"
        "</Execute>\n"
        "This filters out low value transactions."
    )

    code, narrative = _extract_deepanalyze_content(raw_response)
    assert code == "df = df.filter(pl.col('amount') > 1000)"
    assert "Here is the requested Polars query" in narrative or "This filters out" in narrative


def test_structural_query_caching(tmp_path):
    """Verify in-memory structural query hash caching in BiomimeticBrain."""
    storage_file = str(tmp_path / "test_memory.json")
    b = BiomimeticBrain(storage_path=storage_file)

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    prompt = "Filter rows where a > 1"
    code = "df = df.filter(pl.col('a') > 1)"

    # Initially empty
    assert b.get_cached_query(prompt, df) is None

    # Cache query
    b.cache_verified_query(prompt, df, code)

    # Retrieval
    cached = b.get_cached_query(prompt, df)
    assert cached == code

    # Same prompt with different schema should not hit
    df_diff = pl.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    assert b.get_cached_query(prompt, df_diff) is None

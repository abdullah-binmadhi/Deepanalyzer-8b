"""Tests for Optimal High-Throughput Architecture Blueprint:
- orjson Rust serialization & numpy type safety
- Progressive FastEmbed / TF-IDF semantic search engine
- Cross-lingual semantic join
"""

import datetime
import numpy as np
import pandas as pd
import polars as pl
import pytest

from deepanalyze.brain import BiomimeticBrain, _json_dumps, _json_loads
from deepanalyze.enricher import (
    filter_by_semantic_meaning,
    cross_lingual_semantic_join,
    _compute_semantic_vectors
)


def test_orjson_numpy_type_serialization(tmp_path):
    """Verify orjson serializes numpy scalars, arrays, and datetimes without TypeError."""
    payload = {
        "int64_val": np.int64(42),
        "float32_val": np.float32(3.14),
        "array_val": np.array([1, 2, 3]),
        "timestamp": datetime.datetime.now()
    }

    serialized = _json_dumps(payload)
    assert isinstance(serialized, str)
    assert "42" in serialized

    deserialized = _json_loads(serialized)
    assert deserialized["int64_val"] == 42
    assert abs(deserialized["float32_val"] - 3.14) < 0.01


def test_orjson_brain_persistence(tmp_path):
    """Verify BiomimeticBrain memory loads and saves cleanly with orjson."""
    storage_path = str(tmp_path / "orjson_memory.json")
    brain = BiomimeticBrain(storage_path=storage_path)

    df = pl.DataFrame({
        "num_col": [np.int64(10), np.int64(20)],
        "float_col": [np.float64(1.5), np.float64(2.5)]
    })

    brain.log_execution_delta(
        df=df,
        code="df.select(pl.col('num_col') * 2)",
        success=True,
        duration_ms=np.float32(12.5)
    )

    # Reload from disk
    brain2 = BiomimeticBrain(storage_path=storage_path)
    assert len(brain2.memory.get("hardware_profiles", [])) == 1
    assert brain2.memory["hardware_profiles"][0]["duration_ms"] == 12.5


def test_semantic_vector_fallback():
    """Verify semantic vector computation works seamlessly."""
    texts = [
        "Urgent safety recall on vehicle airbag",
        "Routine quarterly maintenance report",
        "Customer invoice payment overdue"
    ]

    vectors = _compute_semantic_vectors(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0


def test_semantic_filter_execution():
    """Verify semantic intent filter returns most relevant rows."""
    df = pl.DataFrame({
        "issue_description": [
            "Brake fluid leaking near front wheels",
            "Billing discrepancy in monthly invoice",
            "Steering wheel vibrating at high speed",
            "Website login button unresponsive"
        ],
        "severity": ["High", "Low", "High", "Low"]
    })

    filtered = filter_by_semantic_meaning(df, query="car mechanical brake defect", top_k=2)
    assert len(filtered) == 2
    # First row should be the brake issue
    descriptions = filtered["issue_description"].to_list()
    assert "Brake fluid leaking near front wheels" in descriptions


def test_cross_lingual_semantic_join():
    """Verify cross-lingual join matches similar conceptual terms."""
    df_en = pl.DataFrame({
        "product_id": [1, 2],
        "name_en": ["Laptop Computer", "Wireless Mouse"]
    })

    df_ar = pl.DataFrame({
        "sku": [101, 102],
        "name_translit": ["Laptop Computar", "Mouse Wireless Device"]
    })

    joined = cross_lingual_semantic_join(df_en, df_ar, left_on="name_en", right_on="name_translit", threshold=0.1)
    assert len(joined) == 2
    assert "_weave_similarity" in joined.columns

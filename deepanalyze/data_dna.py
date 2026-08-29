"""Pillar 1: Data DNA Archetype Fingerprinting Engine (<5ms Profiler)
Computes zero-cost structural and content heuristics to classify datasets into
precise enterprise archetypes with deterministic schema signatures.
"""

import hashlib
import json
import re
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

# Archetype Constants
ARCHETYPE_ERP_HIERARCHICAL = "ERP_HIERARCHICAL_LEDGER"
ARCHETYPE_WIDE_TEMPORAL = "WIDE_TEMPORAL_MATRIX"
ARCHETYPE_SEMI_STRUCTURED_JSON = "SEMI_STRUCTURED_JSON_LOG"
ARCHETYPE_MESSY_TABULAR = "MESSY_DENORMALIZED_TABULAR"
ARCHETYPE_CLEAN_TABULAR = "CLEAN_ANALYTICAL_TABLE"


def generate_schema_signature(df_obj) -> str:
    """Generates a deterministic SHA-256 schema signature from normalized column names and types."""
    if df_obj is None:
        return "empty_schema"

    cols = []
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        cols = [f"{c}:{str(df_obj.schema[c])}" for c in df_obj.columns]
    elif isinstance(df_obj, pd.DataFrame):
        cols = [f"{c}:{str(df_obj[c].dtype)}" for c in df_obj.columns]
    elif hasattr(df_obj, "columns"):
        cols = [str(c) for c in df_obj.columns]

    norm_header = "|".join(sorted(cols))
    return hashlib.sha256(norm_header.encode("utf-8")).hexdigest()[:16]


def compute_data_dna(df_obj) -> dict:
    """Computes a multi-dimensional Data DNA vector in <5ms to classify dataset archetype."""
    if df_obj is None:
        return {"archetype": ARCHETYPE_CLEAN_TABULAR, "confidence": 1.0, "metrics": {}}

    cols = [str(c) for c in (df_obj.columns if hasattr(df_obj, "columns") else [])]
    n_cols = len(cols)
    if n_cols == 0:
        return {"archetype": ARCHETYPE_CLEAN_TABULAR, "confidence": 1.0, "metrics": {}}

    # 1. Unnamed header density
    unnamed_count = sum(1 for c in cols if c.startswith("__UNNAMED__") or "unnamed:" in c.lower() or c.strip() == "")
    unnamed_ratio = unnamed_count / max(n_cols, 1)

    # 2. Sample value scanning (fast 25-row slice)
    sample_text_vals = []
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        sample_df = df_obj.head(25)
        for c in sample_df.columns[:15]:
            if sample_df.schema[c] in (pl.String, pl.Utf8, pl.Categorical):
                sample_text_vals.extend([str(v) for v in sample_df[c].drop_nulls().to_list()])
    elif isinstance(df_obj, pd.DataFrame):
        sample_df = df_obj.head(25)
        for c in sample_df.columns[:15]:
            if sample_df[c].dtype == object or sample_df[c].dtype == "string":
                sample_text_vals.extend([str(v) for v in sample_df[c].dropna().tolist()])

    # 3. Currency / Accounting pattern density
    currency_regex = re.compile(r'[\$€£¥₹%]|SAR|AED|USD|EUR|RM|\(\d+[\d,.]*\)', re.I)
    currency_matches = sum(1 for v in sample_text_vals if currency_regex.search(v))
    currency_ratio = currency_matches / max(len(sample_text_vals), 1)

    # 4. JSON dictionary density
    json_matches = sum(1 for v in sample_text_vals if v.strip().startswith("{") and v.strip().endswith("}"))
    json_ratio = json_matches / max(len(sample_text_vals), 1)

    # 5. Temporal column density
    temporal_regex = re.compile(r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|\d{4}|\d{1,2}:\d{2}|hour_\d+|wk\d+)', re.I)
    temporal_cols = [c for c in cols if temporal_regex.search(c.strip())]
    temporal_col_ratio = len(temporal_cols) / max(n_cols, 1)

    # 6. Mojibake / Control char density
    mojibake_regex = re.compile(r'[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]|Ã[©¨§¹¢ê®°£¥]|â[€™€]', re.I)
    mojibake_matches = sum(1 for v in sample_text_vals if mojibake_regex.search(v))
    mojibake_ratio = mojibake_matches / max(len(sample_text_vals), 1)

    # 7. ERP structural keywords in headers/rows
    erp_keywords = {"doc. no", "doc_no", "invoice_total", "full_description", "grand total", "account summary", "uom", "gl-code"}
    erp_matches = sum(1 for c in cols if any(k in c.lower() for k in erp_keywords))
    sample_text_lower = " ".join(sample_text_vals[:50]).lower()
    erp_text_matches = sum(1 for k in erp_keywords if k in sample_text_lower)

    # Classification State Machine
    archetype = ARCHETYPE_CLEAN_TABULAR
    confidence = 0.85

    if unnamed_ratio >= 0.3 or (erp_matches >= 2 or erp_text_matches >= 2):
        archetype = ARCHETYPE_ERP_HIERARCHICAL
        confidence = 0.98 if unnamed_ratio > 0.4 else 0.92
    elif temporal_col_ratio >= 0.35 and len(temporal_cols) >= 3:
        archetype = ARCHETYPE_WIDE_TEMPORAL
        confidence = 0.96
    elif json_ratio >= 0.05:
        archetype = ARCHETYPE_SEMI_STRUCTURED_JSON
        confidence = 0.95
    elif currency_ratio >= 0.05 or mojibake_ratio >= 0.02:
        archetype = ARCHETYPE_MESSY_TABULAR
        confidence = 0.90

    signature = generate_schema_signature(df_obj)

    return {
        "archetype": archetype,
        "confidence": confidence,
        "schema_signature": signature,
        "metrics": {
            "unnamed_ratio": round(unnamed_ratio, 3),
            "currency_ratio": round(currency_ratio, 3),
            "json_ratio": round(json_ratio, 3),
            "temporal_cols_count": len(temporal_cols),
            "mojibake_ratio": round(mojibake_ratio, 3),
            "erp_keyword_matches": erp_matches + erp_text_matches
        }
    }

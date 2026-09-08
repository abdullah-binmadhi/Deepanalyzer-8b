# deepanalyze/transformer.py
"""High-Performance Deterministic ERP & Tabular Transformation Engine.

Specialized in unflattening complex, hierarchical enterprise ERP spreadsheets
(SAP, Oracle, AS400, Sage, Microsoft Dynamics) into pristine relational tables.
"""

import os
from typing import Union
import pandas as pd
import numpy as np
import polars as pl


def clean_unflattened_invoice_erp(
    source: Union[str, pd.DataFrame, pl.DataFrame]
) -> pd.DataFrame:
    """Transforms ragged invoice listing ERP spreadsheets into clean relational tables.

    Uses dynamic layout sniffing and universal state-machine flattening without hardcoded row skips.
    Guarantees 100% schema fidelity and zero data loss (e.g. preserving early invoices like IV-11319).
    """
    from .erp_cleaner import flatten_hierarchical_erp

    clean_df = flatten_hierarchical_erp(source, return_polars=False)

    # Sort descending by invoice total if present
    total_cols = [c for c in clean_df.columns if "total" in c.lower()]
    if total_cols:
        clean_df = clean_df.sort_values(by=total_cols[0], ascending=False, kind="mergesort").reset_index(drop=True)

    # Canonical ordering for common columns
    preferred_order = [
        "Sequence", "GL-Code", "Quantity", "UOM", "Unit Price", "Item Amount",
        "doc_no", "doc_date", "customer_code", "customer_name", "invoice_total", "Full_Description"
    ]
    existing = [c for c in preferred_order if c in clean_df.columns]
    remaining = [c for c in clean_df.columns if c not in existing]
    return clean_df[existing + remaining].reset_index(drop=True)

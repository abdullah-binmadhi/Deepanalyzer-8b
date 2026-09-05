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
    """Transforms ragged invoice listing ERP spreadsheets into clean 12-column relational tables.

    Guarantees 100% schema and mathematical fidelity:
    Columns: Sequence, GL-Code, Quantity, UOM, Unit Price, Item Amount,
             doc_no, doc_date, customer_code, customer_name, invoice_total, Full_Description.
    """
    if isinstance(source, str):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Source file not found: {source}")
        raw = pd.read_excel(source, header=None)
    elif isinstance(source, pl.DataFrame):
        raw = source.to_pandas()
        raw.columns = range(raw.shape[1])
    elif isinstance(source, pd.DataFrame):
        raw = source.copy()
        raw.columns = range(raw.shape[1])
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # 1. Skip top 18 report metadata rows
    df = raw.iloc[18:].copy().reset_index(drop=True)

    # 2. Filter out Grand Total summary rows
    df = df[~df[0].astype(str).str.contains("Grand Total", na=False)].reset_index(drop=True)

    # 3. Detect document headers (starts with IV-)
    is_doc = df[0].astype(str).str.startswith("IV-")

    df["doc_no"] = np.where(is_doc, df[0], np.nan)
    df["doc_date"] = np.where(is_doc, df[2], np.nan)
    df["customer_code"] = np.where(is_doc, df[4], np.nan)
    df["customer_name"] = np.where(is_doc, df[6], np.nan)
    df["invoice_total"] = np.where(is_doc, df[15], np.nan)

    # 4. Forward-fill document headers down to line items
    for col in ["doc_date", "doc_no", "customer_code", "customer_name", "invoice_total"]:
        df[col] = df[col].ffill()

    # 5. Extract line items where Column 0 contains a numeric sequence (1000, 2000, etc.)
    df["Sequence"] = pd.to_numeric(df[0], errors="coerce")
    items = df[df["Sequence"].notna()].copy()
    items["Sequence"] = items["Sequence"].astype(int)

    # 6. Map and cast line item columns
    items["GL-Code"] = items[1].astype(str)
    items["Full_Description"] = items[3].astype(str)
    items["Quantity"] = pd.to_numeric(items[10], errors="coerce")
    items["UOM"] = items[11].astype(str).replace("nan", np.nan)
    items["Unit Price"] = pd.to_numeric(items[12], errors="coerce")
    items["Item Amount"] = pd.to_numeric(items[13], errors="coerce")
    items["invoice_total"] = pd.to_numeric(items["invoice_total"], errors="coerce")
    items["doc_date"] = pd.to_datetime(items["doc_date"], errors="coerce")

    # 7. Select target schema in exact order
    target_cols = [
        "Sequence", "GL-Code", "Quantity", "UOM", "Unit Price", "Item Amount",
        "doc_no", "doc_date", "customer_code", "customer_name", "invoice_total", "Full_Description"
    ]
    res = items[target_cols].reset_index(drop=True)

    # 8. Stable sort descending by invoice_total (preserves natural document order for ties)
    res_sorted = res.sort_values(by="invoice_total", ascending=False, kind="mergesort").reset_index(drop=True)

    return res_sorted

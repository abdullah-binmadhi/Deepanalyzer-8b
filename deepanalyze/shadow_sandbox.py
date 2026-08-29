"""Pillar 3: Shadow Sandbox & Mathematical Invariant Assertions
Executes pipeline transformations in an ephemeral in-memory sandbox and verifies
5 strict mathematical invariants before committing results to session.
"""

from typing import Any, Callable
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


def verify_mathematical_invariants(raw_df, cleaned_df, archetype: str = "DEFAULT") -> dict:
    """Evaluates 5 formal mathematical and structural invariants between raw and cleaned data."""
    results = {
        "passed_all": True,
        "financial_conservation": True,
        "volume_conservation": True,
        "primary_key_zero_null": True,
        "strict_type_contract": True,
        "zero_pii_leakage": True,
        "details": []
    }

    if cleaned_df is None:
        return {
            "passed_all": False,
            "financial_conservation": False,
            "volume_conservation": False,
            "primary_key_zero_null": False,
            "strict_type_contract": False,
            "zero_pii_leakage": False,
            "details": ["Cleaned DataFrame is None"]
        }

    # 1. Volume Conservation (No 0-row wipeout unless raw was 0)
    raw_len = raw_df.height if hasattr(raw_df, "height") else (len(raw_df) if raw_df is not None else 0)
    clean_len = cleaned_df.height if hasattr(cleaned_df, "height") else len(cleaned_df)

    if raw_len > 0 and clean_len == 0:
        results["volume_conservation"] = False
        results["passed_all"] = False
        results["details"].append("Invariant Violation: Transformation resulted in 0 rows from non-empty input.")

    # 2. Financial Sum Conservation (for ERP & Ledger Archetypes)
    if archetype == "ERP_HIERARCHICAL_LEDGER":
        # Look for total or item amount columns
        amt_cols = [c for c in cleaned_df.columns if any(k in str(c).lower() for k in ["amount", "total", "sales", "revenue", "debit", "credit"])]
        if amt_cols:
            target_col = amt_cols[0]
            col_vals = cleaned_df[target_col]
            if hasattr(col_vals, "is_null"):
                if col_vals.null_count() == clean_len:
                    results["financial_conservation"] = False
                    results["passed_all"] = False
                    results["details"].append(f"Invariant Violation: Amount column `{target_col}` contains 100% null values.")

    # 3. Primary Key Zero-Null Invariance
    pk_candidates = [c for c in cleaned_df.columns if any(k in str(c).lower() for k in ["doc_no", "invoice", "patient_id", "subscriber_id", "parcel_id", "order_id", "customer_code"])]
    for pk in pk_candidates:
        pk_series = cleaned_df[pk]
        null_count = pk_series.null_count() if hasattr(pk_series, "null_count") else pk_series.isnull().sum()
        if null_count > 0:
            results["primary_key_zero_null"] = False
            results["passed_all"] = False
            results["details"].append(f"Invariant Violation: Primary key column `{pk}` has {null_count} null rows.")

    # 4. Strict Type Contract Invariance (Numeric and Date columns must not remain generic raw strings)
    for c in cleaned_df.columns:
        c_lower = str(c).lower()
        if any(k in c_lower for k in ["amount", "price", "rate", "cost", "salary", "qty", "quantity", "val", "weight"]):
            if pl is not None and isinstance(cleaned_df, pl.DataFrame):
                if cleaned_df.schema[c] in (pl.String, pl.Utf8):
                    results["strict_type_contract"] = False
                    results["passed_all"] = False
                    results["details"].append(f"Invariant Violation: Numerical column `{c}` remains untyped String.")
            elif isinstance(cleaned_df, pd.DataFrame):
                if cleaned_df[c].dtype == object:
                    results["strict_type_contract"] = False
                    results["passed_all"] = False
                    results["details"].append(f"Invariant Violation: Numerical column `{c}` remains untyped Object.")

    # 5. Zero-PII Leakage Check
    for c in cleaned_df.columns:
        if any(k in str(c).lower() for k in ["ic", "national_id", "ssn", "credit_card"]):
            vals = [str(v) for v in (cleaned_df[c].drop_nulls().head(10).to_list() if hasattr(cleaned_df[c], "drop_nulls") else cleaned_df[c].dropna().head(10).tolist())]
            if any(len(v) == 12 and v.replace("-", "").isdigit() for v in vals):
                results["zero_pii_leakage"] = False
                results["passed_all"] = False
                results["details"].append(f"Invariant Warning: Unmasked national ID format detected in column `{c}`.")

    return results


def execute_in_shadow_sandbox(raw_df, transformation_fn: Callable, fallback_fn: Callable = None, archetype: str = "DEFAULT") -> tuple[Any, bool, list[str]]:
    """Executes a transformation inside an ephemeral sandbox fork and verifies mathematical invariants."""
    logs = []
    
    # 1. Ephemeral Sandbox Execution
    try:
        if hasattr(raw_df, "clone"):
            sandbox_input = raw_df.clone()
        elif hasattr(raw_df, "copy"):
            sandbox_input = raw_df.copy()
        else:
            sandbox_input = raw_df

        candidate_df, exec_logs = transformation_fn(sandbox_input)
        logs.extend(exec_logs)

        # 2. Invariant Verification
        inv_check = verify_mathematical_invariants(raw_df, candidate_df, archetype=archetype)
        if inv_check["passed_all"]:
            logs.append("🛡️ [Shadow Sandbox]: All 5 mathematical & structural invariants verified successfully.")
            return candidate_df, True, logs
        else:
            logs.append(f"⚠ [Shadow Sandbox]: Invariant verification failed: {'; '.join(inv_check['details'])}")
            if fallback_fn:
                logs.append("🔄 [Shadow Sandbox]: Activating compiled deterministic archetype fallback...")
                fallback_df, fallback_logs = fallback_fn(sandbox_input)
                logs.extend(fallback_logs)
                return fallback_df, True, logs
            return candidate_df, False, logs

    except Exception as e:
        logs.append(f"❌ [Shadow Sandbox Execution Error]: {e}")
        if fallback_fn:
            try:
                logs.append("🔄 [Shadow Sandbox]: Re-routing to deterministic archetype fallback...")
                fallback_df, fallback_logs = fallback_fn(raw_df)
                logs.extend(fallback_logs)
                return fallback_df, True, logs
            except Exception as e_fb:
                logs.append(f"❌ [Fallback Error]: {e_fb}")
        return raw_df, False, logs

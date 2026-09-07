"""DeepAnalyze Automated Data Engineering Engine.

Provides automated domain feature profiling, synthetic macro engineering prompt generation,
and local 8B code stitching to transform cleaned data into analytical/ML-ready features.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import polars as pl


def profile_engineering_opportunities(df: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Scans cleaned dataset to discover high-value feature engineering opportunities.

    Categorizes opportunities into:
      - 'temporal': Date/time extraction, seasonality, weekend flags.
      - 'numerical': Ratios, log transformations, normalization, outlier flags.
      - 'categorical': High-cardinality grouping, frequency encoding, interactions.
      - 'text': String lengths, digit ratios, token extraction.
    """
    if hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame):
        try:
            pl_df = pl.from_pandas(df)
        except Exception:
            pl_df = pl.DataFrame(df)
    else:
        pl_df = df

    opportunities: Dict[str, List[Dict[str, Any]]] = {
        "temporal": [],
        "numerical": [],
        "categorical": [],
        "text": []
    }

    schema = pl_df.schema

    for col in pl_df.columns:
        dtype = schema[col]
        str_dtype = str(dtype).lower()

        # 1. Temporal Detection
        if "date" in str_dtype or "time" in str_dtype or any(term in col.lower() for term in ("date", "time", "created", "timestamp", "year")):
            opportunities["temporal"].append({
                "column": col,
                "suggested_features": [
                    f"{col}_day_of_week",
                    f"{col}_month",
                    f"{col}_is_weekend",
                    f"{col}_quarter"
                ],
                "rationale": "Capture temporal seasonality, cyclical patterns, and business day effects."
            })

        # 2. Numerical Detection
        elif "int" in str_dtype or "float" in str_dtype or "decimal" in str_dtype:
            try:
                min_val = pl_df[col].drop_nulls().min()
                max_val = pl_df[col].drop_nulls().max()
                has_skew = False
                if min_val is not None and max_val is not None and min_val > 0 and (max_val / (min_val + 1e-5)) > 100:
                    has_skew = True

                suggs = [f"{col}_outlier_iqr_flag"]
                if has_skew:
                    suggs.append(f"{col}_log1p")
                suggs.append(f"{col}_zscore_std")

                opportunities["numerical"].append({
                    "column": col,
                    "min": min_val,
                    "max": max_val,
                    "has_skew": has_skew,
                    "suggested_features": suggs,
                    "rationale": "Stabilize heavy-tailed variance and isolate anomalous spikes."
                })
            except Exception:
                pass

        # 3. Categorical & Text Detection
        elif "utf8" in str_dtype or "str" in str_dtype or "cat" in str_dtype:
            n_unique = pl_df[col].n_unique()
            try:
                avg_len = pl_df[col].drop_nulls().str.len_bytes().mean() or 0
                sample_vals = [str(v) for v in pl_df[col].drop_nulls()[:5]]
                has_spaces = any(" " in s for s in sample_vals)
            except Exception:
                avg_len = 0
                has_spaces = False

            if avg_len > 15 or (has_spaces and n_unique == pl_df.height):
                # Free-text detection
                opportunities["text"].append({
                    "column": col,
                    "cardinality": n_unique,
                    "suggested_features": [
                        f"{col}_char_len",
                        f"{col}_word_count"
                    ],
                    "rationale": "Extract structural token density metrics from free-text attributes."
                })
            elif 2 <= n_unique <= 100:
                opportunities["categorical"].append({
                    "column": col,
                    "cardinality": n_unique,
                    "suggested_features": [
                        f"{col}_freq_encoded",
                        f"{col}_one_hot"
                    ],
                    "rationale": "Vectorize discrete categories for statistical clustering and regression."
                })

    return opportunities


def build_engineering_briefing(df: Any, dataset_name: str = "cleaned_data", user_goal: str = "") -> str:
    """Creates a sanitized Engineering Briefing prompt for Frontier models."""
    if hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame):
        try:
            pl_df = pl.from_pandas(df)
        except Exception:
            pl_df = pl.DataFrame(df)
    else:
        pl_df = df

    opps = profile_engineering_opportunities(pl_df)

    briefing = [
        f"# Feature Engineering & Predictive Enhancement Briefing: {dataset_name}",
        f"**Shape**: {pl_df.height} rows x {pl_df.width} columns",
        f"**User Objective**: {user_goal or 'General analytics, clustering, and predictive feature enrichment'}\n",
        "## Cleaned Dataset Schema & Statistics",
        "```json",
        "{"
    ]

    col_entries = []
    for col in pl_df.columns:
        dtype = str(pl_df.schema[col])
        n_null = pl_df[col].null_count()
        n_uniq = pl_df[col].n_unique()
        col_entries.append(f'  "{col}": {{"type": "{dtype}", "nulls": {n_null}, "unique": {n_uniq}}}')
    briefing.append(",\n".join(col_entries))
    briefing.append("}\n```\n")

    briefing.append("## Discovered Feature Engineering Candidates")
    for category, items in opps.items():
        if items:
            briefing.append(f"### {category.capitalize()} Dimensions")
            for item in items[:4]:
                col_name = item["column"]
                suggs = ", ".join(item.get("suggested_features", []))
                rat = item.get("rationale", "")
                briefing.append(f"- `{col_name}`: Propose [{suggs}] ({rat})")

    briefing.append("\n## Engineering Requirements")
    briefing.append("1. Write an executable Python function or script using **Polars** (or Pandas if preferred).")
    briefing.append("2. Assume input DataFrame is already loaded as variable `df`.")
    briefing.append("3. Create 3 to 6 high-value derived features that enhance business insight.")
    briefing.append("4. Ensure all denominators protect against division by zero (e.g. `(a / (b + 1e-6))`).")
    briefing.append("5. Store transformed DataFrame back in `df`.")

    return "\n".join(briefing)


def apply_quick_features(df: Any, selected_types: Optional[List[str]] = None) -> Tuple[Any, List[str]]:
    """Applies high-speed deterministic feature engineering in RAM using pure Polars.

    Returns:
        (transformed_df, list_of_new_column_names)
    """
    is_pandas = hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame)
    if is_pandas:
        try:
            pl_df = pl.from_pandas(df)
        except Exception:
            pl_df = pl.DataFrame(df)
    else:
        pl_df = df

    selected_types = selected_types or ["temporal", "numerical", "categorical", "text"]
    opps = profile_engineering_opportunities(pl_df)
    transformed_df = pl_df.clone()
    new_cols = []

    # 1. Temporal
    if "temporal" in selected_types:
        for item in opps.get("temporal", []):
            col = item["column"]
            try:
                # Try parsing if string
                col_type = str(transformed_df.schema[col]).lower()
                parsed_col = pl.col(col)
                if "utf8" in col_type or "str" in col_type:
                    parsed_col = pl.col(col).str.to_datetime(strict=False)

                transformed_df = transformed_df.with_columns([
                    parsed_col.dt.weekday().alias(f"{col}_weekday"),
                    (parsed_col.dt.weekday() >= 5).cast(pl.Int32).alias(f"{col}_is_weekend"),
                    parsed_col.dt.month().alias(f"{col}_month")
                ])
                new_cols.extend([f"{col}_weekday", f"{col}_is_weekend", f"{col}_month"])
            except Exception:
                pass

    # 2. Numerical
    if "numerical" in selected_types:
        for item in opps.get("numerical", []):
            col = item["column"]
            try:
                mean = transformed_df[col].drop_nulls().mean()
                std = transformed_df[col].drop_nulls().std()
                if mean is not None and std is not None and std > 0:
                    transformed_df = transformed_df.with_columns([
                        ((pl.col(col) - mean) / std).alias(f"{col}_zscore")
                    ])
                    new_cols.append(f"{col}_zscore")

                if item.get("has_skew"):
                    transformed_df = transformed_df.with_columns([
                        (pl.col(col) + 1.0).log().alias(f"{col}_log1p")
                    ])
                    new_cols.append(f"{col}_log1p")
            except Exception:
                pass

    # 3. Categorical frequency encoding
    if "categorical" in selected_types:
        for item in opps.get("categorical", [])[:3]:
            col = item["column"]
            try:
                counts = transformed_df.group_by(col).len().rename({"len": f"{col}_freq"})
                transformed_df = transformed_df.join(counts, on=col, how="left")
                new_cols.append(f"{col}_freq")
            except Exception:
                pass

    # 4. Text metrics
    if "text" in selected_types:
        for item in opps.get("text", [])[:2]:
            col = item["column"]
            try:
                transformed_df = transformed_df.with_columns([
                    pl.col(col).str.len_bytes().alias(f"{col}_char_len"),
                    pl.col(col).str.count_matches(r"\s+").fill_null(0).alias(f"{col}_space_count")
                ])
                new_cols.extend([f"{col}_char_len", f"{col}_space_count"])
            except Exception:
                pass

    if is_pandas:
        try:
            return transformed_df.to_pandas(), new_cols
        except Exception:
            return transformed_df, new_cols
    return transformed_df, new_cols


def stitch_code_with_local_model(
    frontier_code: str,
    target_df: pl.DataFrame,
    df_var: str = "df",
    objective_hint: str = ""
) -> Tuple[bool, str, str]:
    """Uses local DeepAnalyze 8B model to adapt and stitch Frontier macro code to exact local columns.

    Returns:
        (success, stitched_code, diagnosis_notes)
    """
    from .client import request_model_fix

    schema_info = {
        "columns": list(target_df.columns),
        "shape": target_df.shape,
        "dtypes": {col: str(target_df.schema[col]) for col in target_df.columns}
    }

    stitch_prompt = (
        f"You are the DeepAnalyze Local Code Stitcher. A frontier model generated this feature engineering code:\n"
        f"```python\n{frontier_code}\n```\n\n"
        f"The local dataset variable is `{df_var}` with EXACT columns:\n{list(target_df.columns)}\n\n"
        f"Task:\n"
        f"1. Replace any assumed or misspelled column names with the matching exact column name from the local schema.\n"
        f"2. Ensure division by zero protection (`+ 1e-6`).\n"
        f"3. Return ONLY valid, executable Python code (Polars or Pandas) modifying `{df_var}` in place."
    )

    try:
        diag, stitched_code = request_model_fix(
            failed_code=frontier_code,
            traceback_str="Feature Engineering Column Alignment & Stitching Request",
            autopsy_str=stitch_prompt,
            custom_prompt=objective_hint or "Stitch and align feature engineering code to local columns.",
            target_name=df_var,
            schema_info=schema_info,
            timeout=120.0
        )
        return True, stitched_code, diag
    except Exception as e:
        return False, frontier_code, f"Local model stitching error: {str(e)}"

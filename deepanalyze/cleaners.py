"""
DeepAnalyze Cleaners & Transformation Subsystem
Modular suite of 8 specialized data cleaning, sanitization, and restructuring engines.
"""

import difflib
import html
import json
import re
import unicodedata
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


# =============================================================================
# 1. UNICODE & MOJIBAKE SANITIZER (--ftfy)
# =============================================================================
MOJIBAKE_MAP = {
    "Ã©": "é", "Ã¨": "è", "Ã ": "à", "Ã§": "ç", "Ã¹": "ù", "Ã¢": "â", "Ãª": "ê",
    "Ã®": "î", "Ã´": "ô", "Ã»": "û", "Ã«": "ë", "Ã¯": "ï", "Ã¼": "ü", "Ã¶": "ö",
    "Ã¤": "ä", "Ã\x9f": "ß", "Ã±": "ñ", "Ã¡": "á", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-",
    "â€¦": "...", "Â©": "©", "Â®": "®", "Â°": "°", "Â£": "£", "Â¥": "¥", "â‚¬": "€"
}

def sanitize_unicode_string(val: str) -> str:
    """Repairs Mojibake, strips invisible zero-width chars, non-breaking spaces, and unescapes HTML."""
    if not isinstance(val, str):
        return val
    
    # 1. Fix Mojibake substitutions
    for bad, good in MOJIBAKE_MAP.items():
        if bad in val:
            val = val.replace(bad, good)

    # 2. HTML Unescape (&amp; -> &, &quot; -> ", &#39; -> ')
    if "&" in val:
        val = html.unescape(val)

    # 3. Strip zero-width & non-breaking characters
    val = re.sub(r'[\u200b\u200c\u200d\ufeff\u202a-\u202e]', '', val)
    val = val.replace('\xa0', ' ').replace('\u202f', ' ')

    # 4. Unicode NFC Normalization
    val = unicodedata.normalize("NFC", val)
    return val.strip()

def sanitize_unicode_and_mojibake(df_obj):
    """Sanitizes text columns across all string/Utf8 fields in Polars or Pandas."""
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        df = df_obj.clone()
        for col in df.columns:
            if df.schema[col] in (pl.String, pl.Utf8, pl.Categorical):
                vals = df[col].to_list()
                clean_vals = [sanitize_unicode_string(v) if isinstance(v, str) else v for v in vals]
                df = df.with_columns(pl.Series(col, clean_vals))
        return df
    elif isinstance(df_obj, pd.DataFrame):
        df = df_obj.copy()
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                df[col] = df[col].map(lambda x: sanitize_unicode_string(x) if isinstance(x, str) else x)
        return df
    return df_obj


# =============================================================================
# 2. ENTITY RESOLUTION & FUZZY DEDUPLICATION (--fuzzy-clean)
# =============================================================================
def _get_acronym_initials(text: str) -> str:
    words = [w for w in re.split(r'[^a-zA-Z0-9]+', text) if w]
    return "".join(w[0].lower() for w in words) if len(words) > 1 else ""

def fuzzy_harmonize_categories(df_obj, threshold: float = 0.85):
    """Clusters and unifies near-identical categorical string variants (and acronyms) to dominant canonical categories."""
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        df = df_obj.clone()
        for col in df.columns:
            if df.schema[col] in (pl.String, pl.Utf8, pl.Categorical):
                val_counts = df[col].drop_nulls().value_counts()
                if val_counts.height < 2 or val_counts.height > 1000:
                    continue
                
                count_col = "count" if "count" in val_counts.columns else "counts"
                sorted_vals = val_counts.sort(count_col, descending=True)[col].to_list()
                
                canonical_map = {}
                canonical_list = []
                for val in sorted_vals:
                    val_str = str(val).strip()
                    val_lower = val_str.lower()
                    v_clean = re.sub(r'[^a-z0-9]', '', val_lower)
                    v_initials = _get_acronym_initials(val_str)
                    matched = False
                    for canon in canonical_list:
                        canon_str = str(canon).strip()
                        c_clean = re.sub(r'[^a-z0-9]', '', canon_str.lower())
                        c_initials = _get_acronym_initials(canon_str)
                        
                        # 1. Clean alphanumeric match (e.g. "U.S.A." == "USA")
                        # 2. Acronym match (e.g. "USA" / "US" matches initials of "United States")
                        # 3. String edit distance ratio >= threshold (e.g. "Unted States" vs "United States")
                        is_initials_match = (v_clean in (c_initials, c_initials + 'a') and len(v_clean) >= 2) or (c_clean in (v_initials, v_initials + 'a') and len(c_clean) >= 2)
                        if c_clean == v_clean or is_initials_match or difflib.SequenceMatcher(None, val_lower, canon_str.lower()).ratio() >= threshold:
                            canonical_map[val] = canon
                            matched = True
                            break
                    if not matched:
                        canonical_list.append(val)
                        canonical_map[val] = val

                series_vals = df[col].to_list()
                mapped_vals = [canonical_map.get(v, v) if v is not None else None for v in series_vals]
                df = df.with_columns(pl.Series(col, mapped_vals))
        return df

    elif isinstance(df_obj, pd.DataFrame):
        df = df_obj.copy()
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                val_counts = df[col].dropna().value_counts()
                if len(val_counts) < 2 or len(val_counts) > 1000:
                    continue
                sorted_vals = val_counts.index.tolist()
                canonical_map = {}
                canonical_list = []
                for val in sorted_vals:
                    val_str = str(val).strip()
                    val_lower = val_str.lower()
                    v_clean = re.sub(r'[^a-z0-9]', '', val_lower)
                    v_initials = _get_acronym_initials(val_str)
                    matched = False
                    for canon in canonical_list:
                        canon_str = str(canon).strip()
                        c_clean = re.sub(r'[^a-z0-9]', '', canon_str.lower())
                        c_initials = _get_acronym_initials(canon_str)
                        is_initials_match = (v_clean in (c_initials, c_initials + 'a') and len(v_clean) >= 2) or (c_clean in (v_initials, v_initials + 'a') and len(c_clean) >= 2)
                        if c_clean == v_clean or is_initials_match or difflib.SequenceMatcher(None, val_lower, canon_str.lower()).ratio() >= threshold:
                            canonical_map[val] = canon
                            matched = True
                            break
                    if not matched:
                        canonical_list.append(val)
                        canonical_map[val] = val
                df[col] = df[col].map(lambda x: canonical_map.get(x, x) if pd.notna(x) else x)
        return df
    return df_obj


# =============================================================================
# 3. SEMI-STRUCTURED & NESTED JSON/STRUCT EXPLODER (--explode)
# =============================================================================
def explode_nested_json(df_obj):
    """Detects embedded JSON dictionaries/lists in string columns and unrolls them into top-level columns."""
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        df = df_obj.clone()
        for col in df.columns:
            if df.schema[col] in (pl.String, pl.Utf8):
                sample_vals = [str(v).strip() for v in df[col].drop_nulls().head(10).to_list()]
                is_json_dict = all(v.startswith("{") and v.endswith("}") for v in sample_vals if v)
                if is_json_dict and len(sample_vals) > 0:
                    parsed_rows = []
                    for val in df[col].to_list():
                        if val is None:
                            parsed_rows.append({})
                        else:
                            try:
                                parsed_rows.append(json.loads(val))
                            except Exception:
                                parsed_rows.append({})
                    if parsed_rows:
                        all_keys = []
                        for r in parsed_rows:
                            for k in r.keys():
                                if k not in all_keys:
                                    all_keys.append(k)
                        for k in all_keys:
                            new_col_name = f"{col}_{k}"
                            col_vals = [r.get(k, None) for r in parsed_rows]
                            df = df.with_columns(pl.Series(new_col_name, col_vals))
                        df = df.drop(col)
        return df

    elif isinstance(df_obj, pd.DataFrame):
        df = df_obj.copy()
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                sample_vals = [str(v).strip() for v in df[col].dropna().head(10).tolist()]
                is_json_dict = all(v.startswith("{") and v.endswith("}") for v in sample_vals if v)
                if is_json_dict and len(sample_vals) > 0:
                    parsed_series = df[col].map(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("{") else (x if isinstance(x, dict) else {}))
                    json_df = pd.json_normalize(parsed_series)
                    json_df.columns = [f"{col}_{k}" for k in json_df.columns]
                    df = pd.concat([df.drop(columns=[col]), json_df], axis=1)
        return df
    return df_obj


# =============================================================================
# 4. WIDE-TO-LONG MATRIX UNPIVOTER (--unpivot)
# =============================================================================
def unpivot_temporal_matrix(df_obj):
    """Detects wide temporal headers (months, quarters, years) and unpivots them into tidy rows."""
    temporal_pattern = re.compile(
        r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|fy\d{2,4}|\d{4}|y\d{4}|\d{4}[-_/]\d{2})',
        re.IGNORECASE
    )
    cols = df_obj.columns if hasattr(df_obj, "columns") else []
    temporal_cols = [c for c in cols if temporal_pattern.search(str(c).strip())]
    id_cols = [c for c in cols if c not in temporal_cols]

    if len(temporal_cols) >= 3 and id_cols:
        if pl is not None and isinstance(df_obj, pl.DataFrame):
            return df_obj.unpivot(index=id_cols, on=temporal_cols, variable_name="period", value_name="value")
        elif isinstance(df_obj, pd.DataFrame):
            return pd.melt(df_obj, id_vars=id_cols, value_vars=temporal_cols, var_name="period", value_name="value")
    return df_obj


# =============================================================================
# 5. MIXED UNITS & CURRENCY NORMALIZER (--convert-units)
# =============================================================================
UNIT_CONVERSIONS = {
    # Mass to kg
    "g": 0.001, "gram": 0.001, "grams": 0.001, "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592, "oz": 0.0283495, "ounce": 0.0283495,
    # Length to meter
    "m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "centimeter": 0.01, "mm": 0.001,
    "km": 1000.0, "kilometer": 1000.0, "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "yd": 0.9144, "yard": 0.9144, "mi": 1609.34, "mile": 1609.34,
}

def parse_unit_value(s: str) -> float:
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s_clean = s.strip().replace(",", "")
    is_neg = s_clean.startswith("(") and s_clean.endswith(")")
    if is_neg:
        s_clean = s_clean[1:-1].strip()

    match = re.match(r'^[^\d-]*(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+)?$', s_clean)
    if match:
        num = float(match.group(1))
        if is_neg: num = -num
        unit = (match.group(2) or "").lower()
        multiplier = UNIT_CONVERSIONS.get(unit, 1.0)
        return num * multiplier
    
    num_sub = re.sub(r'[^0-9.-]', '', s_clean)
    if num_sub:
        try:
            val = float(num_sub)
            return -val if is_neg else val
        except ValueError:
            return np.nan
    return np.nan

def normalize_units_and_currencies(df_obj):
    """Normalizes columns with mixed unit or currency representations."""
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        df = df_obj.clone()
        for col in df.columns:
            if df.schema[col] in (pl.String, pl.Utf8):
                sample_vals = df[col].drop_nulls().head(10).to_list()
                if any(re.search(r'[\$€£¥₹]|SAR|AED|USD|EUR|kg|lbs|gram|meter|km|miles', str(v), re.I) for v in sample_vals):
                    parsed_vals = [parse_unit_value(v) for v in df[col].to_list()]
                    df = df.with_columns(pl.Series(col, parsed_vals).cast(pl.Float64))
        return df
    elif isinstance(df_obj, pd.DataFrame):
        df = df_obj.copy()
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                sample_vals = df[col].dropna().head(10).tolist()
                if any(re.search(r'[\$€£¥₹]|SAR|AED|USD|EUR|kg|lbs|gram|meter|km|miles', str(v), re.I) for v in sample_vals):
                    df[col] = df[col].map(parse_unit_value).astype(float)
        return df
    return df_obj


# =============================================================================
# 6. STATISTICAL OUTLIER & TYPO GUARD (--winsorize)
# =============================================================================
def winsorize_numeric_outliers(df_obj, lower_p: float = 0.01, upper_p: float = 0.99):
    """Winsorizes extreme data entry outliers to the specified percentiles across numeric columns."""
    if pl is not None and isinstance(df_obj, pl.DataFrame):
        df = df_obj.clone()
        for col in df.columns:
            dtype = df.schema[col]
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                s = df[col].drop_nulls()
                if s.len() >= 10:
                    low_val = s.quantile(lower_p)
                    high_val = s.quantile(upper_p)
                    if low_val is not None and high_val is not None and low_val < high_val:
                        df = df.with_columns(
                            pl.col(col).clip(lower_bound=low_val, upper_bound=high_val).alias(col)
                        )
        return df
    elif isinstance(df_obj, pd.DataFrame):
        df = df_obj.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                if len(s) >= 10:
                    low_val = s.quantile(lower_p)
                    high_val = s.quantile(upper_p)
                    if pd.notna(low_val) and pd.notna(high_val) and low_val < high_val:
                        df[col] = df[col].clip(lower=low_val, upper=high_val)
        return df
    return df_obj


# =============================================================================
# 7. AUTOMATIC DATA TYPE & BOOLEAN ASSERTER (--auto-type)
# =============================================================================
def auto_cast_data_types(df_obj):
    """Coerces string columns to booleans, floats, or integers where possible without losing data."""
    bool_true = {"true", "t", "yes", "y", "1"}
    bool_false = {"false", "f", "no", "n", "0"}

    if pl is not None and isinstance(df_obj, pl.DataFrame):
        df = df_obj.clone()
        for col in df.columns:
            if df.schema[col] in (pl.String, pl.Utf8):
                clean_s = df[col].drop_nulls()
                if clean_s.len() == 0:
                    continue
                vals_lower = set(str(v).strip().lower() for v in clean_s.head(100).to_list())
                if vals_lower.issubset(bool_true | bool_false) and len(vals_lower) > 0:
                    df = df.with_columns(
                        pl.when(pl.col(col).str.to_lowercase().is_in(list(bool_true)))
                          .then(True)
                          .when(pl.col(col).str.to_lowercase().is_in(list(bool_false)))
                          .then(False)
                          .otherwise(None)
                          .alias(col)
                    )
                elif all(re.match(r'^-?\d+(?:\.\d+)?$', str(v).strip().replace(',', '')) for v in clean_s.head(50).to_list()):
                    try:
                        df = df.with_columns(pl.col(col).str.replace_all(',', '').cast(pl.Float64, strict=False))
                    except Exception:
                        pass
        return df

    elif isinstance(df_obj, pd.DataFrame):
        df = df_obj.copy()
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                clean_s = df[col].dropna()
                if len(clean_s) == 0:
                    continue
                vals_lower = set(str(v).strip().lower() for v in clean_s.head(100).tolist())
                if vals_lower.issubset(bool_true | bool_false) and len(vals_lower) > 0:
                    df[col] = df[col].map(lambda x: True if str(x).strip().lower() in bool_true else (False if str(x).strip().lower() in bool_false else np.nan))
                elif all(re.match(r'^-?\d+(?:\.\d+)?$', str(v).strip().replace(',', '')) for v in clean_s.head(50).tolist()):
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        return df
    return df_obj


# =============================================================================
# 8. RELATIONAL MULTI-TABLE AUTO-LINKER (--stitch)
# =============================================================================
def auto_stitch_dataframes(dfs_dict: dict) -> tuple[object, list[str]]:
    """Infers primary-foreign key relationships across session DataFrames and merges them."""
    if len(dfs_dict) < 2:
        for k, v in dfs_dict.items():
            return v, [f"Only 1 DataFrame `{k}` present. No join necessary."]

    table_names = list(dfs_dict.keys())
    primary_name = table_names[0]
    base_df = dfs_dict[primary_name]
    join_log = []

    for other_name in table_names[1:]:
        other_df = dfs_dict[other_name]
        common_cols = [c for c in base_df.columns if c in other_df.columns and not str(c).startswith("column_")]
        if common_cols:
            join_key = common_cols[0]
            try:
                if pl is not None and isinstance(base_df, pl.DataFrame) and isinstance(other_df, pl.DataFrame):
                    base_df = base_df.join(other_df, on=join_key, how="left", suffix=f"_{other_name}")
                    join_log.append(f"🔗 Linked `{primary_name}` ↔ `{other_name}` on key `{join_key}` (Left Join)")
                elif isinstance(base_df, pd.DataFrame) and isinstance(other_df, pd.DataFrame):
                    base_df = base_df.merge(other_df, on=join_key, how="left", suffixes=("", f"_{other_name}"))
                    join_log.append(f"🔗 Linked `{primary_name}` ↔ `{other_name}` on key `{join_key}` (Left Join)")
            except Exception as e_join:
                join_log.append(f"⚠ Failed join between `{primary_name}` and `{other_name}`: {e_join}")
        else:
            join_log.append(f"⚪ No matching keys found between `{primary_name}` and `{other_name}`.")

    return base_df, join_log

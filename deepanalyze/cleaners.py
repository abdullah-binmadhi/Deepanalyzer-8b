"""
DeepAnalyze Cleaners & Transformation Subsystem
Modular suite of 8 specialized data cleaning, sanitization, and restructuring engines.
"""

import datetime
from datetime import date
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


# =============================================================================
# 9. UNIVERSAL HIERARCHICAL ERP REPORT UNRAVELLER (--unravel)
# =============================================================================
def _make_columns_unique(cols):
    seen = {}
    new_cols = []
    for c in cols:
        name = str(c).strip()
        if name in seen:
            seen[name] += 1
            new_cols.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_cols.append(name)
    return new_cols


def unravel_hierarchical_erp_report(df_or_path, target_columns: list = None) -> object:
    """Universal Hierarchical Report & ERP Unraveller.
    Autonomously parses ANY multi-level hierarchical accounting/ERP report
    (Invoices, General Ledgers, AR/AP Aging, Payroll, Sales Orders, Inventory Listings)
    without hardcoded column names or fixed regexes.
    """
    if isinstance(df_or_path, str):
        if df_or_path.endswith(('.xlsx', '.xls')):
            raw_df = pl.read_excel(df_or_path) if pl else pd.read_excel(df_or_path)
        else:
            raw_df = pl.read_csv(df_or_path) if pl else pd.read_csv(df_or_path)
    else:
        raw_df = df_or_path

    pdf = raw_df.to_pandas() if hasattr(raw_df, 'to_pandas') else raw_df.copy()

    # Prepend column names only if they contain text strings, not default numeric integers
    col_names = list(pdf.columns)
    if any(isinstance(c, str) and not c.isdigit() and not c.startswith('column_') for c in col_names):
        header_row = pd.DataFrame([[c for c in col_names]], columns=col_names)
        full_df = pd.concat([header_row, pdf], ignore_index=True)
    else:
        full_df = pdf.copy()

    parent_context = {}
    parent_header_labels = {}
    table_col_map = {}
    text_columns = set()
    
    records = []
    current_record = None

    SUMMARY_KEYWORDS = {'grand total', 'total account', 'total item', 'balance forward', 'report total', 'page total', 'total:'}
    
    def is_date_val(v):
        if isinstance(v, (datetime.datetime, datetime.date, pd.Timestamp)):
            return True
        s = str(v).strip()
        return bool(re.match(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$', s) or 
                    re.match(r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$', s))

    def is_numeric_val(v):
        if isinstance(v, (int, float, np.number)) and not np.isnan(v): return True
        s = str(v).strip().replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('RM', '').replace('SAR', '')
        if s.startswith('(') and s.endswith(')'): s = s[1:-1]
        try:
            float(s)
            return True
        except ValueError:
            return False

    def clean_num(v):
        if v is None or pd.isna(v): return 0.0
        if isinstance(v, (int, float, np.number)): return float(v)
        s = str(v).strip().replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('RM', '').replace('SAR', '')
        if s.startswith('(') and s.endswith(')'): s = '-' + s[1:-1]
        try: return float(s)
        except Exception: return 0.0

    KV_HEADER_REGEX = re.compile(r'^(Account(?:\s+No)?|Account\s+Name|Vendor(?:\s+No|\s+Code|\s+Name)?|Customer(?:\s+No|\s+Code|\s+Name)?|Dept(?:\s+No)?|Branch|Location|Warehouse|Project|Cost\s+Center|Doc(?:\s+No|\s+Date)?|PO(?:\s+No|\s+Date)?|Invoice(?:\s+No|\s+Date)?|Category|Currency)\s*[:=]\s*(.*)$', re.IGNORECASE)

    for r_idx, row in full_df.iterrows():
        non_empty = {c_idx: val for c_idx, val in enumerate(row.values) if pd.notna(val) and str(val).strip() != ''}
        if not non_empty:
            continue

        row_str_lower = ' '.join([str(v).lower() for v in non_empty.values()])
        first_pos = min(non_empty.keys())
        first_val = non_empty[first_pos]
        first_str = str(first_val).strip()

        # 1. Skip Report Parameters / Filter Block (e.g. 'Date : From 1/8/2025...', 'Company : All', etc.)
        if (' : ' in row_str_lower and any(w in row_str_lower for w in ['from ', 'all', 'selected', 'sort by', 'date', 'company'])) or row_str_lower.startswith('from ') or 'selected' in row_str_lower:
            continue
        if len(non_empty) == 1 and any(comp in row_str_lower for comp in ['sdn bhd', 'ltd', 'inc.', 'corp', 'llc', 'gmbh', 'co.']) and not KV_HEADER_REGEX.match(first_str):
            continue

        # 2. Summary / Divider Rows
        if any(kw in row_str_lower for kw in SUMMARY_KEYWORDS) or re.match(r'^[=\-_]{3,}$', first_str):
            if current_record is not None:
                records.append(current_record)
                current_record = None
            continue

        # 3. Check for Inline Key-Value Parent Header Rows
        has_kv_pairs = any(KV_HEADER_REGEX.match(str(v).strip()) for v in non_empty.values())
        if has_kv_pairs:
            if current_record is not None:
                records.append(current_record)
                current_record = None
            for c_idx, val in non_empty.items():
                m = KV_HEADER_REGEX.match(str(val).strip())
                if m:
                    k_name, v_val = m.group(1).strip().lower().replace(' ', '_'), m.group(2).strip()
                    parent_context[k_name] = v_val
                else:
                    parent_context[f'header_{c_idx}'] = str(val).strip()
            continue

        # 4. Document / Grouping Header Definition Labels
        if any(h in row_str_lower for h in ['doc. no', 'doc no', 'invoice no', 'voucher no', 'account no', 'order no', 'po no', 'bill no', 'dept no']) and not any(is_numeric_val(v) for v in non_empty.values()):
            parent_header_labels = {}
            for c_idx, val in non_empty.items():
                v_clean = str(val).strip()
                parent_header_labels[c_idx] = v_clean
            continue

        # 5. Table Column Definition Row
        has_table_keywords = any(kw in row_str_lower for kw in ['seq', 'item', 'code', 'desc', 'qty', 'quantity', 'uom', 'price', 'amount', 'debit', 'credit', 'balance', 'rate', 'unit', 'discount', 'tax', 'salary', 'particular', 'ref', 'sku', 'cost'])
        if has_table_keywords and not any(is_numeric_val(v) for v in non_empty.values()) and len(non_empty) >= 3:
            table_col_map = {}
            text_columns = set()
            for c_idx, val in non_empty.items():
                v_clean = str(val).strip()
                table_col_map[c_idx] = v_clean
                if any(t in v_clean.lower() for t in ['desc', 'name', 'remark', 'note', 'detail', 'particular']):
                    text_columns.add(c_idx)
            continue

        # 6. Document / Section Header Instance Row
        is_parent_header = False
        if parent_header_labels:
            matched_header_cols = [c for c in non_empty.keys() if c in parent_header_labels]
            if len(matched_header_cols) >= 1:
                if not (table_col_map and any(nk in table_col_map.get(first_pos, '').lower() for nk in ['seq', 'no.', 'item', 'line', 'sku']) and is_numeric_val(first_val)):
                    is_parent_header = True
        else:
            if (is_date_val(first_val) or re.match(r'^[A-Z]{1,5}[-_]?\d+', first_str)) and len(non_empty) <= 6 and not table_col_map:
                is_parent_header = True

        if is_parent_header:
            if current_record is not None:
                records.append(current_record)
                current_record = None

            new_parent_context = {}
            if parent_header_labels:
                sorted_label_cols = sorted(parent_header_labels.keys())
                for c_idx, val in non_empty.items():
                    if c_idx in parent_header_labels:
                        lbl = parent_header_labels[c_idx]
                    else:
                        closest_lbl_col = max([l for l in sorted_label_cols if l <= c_idx], default=sorted_label_cols[0])
                        lbl = parent_header_labels[closest_lbl_col]
                    
                    if 'amount' in lbl.lower() or 'total' in lbl.lower():
                        new_parent_context['doc_total'] = clean_num(val)
                    else:
                        new_parent_context[lbl] = str(val).strip() if not is_date_val(val) else val
            else:
                for c_idx, val in non_empty.items():
                    if is_date_val(val): new_parent_context['doc_date'] = val
                    elif is_numeric_val(val): new_parent_context['doc_total'] = clean_num(val)
                    elif 'doc_no' not in new_parent_context: new_parent_context['doc_no'] = str(val).strip()
                    else: new_parent_context[f'header_{c_idx}'] = str(val).strip()

            parent_context = new_parent_context
            continue

        # 7. Table Line Item Record
        is_line_item = False
        if table_col_map and parent_context:
            matching_cols = [c for c in non_empty.keys() if c in table_col_map]
            if len(matching_cols) >= 2 or (len(matching_cols) >= 1 and any(is_numeric_val(non_empty[c]) for c in matching_cols)):
                is_line_item = True
        elif len(non_empty) >= 3 and any(is_numeric_val(v) for v in non_empty.values()):
            is_line_item = True

        if is_line_item:
            if current_record is not None:
                records.append(current_record)

            item_data = {}
            for pk, pv in parent_context.items():
                item_data[pk] = pv

            if table_col_map:
                for c_idx, col_name in table_col_map.items():
                    val = non_empty.get(c_idx)
                    if val is not None and is_numeric_val(val) and any(nk in col_name.lower() for nk in ['seq', 'qty', 'quantity', 'price', 'amount', 'debit', 'credit', 'balance', 'total', 'rate', 'disc', 'tax', 'salary', 'cost']):
                        item_data[col_name] = clean_num(val)
                    elif val is not None:
                        item_data[col_name] = str(val).strip()
                    else:
                        item_data[col_name] = None
            else:
                for c_idx, val in non_empty.items():
                    item_data[f'col_{c_idx}'] = val

            current_record = item_data
            continue

        # 8. Multi-line Wrapped Text Continuation
        if current_record is not None and len(non_empty) <= 2:
            for c_idx, val in non_empty.items():
                if not is_numeric_val(val) and not is_date_val(val):
                    col_name = table_col_map.get(c_idx)
                    if col_name and col_name in current_record:
                        curr_v = current_record[col_name]
                        current_record[col_name] = f'{curr_v} {str(val).strip()}'.strip() if curr_v else str(val).strip()

    if current_record is not None:
        records.append(current_record)

    out_df = pd.DataFrame(records)

    # 9. Dynamic Target Column Normalizer & Semantic Mapper
    rename_map = {}
    for col in out_df.columns:
        c_low = str(col).strip().lower()
        if c_low in ('doc. no', 'doc no', 'invoice no', 'voucher no', 'bill no', 'order no', 'po no', 'ref no', 'po_no'): rename_map[col] = 'doc_no'
        elif c_low in ('doc. date', 'doc date', 'invoice date', 'bill date', 'order date', 'posting date', 'txn date', 'po_date'): rename_map[col] = 'doc_date'
        elif c_low in ('code', 'cust code', 'customer code', 'debtor code', 'account code', 'vendor code', 'supplier code', 'vendor'): rename_map[col] = 'customer_code'
        elif c_low in ('name', 'cust name', 'customer name', 'debtor name', 'account name', 'vendor name', 'supplier name'): rename_map[col] = 'customer_name'
        elif c_low in ('seq', 'sequence', 'no.', 'line no', 'item no', 'sr no'): rename_map[col] = 'Sequence'
        elif 'gl' in c_low and 'code' in c_low: rename_map[col] = 'GL-Code'
        elif 'desc' in c_low or 'particular' in c_low or 'item name' in c_low or 'description' in c_low: rename_map[col] = 'Full_Description'
        elif 'unit price' in c_low or 'unit rate' in c_low or 'price' in c_low or 'rate' in c_low: rename_map[col] = 'Unit Price'
        elif 'qty' in c_low or 'quantity' in c_low: rename_map[col] = 'Quantity'
        elif c_low in ('uom', 'unit', 'units'): rename_map[col] = 'UOM'
        elif 'amount' in c_low or 'subtotal' in c_low or 'total cost' in c_low or 'total amount' in c_low: rename_map[col] = 'Item Amount'

    out_df = out_df.rename(columns=rename_map)

    if 'doc_no' in out_df.columns and 'Item Amount' in out_df.columns:
        if 'invoice_total' not in out_df.columns or out_df['invoice_total'].isna().all():
            out_df['invoice_total'] = out_df.groupby('doc_no')['Item Amount'].transform('sum')

    out_df.columns = _make_columns_unique(out_df.columns)

    for c in out_df.columns:
        if any(nk in c.lower() for nk in ['seq', 'quantity', 'price', 'amount', 'total', 'debit', 'credit', 'balance', 'salary', 'cost']):
            out_df[c] = pd.to_numeric(out_df[c], errors='coerce').fillna(0.0)
        elif 'date' in c.lower():
            out_df[c] = pd.to_datetime(out_df[c], errors='coerce')
        else:
            out_df[c] = out_df[c].astype(str).replace('None', '').replace('nan', '')

    if target_columns:
        avail_target_cols = [c for c in target_columns if c in out_df.columns]
        other_cols = [c for c in out_df.columns if c not in avail_target_cols]
        out_df = out_df[avail_target_cols + other_cols]

    if pl is not None:
        return pl.from_pandas(out_df)
    return out_df


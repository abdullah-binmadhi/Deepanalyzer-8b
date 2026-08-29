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


def sanitize_dirty_numeric_series(val) -> float:
    """Robust sanitization of dirty currency, percentage, and accounting strings."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in ("n/a", "none", "null", "missing", "nan", "-"):
        return np.nan
    # Check accounting negative in parentheses: (1,250.50) -> -1250.50
    is_neg = s.startswith("(") and s.endswith(")")
    if is_neg:
        s = s[1:-1].strip()
    # Strip currencies, percentages, and commas
    s = re.sub(r'[\$€£¥₹%]|SAR|AED|USD|EUR|GBP|RM|\s+', '', s, flags=re.I).replace(',', '')
    try:
        f = float(s)
        return -f if is_neg else f
    except (ValueError, TypeError):
        return np.nan


def parse_mixed_datetime_series(s_series) -> pd.Series:
    """Auto-coerces mixed date formats into uniform ISO-8601 datetimes."""
    return pd.to_datetime(s_series, errors='coerce', dayfirst=True)


# =============================================================================
# 7. AUTOMATIC DATA TYPE & BOOLEAN ASSERTER (--auto-type)
# =============================================================================
def auto_cast_data_types(df_obj):
    """Coerces string columns to booleans, floats, integers, or datetimes without losing data."""
    bool_true = {"true", "t", "yes", "y", "1"}
    bool_false = {"false", "f", "no", "n", "0"}

    if pl is not None and isinstance(df_obj, pl.DataFrame):
        pdf = df_obj.to_pandas()
        for col in pdf.columns:
            if not pd.api.types.is_numeric_dtype(pdf[col]) and not pd.api.types.is_datetime64_any_dtype(pdf[col]):
                sample_vals = [str(v).strip() for v in pdf[col].dropna().head(20).tolist()]
                if not sample_vals:
                    continue
                # 1. Check Boolean
                if set(v.lower() for v in sample_vals).issubset(bool_true | bool_false):
                    pdf[col] = pdf[col].astype(str).str.strip().str.lower().isin(bool_true)
                # 2. Check Currency / Dirty Numeric
                elif any(re.search(r'[\$€£¥₹%]|SAR|AED|USD|EUR|RM|\(\d+[\d,.]*\)', str(v), re.I) for v in sample_vals):
                    coerced = pdf[col].map(sanitize_dirty_numeric_series)
                    if coerced.notna().sum() >= len(pdf) * 0.5:
                        pdf[col] = pd.to_numeric(coerced, errors='coerce')
                # 3. Check Datetime
                elif any(re.search(r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', str(v)) for v in sample_vals):
                    parsed_dt = pd.to_datetime(pdf[col], errors='coerce')
                    if parsed_dt.notna().sum() >= len(pdf) * 0.5:
                        pdf[col] = parsed_dt
                # 4. Standard float/int cast
                else:
                    try:
                        numeric_s = pd.to_numeric(pdf[col].astype(str).str.replace(',', ''), errors='coerce')
                        if numeric_s.notna().sum() >= len(pdf) * 0.6:
                            pdf[col] = numeric_s
                    except Exception:
                        pass
        return pl.from_pandas(pdf)
    elif isinstance(df_obj, pd.DataFrame):
        pdf = df_obj.copy()
        for col in pdf.columns:
            if not pd.api.types.is_numeric_dtype(pdf[col]) and not pd.api.types.is_datetime64_any_dtype(pdf[col]):
                sample_vals = [str(v).strip() for v in pdf[col].dropna().head(20).tolist()]
                if not sample_vals:
                    continue
                if set(v.lower() for v in sample_vals).issubset(bool_true | bool_false):
                    pdf[col] = pdf[col].astype(str).str.strip().str.lower().isin(bool_true)
                elif any(re.search(r'[\$€£¥₹%]|SAR|AED|USD|EUR|RM|\(\d+[\d,.]*\)', str(v), re.I) for v in sample_vals):
                    coerced = pdf[col].map(sanitize_dirty_numeric_series)
                    if coerced.notna().sum() >= len(pdf) * 0.5:
                        pdf[col] = pd.to_numeric(coerced, errors='coerce')
                elif any(re.search(r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', str(v)) for v in sample_vals):
                    parsed_dt = pd.to_datetime(pdf[col], errors='coerce')
                    if parsed_dt.notna().sum() >= len(pdf) * 0.5:
                        pdf[col] = parsed_dt
                else:
                    try:
                        numeric_s = pd.to_numeric(pdf[col].astype(str).str.replace(',', ''), errors='coerce')
                        if numeric_s.notna().sum() >= len(pdf) * 0.6:
                            pdf[col] = numeric_s
                    except Exception:
                        pass
        return pdf
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

    col_names = list(pdf.columns)
    if any(isinstance(c, str) and not c.isdigit() and not c.startswith('column_') and not c.startswith('__UNNAMED__') for c in col_names):
        header_row = pd.DataFrame([[c for c in col_names]], columns=col_names)
        full_df = pd.concat([header_row, pdf], ignore_index=True)
    else:
        full_df = pdf.copy()

    parent_context = {}
    parent_header_labels = {}
    table_col_map = {}
    
    records = []
    current_record = None

    SUMMARY_KEYWORDS = {
        'grand total', 'total account', 'total item', 'balance forward',
        'report total', 'page total', 'total:', 'account summary', 'item code summary'
    }

    KV_HEADER_REGEX = re.compile(
        r'^(Account(?:\s+No)?|Account\s+Name|Vendor(?:\s+No|\s+Code|\s+Name)?|'
        r'Customer(?:\s+No|\s+Code|\s+Name)?|Dept(?:\s+No)?|Branch|Location|Warehouse|'
        r'Project|Cost\s+Center|Doc(?:\s+No|\s+Date)?|PO(?:\s+No|\s+Date)?|Invoice(?:\s+No|\s+Date)?|'
        r'Category|Currency)\s*[:=]\s*(.*)$', re.IGNORECASE
    )

    def is_date_val(v):
        if isinstance(v, (datetime.datetime, datetime.date, pd.Timestamp)):
            return True
        s = str(v).strip()
        return bool(
            re.match(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$', s) or 
            re.match(r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$', s)
        )

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

    in_summary_section = False

    for r_idx, row in full_df.iterrows():
        non_empty = {c_idx: val for c_idx, val in enumerate(row.values) if pd.notna(val) and str(val).strip() != ''}
        if not non_empty:
            continue

        row_str_lower = ' '.join([str(v).lower() for v in non_empty.values()])
        first_pos = min(non_empty.keys())
        first_val = non_empty[first_pos]
        first_str = str(first_val).strip()

        # 1. Stop at Trailing Summary Sections (Account Summary, Item Summary, Grand Total Amount)
        if any(kw in row_str_lower for kw in ['account summary', 'item code summary', 'grand total amount', 'total item(s) :']):
            in_summary_section = True
            current_record = None
            continue

        if in_summary_section:
            continue

        # 2. Skip Report Parameters, Page Breaks, Print Timestamps, & Filter Blocks
        is_page_break = any(p in row_str_lower for p in ['page ', 'page:', 'p. ']) and any(w in row_str_lower for w in [' of ', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        is_print_meta = any(m in row_str_lower for m in ['printed on', 'run date', 'report id', 'user id', 'print date', 'system timestamp', 'as at ', 'sort by', 'invoice listing', 'incl cancelled', 'co category'])
        is_filter_block = (' : ' in row_str_lower and any(w in row_str_lower for w in ['from ', 'all', 'selected', 'sort by', 'date', 'company'])) or row_str_lower.startswith('from ') or 'selected' in row_str_lower

        if is_page_break or is_print_meta or is_filter_block:
            continue
        if len(non_empty) == 1 and any(comp in row_str_lower for comp in ['sdn bhd', 'ltd', 'inc.', 'corp', 'llc', 'gmbh', 'co.']) and not KV_HEADER_REGEX.match(first_str):
            continue

        # 3. Subtotal & Divider Rows (Drop to prevent double-counting)
        is_subtotal = any(kw in row_str_lower for kw in ['subtotal', 'sub-total', 'total for', 'balance c/f', 'balance b/f', 'carried forward', 'brought forward'])
        if is_subtotal or any(kw in row_str_lower for kw in SUMMARY_KEYWORDS) or re.match(r'^[=\-_]{3,}$', first_str):
            current_record = None
            continue

        # 4. Check for Inline Key-Value Parent Header Rows (e.g. Account No: 1000-00, PO No: PO-99881)
        has_kv_pairs = any(KV_HEADER_REGEX.match(str(v).strip()) for v in non_empty.values())
        if has_kv_pairs:
            current_record = None
            new_parent_context = {}
            for c_idx, val in non_empty.items():
                m = KV_HEADER_REGEX.match(str(val).strip())
                if m:
                    k_name, v_val = m.group(1).strip().lower().replace(' ', '_'), m.group(2).strip()
                    new_parent_context[k_name] = v_val
                else:
                    new_parent_context[f'header_{c_idx}'] = str(val).strip()
            parent_context = new_parent_context
            continue

        # 5. Flat Report Table Header Definition (e.g. Invoice No, Date, Customer, Amount)
        if any(h in row_str_lower for h in ['doc. no', 'doc no', 'invoice no', 'voucher no']) and any(a in row_str_lower for a in ['amount', 'price', 'total', 'debit', 'credit']) and not any(s in row_str_lower for s in ['seq', 'item code', 'uom', 'unit price']):
            if any(c in row_str_lower for a in ['code', 'cust code', 'customer code'] for c in [a]):
                parent_header_labels = {c_idx: str(val).strip() for c_idx, val in non_empty.items()}
            else:
                table_col_map = {c_idx: str(val).strip() for c_idx, val in non_empty.items()}
            continue

        # 6. Document / Grouping Header Definition Labels (e.g. Doc. No, Doc. Date, Code, Name, Amount (RM))
        if any(h in row_str_lower for h in ['doc. no', 'doc no', 'invoice no', 'voucher no', 'account no', 'order no', 'po no', 'bill no', 'dept no']) and not any(is_numeric_val(v) for v in non_empty.values()):
            parent_header_labels = {c_idx: str(val).strip() for c_idx, val in non_empty.items()}
            continue

        # 7. Table Column Definition Row (e.g. Seq, GL Code, Description, Quantity, UOM, Unit Price, Amount)
        has_table_keywords = any(kw in row_str_lower for kw in ['seq', 'line no', 'sku', 'item', 'desc', 'particulars', 'debit', 'credit'])
        if has_table_keywords and not any(is_numeric_val(v) for v in non_empty.values()) and len(non_empty) >= 3:
            table_col_map = {c_idx: str(val).strip() for c_idx, val in non_empty.items()}
            continue

        # 8. Document Header Instance Row (e.g. IV-11319, 2025-08-01, 300-10110, 10108 TS, 1018.29)
        is_doc_instance = False
        if parent_header_labels:
            if re.match(r'^[A-Z]{1,5}[-_]?\d+', first_str) or any(is_date_val(v) for v in non_empty.values()):
                if not (first_str.isdigit() and int(first_str) >= 100):
                    is_doc_instance = True
        elif not table_col_map and re.match(r'^[A-Z]{1,5}[-_]?\d+', first_str) and len(non_empty) <= 6:
            is_doc_instance = True

        if is_doc_instance:
            current_record = None
            new_parent_context = {}
            if parent_header_labels:
                sorted_labels = sorted(parent_header_labels.keys())
                for c_idx, val in non_empty.items():
                    nearest_l = min(sorted_labels, key=lambda l: abs(l - c_idx))
                    lbl = parent_header_labels[nearest_l]
                    lbl_lower = lbl.lower()
                    if is_date_val(val):
                        new_parent_context['doc_date'] = val
                    elif ('amount' in lbl_lower or 'total' in lbl_lower) and is_numeric_val(val):
                        new_parent_context['invoice_total'] = clean_num(val)
                    elif re.match(r'^[A-Z]{1,5}[-_]?\d+', str(val).strip()) and 'doc_no' not in new_parent_context:
                        new_parent_context['doc_no'] = str(val).strip()
                    elif 'code' in lbl_lower or re.match(r'^\d{3,}[-_][A-Za-z0-9]+', str(val).strip()):
                        new_parent_context['customer_code'] = str(val).strip()
                    elif 'name' in lbl_lower or 'customer' in lbl_lower or 'debtor' in lbl_lower:
                        if 'customer_name' not in new_parent_context:
                            new_parent_context['customer_name'] = str(val).strip()
                        else:
                            new_parent_context['customer_name'] += f" {str(val).strip()}"
                    else:
                        new_parent_context[lbl] = str(val).strip()
            else:
                for c_idx, val in non_empty.items():
                    if is_date_val(val): new_parent_context['doc_date'] = val
                    elif is_numeric_val(val): new_parent_context['invoice_total'] = clean_num(val)
                    elif re.match(r'^[A-Z]{1,5}[-_]?\d+', str(val).strip()) and 'doc_no' not in new_parent_context:
                        new_parent_context['doc_no'] = str(val).strip()
                    else:
                        new_parent_context[f'header_{c_idx}'] = str(val).strip()

            parent_context = new_parent_context
            continue

        # 9. Table Line Item Record
        is_line_item = False
        if table_col_map:
            if first_str.isdigit() and int(first_str) >= 100:
                is_line_item = True
            elif len(non_empty) >= 3 and any(is_numeric_val(v) for v in non_empty.values()) and not is_doc_instance:
                if not re.match(r'^[A-Z]{1,5}[-_]?\d+', first_str):
                    is_line_item = True
                elif not parent_header_labels:
                    is_line_item = True

        if is_line_item:
            item_data = dict(parent_context)
            if table_col_map:
                sorted_cols = sorted(table_col_map.keys())
                for c_idx, val in non_empty.items():
                    if c_idx in table_col_map:
                        col_name = table_col_map[c_idx]
                    else:
                        nearest_c = min(sorted_cols, key=lambda sc: abs(sc - c_idx))
                        col_name = table_col_map[nearest_c]

                    if is_numeric_val(val) and any(nk in col_name.lower() for nk in ['seq', 'qty', 'quantity', 'price', 'amount', 'debit', 'credit', 'balance', 'total', 'rate', 'disc', 'tax', 'salary', 'cost']):
                        item_data[col_name] = clean_num(val)
                    else:
                        item_data[col_name] = str(val).strip()
            else:
                for c_idx, val in non_empty.items():
                    item_data[f'col_{c_idx}'] = clean_num(val) if is_numeric_val(val) else str(val).strip()

            records.append(item_data)
            current_record = item_data
            continue

        # 10. Multi-line Wrapped Text Continuation
        if current_record is not None and len(non_empty) <= 2:
            for c_idx, val in non_empty.items():
                if not is_numeric_val(val) and not is_date_val(val):
                    target_col = None
                    if table_col_map and c_idx in table_col_map:
                        target_col = table_col_map[c_idx]
                    elif table_col_map:
                        sorted_cols = sorted(table_col_map.keys())
                        nearest_c = min(sorted_cols, key=lambda sc: abs(sc - c_idx))
                        target_col = table_col_map[nearest_c]

                    if not target_col or target_col not in current_record:
                        target_col = next((c for c in current_record if any(t in c.lower() for t in ['desc', 'particular', 'item description', 'note']) and 'customer' not in c.lower() and 'vendor' not in c.lower()), None)

                    if target_col and target_col in current_record:
                        curr_v = current_record[target_col]
                        current_record[target_col] = f'{curr_v} {str(val).strip()}'.strip() if curr_v else str(val).strip()

    if not records:
        return pdf

    out_df = pd.DataFrame(records)

    # 11. Semantic Column Normalizer & Standardizer
    rename_map = {}
    for col in out_df.columns:
        c_low = str(col).strip().lower()
        if c_low in ('doc. no', 'doc no', 'invoice no', 'voucher no', 'bill no', 'order no', 'po no', 'ref no', 'po_no', 'ref_no'): rename_map[col] = 'doc_no'
        elif c_low in ('doc. date', 'doc date', 'invoice date', 'bill date', 'order date', 'posting date', 'txn date', 'po_date', 'date'):
            if 'doc_date' not in rename_map.values(): rename_map[col] = 'doc_date'
        elif c_low in ('code', 'cust code', 'customer code', 'debtor code', 'account code', 'vendor code', 'supplier code', 'vendor'): rename_map[col] = 'customer_code'
        elif c_low in ('name', 'cust name', 'customer name', 'debtor name', 'account name', 'vendor name', 'supplier name', 'customer'): rename_map[col] = 'customer_name'
        elif c_low in ('seq', 'sequence', 'no.', 'line no', 'item no', 'sr no'): rename_map[col] = 'Sequence'
        elif 'gl' in c_low and 'code' in c_low: rename_map[col] = 'GL-Code'
        elif 'desc' in c_low or 'particular' in c_low or 'item name' in c_low or 'description' in c_low: rename_map[col] = 'Full_Description'
        elif 'unit price' in c_low or 'unit rate' in c_low or 'price' in c_low or 'rate' in c_low: rename_map[col] = 'Unit Price'
        elif 'qty' in c_low or 'quantity' in c_low: rename_map[col] = 'Quantity'
        elif c_low in ('uom', 'unit', 'units'): rename_map[col] = 'UOM'
        elif 'amount' in c_low or 'subtotal' in c_low or 'total cost' in c_low or 'total amount' in c_low:
            if col != 'invoice_total':
                rename_map[col] = 'Item Amount'

    out_df = out_df.rename(columns=rename_map)

    if 'doc_no' in out_df.columns and 'Item Amount' in out_df.columns:
        if 'invoice_total' not in out_df.columns or out_df['invoice_total'].isna().all() or (out_df['invoice_total'] == 0).all():
            out_df['invoice_total'] = out_df.groupby('doc_no')['Item Amount'].transform('sum')

    seen = {}
    new_cols = []
    for c in out_df.columns:
        name = str(c).strip()
        if name in seen:
            seen[name] += 1
            new_cols.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_cols.append(name)
    out_df.columns = new_cols

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


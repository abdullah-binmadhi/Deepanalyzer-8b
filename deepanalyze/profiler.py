"""DeepAnalyze: Deep Data Exploration, Workbook Topology & Autonomous Prompt Synthesis Engine."""

from dataclasses import dataclass, field
from enum import Enum
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl


class SheetRole(str, Enum):
    """Semantic role of an individual sheet within an enterprise workbook."""
    TRANSACTION_LEDGER = "TRANSACTION_LEDGER"
    PIVOT_TABLE = "PIVOT_TABLE"
    LOOKUP_DIMENSION = "LOOKUP_DIMENSION"
    METADATA_BLOCK = "METADATA_BLOCK"


# Regex detectors for column-level data quality anomalies
DATE_PATTERNS = [
    (re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"), "YYYY-MM-DD (ISO)"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"), "DD/MM/YYYY or MM/DD/YYYY"),
    (re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}"), "DD-Mon-YYYY (e.g. 15-Aug-2025)"),
    (re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"), "Month DD, YYYY (e.g. August 15, 2025)"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}"), "Timestamp ISO-8601"),
]

ACCOUNTING_NEGATIVE_PATTERN = re.compile(r"^\s*\(\s*[\$€£SARa-zA-Z]*\s*[\d,]+(\.\d+)?\s*\)\s*$|^\s*[\d,]+(\.\d+)?\s*-\s*$")
DIRTY_CURRENCY_PATTERN = re.compile(r"[\$€£¥₹₽₩₺₴]|(?:\b(?:SAR|AED|USD|EUR|GBP|INR|CAD|AUD|CHF|JPY|CNY|KWD|BHD|OMR|QAR|EGP)\b)|ريال", re.IGNORECASE)
SUBTOTAL_KEYWORD_PATTERN = re.compile(r"\b(total|subtotal|grand\s*total|summary|balance|all)\b", re.IGNORECASE)
MONTH_HEADER_PATTERN = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|20\d\d)", re.IGNORECASE)


@dataclass
class ColumnProfile:
    """Statistical and semantic diagnostic profile for a single column."""
    name: str
    dtype: str
    total_rows: int
    null_count: int
    null_pct: float
    cardinality: int
    cardinality_ratio: float
    inferred_role: str
    date_formats: List[str] = field(default_factory=list)
    has_accounting_negatives: bool = False
    has_dirty_currency: bool = False
    has_whitespace_issues: bool = False
    sample_patterns: List[str] = field(default_factory=list)


@dataclass
class SheetProfile:
    """Architectural profile of a spreadsheet table or workbook sheet."""
    sheet_name: str
    row_count: int
    col_count: int
    role: SheetRole
    header_row_offset: int = 0
    subtotal_rows: List[int] = field(default_factory=list)
    columns: List[ColumnProfile] = field(default_factory=list)
    df: Optional[pl.DataFrame] = None


@dataclass
class ForeignKeyCandidate:
    """Inferred relational join key linking two sheets."""
    from_sheet: str
    from_col: str
    to_sheet: str
    to_col: str
    overlap_pct: float
    is_name_match: bool


@dataclass
class WorkbookTopology:
    """Global multi-sheet architecture, relational linkages, and engineering plan."""
    file_path: str
    sheets: Dict[str, SheetProfile] = field(default_factory=dict)
    primary_sheet: str = ""
    foreign_keys: List[ForeignKeyCandidate] = field(default_factory=list)
    recommended_pipeline_steps: List[str] = field(default_factory=list)


def detect_date_formats(sample_values: List[str]) -> List[str]:
    """Detects distinct date and timestamp formats present in text samples."""
    formats_found: Set[str] = set()
    for val in sample_values:
        val_str = str(val).strip()
        for regex, fmt_name in DATE_PATTERNS:
            if regex.search(val_str):
                formats_found.add(fmt_name)
    return sorted(list(formats_found))


def detect_accounting_negatives(sample_values: List[str]) -> bool:
    """Checks if values contain parentheses indicating accounting negative numbers."""
    for val in sample_values:
        if ACCOUNTING_NEGATIVE_PATTERN.search(str(val)):
            return True
    return False


def detect_dirty_currency(sample_values: List[str]) -> bool:
    """Checks if numeric text fields contain embedded currency symbols or codes."""
    for val in sample_values:
        if DIRTY_CURRENCY_PATTERN.search(str(val)):
            return True
    return False


def detect_whitespace_issues(sample_values: List[str]) -> bool:
    """Detects trailing, leading, or irregular non-breaking whitespace."""
    for val in sample_values:
        val_str = str(val)
        if len(val_str) != len(val_str.strip()):
            return True
        if "\u00a0" in val_str or "\u2009" in val_str or "\t" in val_str or "\n" in val_str or "\r" in val_str:
            return True
    return False


def infer_column_role(name: str, cardinality_ratio: float, dtype: str, null_pct: float, sample_strs: List[str]) -> str:
    """Infers semantic role of a column based on heuristics."""
    name_lower = name.lower()
    name_tokens = name_lower.replace("_", " ").replace("-", " ")

    if cardinality_ratio >= 0.98 and null_pct == 0.0:
        return "Primary Key Candidate"

    if re.search(r"\b(zip|postal|age|gender|sex|dob|birth|nationality)\b", name_tokens):
        return "Quasi-Identifier"

    if re.search(r"\b(id|code|no|num|number|key|doc|inv)\b", name_tokens):
        return "Business Key / Foreign Identifier"

    if re.search(r"\b(name|patient|customer|client|employee|user|person|doctor|nurse)\b", name_tokens):
        return "Direct Identifier / Personal Identity"

    if re.search(r"\b(department|dept|branch)\b", name_tokens):
        return "Quasi-Identifier"

    dtype_lower = dtype.lower()
    if "int" in dtype_lower or "float" in dtype_lower or "decimal" in dtype_lower:
        return "Numeric Metric / Quantitative"

    # Check for long narrative texts
    avg_len = sum(len(s) for s in sample_strs) / max(len(sample_strs), 1)
    if avg_len > 40:
        return "Free-Text Narrative / Clinical Note"

    if cardinality_ratio < 0.20:
        return "Categorical / Dimension"

    return "Descriptive Attribute"


def profile_dataframe(df: pl.DataFrame, name: str = "df") -> SheetProfile:
    """Extracts in-depth zero-PII column diagnostics and data anomalies for a DataFrame."""
    total_rows = max(df.height, 1)
    column_profiles: List[ColumnProfile] = []

    for col_name in df.columns:
        series = df[col_name]
        null_count = series.null_count()
        null_pct = round((null_count / total_rows) * 100.0, 1)
        dtype_str = str(series.dtype)

        # Drop nulls for non-null sample inspection
        non_null_s = series.drop_nulls()
        cardinality = non_null_s.n_unique() if non_null_s.len() > 0 else 0
        card_ratio = round(cardinality / total_rows, 4)

        # Convert up to 100 samples to strings for pattern detection
        sample_slice = non_null_s.head(100).to_list()
        sample_strs = [str(x) for x in sample_slice]

        date_fmts = detect_date_formats(sample_strs)
        has_acct_neg = detect_accounting_negatives(sample_strs)
        has_dirty_curr = detect_dirty_currency(sample_strs)
        has_ws = detect_whitespace_issues(sample_strs)
        role = infer_column_role(col_name, card_ratio, dtype_str, null_pct, sample_strs)

        column_profiles.append(ColumnProfile(
            name=col_name,
            dtype=dtype_str,
            total_rows=total_rows,
            null_count=null_count,
            null_pct=null_pct,
            cardinality=cardinality,
            cardinality_ratio=card_ratio,
            inferred_role=role,
            date_formats=date_fmts,
            has_accounting_negatives=has_acct_neg,
            has_dirty_currency=has_dirty_curr,
            has_whitespace_issues=has_ws,
            sample_patterns=[]
        ))

    # Determine sheet role and subtotal rows
    role, header_offset, subtotal_rows = detect_sheet_role(df, name)

    return SheetProfile(
        sheet_name=name,
        row_count=df.height,
        col_count=df.width,
        role=role,
        header_row_offset=header_offset,
        subtotal_rows=subtotal_rows,
        columns=column_profiles,
        df=df
    )


def detect_sheet_role(df: pl.DataFrame, sheet_name: str) -> Tuple[SheetRole, int, List[int]]:
    """Inspects table dimensions, header names, and row patterns to assign a semantic SheetRole."""
    sheet_name_lower = sheet_name.lower()
    row_count = df.height
    col_count = df.width

    subtotal_rows: List[int] = []
    # Check first column for subtotal keywords
    if col_count > 0:
        first_col = df[df.columns[0]]
        for idx, val in enumerate(first_col.head(min(row_count, 500)).to_list()):
            if val is not None and SUBTOTAL_KEYWORD_PATTERN.search(str(val)):
                subtotal_rows.append(idx)

    # Check for pivot table signs:
    # 1. Sheet name contains 'pivot', 'summary', 'report', 'budget'
    # 2. Columns are month names, quarters, or year numbers
    month_col_count = sum(1 for c in df.columns if MONTH_HEADER_PATTERN.search(c))
    if month_col_count >= 3 or ("pivot" in sheet_name_lower or "summary" in sheet_name_lower):
        if row_count < 100:
            return SheetRole.PIVOT_TABLE, 0, subtotal_rows

    # Check for dimension / lookup table:
    # 1. Medium row count (< 1000)
    # 2. Sheet name has 'lookup', 'dim', 'customer', 'dept', 'master', 'codes'
    # 3. High uniqueness in column 0
    if ("lookup" in sheet_name_lower or "dim" in sheet_name_lower or "master" in sheet_name_lower) and row_count < 2000:
        return SheetRole.LOOKUP_DIMENSION, 0, subtotal_rows

    # Check for metadata block:
    if (row_count <= 5 and col_count <= 2) or (
        ("metadata" in sheet_name_lower or "cover" in sheet_name_lower or "info" in sheet_name_lower or "readme" in sheet_name_lower)
        and row_count <= 15
    ):
        return SheetRole.METADATA_BLOCK, 0, subtotal_rows

    # Check header row offset (ERP unflattened metadata rows at the top)
    header_offset = 0
    if col_count >= 2:
        for r_idx in range(min(row_count, 25)):
            row_vals = [df[c][r_idx] for c in df.columns]
            non_nulls = sum(1 for v in row_vals if v is not None and str(v).strip() != "")
            # If a row has very few values compared to total columns, it might be top metadata
            if non_nulls <= 2 and col_count >= 8:
                header_offset = r_idx + 1
            elif non_nulls >= (col_count * 0.6):
                break

    return SheetRole.TRANSACTION_LEDGER, header_offset, subtotal_rows


def detect_foreign_keys(sheets: Dict[str, SheetProfile]) -> List[ForeignKeyCandidate]:
    """Identifies candidate join keys between sheets using column name similarity and value overlap."""
    candidates: List[ForeignKeyCandidate] = []
    sheet_names = list(sheets.keys())

    for i in range(len(sheet_names)):
        for j in range(i + 1, len(sheet_names)):
            s1_name = sheet_names[i]
            s2_name = sheet_names[j]
            s1 = sheets[s1_name]
            s2 = sheets[s2_name]

            if s1.df is None or s2.df is None:
                continue

            # Compare columns
            for col1 in s1.columns:
                c1_clean = re.sub(r"[^a-zA-Z0-9]", "", col1.name.lower())
                c1_vals: Set[str] = set()
                try:
                    c1_vals = set(s1.df[col1.name].drop_nulls().head(200).cast(pl.Utf8).to_list())
                except Exception:
                    continue

                if len(c1_vals) == 0:
                    continue

                for col2 in s2.columns:
                    c2_clean = re.sub(r"[^a-zA-Z0-9]", "", col2.name.lower())
                    name_match = (c1_clean == c2_clean) or (c1_clean in c2_clean) or (c2_clean in c1_clean)

                    c2_vals: Set[str] = set()
                    try:
                        c2_vals = set(s2.df[col2.name].drop_nulls().head(200).cast(pl.Utf8).to_list())
                    except Exception:
                        continue

                    if len(c2_vals) == 0:
                        continue

                    intersection = len(c1_vals.intersection(c2_vals))
                    overlap_pct = round((intersection / min(len(c1_vals), len(c2_vals))) * 100.0, 1)

                    if name_match and overlap_pct >= 20.0:
                        candidates.append(ForeignKeyCandidate(
                            from_sheet=s1_name,
                            from_col=col1.name,
                            to_sheet=s2_name,
                            to_col=col2.name,
                            overlap_pct=overlap_pct,
                            is_name_match=True
                        ))
                    elif overlap_pct >= 60.0 and len(c1_vals) >= 3:
                        candidates.append(ForeignKeyCandidate(
                            from_sheet=s1_name,
                            from_col=col1.name,
                            to_sheet=s2_name,
                            to_col=col2.name,
                            overlap_pct=overlap_pct,
                            is_name_match=name_match
                        ))

    return candidates


def find_header_row(df_pd: Any) -> Tuple[int, List[str]]:
    """Detects the true table header row in a raw header=None DataFrame."""
    if len(df_pd) == 0:
        return 0, [str(c) for c in df_pd.columns]

    max_cols = len(df_pd.columns)

    # Check first 30 rows
    for r_idx in range(min(len(df_pd), 30)):
        row_vals = df_pd.iloc[r_idx].to_list()
        str_vals = [
            str(v).strip() for v in row_vals
            if v is not None and str(v).strip() not in ("", "nan", "None")
        ]
        if len(str_vals) >= max(2, int(max_cols * 0.4)):
            non_numeric = sum(1 for v in str_vals if not re.match(r"^-?\d+(\.\d+)?$", v))
            if non_numeric >= len(str_vals) * 0.7:
                seen: Dict[str, int] = {}
                clean_cols = []
                for idx, v in enumerate(row_vals):
                    raw_str = (
                        str(v).strip()
                        if v is not None and str(v).strip() not in ("", "nan", "None")
                        else f"Column{idx+1}"
                    )
                    if raw_str in seen:
                        seen[raw_str] += 1
                        clean_cols.append(f"{raw_str}_{seen[raw_str]}")
                    else:
                        seen[raw_str] = 0
                        clean_cols.append(raw_str)
                return r_idx, clean_cols

    return 0, [f"Column{i+1}" for i in range(max_cols)]


def profile_workbook(file_path: str) -> WorkbookTopology:
    """Inspects an entire Excel workbook across all sheets, building full topology diagnostics."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Workbook file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    sheets_dict: Dict[str, SheetProfile] = {}

    if ext in (".xlsx", ".xls", ".xlsm"):
        import pandas as pd
        sheet_names = []
        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
        except Exception:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                sheet_names = wb.sheetnames
                wb.close()
            except Exception:
                sheet_names = ["Sheet1"]

        for sname in sheet_names:
            try:
                # Ingest with header=None to preserve 100% of rows and columns
                df_raw = pd.read_excel(file_path, sheet_name=sname, header=None)
                h_idx, clean_cols = find_header_row(df_raw)
                header_offset = h_idx

                df_data = df_raw.iloc[h_idx + 1:].copy() if len(df_raw) > h_idx else df_raw.copy()
                df_data.columns = clean_cols
                for c in df_data.columns:
                    if df_data[c].dtype == "object":
                        df_data[c] = df_data[c].map(lambda x: str(x) if pd.notna(x) else None)
                pl_df = pl.from_pandas(df_data)
                profile = profile_dataframe(pl_df, name=sname)
                profile.header_row_offset = header_offset
                sheets_dict[sname] = profile
            except Exception:
                continue

    # Identify primary sheet (highest row count by default)
    primary_sheet = ""
    max_rows = -1
    for sname, profile in sheets_dict.items():
        if profile.row_count > max_rows:
            max_rows = profile.row_count
            primary_sheet = sname

    foreign_keys = detect_foreign_keys(sheets_dict)

    # Compile recommended pipeline steps
    steps: List[str] = []
    pivot_sheets = [s for s, p in sheets_dict.items() if p.role == SheetRole.PIVOT_TABLE]
    lookup_sheets = [s for s, p in sheets_dict.items() if p.role == SheetRole.LOOKUP_DIMENSION]

    if primary_sheet and sheets_dict[primary_sheet].header_row_offset > 0:
        steps.append(f"Skip top {sheets_dict[primary_sheet].header_row_offset} metadata rows in '{primary_sheet}' and promote row as header.")

    for ps in pivot_sheets:
        sub_info = f" (filter out subtotal rows: {sheets_dict[ps].subtotal_rows})" if sheets_dict[ps].subtotal_rows else ""
        steps.append(f"Unpivot wide columns in pivot sheet '{ps}' to long format{sub_info}.")

    for fk in foreign_keys:
        steps.append(f"Merge '{fk.to_sheet}' into '{fk.from_sheet}' using join keys `{fk.from_col}` == `{fk.to_col}` ({fk.overlap_pct}% key match).")

    return WorkbookTopology(
        file_path=file_path,
        sheets=sheets_dict,
        primary_sheet=primary_sheet,
        foreign_keys=foreign_keys,
        recommended_pipeline_steps=steps
    )


def generate_engineering_briefing(topology: WorkbookTopology, user_prompt: str = "") -> str:
    """Synthesizes zero-PII architectural and statistical findings into a complete engineering briefing."""
    sections: List[str] = []

    sections.append("### AUTONOMOUS DATA ENGINEERING & TOPOLOGY BRIEFING")
    sections.append(f"**Target Architecture:** Multi-Sheet Enterprise Workbook ({len(topology.sheets)} sheets detected)")
    if user_prompt:
        sections.append(f"**User Objective:** {user_prompt}\n")

    # 1. Sheet Architecture Summary
    sections.append("#### 1. Workbook Topology & Sheet Roles:")
    for sname, profile in topology.sheets.items():
        role_label = profile.role.value
        dim_str = f"{profile.row_count:,} rows x {profile.col_count} columns"
        sections.append(f"- **Sheet '{sname}'** ({dim_str}) -> Role: `{role_label}`")
        if profile.header_row_offset > 0:
            sections.append(f"  * Detected top metadata offset: true table headers begin at row {profile.header_row_offset + 1}.")
        if profile.subtotal_rows:
            sections.append(f"  * Detected {len(profile.subtotal_rows)} summary/subtotal row(s) (e.g. at index {profile.subtotal_rows[:3]}).")

    # 2. Relational Linkage
    if topology.foreign_keys:
        sections.append("\n#### 2. Inferred Relational Join Keys:")
        for fk in topology.foreign_keys:
            name_status = "Exact/Similar Name" if fk.is_name_match else "Value Overlap Inference"
            sections.append(f"- Link: `{fk.from_sheet}.{fk.from_col}` <-> `{fk.to_sheet}.{fk.to_col}` ({fk.overlap_pct}% match, {name_status})")
    else:
        sections.append("\n#### 2. Relational Linkage:")
        sections.append("- Single consolidated table structure or independent dimension sheets.")

    # 3. Column-Level Data Quality Findings & Anomalies
    sections.append("\n#### 3. Field-Level Data Quality Anomalies & Format Variations:")
    anomaly_count = 0
    for sname, profile in topology.sheets.items():
        for col in profile.columns:
            anomalies: List[str] = []
            if len(col.date_formats) > 1:
                anomalies.append(f"Mixed date formats: {', '.join(col.date_formats)}")
            elif len(col.date_formats) == 1:
                anomalies.append(f"Format: {col.date_formats[0]}")

            if col.has_accounting_negatives:
                anomalies.append("Contains accounting negative brackets `(1,000.00)`")
            if col.has_dirty_currency:
                anomalies.append("Contains currency symbols/codes needing stripping")
            if col.has_whitespace_issues:
                anomalies.append("Contains leading/trailing or non-breaking whitespace")
            if col.null_pct >= 30.0 and col.inferred_role == "Business Key / Foreign Identifier":
                anomalies.append(f"{col.null_pct}% nulls (forward-fill candidate)")

            if anomalies:
                anomaly_count += 1
                sections.append(f"- **[{sname}] `{col.name}`** ({col.inferred_role}): {'; '.join(anomalies)}")

    if anomaly_count == 0:
        sections.append("- Clean column types detected with zero irregular formatting anomalies.")

    # 4. Actionable Transformation Checklist
    sections.append("\n#### 4. Actionable Cleaning & Engineering Checklist:")
    if topology.recommended_pipeline_steps:
        for idx, step in enumerate(topology.recommended_pipeline_steps, 1):
            sections.append(f"{idx}. {step}")
    else:
        sections.append("1. Standardize column names to clean snake_case.")
        sections.append("2. Cast date columns to ISO format (YYYY-MM-DD) using null-safe parsing.")
        sections.append("3. Strip currency symbols and parse numeric metrics cleanly.")
        sections.append("4. Remove duplicate rows and return the final DataFrame assigned to `df`.")

    return "\n".join(sections)

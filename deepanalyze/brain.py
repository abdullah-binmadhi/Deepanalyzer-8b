"""DeepAnalyze: 7-Brain Cognitive Resonance Engine.

Operates on raw data physics—Shannon entropy, matrix geometry, statistical morphology,
forensic pathology, relational cryptography, and brute-force algebraic invariant discovery—to
autonomously profile any dataset and synthesize deterministic, frontier-grade Data Engineering prompts.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class CognitiveBlackboard:
    """Shared state bus for all 7 cognitive sub-engines."""

    filepath: str = ""
    filename: str = "dataset"
    shape: Tuple[int, int] = (0, 0)
    columns: List[str] = field(default_factory=list)

    # Topology (Brain 1)
    header_row_index: int = 0
    metadata_rows: List[int] = field(default_factory=list)
    ragged_continuation_cols: List[Union[int, str]] = field(default_factory=list)
    footer_start_index: Optional[int] = None
    embedded_table_bounds: Optional[Dict[str, int]] = None

    # Morphology & Types (Brain 2)
    column_profiles: Dict[Union[int, str], Dict[str, Any]] = field(default_factory=dict)

    # Pathology (Brain 3)
    type_contaminations: List[Dict[str, Any]] = field(default_factory=list)
    cross_column_leaks: List[Dict[str, Any]] = field(default_factory=list)
    skewed_columns: List[Union[int, str]] = field(default_factory=list)
    outlier_metrics: Dict[Union[int, str], Dict[str, Any]] = field(default_factory=dict)

    # Relational & Cardinality (Brain 4)
    candidate_primary_keys: List[Union[int, str]] = field(default_factory=list)
    composite_primary_keys: List[Tuple[Union[int, str], ...]] = field(default_factory=list)
    hierarchical_dependencies: List[Tuple[Union[int, str], Union[int, str]]] = field(default_factory=list)

    # Mathematical Physics (Brain 5)
    algebraic_laws: List[str] = field(default_factory=list)

    # Feature Alchemy (Brain 6)
    engineered_features: List[Dict[str, str]] = field(default_factory=list)

    # Final Output (Brain 7)
    internal_monologue: List[str] = field(default_factory=list)
    master_prompt: str = ""


# Universal Arabic-Indic Digit & Separator translation table
# Translates Eastern Arabic numerals ٠١٢٣٤٥٦٧٨٩ to 0123456789 and Arabic decimal comma ٫ to .
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")

# Invisible BiDi (Bidirectional) Unicode directional markers and control characters
# \u200e (LRM), \u200f (RLM), \u202a-\u202e (Embedding/Overrides), \u061c (ALM), \ufeff (BOM)
BIDI_CHARS = re.compile(r"[\u200e\u200f\u202a-\u202e\u061c\ufeff]")


def normalize_bilingual_cell(val: Any) -> Any:
    """Normalizes Eastern Arabic numerals, Arabic decimal separators, and invisible BiDi marks."""
    if not isinstance(val, str):
        return val
    cleaned = BIDI_CHARS.sub("", val).strip()
    cleaned = cleaned.replace("٬", "")
    return cleaned.translate(ARABIC_INDIC_DIGITS)


def calculate_entropy(series: pd.Series) -> float:
    """Calculates normalized Shannon Entropy for a column, bounded in [0.0, 1.0]."""
    clean_series = series.dropna()
    if clean_series.empty:
        return 0.0
    counts = clean_series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    entropy = -float(np.sum(counts * np.log2(counts)))
    max_entropy = math.log2(len(clean_series)) if len(clean_series) > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _get_data_start_row(df: pd.DataFrame, bb: CognitiveBlackboard) -> int:
    """Returns the zero-based index of the first real data row."""
    has_named_cols = any(not str(c).isdigit() for c in df.columns)
    if has_named_cols and bb.header_row_index == 0:
        return 0
    return bb.header_row_index + 1 if bb.header_row_index < df.shape[0] - 1 else bb.header_row_index


class Brain1TopologicalCartographer:
    """Maps physical geometry, density drops, header cutoffs, and ragged boundaries."""

    METADATA_KEYWORDS = [
        "report", "filter", "criteria", "parameters", "sort", "date :",
        "تصفية", "فلتر", "معايير", "التاريخ :", "ترتيب", "تقرير", "بيانات", "كشف"
    ]
    FOOTER_KEYWORDS = [
        "grand total", "subtotal", "total summary", "end of report", "page 1 of", "summary", "total",
        "المجموع", "الإجمالي", "إجمالي التقرير", "صافي", "ملخص", "الاجمالي الكلي", "المجموع الكلي", "إجمالي"
    ]

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard) -> None:
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return

        has_named_cols = any(not str(c).isdigit() for c in df.columns)

        # 1. Header & Metadata Boundary via Density & Type Variance
        row_densities = df.notna().sum(axis=1).values
        max_density = int(row_densities.max()) if len(row_densities) > 0 else 0

        meta_rows: List[int] = []
        header_idx = 0
        if max_density > 0 and not has_named_cols:
            scan_depth = min(50, n_rows)
            for i in range(scan_depth):
                density = row_densities[i]
                row_str = " ".join(df.iloc[i].dropna().astype(str)).lower()
                is_meta_kw = any(kw in row_str for kw in self.METADATA_KEYWORDS)
                if density < (max_density * 0.5) or is_meta_kw:
                    meta_rows.append(i)
                elif density >= (max_density * 0.75):
                    row_vals = [str(x).strip() for x in df.iloc[i].dropna()]
                    string_ratio = sum(1 for v in row_vals if not re.match(r"^[-+]?\d+(?:\.\d+)?$", v)) / (len(row_vals) or 1)
                    if string_ratio > 0.5:
                        header_idx = i
                        break
        elif max_density > 0 and has_named_cols:
            # Check if top rows inside table are metadata blocks
            for i in range(min(5, n_rows)):
                row_str = " ".join(df.iloc[i].dropna().astype(str)).lower()
                is_meta_kw = any(kw in row_str for kw in self.METADATA_KEYWORDS)
                if row_densities[i] < (max_density * 0.5) or is_meta_kw:
                    meta_rows.append(i)
                else:
                    break

        bb.metadata_rows = meta_rows
        bb.header_row_index = header_idx

        # 2. Ragged Continuation (Orphaned wrapped text cells)
        for c in range(n_cols):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            solo_mask = (df.notna().sum(axis=1) == 1) & (df.iloc[:, c].notna())
            solo_count = int(solo_mask.sum())
            if solo_count > max(2, int(n_rows * 0.01)):
                lengths = df.iloc[:, c].dropna().astype(str).str.len()
                if lengths.mean() > 10:
                    bb.ragged_continuation_cols.append(col_key)

        # 3. Footer Boundary (Summary / Grand Total rows)
        footer_scan_depth = min(50, n_rows)
        for i in range(n_rows - 1, max(-1, n_rows - footer_scan_depth), -1):
            row_str = " ".join(df.iloc[i].dropna().astype(str)).lower()
            if any(kw in row_str for kw in self.FOOTER_KEYWORDS):
                bb.footer_start_index = i
                break


class Brain2MorphologicalTypologist:
    """Classifies column roles using Shannon entropy, character signatures, and fuzzy coercion."""

    REGEX_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    REGEX_IPV4 = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
    REGEX_DATE = re.compile(r"^\d{2,4}[-/.]\d{1,2}[-/.]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
    REGEX_HIJRI_DATE = re.compile(
        r"(?:^\d{1,2}[-/.]\d{1,2}[-/.]1[34]\d{2}(?:\s*هـ)?$)|"
        r"(?:^1[34]\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s*هـ)?$)|"
        r"(?:.*(?:محرم|صفر|ربيع الأول|ربيع الثاني|جمادى الأولى|جمادى الآخرة|رجب|شعبان|رمضان|شوال|ذو القعدة|ذو الحجة).*\b1[34]\d{2}\b)",
        re.UNICODE
    )
    REGEX_COMPOSITE = re.compile(r"^[\w.]+\s*[/|x_\-]\s*[\w.]+$", re.UNICODE)
    REGEX_CURRENCY = re.compile(r"[$€£¥₹]|SAR|AED|KWD|BHD|OMR|QAR|ر\.س|ريال|د\.إ|^\(\s*[\d,.]+\s*\)$", re.UNICODE)

    # Statutory Saudi / GCC Regional Identifiers
    REGEX_ZATCA_VAT = re.compile(r"^3\d{13}3$")
    REGEX_SAUDI_CR = re.compile(r"^[1247]\d{9}$")
    REGEX_SAUDI_NID = re.compile(r"^[12]\d{9}$")

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard) -> None:
        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(1000)
        n_cols = df.shape[1]

        for c in range(n_cols):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            series = sample_df.iloc[:, c].dropna().astype(str).astype(object).str.strip()
            if series.empty:
                bb.column_profiles[col_key] = {
                    "col_index": c,
                    "role": "ALL_NULL",
                    "entropy": 0.0,
                    "numeric_ratio": 0.0,
                    "cardinality_ratio": 0.0,
                    "is_composite": False,
                    "is_date": False,
                    "is_hijri": False,
                    "is_currency": False,
                    "mean_str_len": 0.0
                }
                continue

            n_samples = len(series)
            entropy = calculate_entropy(series)
            n_unique = series.nunique()
            cardinality_ratio = n_unique / n_samples

            # Clean currency symbols and accounting brackets, but preserve composite delimiters
            cleaned_num_strings = (
                series.str.replace(r"[$€£¥₹]|SAR|AED|KWD|BHD|OMR|QAR|ر\.س|ريال|د\.إ|,|\s", "", regex=True)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            )
            num_coerced = pd.to_numeric(cleaned_num_strings, errors="coerce")
            numeric_ratio = float(num_coerced.notna().mean())

            # Morphological signature matches
            is_hijri = float(series.str.match(self.REGEX_HIJRI_DATE).mean()) > 0.40 or float(series.str.contains(r"1[34]\d{2}\s*هـ").mean()) > 0.40
            is_date = is_hijri or (float(series.str.match(self.REGEX_DATE).mean()) > 0.45)
            is_uuid = float(series.str.match(self.REGEX_UUID).mean()) > 0.45
            is_ip = float(series.str.match(self.REGEX_IPV4).mean()) > 0.45
            is_composite = float(series.str.match(self.REGEX_COMPOSITE).mean()) > 0.40
            is_currency = float(series.str.contains(self.REGEX_CURRENCY).mean()) > 0.20
            has_spaces = float(series.str.contains(r"\s").mean()) > 0.5
            is_narrative = (series.str.len().mean() > 30) and has_spaces

            # Statutory Regional Entity Matches
            is_zatca_vat = float(series.str.match(self.REGEX_ZATCA_VAT).mean()) > 0.60
            is_saudi_cr = float(series.str.match(self.REGEX_SAUDI_CR).mean()) > 0.60
            is_saudi_nid = float(series.str.match(self.REGEX_SAUDI_NID).mean()) > 0.60
            is_regional_id = is_zatca_vat or is_saudi_cr or is_saudi_nid

            regional_id = None
            if is_zatca_vat:
                regional_id = "ZATCA_VAT_ID"
            elif is_saudi_cr:
                regional_id = "SAUDI_CR"
            elif is_saudi_nid:
                regional_id = "SAUDI_NID"

            # Role taxonomy determination with strict precedence
            role = "UNKNOWN"
            if is_hijri:
                role = "TEMPORAL_HIJRI"
            elif is_date:
                role = "TEMPORAL"
            elif is_uuid or is_ip or is_regional_id:
                role = "PRIMARY_IDENTIFIER"
            elif is_narrative:
                role = "FREE_TEXT_NARRATIVE"
            elif is_composite and not is_currency:
                role = "COMPOSITE_KEY"
            elif cardinality_ratio > 0.95 and n_samples >= 5 and numeric_ratio <= 0.8:
                role = "PRIMARY_IDENTIFIER"
            elif numeric_ratio > 0.8:
                role = "CONTINUOUS_NUMERIC" if cardinality_ratio > 0.25 else "DISCRETE_NUMERIC"
            elif entropy < 0.35 or cardinality_ratio < 0.1:
                role = "CATEGORICAL_DIMENSION"
            else:
                role = "CATEGORICAL_DIMENSION"

            bb.column_profiles[col_key] = {
                "col_index": c,
                "role": role,
                "entropy": round(entropy, 4),
                "numeric_ratio": round(numeric_ratio, 4),
                "cardinality_ratio": round(cardinality_ratio, 4),
                "is_composite": is_composite,
                "is_date": is_date,
                "is_hijri": is_hijri,
                "is_currency": is_currency,
                "is_zatca_vat": is_zatca_vat,
                "is_saudi_cr": is_saudi_cr,
                "is_saudi_nid": is_saudi_nid,
                "regional_id": regional_id,
                "mean_str_len": round(float(series.str.len().mean()), 1)
            }


class Brain3ForensicPathologist:
    """Detects type contamination, composite structures, and statistical outliers/skewness."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard) -> None:
        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(1000)

        for col_key, profile in bb.column_profiles.items():
            c = profile.get("col_index", 0)
            series = sample_df.iloc[:, c].dropna().astype(str).astype(object).str.strip()
            if series.empty:
                continue

            # 1. Type Contamination (e.g. 70% to 98% numeric with noisy text tokens)
            num_ratio = profile["numeric_ratio"]
            if 0.65 < num_ratio < 0.99:
                bb.type_contaminations.append({
                    "col": col_key,
                    "defect": f"Mixed Contaminated Types ({num_ratio:.1%} numeric, {(1 - num_ratio):.1%} string noise)",
                    "action": "Force numeric coercion via `pd.to_numeric(..., errors='coerce')` and map non-numeric artifacts to NaN."
                })

            # 2. Composite Splitting
            if profile["is_composite"] and profile["role"] not in ["TEMPORAL", "PRIMARY_IDENTIFIER"]:
                bb.type_contaminations.append({
                    "col": col_key,
                    "defect": "Delimited composite string structure detected (e.g. 'X/Y', 'A-B').",
                    "action": "Decompose into independent feature columns using regex capture groups."
                })

            # 3. Skewness & Outlier Detection on Numeric Columns
            if "NUMERIC" in profile["role"]:
                clean_num_series = series.str.replace(r"[^\d.-]", "", regex=True)
                nums = pd.to_numeric(clean_num_series, errors="coerce").dropna()
                if len(nums) >= 8:
                    q25 = float(nums.quantile(0.25))
                    q75 = float(nums.quantile(0.75))
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers_count = int(((nums < lower_bound) | (nums > upper_bound)).sum())
                    outlier_pct = round((outliers_count / len(nums)) * 100, 1)

                    std_val = float(nums.std())
                    skew_val = float(nums.skew()) if std_val > 1e-6 else 0.0

                    if abs(skew_val) > 1.5 and not math.isnan(skew_val):
                        bb.skewed_columns.append(col_key)

                    if outlier_pct >= 3.0:
                        bb.outlier_metrics[col_key] = {
                            "skew": round(skew_val, 2),
                            "iqr": round(iqr, 2),
                            "outlier_pct": outlier_pct,
                            "action": "Apply log1p transformation or robust IQR clipping."
                        }


class Brain4RelationalCryptographer:
    """Discovers implied primary keys, functional hierarchies, and relational dependencies."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard) -> None:
        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(1000)

        # 1. Candidate Primary Keys (Uniqueness >= 98% and non-null)
        for col_key, profile in bb.column_profiles.items():
            if profile["cardinality_ratio"] >= 0.98 and profile["role"] not in ["ALL_NULL", "CONTINUOUS_NUMERIC"]:
                bb.candidate_primary_keys.append(col_key)

        # 2. Composite Candidate Keys (if no single primary key found)
        if not bb.candidate_primary_keys and sample_df.shape[1] >= 2 and len(sample_df) >= 10:
            high_card_cols = [
                k for k, p in bb.column_profiles.items()
                if p["cardinality_ratio"] > 0.4 and p["role"] != "ALL_NULL"
            ]
            for a, b in itertools.combinations(high_card_cols[:6], 2):
                idx_a = bb.column_profiles[a]["col_index"]
                idx_b = bb.column_profiles[b]["col_index"]
                combined = sample_df.iloc[:, idx_a].astype(str) + "||" + sample_df.iloc[:, idx_b].astype(str)
                if combined.nunique() == len(sample_df):
                    bb.composite_primary_keys.append((a, b))
                    break

        # 3. Hierarchical Functional Dependencies (B -> A)
        cat_cols = [
            k for k, p in bb.column_profiles.items()
            if p["role"] in ["CATEGORICAL_DIMENSION", "DISCRETE_NUMERIC"]
        ]
        for a, b in itertools.permutations(cat_cols[:10], 2):
            idx_a = bb.column_profiles[a]["col_index"]
            idx_b = bb.column_profiles[b]["col_index"]
            s_a = sample_df.iloc[:, idx_a]
            s_b = sample_df.iloc[:, idx_b]

            n_a = s_a.nunique()
            n_b = s_b.nunique()
            if 1 < n_a < n_b and n_b >= 3:
                # If grouping by B always yields exactly 1 unique value of A: B -> A
                sub_df = pd.DataFrame({"a": s_a.values, "b": s_b.values}).dropna()
                if not sub_df.empty:
                    groupby_uniques = sub_df.groupby("b", observed=False)["a"].nunique()
                    if (groupby_uniques == 1).all():
                        bb.hierarchical_dependencies.append((a, b))
                        if len(bb.hierarchical_dependencies) >= 3:
                            break


class Brain5MathematicalPhysicist:
    """Audits algebraic invariants (A * B ≈ C, A + B ≈ C) and statutory VAT identities across numerical dimensions."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard) -> None:
        if not bb.column_profiles:
            for c_idx, col in enumerate(df.columns):
                s = pd.to_numeric(df[col], errors="coerce")
                if float(s.notna().mean()) > 0.7:
                    bb.column_profiles[col] = {"col_index": c_idx, "numeric_ratio": float(s.notna().mean()), "role": "CONTINUOUS_NUMERIC"}

        num_keys = [
            k for k, p in bb.column_profiles.items()
            if p.get("numeric_ratio", 0.0) > 0.7 or "NUMERIC" in p.get("role", "")
        ]
        if len(num_keys) < 2:
            return

        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(500)

        num_matrix: Dict[Union[int, str], pd.Series] = {}
        for k in num_keys[:12]:
            idx = bb.column_profiles[k]["col_index"]
            clean_s = (
                sample_df.iloc[:, idx].astype(str)
                .str.replace(r"[$€£¥₹]|SAR|AED|KWD|BHD|OMR|QAR|ر\.س|ريال|د\.إ|,|\s", "", regex=True)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            )
            num_matrix[k] = pd.to_numeric(clean_s, errors="coerce")

        min_records = min(10, max(3, int(sample_df.shape[0] * 0.4)))

        # 1. Statutory Tax / VAT Invariants (Pairwise Check: ZATCA 15% and GCC 5%)
        for a, b in itertools.permutations(list(num_matrix.keys()), 2):
            s_a, s_b = num_matrix[a], num_matrix[b]
            mask_pair = s_a.notna() & s_b.notna() & (s_a > 0) & (s_b > 0)
            if int(mask_pair.sum()) < min_records:
                continue

            vals_a = s_a[mask_pair].values
            vals_b = s_b[mask_pair].values

            # 15% ZATCA VAT Gross Total: B ≈ A * 1.15
            rel_diff_vat15_gross = np.abs((vals_a * 1.15) - vals_b) / (np.abs(vals_b) + 1e-9)
            if float((rel_diff_vat15_gross < 0.02).mean()) > 0.85:
                law_str = f"Statutory Tax Invariant: `{b}` ≈ `{a}` * 1.15 (15% ZATCA VAT Gross Compliance - Validated across {int(mask_pair.sum())} records)"
                if law_str not in bb.algebraic_laws:
                    bb.algebraic_laws.append(law_str)

            # 15% ZATCA VAT Tax Amount: B ≈ A * 0.15
            rel_diff_vat15_tax = np.abs((vals_a * 0.15) - vals_b) / (np.abs(vals_b) + 1e-9)
            if float((rel_diff_vat15_tax < 0.02).mean()) > 0.85:
                law_str = f"Statutory Tax Invariant: `{b}` ≈ `{a}` * 0.15 (15% ZATCA Tax Amount - Validated across {int(mask_pair.sum())} records)"
                if law_str not in bb.algebraic_laws:
                    bb.algebraic_laws.append(law_str)

            # 5% GCC VAT Gross Total: B ≈ A * 1.05
            rel_diff_vat5_gross = np.abs((vals_a * 1.05) - vals_b) / (np.abs(vals_b) + 1e-9)
            if float((rel_diff_vat5_gross < 0.02).mean()) > 0.85:
                law_str = f"Statutory Tax Invariant: `{b}` ≈ `{a}` * 1.05 (5% GCC VAT Gross Compliance - Validated across {int(mask_pair.sum())} records)"
                if law_str not in bb.algebraic_laws:
                    bb.algebraic_laws.append(law_str)

        # 2. Permutation Invariants: A * B ≈ C and A + B ≈ C
        if len(num_keys) >= 3:
            for a, b, c in itertools.permutations(list(num_matrix.keys()), 3):
                s_a, s_b, s_c = num_matrix[a], num_matrix[b], num_matrix[c]
                mask = s_a.notna() & s_b.notna() & s_c.notna() & (s_a > 0) & (s_b > 0)
                if int(mask.sum()) < min_records:
                    continue

                vals_a = s_a[mask].values
                vals_b = s_b[mask].values
                vals_c = s_c[mask].values

                # Multiplicative Invariant: A * B ≈ C
                product = vals_a * vals_b
                rel_diff_mult = np.abs(product - vals_c) / (np.abs(vals_c) + 1e-9)
                if float((rel_diff_mult < 0.02).mean()) > 0.85:
                    law_str = f"Multiplicative Law: `{a}` * `{b}` ≈ `{c}` (Validated across {int(mask.sum())} records)"
                    if law_str not in bb.algebraic_laws:
                        bb.algebraic_laws.append(law_str)
                    if len(bb.algebraic_laws) >= 4:
                        return

                # Additive Invariant: A + B ≈ C
                sum_ab = vals_a + vals_b
                rel_diff_add = np.abs(sum_ab - vals_c) / (np.abs(vals_c) + 1e-9)
                if float((rel_diff_add < 0.02).mean()) > 0.85:
                    law_str = f"Additive Law: `{a}` + `{b}` ≈ `{c}` (Validated across {int(mask.sum())} records)"
                    if law_str not in bb.algebraic_laws:
                        bb.algebraic_laws.append(law_str)
                    if len(bb.algebraic_laws) >= 4:
                        return


class Brain6AutonomousFeatureAlchemist:
    """Prescribes universal ML feature engineering based on statistical morphology."""

    def execute(self, bb: CognitiveBlackboard) -> None:
        for col_key, profile in bb.column_profiles.items():
            if profile["role"] == "TEMPORAL":
                bb.engineered_features.append({
                    "feature": f"Temporal Deconstruction (`{col_key}`)",
                    "logic": f"Parse `{col_key}` to datetime64[ns] and extract `day_of_week`, `month`, and `is_weekend`."
                })
            elif profile["role"] == "FREE_TEXT_NARRATIVE":
                bb.engineered_features.append({
                    "feature": f"Narrative Density Metrics (`{col_key}`)",
                    "logic": f"Compute character length and token count from `{col_key}` to capture text density."
                })
            elif profile["is_composite"]:
                bb.engineered_features.append({
                    "feature": f"Composite Decomposition (`{col_key}`)",
                    "logic": f"Deconstruct delimited values in `{col_key}` into independent sub-tokens."
                })

        if bb.skewed_columns:
            bb.engineered_features.append({
                "feature": f"Log1p Transformation ({', '.join([f'`{c}`' for c in bb.skewed_columns[:5]])})",
                "logic": "Apply `np.log1p()` transformation to normalize heavy right-skewed distributions."
            })

        if bb.algebraic_laws:
            first_law = bb.algebraic_laws[0].split(" (")[0]
            bb.engineered_features.append({
                "feature": "Mathematical Invariant Integrity Flag",
                "logic": f"Enforce {first_law} and add boolean discrepancy flag `reconciliation_anomaly_flag`."
            })


class Brain7ExecutiveOrchestrator:
    """Translates the Cognitive Blackboard into an authoritative prompt."""

    def execute(self, bb: CognitiveBlackboard) -> str:
        monologue = [
            f"Topological Cartography: Resolved matrix to {bb.shape[0]:,} rows x {bb.shape[1]} columns.",
            f"Boundary Cutoff: Header boundary identified at row index {bb.header_row_index}.",
        ]

        n_temporal = sum(1 for p in bb.column_profiles.values() if p.get("role") == "TEMPORAL")
        n_numeric = sum(1 for p in bb.column_profiles.values() if "NUMERIC" in p.get("role", ""))
        n_cat = sum(1 for p in bb.column_profiles.values() if "CATEGORICAL" in p.get("role", ""))
        monologue.append(
            f"Morphological Entropy: Identified {n_temporal} temporal axes, {n_numeric} quantitative tensors, "
            f"and {n_cat} categorical dimensions."
        )

        if bb.candidate_primary_keys:
            monologue.append(f"Relational Cryptography: Confirmed candidate primary keys: {bb.candidate_primary_keys[:3]}.")
        elif bb.composite_primary_keys:
            monologue.append(f"Relational Cryptography: Identified composite primary key pair: {bb.composite_primary_keys[0]}.")

        if bb.hierarchical_dependencies:
            dep_strs = [f"`{b}` -> `{a}`" for a, b in bb.hierarchical_dependencies[:2]]
            monologue.append(f"Functional Hierarchy: Discovered functional dependencies: {', '.join(dep_strs)}.")

        if bb.algebraic_laws:
            monologue.append(f"Mathematical Physics: Discovered {bb.algebraic_laws[0]}.")

        if bb.type_contaminations:
            monologue.append(f"Forensic Pathology: Detected {len(bb.type_contaminations)} type/structural contaminations.")

        bb.internal_monologue = monologue

        prompt_lines = [
            "### SYSTEM ROLE & OBJECTIVE",
            f"You are an expert Senior Data Engineer. Write a self-contained, deterministic Python (Pandas/NumPy) "
            f"pipeline to clean, flatten, and engineer features for `{bb.filename}`.",
            "\n---",
            "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)",
        ]
        for m in bb.internal_monologue:
            prompt_lines.append(f"* {m}")

        prompt_lines.extend([
            "\n---",
            "### 1. DATASET TOPOLOGY & BOUNDARIES",
            f"* **Source Dimensions**: {bb.shape[0]:,} rows x {bb.shape[1]} columns.",
            f"* **Header Cutoff**: Tabular headers begin at row index {bb.header_row_index}. Discard prior metadata rows."
        ])

        if bb.footer_start_index:
            prompt_lines.append(f"* **Summary Footers**: Footers/subtotals detected starting at row {bb.footer_start_index}. Prune prior to execution.")
        else:
            prompt_lines.append("* **Summary Footers**: No static trailing totals detected.")

        if bb.ragged_continuation_cols:
            prompt_lines.append(
                f"* **Ragged Continuations**: Columns {bb.ragged_continuation_cols} contain orphaned wrapped text. "
                f"Forward-fill empty structural anchors and concatenate these strings upward."
            )

        prompt_lines.extend([
            "\n---",
            "### 2. PATHOLOGY REPAIR PROTOCOLS",
        ])
        if bb.type_contaminations:
            for p in bb.type_contaminations:
                prompt_lines.append(f"* **Column `{p['col']}`**: {p['defect']} -> {p['action']}")
        else:
            prompt_lines.append("* No deep type contaminations detected. Standardize missing tokens.")

        if bb.algebraic_laws:
            prompt_lines.extend([
                "\n---",
                "### 3. MATHEMATICAL INVARIANTS",
                f"* **Algebraic Law**: {bb.algebraic_laws[0]}.",
                "* Enforce this invariant in local execution and create boolean flag `reconciliation_anomaly_flag` for any violating rows."
            ])

        prompt_lines.extend([
            "\n---",
            "### 4. ALGORITHMIC FEATURE ENGINEERING",
            "Based on statistical morphology, generate the following features:"
        ])
        if bb.engineered_features:
            for feat in bb.engineered_features:
                prompt_lines.append(f"* **{feat['feature']}**: {feat['logic']}")
        else:
            prompt_lines.append("* Standardize all categorical strings to title-case and extract temporal epochs if datetime features exist.")

        prompt_lines.extend([
            "\n---",
            "### 5. AST SECURITY FIREWALL CONSTRAINTS",
            "* Do NOT use network libraries (`socket`, `requests`, `urllib`, `httpx`).",
            "* Do NOT access system environments (`os.environ`) or OS filepaths.",
            "* Do NOT use side-channel sleep calls (`time.sleep`).",
            "* Output executable, vectorized Python code using pre-injected standard libraries (`pd`, `np`, `re`)."
        ])

        bb.master_prompt = "\n".join(prompt_lines)
        return bb.master_prompt


class DynamicResonanceEngine:
    """Master Orchestrator triggering the 7-Brain Cognitive Resonance Hive Mind."""

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame, pl.DataFrame],
        filename: Optional[str] = None
    ):
        self.filename = filename or "dataset"
        self.filepath = ""

        if isinstance(data_source, (str, Path)):
            self.filepath = str(data_source)
            self.filename = filename or Path(data_source).name
            self.df_raw = self._load_file(data_source)
        elif isinstance(data_source, pl.DataFrame):
            self.df_raw = data_source.to_pandas()
        elif isinstance(data_source, pd.DataFrame):
            self.df_raw = data_source.copy()
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        # Ingestion sanitization: normalize Eastern Arabic numerals, decimal/thousands separators, and BiDi marks
        for col in self.df_raw.columns:
            if pd.api.types.is_string_dtype(self.df_raw[col]) or self.df_raw[col].dtype == object:
                self.df_raw[col] = self.df_raw[col].map(normalize_bilingual_cell)

        # Track existing column headers if not numeric range
        cols = [normalize_bilingual_cell(str(c)) for c in self.df_raw.columns]
        self.bb = CognitiveBlackboard(
            filepath=self.filepath,
            filename=self.filename,
            shape=self.df_raw.shape,
            columns=cols
        )

    def _load_file(self, path: Union[str, Path]) -> pd.DataFrame:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in [".xlsx", ".xlsm"]:
            return pd.read_excel(p, header=None)
        elif ext == ".xls":
            try:
                return pd.read_excel(p, header=None, engine="calamine")
            except Exception:
                return pd.read_excel(p, header=None)
        elif ext == ".parquet":
            return pd.read_parquet(p)
        elif ext == ".json":
            return pd.read_json(p)
        else:
            return pd.read_csv(p, header=None)

    def think_and_synthesize(self) -> str:
        """Executes the synchronous 7-Brain cognitive loop."""
        Brain1TopologicalCartographer().execute(self.df_raw, self.bb)
        Brain2MorphologicalTypologist().execute(self.df_raw, self.bb)
        Brain3ForensicPathologist().execute(self.df_raw, self.bb)
        Brain4RelationalCryptographer().execute(self.df_raw, self.bb)
        Brain5MathematicalPhysicist().execute(self.df_raw, self.bb)
        Brain6AutonomousFeatureAlchemist().execute(self.bb)
        return Brain7ExecutiveOrchestrator().execute(self.bb)

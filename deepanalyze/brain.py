"""DeepAnalyze: 18-Brain Omni-Cognitive Resonance Engine (Left Hemisphere: 14 Data Physics Brains + Right Hemisphere: 4 Startup Colleague EQ Brains).
Combines Left Hemisphere cold data physics with Right Hemisphere emotional intelligence, friction analysis, and humble startup colleague persona wrapper.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class CognitiveBlackboard:
    """Shared state bus for all 14 cognitive sub-engines with Stigmergic Bayesian consensus."""

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

    # Stigmergic Belief Tensors & Probabilities (Col Key -> Belief Type -> Probability [0.0, 1.0])
    column_beliefs: Dict[Union[int, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    # Pathology (Brain 3)
    type_contaminations: List[Dict[str, Any]] = field(default_factory=list)
    cross_column_leaks: List[Dict[str, Any]] = field(default_factory=list)
    skewed_columns: List[Union[int, str]] = field(default_factory=list)
    outlier_metrics: Dict[Union[int, str], Dict[str, Any]] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    # Relational & Cardinality (Brain 4)
    candidate_primary_keys: List[Union[int, str]] = field(default_factory=list)
    composite_primary_keys: List[Tuple[Union[int, str], ...]] = field(default_factory=list)
    hierarchical_dependencies: List[Tuple[Union[int, str], Union[int, str]]] = field(default_factory=list)

    # Mathematical Physics (Brain 5)
    algebraic_laws: List[str] = field(default_factory=list)

    # Feature Alchemy & Directives (Brain 6)
    engineered_features: List[Dict[str, str]] = field(default_factory=list)
    feature_directives: List[Dict[str, str]] = field(default_factory=list)

    # Spatial Cartography (Brain 8)
    spatial_profiles: Dict[str, Any] = field(default_factory=dict)

    # Chronometric Signal Processing (Brain 9)
    chronometric_profiles: Dict[str, Any] = field(default_factory=dict)

    # Process & State Modeling (Brain 10)
    process_models: Dict[str, Any] = field(default_factory=dict)

    # Tensor Semantics (Brain 11)
    tensor_profiles: Dict[str, Any] = field(default_factory=dict)

    # Graph & Network Topology (Brain 12)
    graph_topology: Dict[str, Any] = field(default_factory=dict)

    # Statutory Governance & Privacy (Brain 13)
    compliance_overrides: List[str] = field(default_factory=list)

    # Cryptographic Sentinel & Surrogates (Brain 14)
    cryptographic_signatures: List[str] = field(default_factory=list)

    # Final Output & Monologue (Brain 7)
    internal_monologue: List[str] = field(default_factory=list)
    master_prompt: str = ""

    # Right Hemisphere EQ, Socratic Inquiry & Persona (Brains 15-18)
    colleague_questions: List[str] = field(default_factory=list)
    persona_directives: List[str] = field(default_factory=list)
    detective_insights: List[str] = field(default_factory=list)
    friction_score: int = 0

    # Ouroboros Self-Healing State
    ouroboros_traceback: Optional[str] = None
    ouroboros_repair_prompt: Optional[str] = None

    def add_belief(self, col_key: Union[int, str], belief_type: str, confidence: float, reasoning: str = "") -> None:
        """Bayesian-inspired additive probability update: P(A or B) = P(A) + P(B) - P(A)*P(B)."""
        current = self.column_beliefs[col_key][belief_type]
        updated = current + confidence - (current * confidence)
        self.column_beliefs[col_key][belief_type] = float(np.clip(updated, 0.0, 1.0))

    def get_dominant_belief(self, col_key: Union[int, str]) -> str:
        """Returns the classification with the highest probabilistic confidence."""
        beliefs = self.column_beliefs.get(col_key, {})
        if not beliefs:
            return self.column_profiles.get(col_key, {}).get("role", "UNKNOWN")
        return max(beliefs, key=beliefs.get)


# Alias StigmergicBlackboard to CognitiveBlackboard for unified architecture
StigmergicBlackboard = CognitiveBlackboard


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


# Alias sanitize_cell to normalize_bilingual_cell
sanitize_cell = normalize_bilingual_cell


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


class BaseCognitiveBrain:
    """Base class for all 14 cognitive sub-engines."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> Any:
        raise NotImplementedError


# ==============================================================================
# BRAIN 1: TOPOLOGICAL CARTOGRAPHER
# ==============================================================================
class Brain1TopologicalCartographer(BaseCognitiveBrain):
    """Maps physical geometry, density drops, header cutoffs, and ragged boundaries."""

    METADATA_KEYWORDS = [
        "report", "filter", "criteria", "parameters", "sort", "date :",
        "تصفية", "فلتر", "معايير", "التاريخ :", "ترتيب", "تقرير", "بيانات", "كشف"
    ]
    FOOTER_KEYWORDS = [
        "total", "grand total", "subtotal", "summary", "end of report", "average", "count",
        "المجموع", "الإجمالي", "إجمالي", "صافي", "ملخص", "المحصلة", "النهائي", "كلي"
    ]

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return

        row_densities = df.notna().sum(axis=1).values
        max_density = float(row_densities.max()) if len(row_densities) > 0 else 0.0

        # 1. Header Boundary Detection
        meta_rows: List[int] = []
        header_idx = 0

        scan_limit = min(50, n_rows)
        for i in range(scan_limit):
            row_vals = df.iloc[i].dropna().astype(str).str.strip()
            row_str = " ".join(row_vals).lower()

            if any(kw in row_str for kw in self.METADATA_KEYWORDS):
                meta_rows.append(i)
                continue

            density = float(row_densities[i])
            if max_density > 0 and density < (max_density * 0.40):
                meta_rows.append(i)
            elif max_density > 0 and density >= (max_density * 0.70):
                str_count = sum(1 for v in row_vals if not v.replace(".", "", 1).isdigit())
                if str_count / max(1, len(row_vals)) > 0.5:
                    header_idx = i
                    break

        bb.metadata_rows = meta_rows
        bb.header_row_index = header_idx

        # 2. Ragged Continuation (Multiline wrap)
        for c in range(n_cols):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            solo_mask = (df.notna().sum(axis=1) == 1) & (df.iloc[:, c].notna())
            if float(solo_mask.sum()) > (n_rows * 0.008):
                lengths = df.iloc[:, c].dropna().astype(str).str.len()
                if not lengths.empty and lengths.mean() > 8:
                    bb.ragged_continuation_cols.append(col_key)

        # 3. Footer Boundary
        for i in range(n_rows - 1, max(0, n_rows - 50), -1):
            row_str = " ".join(df.iloc[i].dropna().astype(str)).lower()
            if any(kw in row_str for kw in self.FOOTER_KEYWORDS):
                bb.footer_start_index = i
                break


# ==============================================================================
# BRAIN 2: MORPHOLOGICAL TYPOLOGIST
# ==============================================================================
class Brain2MorphologicalTypologist(BaseCognitiveBrain):
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

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(sample_size)
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
                bb.add_belief(col_key, "TEMPORAL_HIJRI", 0.95, "Hijri calendar format detected.")
                bb.add_belief(col_key, "TEMPORAL", 0.90, "Temporal date syntax.")
            elif is_date:
                role = "TEMPORAL"
                bb.add_belief(col_key, "TEMPORAL", 0.90, "Standard ISO/gregorian date syntax.")
            elif is_uuid or is_ip or is_regional_id:
                role = "PRIMARY_IDENTIFIER"
                bb.add_belief(col_key, "PRIMARY_IDENTIFIER", 0.95, "Explicit structural/regional identifier.")
            elif is_narrative:
                role = "FREE_TEXT_NARRATIVE"
                bb.add_belief(col_key, "FREE_TEXT_NARRATIVE", 0.90, "High average length with whitespace.")
            elif is_composite and not is_currency:
                role = "COMPOSITE_KEY"
                bb.add_belief(col_key, "COMPOSITE_KEY", 0.90, "Delimited multi-part composite key.")
            elif cardinality_ratio > 0.95 and n_samples >= 5 and numeric_ratio <= 0.8:
                role = "PRIMARY_IDENTIFIER"
                bb.add_belief(col_key, "PRIMARY_IDENTIFIER", 0.88, "High cardinality non-numeric identifier.")
            elif numeric_ratio > 0.8:
                role = "CONTINUOUS_NUMERIC" if cardinality_ratio > 0.25 else "DISCRETE_NUMERIC"
                bb.add_belief(col_key, role, min(0.85, numeric_ratio * 0.85), "High numeric conversion ratio.")
            elif entropy < 0.35 or cardinality_ratio < 0.1:
                role = "CATEGORICAL_DIMENSION"
                bb.add_belief(col_key, "CATEGORICAL_DIMENSION", 0.85, "Low entropy categorical distribution.")
            else:
                role = "CATEGORICAL_DIMENSION"
                bb.add_belief(col_key, "CATEGORICAL_DIMENSION", 0.70, "General dimensional string.")

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


# ==============================================================================
# BRAIN 3: FORENSIC PATHOLOGIST
# ==============================================================================
class Brain3ForensicPathologist(BaseCognitiveBrain):
    """Detects type contamination, composite structures, and statistical outliers/skewness."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(sample_size)

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
                    "numeric_ratio": num_ratio,
                    "defect": "Contaminated numeric column containing string artifacts or missing sentinels",
                    "action": "Coerce with `pd.to_numeric(errors='coerce')` and handle NaN values explicitly."
                })
                bb.anomalies.append({
                    "col": col_key,
                    "defect": "Contaminated numeric column with mixed text tokens.",
                    "action": "Coerce to numeric, mapping non-numeric tokens to NaN."
                })

            # 2. Composite Delimited Strings (e.g. "120/80", "12GB/256GB")
            if profile.get("is_composite") and not profile.get("is_currency"):
                bb.type_contaminations.append({
                    "col": col_key,
                    "defect": "Delimited composite string structure detected (e.g. 'X/Y', 'A-B').",
                    "action": "Decompose into independent feature columns using regex capture groups."
                })
                bb.anomalies.append({
                    "col": col_key,
                    "defect": "Composite delimited string (e.g., Blood Pressure/Storage/Arabic ID).",
                    "action": "Decompose into independent sub-token fields using regex capture groups."
                })
                bb.add_belief(col_key, "COMPOSITE_METRIC", 0.95, "Slash/Dash/Hyphen delimited composite.")

            # 3. Statistical Skewness & Outliers for Continuous Numerics
            if "NUMERIC" in profile.get("role", ""):
                clean_nums = (
                    series.str.replace(r"[$€£¥₹]|SAR|AED|KWD|BHD|OMR|QAR|ر\.س|ريال|د\.إ|,|\s", "", regex=True)
                    .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                )
                nums = pd.to_numeric(clean_nums, errors="coerce").dropna()
                if len(nums) >= 8:
                    mean_val = float(nums.mean())
                    std_val = float(nums.std())
                    skew_val = float(nums.skew()) if std_val > 1e-6 else 0.0
                    if abs(skew_val) > 1.5 and not math.isnan(skew_val):
                        bb.skewed_columns.append(col_key)

                    q25, q75 = float(nums.quantile(0.25)), float(nums.quantile(0.75))
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_bound = q25 - (1.5 * iqr)
                        upper_bound = q75 + (1.5 * iqr)
                        n_outliers = int(((nums < lower_bound) | (nums > upper_bound)).sum())
                        if n_outliers > 0:
                            bb.outlier_metrics[col_key] = {
                                "iqr": round(iqr, 2),
                                "outliers_count": n_outliers,
                                "bounds": (round(lower_bound, 2), round(upper_bound, 2))
                            }


# ==============================================================================
# BRAIN 4: RELATIONAL CRYPTOGRAPHER
# ==============================================================================
class Brain4RelationalCryptographer(BaseCognitiveBrain):
    """Discovers Candidate Primary Keys, Composite Keys, and Functional Hierarchies."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample_df = df.iloc[start_row:].head(sample_size)
        n_rows = sample_df.shape[0]
        if n_rows < 5:
            return

        # 1. Candidate Primary Keys
        for col_key, profile in bb.column_profiles.items():
            if profile.get("cardinality_ratio", 0) >= 0.98 and profile.get("role") != "FREE_TEXT_NARRATIVE":
                c = profile.get("col_index", 0)
                series = sample_df.iloc[:, c].dropna()
                if len(series) == n_rows and series.nunique() == n_rows:
                    bb.candidate_primary_keys.append(col_key)
                    bb.add_belief(col_key, "PRIMARY_KEY", 0.95, "100% uniqueness without nulls.")

        # 2. Composite Key Discovery
        if not bb.candidate_primary_keys and len(bb.column_profiles) >= 2:
            dim_keys = [
                k for k, p in bb.column_profiles.items()
                if p.get("role") in ("CATEGORICAL_DIMENSION", "TEMPORAL", "DISCRETE_NUMERIC")
            ]
            for a, b in itertools.combinations(dim_keys[:6], 2):
                idx_a = bb.column_profiles[a]["col_index"]
                idx_b = bb.column_profiles[b]["col_index"]
                combined = sample_df.iloc[:, idx_a].astype(str) + "||" + sample_df.iloc[:, idx_b].astype(str)
                if combined.nunique() == n_rows:
                    bb.composite_primary_keys.append((a, b))
                    bb.add_belief(a, "COMPOSITE_PRIMARY_KEY", 0.85, f"Pairwise primary key with {b}")
                    bb.add_belief(b, "COMPOSITE_PRIMARY_KEY", 0.85, f"Pairwise primary key with {a}")
                    break

        # 3. Hierarchical Functional Dependencies (B -> A)
        cat_keys = [
            k for k, p in bb.column_profiles.items()
            if p.get("role") == "CATEGORICAL_DIMENSION" and 0.01 < p.get("cardinality_ratio", 0) < 0.60
        ]
        if len(cat_keys) >= 2:
            for a, b in itertools.permutations(cat_keys[:8], 2):
                idx_a = bb.column_profiles[a]["col_index"]
                idx_b = bb.column_profiles[b]["col_index"]
                sub_df = sample_df.iloc[:, [idx_a, idx_b]].dropna()
                sub_df.columns = ["a", "b"]
                if not sub_df.empty and sub_df["b"].nunique() > 1:
                    groupby_uniques = sub_df.groupby("b", observed=False)["a"].nunique()
                    if (groupby_uniques == 1).all():
                        bb.hierarchical_dependencies.append((a, b))
                        if len(bb.hierarchical_dependencies) >= 3:
                            break


# ==============================================================================
# BRAIN 5: MATHEMATICAL PHYSICIST
# ==============================================================================
class Brain5MathematicalPhysicist(BaseCognitiveBrain):
    """Audits algebraic invariants (A * B ≈ C, A + B ≈ C) and statutory VAT identities across numerical dimensions."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
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
        sample_df = df.iloc[start_row:].head(min(500, sample_size))

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


# ==============================================================================
# BRAIN 6: AUTONOMOUS FEATURE ALCHEMIST
# ==============================================================================
class Brain6AutonomousFeatureAlchemist(BaseCognitiveBrain):
    """Prescribes universal ML feature engineering based on statistical morphology."""

    def execute(self, df_or_bb: Any, bb: Optional[CognitiveBlackboard] = None, sample_size: int = 1500) -> None:
        target_bb = bb if bb is not None else (df_or_bb if isinstance(df_or_bb, CognitiveBlackboard) else None)
        if target_bb is None:
            return

        for col_key, profile in target_bb.column_profiles.items():
            if profile.get("role") in ("TEMPORAL", "TEMPORAL_HIJRI"):
                target_bb.engineered_features.append({
                    "feature": f"Temporal Deconstruction (`{col_key}`)",
                    "logic": f"Parse `{col_key}` to datetime64[ns] and extract `day_of_week`, `month`, and `is_weekend`."
                })
            elif profile.get("role") == "FREE_TEXT_NARRATIVE":
                target_bb.engineered_features.append({
                    "feature": f"Narrative Density Metrics (`{col_key}`)",
                    "logic": f"Compute character length and token count from `{col_key}` to capture text density."
                })
            elif profile.get("is_composite"):
                target_bb.engineered_features.append({
                    "feature": f"Composite Decomposition (`{col_key}`)",
                    "logic": f"Deconstruct delimited values in `{col_key}` into independent sub-tokens."
                })

        if target_bb.skewed_columns:
            target_bb.engineered_features.append({
                "feature": f"Log1p Transformation ({', '.join([f'`{c}`' for c in target_bb.skewed_columns[:5]])})",
                "logic": "Apply `np.log1p()` transformation to normalize heavy right-skewed distributions."
            })

        if target_bb.algebraic_laws:
            first_law = target_bb.algebraic_laws[0].split(" (")[0]
            target_bb.engineered_features.append({
                "feature": "Mathematical Invariant Integrity Flag",
                "logic": f"Enforce {first_law} and add boolean discrepancy flag `reconciliation_anomaly_flag`."
            })


# ==============================================================================
# BRAIN 8: SPATIAL CARTOGRAPHER
# ==============================================================================
class Brain8SpatialCartographer(BaseCognitiveBrain):
    """Detects geospatial vectors, bounding box boundaries, and projection distortions."""

    REGEX_LAT = re.compile(r"^(?:lat|latitude|y_coord|y_pos|خط_العرض)$", re.IGNORECASE)
    REGEX_LON = re.compile(r"^(?:lon|lng|long|longitude|x_coord|x_pos|خط_الطول)$", re.IGNORECASE)
    REGEX_COORD_PAIR = re.compile(r"^\(?\s*([-+]?\d{1,2}(?:\.\d+)?)\s*,\s*([-+]?\d{1,3}(?:\.\d+)?)\s*\)?$")

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample = df.iloc[start_row:].head(sample_size)
        n_cols = df.shape[1]

        lat_cols: List[Union[int, str]] = []
        lon_cols: List[Union[int, str]] = []
        pair_cols: List[Union[int, str]] = []

        for c in range(n_cols):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            series = sample.iloc[:, c].dropna().astype(str).astype(object).str.strip()
            if series.empty:
                continue

            # 1. Single-column coordinate pair (e.g. "24.7136, 46.6753")
            pair_ratio = float(series.str.match(self.REGEX_COORD_PAIR).mean())
            if pair_ratio > 0.40:
                pair_cols.append(col_key)
                bb.add_belief(col_key, "SPATIAL_COORD", 0.95, "Coordinate pair string detected.")
                continue

            # 2. Independent lat/long numeric series
            clean_s = pd.to_numeric(series.str.replace(r"[^\d.-]", "", regex=True), errors="coerce").dropna()
            if len(clean_s) >= 5 and (len(clean_s) / len(series)) > 0.8:
                col_name_str = str(col_key).lower()
                is_named_lat = bool(self.REGEX_LAT.search(col_name_str))
                is_named_lon = bool(self.REGEX_LON.search(col_name_str))

                in_lat_range = float(((clean_s >= -90.0) & (clean_s <= 90.0)).mean()) > 0.95
                in_lon_range = float(((clean_s >= -180.0) & (clean_s <= 180.0)).mean()) > 0.95

                if is_named_lat or (in_lat_range and not is_named_lon and "lat" in col_name_str):
                    lat_cols.append(col_key)
                    bb.add_belief(col_key, "SPATIAL_LATITUDE", 0.95, "Latitude spatial coordinate range.")
                    bb.add_belief(col_key, "SPATIAL_COORD", 0.92, "Latitude spatial coordinate range.")
                elif is_named_lon or (in_lon_range and not is_named_lat and ("lon" in col_name_str or "lng" in col_name_str)):
                    lon_cols.append(col_key)
                    bb.add_belief(col_key, "SPATIAL_LONGITUDE", 0.95, "Longitude spatial coordinate range.")
                    bb.add_belief(col_key, "SPATIAL_COORD", 0.92, "Longitude spatial coordinate range.")

        if lat_cols or lon_cols or pair_cols:
            bb.spatial_profiles = {
                "lat_cols": lat_cols,
                "lon_cols": lon_cols,
                "pair_cols": pair_cols,
                "coordinates": {
                    "lat_col": lat_cols[0] if lat_cols else None,
                    "lon_col": lon_cols[0] if lon_cols else None
                }
            }
            bb.feature_directives.append({
                "feature": "Spatial Geo-Engineering (Haversine & Bounding Box)",
                "logic": f"Project geospatial coordinates ({lat_cols or pair_cols}, {lon_cols}) into metric CRS (EPSG:3857) or compute Haversine distances & H3 hex-bins."
            })


# ==============================================================================
# BRAIN 9: CHRONOMETRIC SIGNAL PROCESSOR
# ==============================================================================
class Brain9ChronometricSignalProcessor(BaseCognitiveBrain):
    """Detects time-series interval periodicity via FFT, volatility clustering, and homoscedasticity."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample = df.iloc[start_row:].head(sample_size)

        temporal_cols = [
            c for c, p in bb.column_profiles.items()
            if p.get("role") in ("TEMPORAL", "TEMPORAL_HIJRI") or p.get("is_date")
        ]
        numeric_cols = [
            c for c, p in bb.column_profiles.items()
            if "NUMERIC" in p.get("role", "")
        ]

        if not temporal_cols or not numeric_cols:
            return

        date_col = temporal_cols[0]
        c_idx = bb.column_profiles[date_col].get("col_index", 0)
        s_date = pd.to_datetime(sample.iloc[:, c_idx], errors="coerce").dropna().sort_values()

        if len(s_date) < 10:
            return

        deltas = s_date.diff().dropna()
        median_delta = deltas.median()
        is_uniform = bool((deltas == median_delta).mean() > 0.70)

        dominant_period = None
        num_col = numeric_cols[0]
        n_idx = bb.column_profiles[num_col].get("col_index", 0)
        s_num = pd.to_numeric(sample.iloc[:, n_idx], errors="coerce").dropna().values

        if len(s_num) >= 32:
            fft_vals = np.abs(np.fft.rfft(s_num - np.mean(s_num)))
            if len(fft_vals) > 2:
                top_freq_idx = int(np.argmax(fft_vals[1:])) + 1
                if top_freq_idx > 0 and fft_vals[top_freq_idx] > (1.8 * float(np.median(fft_vals))):
                    dominant_period = round(len(s_num) / top_freq_idx, 1)

        bb.chronometric_profiles = {
            "date_col": date_col,
            "median_delta": str(median_delta),
            "is_uniform": is_uniform,
            "dominant_period": dominant_period,
            date_col: {"cadence": str(median_delta), "is_uniform": is_uniform}
        }
        bb.add_belief(date_col, "TEMPORAL_CHRONOMETRIC", 0.95, f"Temporal delta {median_delta}")
        bb.add_belief(date_col, "CHRONOMETRIC_TIME_SERIES", 0.90, f"Temporal delta {median_delta}")

        desc = (
            f"Apply Fourier feature extraction (detected period: {dominant_period}), lag features, and rolling standard deviation."
            if dominant_period
            else "Extract chronological lag features, rolling windows, and evaluate variance stability over time."
        )
        bb.feature_directives.append({
            "feature": f"Chronometric Signal Processing (`{date_col}` -> `{num_col}`)",
            "logic": desc
        })


# ==============================================================================
# BRAIN 10: PROCESS & STATE MODELER
# ==============================================================================
class Brain10ProcessStateModeler(BaseCognitiveBrain):
    """Detects event log sequence mining patterns, transition matrices, and concurrency anomalies."""

    REGEX_CASE = re.compile(r"^(?:case|trace|order|claim|ticket|visit|session|patient)[_\s]?(?:id|num|no)?$", re.IGNORECASE)
    REGEX_ACTIVITY = re.compile(r"^(?:activity|action|status|stage|step|event|state|operation|الحالة|المرحلة)$", re.IGNORECASE)

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample = df.iloc[start_row:].head(sample_size)
        n_cols = df.shape[1]

        case_col = None
        activity_col = None
        timestamp_col = None

        for c in range(n_cols):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            name_str = str(col_key).lower()
            profile = bb.column_profiles.get(col_key, {})

            if profile.get("role") in ("TEMPORAL", "TEMPORAL_HIJRI") and timestamp_col is None:
                timestamp_col = col_key
            elif self.REGEX_CASE.search(name_str) and case_col is None:
                case_col = col_key
            elif self.REGEX_ACTIVITY.search(name_str) or "status" in name_str or "state" in name_str or "stage" in name_str or (profile.get("role") in ("CATEGORICAL_DIMENSION", "COMPOSITE_KEY") and profile.get("entropy", 0) < 0.8):
                if activity_col is None and col_key != case_col:
                    activity_col = col_key

        if activity_col:
            c_act_idx = bb.column_profiles[activity_col].get("col_index", 0) if activity_col in bb.column_profiles else (bb.columns.index(activity_col) if activity_col in bb.columns else 0)
            states = list(sample.iloc[:, c_act_idx].dropna().unique())
            bb.add_belief(activity_col, "PROCESS_STATE", 0.95, "Discrete lifecycle workflow states.")
            bb.process_models[activity_col] = {
                "states": states,
                "case_col": case_col,
                "timestamp_col": timestamp_col,
            }
            if case_col:
                bb.add_belief(case_col, "PROCESS_CASE_ID", 0.92, "Process instance case identifier.")

            concurrency_anomalies = 0
            if case_col and timestamp_col:
                c_case_idx = bb.column_profiles[case_col].get("col_index", 0)
                c_time_idx = bb.column_profiles[timestamp_col].get("col_index", 0)
                pair_series = sample.iloc[:, c_case_idx].astype(str) + "||" + sample.iloc[:, c_time_idx].astype(str)
                concurrency_anomalies = int((pair_series.value_counts() > 1).sum())
                bb.process_models["concurrency_anomalies"] = concurrency_anomalies

            bb.feature_directives.append({
                "feature": f"Process State Modeling (`{activity_col}`)",
                "logic": f"Model transition states {states[:5]}, construct Directly-Follows Graph (DFG) and detect process bottlenecks."
            })


# ==============================================================================
# BRAIN 11: TENSOR SEMANTICIST
# ==============================================================================
class Brain11TensorSemanticist(BaseCognitiveBrain):
    """Detects vector embeddings, intrinsic dimensionality, and Manifold Hypothesis preservation."""

    REGEX_VECTOR_STRING = re.compile(r"^\[\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?){7,}\s*\]$")

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample = df.iloc[start_row:].head(sample_size)
        n_cols = df.shape[1]

        embedding_cols: List[Union[int, str]] = []
        for c in range(n_cols):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            series = sample.iloc[:, c].dropna().astype(str).astype(object).str.strip()
            if series.empty:
                continue
            is_vector_str = float(series.str.match(self.REGEX_VECTOR_STRING).mean()) > 0.40
            if is_vector_str:
                embedding_cols.append(col_key)
                bb.add_belief(col_key, "TENSOR_EMBEDDING", 0.95, "Vector embedding array string.")

        # Multi-column embedding blocks (e.g. emb_dim_0 .. emb_dim_N)
        regex_emb_name = re.compile(r"^(?:emb|embedding|dim|vec|vector|latent|pca)[_\w]*\d+$", re.IGNORECASE)
        named_emb_cols = [
            (bb.columns[c] if c < len(bb.columns) else c) for c in range(n_cols)
            if regex_emb_name.search(str(bb.columns[c] if c < len(bb.columns) else c))
        ]
        if len(named_emb_cols) >= 4:
            embedding_cols.extend(named_emb_cols)
            for col_k in named_emb_cols:
                bb.add_belief(col_k, "TENSOR_EMBEDDING", 0.90, "Multi-dimensional tensor embedding dimension.")

        if embedding_cols:
            bb.tensor_profiles = {
                "embedding_cols": embedding_cols,
                "dimension_count": len(embedding_cols),
                "preservation_method": "UMAP / TruncatedSVD"
            }
            bb.feature_directives.append({
                "feature": f"Tensor Manifold Dimensionality Reduction ({len(embedding_cols)} dimensions)",
                "logic": "Parse vector embeddings, evaluate intrinsic dimensionality, and apply non-linear UMAP or TruncatedSVD to preserve manifold topology."
            })


# ==============================================================================
# BRAIN 12: GRAPH & NETWORK TOPOLOGIST
# ==============================================================================
class Brain12GraphNetworkTopologist(BaseCognitiveBrain):
    """Detects graph network topologies, node linkages, and degree centrality distributions."""

    REGEX_SRC = re.compile(r"^(?:source|src|sender|from|caller|parent|المصدر)[_\s]?(?:id|num|no)?$", re.IGNORECASE)
    REGEX_DST = re.compile(r"^(?:target|dst|dest|receiver|recipient|to|callee|child|الوجهة|الهدف)[_\s]?(?:id|num|no)?$", re.IGNORECASE)

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample = df.iloc[start_row:].head(sample_size)

        id_cols = [
            c for c, p in bb.column_profiles.items()
            if p.get("role") in ("PRIMARY_IDENTIFIER", "CATEGORICAL_DIMENSION", "COMPOSITE_KEY") or self.REGEX_SRC.search(str(c)) or self.REGEX_DST.search(str(c))
        ]

        src_dst_pairs = []
        for a, b in itertools.permutations(id_cols[:8], 2):
            name_a, name_b = str(a).lower(), str(b).lower()
            is_name_link = bool(self.REGEX_SRC.search(name_a) and self.REGEX_DST.search(name_b))

            idx_a = bb.column_profiles[a].get("col_index", 0) if a in bb.column_profiles else (bb.columns.index(a) if a in bb.columns else 0)
            idx_b = bb.column_profiles[b].get("col_index", 0) if b in bb.column_profiles else (bb.columns.index(b) if b in bb.columns else 1)
            vals_a = set(sample.iloc[:, idx_a].dropna().astype(str))
            vals_b = set(sample.iloc[:, idx_b].dropna().astype(str))

            if vals_a and vals_b:
                overlap = len(vals_a.intersection(vals_b)) / min(len(vals_a), len(vals_b))
                if is_name_link or overlap > 0.15:
                    src_dst_pairs.append((a, b, round(overlap, 3)))
                    bb.add_belief(a, "NETWORK_SOURCE_NODE", 0.88, f"Network linkage to {b}")
                    bb.add_belief(b, "NETWORK_TARGET_NODE", 0.88, f"Network linkage from {a}")
                    break

        if src_dst_pairs:
            src, dst, ov = src_dst_pairs[0]
            bb.graph_topology = {
                "source": src,
                "target": dst,
                "overlap": ov,
                "edges": {"source_col": src, "target_col": dst}
            }
            bb.feature_directives.append({
                "feature": f"Graph Topology & Node Centrality (`{src}` -> `{dst}`)",
                "logic": f"Construct directed network graph, extract PageRank and in/out degree centrality features across shared entity universe."
            })


# ==============================================================================
# BRAIN 13: STATUTORY ARBITER
# ==============================================================================
class Brain13StatutoryArbiter(BaseCognitiveBrain):
    """Audits data privacy risk vectors, quasi-identifier linkage attacks, and compliance overrides."""

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        # 1. Statutory Regional & Tax Identifiers
        for c, p in bb.column_profiles.items():
            col_name = str(c).lower()
            if p.get("is_zatca_vat") or p.get("regional_id") == "ZATCA_VAT_ID" or "vat" in col_name or "tax" in col_name:
                bb.compliance_overrides.append(
                    f"ZATCA / Statutory Tax Compliance: Column `{c}` contains tax registration/VAT identifiers. Enforce statutory retention and deterministic tokenization."
                )
            elif p.get("is_saudi_nid") or p.get("is_saudi_cr") or "id_card" in col_name or "national_id" in col_name or "cr_no" in col_name:
                bb.compliance_overrides.append(
                    f"NDMO / Regional Identifier Privacy Override: Column `{c}` contains national identification or commercial registry numbers. Restrict direct transmission and enforce cryptographic masking."
                )
            elif "iban" in col_name or "account" in col_name or "swift" in col_name:
                bb.compliance_overrides.append(
                    f"SAMA / Financial Banking Compliance: Column `{c}` contains banking coordinates. Enforce cryptographic tokenization."
                )

        # 2. Spatial Privacy Override
        if bb.spatial_profiles:
            coords = bb.spatial_profiles.get("lat_cols", []) + bb.spatial_profiles.get("pair_cols", [])
            bb.compliance_overrides.append(
                f"Statutory Privacy Override on Coordinates {coords}: Mandate H3 Hex-Binning or Laplace Differential Privacy noise. Do not emit precise GPS points."
            )

        # 3. Quasi-Identifier Linkage Vulnerability
        cat_cols = [
            c for c, p in bb.column_profiles.items()
            if p.get("role") == "CATEGORICAL_DIMENSION" and 0.05 < p.get("cardinality_ratio", 0) < 0.5
        ]
        if len(cat_cols) >= 3:
            bb.compliance_overrides.append(
                f"Quasi-Identifier Linkage Attack Risk: Columns {cat_cols[:3]} form a quasi-identifier vector. Ensure k-anonymity (k >= 5) before external aggregation."
            )


# ==============================================================================
# BRAIN 14: CRYPTOGRAPHIC SENTINEL
# ==============================================================================
class Brain14CryptographicSentinel(BaseCognitiveBrain):
    """Identifies structural cryptographic surrogates and prevents LLM semantic hallucination."""

    REGEX_TOKEN_SURROGATE = re.compile(r"(?:<[A-Z_]+(?:_\d+)?>|TOK_[A-Z0-9_]+|SURR_[A-Z0-9_]+|ANON_[A-Z0-9_]+)")
    REGEX_ALPHANUM_MASK = re.compile(r"(?:X{3,}|[x*]{3,})")
    REGEX_DUMMY_NUMERIC = re.compile(r"^(?:9,999(?:\.00)?|99999|-999|-9999)$")

    def execute(self, df: pd.DataFrame, bb: CognitiveBlackboard, sample_size: int = 1500) -> None:
        start_row = _get_data_start_row(df, bb)
        sample = df.iloc[start_row:].head(min(150, sample_size))

        for c in range(sample.shape[1]):
            col_key = bb.columns[c] if c < len(bb.columns) else c
            series_str = sample.iloc[:, c].dropna().astype(str)
            if series_str.empty:
                continue
            matched_tokens = series_str[series_str.str.contains(self.REGEX_TOKEN_SURROGATE, regex=True)]
            if len(matched_tokens) > 0:
                sig = f"Deterministic Token Surrogates: Column `{col_key}` contains surrogate tokens ('{matched_tokens.iloc[0]}')."
                if sig not in bb.cryptographic_signatures:
                    bb.cryptographic_signatures.append(sig)
                bb.add_belief(col_key, "CRYPTOGRAPHIC_SURROGATE", 0.98, "Surrogate token format.")

        sample_txt = " ".join(sample.fillna("").astype(str).values.flatten())

        if self.REGEX_ALPHANUM_MASK.search(sample_txt):
            sig = "Alphanumeric Masking: 'XXXX' or '***' surrogate patterns detected."
            if sig not in bb.cryptographic_signatures:
                bb.cryptographic_signatures.append(sig)

        if self.REGEX_DUMMY_NUMERIC.search(sample_txt):
            sig = "Standardized Numeric Placeholders: '9,999.00' or '-999' dummy masks detected."
            if sig not in bb.cryptographic_signatures:
                bb.cryptographic_signatures.append(sig)

        if bb.cryptographic_signatures:
            for c, p in bb.column_profiles.items():
                if p.get("role") == "FREE_TEXT_NARRATIVE":
                    idx = p.get("col_index", 0)
                    col_str = " ".join(sample.iloc[:, idx].astype(str))
                    if self.REGEX_ALPHANUM_MASK.search(col_str) or self.REGEX_TOKEN_SURROGATE.search(col_str):
                        bb.add_belief(c, "CRYPTOGRAPHIC_SURROGATE", 0.98, "Surrogate mask detected in text.")
                        bb.compliance_overrides.append(
                            f"Suppress Semantic NLP on Column `{c}`: Contains surrogate tokens. Disallow sentiment analysis or semantic extraction; enforce structural/mathematical processing only."
                        )


# ==============================================================================
# BRAIN 7: EXECUTIVE ORCHESTRATOR
# ==============================================================================
class Brain7ExecutiveOrchestrator(BaseCognitiveBrain):
    """Translates the Cognitive Blackboard into an authoritative prompt."""

    def execute(self, df_or_bb: Any, bb: Optional[CognitiveBlackboard] = None, sample_size: int = 0) -> str:
        target_bb = bb if bb is not None else (df_or_bb if isinstance(df_or_bb, CognitiveBlackboard) else None)
        if target_bb is None:
            return ""

        monologue = [
            f"Topological Cartography: Resolved matrix to {target_bb.shape[0]:,} rows x {target_bb.shape[1]} columns.",
            f"Boundary Cutoff: Header boundary identified at row index {target_bb.header_row_index}.",
        ]

        n_temporal = sum(1 for p in target_bb.column_profiles.values() if p.get("role") in ("TEMPORAL", "TEMPORAL_HIJRI"))
        n_numeric = sum(1 for p in target_bb.column_profiles.values() if "NUMERIC" in p.get("role", ""))
        n_cat = sum(1 for p in target_bb.column_profiles.values() if "CATEGORICAL" in p.get("role", ""))
        monologue.append(
            f"Morphological Entropy: Identified {n_temporal} temporal axes, {n_numeric} quantitative tensors, "
            f"and {n_cat} categorical dimensions."
        )

        if target_bb.candidate_primary_keys:
            monologue.append(f"Relational Cryptography: Confirmed candidate primary keys: {target_bb.candidate_primary_keys[:3]}.")
        elif target_bb.composite_primary_keys:
            monologue.append(f"Relational Cryptography: Identified composite primary key pair: {target_bb.composite_primary_keys[0]}.")

        if target_bb.hierarchical_dependencies:
            dep_strs = [f"`{b}` -> `{a}`" for a, b in target_bb.hierarchical_dependencies[:2]]
            monologue.append(f"Functional Hierarchy: Discovered functional dependencies: {', '.join(dep_strs)}.")

        if target_bb.algebraic_laws:
            monologue.append(f"Mathematical Physics: Discovered {target_bb.algebraic_laws[0]}.")

        if target_bb.type_contaminations:
            monologue.append(f"Forensic Pathology: Detected {len(target_bb.type_contaminations)} type/structural contaminations.")

        if target_bb.spatial_profiles:
            monologue.append("Spatial Cartography: Geospatial coordinate axes confirmed.")

        if target_bb.chronometric_profiles:
            monologue.append(f"Chronometric Analysis: Temporal delta {target_bb.chronometric_profiles.get('median_delta')}.")

        if target_bb.process_models:
            monologue.append("Process Modeling: Case-activity event log sequence topology confirmed.")

        if target_bb.tensor_profiles:
            monologue.append("Tensor Semantics: Multi-dimensional embedding array detected.")

        if target_bb.graph_topology:
            monologue.append("Network Topology: Relational entity edge linkages discovered.")

        target_bb.internal_monologue = monologue

        prompt_lines = [
            "### SYSTEM ROLE & OBJECTIVE",
            f"You are an expert Senior Data Engineer & ML Architect. Write a self-contained, deterministic Python (Pandas/Polars/NumPy) "
            f"pipeline to clean, flatten, and engineer features for `{target_bb.filename}`.",
            "\n---",
            "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)",
        ]
        for m in target_bb.internal_monologue:
            prompt_lines.append(f"* {m}")

        prompt_lines.extend([
            "\n---",
            "### 1. DATASET TOPOLOGY & BOUNDARIES",
            f"* **Source Dimensions**: {target_bb.shape[0]:,} rows x {target_bb.shape[1]} columns.",
            f"* **Header Cutoff**: Tabular headers begin at row index {target_bb.header_row_index}. Discard prior metadata rows."
        ])

        if target_bb.footer_start_index:
            prompt_lines.append(f"* **Summary Footers**: Footers/subtotals detected starting at row {target_bb.footer_start_index}. Prune prior to execution.")
        else:
            prompt_lines.append("* **Summary Footers**: No static trailing totals detected.")

        if target_bb.ragged_continuation_cols:
            prompt_lines.append(
                f"* **Ragged Continuations**: Columns {target_bb.ragged_continuation_cols} contain orphaned wrapped text. "
                f"Forward-fill empty structural anchors and concatenate these strings upward."
            )

        prompt_lines.extend([
            "\n---",
            "### 2. PATHOLOGY REPAIR PROTOCOLS",
        ])
        if target_bb.type_contaminations:
            for p in target_bb.type_contaminations:
                prompt_lines.append(f"* **Column `{p['col']}`**: {p['defect']} -> {p['action']}")
        else:
            prompt_lines.append("* No deep type contaminations detected. Standardize missing tokens.")

        if target_bb.algebraic_laws:
            prompt_lines.extend([
                "\n---",
                "### 3. MATHEMATICAL INVARIANTS",
                f"* **Algebraic Law**: {target_bb.algebraic_laws[0]}.",
                "* Enforce this invariant in local execution and create boolean flag `reconciliation_anomaly_flag` for any violating rows."
            ])

        prompt_lines.extend([
            "\n---",
            "### 4. ALGORITHMIC FEATURE ENGINEERING",
            "Based on statistical morphology, generate the following features:"
        ])
        if target_bb.engineered_features:
            for feat in target_bb.engineered_features:
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

        if target_bb.feature_directives:
            prompt_lines.extend([
                "\n---",
                "### 6. ADVANCED MULTI-MODAL DIRECTIVES",
            ])
            for d in target_bb.feature_directives:
                prompt_lines.append(f"* **{d['feature']}**: {d['logic']}")

        if target_bb.cryptographic_signatures or target_bb.compliance_overrides:
            prompt_lines.extend([
                "\n---",
                "### 7. COMPLIANCE & CRYPTOGRAPHIC CONSTRAINTS",
            ])
            for s in target_bb.cryptographic_signatures:
                prompt_lines.append(f"* **Surrogate Geometry**: {s}")
            for o in target_bb.compliance_overrides:
                prompt_lines.append(f"* **COMPLIANCE OVERRIDE**: {o}")

        target_bb.master_prompt = "\n".join(prompt_lines)
        return target_bb.master_prompt


# ==============================================================================
# BRAIN 15: SOCRATIC INQUIRER (THE "WHAT IF?" ENGINE)
# ==============================================================================
class Brain15SocraticInquirer(BaseCognitiveBrain):
    """The 'What If?' Engine: Formulates curious questions for statistical outliers and ambiguities."""

    def execute(self, df_or_bb: Any, bb: Optional[CognitiveBlackboard] = None, sample_size: int = 1500) -> None:
        target_bb = bb if bb is not None else (df_or_bb if isinstance(df_or_bb, CognitiveBlackboard) else None)
        if target_bb is None:
            return

        for anomaly in target_bb.anomalies:
            col = anomaly.get("col")
            defect = str(anomaly.get("defect", ""))
            action = str(anomaly.get("action", ""))
            if "Mixed" in defect or "Contaminated" in defect:
                target_bb.colleague_questions.append(
                    f"I noticed Column `{col}` is mostly numbers, but has some unexpected text scattered in. "
                    f"I'm going to isolate the text just in case it's an error code or special flag we need to review later."
                )
            elif "Composite" in defect:
                target_bb.colleague_questions.append(
                    f"Column `{col}` has composite/slashed values (e.g. 140/90 or 12GB/256GB). I've planned to split them up for easier arithmetic, "
                    f"but let me know if those represent something specific I should name differently!"
                )
            else:
                target_bb.colleague_questions.append(
                    f"I noticed an interesting pattern in Column `{col}`: {defect}. I went ahead and planned: {action}. Does that work for you, or should we handle it another way?"
                )

        for p in target_bb.type_contaminations:
            col = p.get("col")
            if not any(f"`{col}`" in q for q in target_bb.colleague_questions):
                target_bb.colleague_questions.append(
                    f"Hey, Column `{col}` has mixed text and numbers. I've set it up to parse cleanly without dropping any real records."
                )

        if target_bb.algebraic_laws:
            target_bb.colleague_questions.append(
                f"The calculations in this dataset seem to follow `{target_bb.algebraic_laws[0]}`. I've flagged any rows that don't match just in case they're custom discounts or special promos!"
            )


# ==============================================================================
# BRAIN 16: EMPATHETIC TRANSLATOR (THE "NO EGO" PEDAGOGY ENGINE)
# ==============================================================================
class Brain16EmpatheticTranslator(BaseCognitiveBrain):
    """The Pedagogy Engine: Calculates cognitive load, scores export friction, and enforces anti-jargon rules."""

    def execute(self, df_or_bb: Any, bb: Optional[CognitiveBlackboard] = None, sample_size: int = 1500) -> None:
        target_bb = bb if bb is not None else (df_or_bb if isinstance(df_or_bb, CognitiveBlackboard) else None)
        if target_bb is None:
            return

        friction_score = (
            len(target_bb.anomalies)
            + len(target_bb.type_contaminations)
            + len(target_bb.ragged_continuation_cols)
            + (2 if (target_bb.header_row_index and target_bb.header_row_index > 5) else 0)
            + (1 if target_bb.footer_start_index else 0)
        )
        target_bb.friction_score = friction_score

        if friction_score >= 3:
            h_row = target_bb.header_row_index or 0
            target_bb.persona_directives.append(
                f"EMPATHY TRIGGER: The user is dealing with a highly fragmented, painful data export (Friction Score: {friction_score}). "
                f"Acknowledge the messiness upfront with humble camaraderie: 'Man, these system exports are always such a headache—it looks like "
                f"the actual data doesn\\'t even start until row {h_row} because of all that header junk. I went ahead and sliced all that off for you.'"
            )

        target_bb.persona_directives.append(
            "ANTI-JARGON RULE: Never use academic data science terms (e.g., 'heteroscedasticity', 'tensor', 'imputation', 'homoscedasticity'). "
            "Brush off complex structural fixes casually, e.g., 'I did some quick cleanup to bundle those orphaned rows back together.'"
        )


# ==============================================================================
# BRAIN 17: INTUITIVE DETECTIVE (THE FUZZY LOGIC ENGINE)
# ==============================================================================
class Brain17IntuitiveDetective(BaseCognitiveBrain):
    """The Fuzzy Logic Engine: Infers human behavioral intent and business priority flags from free-text."""

    REGEX_BEHAVIORAL = re.compile(
        r"(?i)\b(asap|urgent|error|test|check|review|vip|priority|critical|pending|hold|fix|عاجل|مهم|فحص|مراجعة|خطأ|تنبيه)\b"
    )

    def execute(self, df_or_bb: Any, bb: Optional[CognitiveBlackboard] = None, sample_size: int = 1500) -> None:
        target_bb = bb if bb is not None else (df_or_bb if isinstance(df_or_bb, CognitiveBlackboard) else None)
        if target_bb is None:
            return

        df_inst = df_or_bb if isinstance(df_or_bb, pd.DataFrame) else None
        if df_inst is None:
            return

        start_row = _get_data_start_row(df_inst, target_bb)
        sample = df_inst.iloc[start_row:].head(sample_size)
        n_cols = sample.shape[1]

        for c in range(n_cols):
            col_key = target_bb.columns[c] if c < len(target_bb.columns) else c
            name_str = str(col_key).lower()
            role = target_bb.column_profiles.get(col_key, {}).get("role", "")
            dominant = target_bb.get_dominant_belief(col_key)

            is_text_candidate = (
                role in ("FREE_TEXT_NARRATIVE", "CATEGORICAL_DIMENSION", "COMPOSITE_KEY")
                or dominant in ("FREE_TEXT", "FREE_TEXT_NARRATIVE", "CATEGORICAL_DIMENSION")
                or any(k in name_str for k in ("note", "comment", "desc", "remark", "reason", "status", "flag", "text", "ملاحظ", "وصف"))
            )
            if not is_text_candidate:
                continue

            series = sample.iloc[:, c].dropna().astype(str)
            if series.empty:
                continue

            matches = int(series.apply(lambda x: bool(self.REGEX_BEHAVIORAL.search(x))).sum())
            if matches > 0:
                target_bb.detective_insights.append(
                    f"INTUITION TRIGGER: Column `{col_key}` contains human behavioral flags like 'urgent', 'asap', or 'review' ({matches} records). "
                    f"Casually point this out: 'By the way, I saw some urgent/review notes in column `{col_key}`. "
                    f"I went ahead and created a quick helper flag for those so you can pull them up easily.'"
                )


# ==============================================================================
# BRAIN 18: NARRATIVE WEAVER (THE STARTUP COLLEAGUE ORCHESTRATOR)
# ==============================================================================
class Brain18NarrativeWeaver(BaseCognitiveBrain):
    """The Master Orchestrator: Wraps data physics in the 'Humble Startup Colleague' persona."""

    def execute(self, df_or_bb: Any, bb: Optional[CognitiveBlackboard] = None, sample_size: int = 0) -> str:
        target_bb = bb if bb is not None else (df_or_bb if isinstance(df_or_bb, CognitiveBlackboard) else None)
        if target_bb is None:
            return ""

        # Populate internal monologue if not already done
        if not target_bb.internal_monologue:
            Brain7ExecutiveOrchestrator().execute(df_or_bb, target_bb)

        prompt_lines = [
            "### SYSTEM ROLE & OBJECTIVE",
            "### SYSTEM ROLE & PERSONA",
            f"You are a brilliant, highly-curious, but incredibly humble Senior Data Engineer at a fast-paced startup. "
            f"The user is your respected peer. Write a self-contained, deterministic Python (Pandas/Polars/NumPy) "
            f"pipeline to clean, flatten, and engineer features for `{target_bb.filename}`.",
            "Your communication style must follow these strict rules:",
            "1. **Tone**: Casual, warm, and collaborative. Use friendly greetings ('Hey!', 'Hi!'). Use conversational contractions (I've, we'll, let's).",
            "2. **Ego**: Zero ego. Never sound like a machine, a lecturer, or an AI. Never use words like 'Therefore', 'Thus', 'In conclusion', or 'As requested'.",
            "3. **Delivery**: Make the math sound effortless and helpful. Instead of 'I applied a forward-fill masking algorithm,' say 'I went ahead and smoothed out those missing values so the math works.'",
            "4. **Closing**: Always end your response with a collaborative, open-ended question asking if they need any tweaks.",
        ]

        if target_bb.persona_directives or target_bb.colleague_questions or target_bb.detective_insights:
            prompt_lines.extend([
                "\n---",
                "### BEHAVIORAL DIRECTIVES (MANDATORY INJECTIONS)",
            ])
            for rule in target_bb.persona_directives:
                prompt_lines.append(f"* {rule}")
            for q in target_bb.colleague_questions:
                prompt_lines.append(f"* MENTION THIS CASUALLY: {q}")
            for insight in target_bb.detective_insights:
                prompt_lines.append(f"* MENTION THIS CASUALLY: {insight}")

        prompt_lines.extend([
            "\n---",
            "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)",
        ])
        for m in target_bb.internal_monologue:
            prompt_lines.append(f"* {m}")

        prompt_lines.extend([
            "\n---",
            "### 1. DATASET TOPOLOGY & BOUNDARIES",
            f"* **Source Dimensions**: {target_bb.shape[0]:,} rows x {target_bb.shape[1]} columns.",
            f"* **Header Cutoff**: Tabular headers begin at row index {target_bb.header_row_index}. Discard prior metadata rows."
        ])

        if target_bb.footer_start_index:
            prompt_lines.append(f"* **Summary Footers**: Footers/subtotals detected starting at row {target_bb.footer_start_index}. Prune prior to execution.")
        else:
            prompt_lines.append("* **Summary Footers**: No static trailing totals detected.")

        if target_bb.ragged_continuation_cols:
            prompt_lines.append(
                f"* **Ragged Continuations**: Columns {target_bb.ragged_continuation_cols} contain orphaned wrapped text. "
                f"Forward-fill empty structural anchors and concatenate these strings upward."
            )

        prompt_lines.extend([
            "\n---",
            "### 2. PATHOLOGY REPAIR PROTOCOLS",
        ])
        if target_bb.type_contaminations:
            for p in target_bb.type_contaminations:
                prompt_lines.append(f"* **Column `{p['col']}`**: {p['defect']} -> {p['action']}")
        else:
            prompt_lines.append("* No deep type contaminations detected. Standardize missing tokens.")

        if target_bb.algebraic_laws:
            prompt_lines.extend([
                "\n---",
                "### 3. MATHEMATICAL INVARIANTS",
                f"* **Algebraic Law**: {target_bb.algebraic_laws[0]}.",
                "* Enforce this invariant in local execution and create boolean flag `reconciliation_anomaly_flag` for any violating rows."
            ])

        prompt_lines.extend([
            "\n---",
            "### 4. ALGORITHMIC FEATURE ENGINEERING",
            "Based on statistical morphology, generate the following features:"
        ])
        if target_bb.engineered_features:
            for feat in target_bb.engineered_features:
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

        if target_bb.feature_directives:
            prompt_lines.extend([
                "\n---",
                "### 6. ADVANCED MULTI-MODAL DIRECTIVES",
            ])
            for d in target_bb.feature_directives:
                prompt_lines.append(f"* **{d['feature']}**: {d['logic']}")

        if target_bb.cryptographic_signatures or target_bb.compliance_overrides:
            prompt_lines.extend([
                "\n---",
                "### 7. COMPLIANCE & CRYPTOGRAPHIC CONSTRAINTS",
            ])
            for s in target_bb.cryptographic_signatures:
                prompt_lines.append(f"* **Surrogate Geometry**: {s}")
            for o in target_bb.compliance_overrides:
                prompt_lines.append(f"* **COMPLIANCE OVERRIDE**: {o}")

        target_bb.master_prompt = "\n".join(prompt_lines)
        return target_bb.master_prompt


# ==============================================================================
# OUROBOROS LOOP: CRASH AUTOPSY & REPAIR PROMPT
# ==============================================================================
def autopsy_traceback(
    error_traceback: str,
    bb: CognitiveBlackboard,
    df: Optional[pd.DataFrame] = None
) -> str:
    """Ouroboros Self-Healing Autopsy: Analyzes pipeline traceback and generates a surgical repair prompt."""
    tb_str = str(error_traceback).strip()
    bb.ouroboros_traceback = tb_str

    err_lines = [line.strip() for line in tb_str.split("\n") if line.strip()]
    last_line = err_lines[-1] if err_lines else "UnknownError"
    error_type = last_line.split(":")[0].strip()
    error_msg = last_line[len(error_type) + 1:].strip() if ":" in last_line else last_line

    offending_col = None
    root_cause = "Pipeline crashed during airlock execution."
    patch_constraint = "Fix exception and ensure safe execution."

    key_match = re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", tb_str)
    if key_match:
        missing_key = key_match.group(1)
        offending_col = missing_key
        actual_cols = [str(c) for c in (bb.columns if bb.columns else (df.columns if df is not None else []))]
        close_matches = [c for c in actual_cols if missing_key.lower() in c.lower() or c.lower() in missing_key.lower()]
        hint = f"Actual available columns: {close_matches[:4]}" if close_matches else f"Available columns: {actual_cols[:8]}"
        root_cause = f"Column `{missing_key}` not found in DataFrame. {hint}"
        chosen_col = close_matches[0] if close_matches else (actual_cols[0] if actual_cols else "col")
        patch_constraint = f"Use existing column from DataFrame headers (e.g. `{chosen_col}`) or verify header promotion index (Header Row: {bb.header_row_index})."

    elif "TypeError" in error_type and ("convert" in error_msg or "float" in error_msg or "numeric" in error_msg):
        root_cause = f"Type coercion failure: {error_msg}. Non-numeric dirty string encountered during arithmetic."
        patch_constraint = "Apply `pd.to_numeric(series.astype(str).str.replace(r'[^\\d.-]', '', regex=True), errors='coerce').fillna(0)` before math operations."

    elif "ZeroDivisionError" in error_type:
        root_cause = "Zero division encountered during ratio or rate calculation."
        patch_constraint = "Stabilize denominators with epsilon: `df['denominator'].replace(0, np.nan)` or `+ 1e-9`."

    repair_prompt = f"""### OUROBOROS AUTONOMOUS CRASH AUTOPSY & SURGICAL REPAIR PROMPT
An unhandled exception occurred during airlock sandbox execution for `{bb.filename}`:
* **Error Type**: `{error_type}`
* **Error Message**: `{error_msg}`
* **Offending Entity**: `{offending_col or 'Data Pipeline Expression'}`
* **Forensic Root Cause**: {root_cause}
* **Algorithmic Patch Constraint**: {patch_constraint}

### SYSTEM DIRECTIVE
You are an expert Senior Data Engineer. Rewrite the complete, corrected Python cleaning script fixing this exact error.
Ensure the output is 100% executable within the AST security firewall (no network, no os.environ)."""

    bb.ouroboros_repair_prompt = repair_prompt
    return repair_prompt


# ==============================================================================
# MASTER ORCHESTRATOR: OMNI-MODAL RESONANCE ENGINE (18-BRAIN ARCHITECTURE)
# ==============================================================================
class OmniModalResonanceEngine:
    """Master Orchestrator triggering the 18-Brain Omni-Cognitive Resonance Hive Mind."""

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

        # Instantiate all 18 cognitive sub-engines (Left & Right Hemispheres)
        self.brain1 = Brain1TopologicalCartographer()
        self.brain2 = Brain2MorphologicalTypologist()
        self.brain3 = Brain3ForensicPathologist()
        self.brain4 = Brain4RelationalCryptographer()
        self.brain5 = Brain5MathematicalPhysicist()
        self.brain6 = Brain6AutonomousFeatureAlchemist()
        self.brain8 = Brain8SpatialCartographer()
        self.brain9 = Brain9ChronometricSignalProcessor()
        self.brain10 = Brain10ProcessStateModeler()
        self.brain11 = Brain11TensorSemanticist()
        self.brain12 = Brain12GraphNetworkTopologist()
        self.brain13 = Brain13StatutoryArbiter()
        self.brain14 = Brain14CryptographicSentinel()
        self.brain15 = Brain15SocraticInquirer()
        self.brain16 = Brain16EmpatheticTranslator()
        self.brain17 = Brain17IntuitiveDetective()
        self.brain18 = Brain18NarrativeWeaver()
        self.brain7 = Brain7ExecutiveOrchestrator()
        self.executive = self.brain18

    @property
    def left_brains(self) -> List[Any]:
        """Returns the 14 Left Brains (data physics, multi-modal, statutory & cryptographic)."""
        return [
            self.brain1, self.brain2, self.brain3, self.brain4,
            self.brain5, self.brain6, self.brain8, self.brain9,
            self.brain10, self.brain11, self.brain12, self.brain13,
            self.brain14, self.brain7
        ]

    @property
    def right_brains(self) -> List[Any]:
        """Returns the 4 Right Brains (EQ, empathy, intuition, and persona narrative)."""
        return [self.brain15, self.brain16, self.brain17, self.brain18]

    @property
    def brains(self) -> List[Any]:
        """Returns the complete 18-Brain council in execution order."""
        return [
            self.brain1, self.brain2, self.brain3, self.brain4,
            self.brain5, self.brain6, self.brain7, self.brain8,
            self.brain9, self.brain10, self.brain11, self.brain12,
            self.brain13, self.brain14, self.brain15, self.brain16,
            self.brain17, self.brain18
        ]

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
        """Executes the synchronous 18-Brain cognitive loop."""
        # 1. Left Hemisphere (Math & Physics)
        self.brain1.execute(self.df_raw, self.bb)
        self.brain2.execute(self.df_raw, self.bb)
        self.brain3.execute(self.df_raw, self.bb)
        self.brain4.execute(self.df_raw, self.bb)
        self.brain5.execute(self.df_raw, self.bb)
        self.brain6.execute(self.df_raw, self.bb)
        self.brain8.execute(self.df_raw, self.bb)
        self.brain9.execute(self.df_raw, self.bb)
        self.brain10.execute(self.df_raw, self.bb)
        self.brain11.execute(self.df_raw, self.bb)
        self.brain12.execute(self.df_raw, self.bb)
        self.brain13.execute(self.df_raw, self.bb)
        self.brain14.execute(self.df_raw, self.bb)
        # 2. Right Hemisphere (EQ, Socratic & Intuitive Persona)
        self.brain15.execute(self.df_raw, self.bb)
        self.brain16.execute(self.df_raw, self.bb)
        self.brain17.execute(self.df_raw, self.bb)
        # 3. Master Synthesis
        return self.executive.execute(self.df_raw, self.bb)


# Backward compatibility alias
DynamicResonanceEngine = OmniModalResonanceEngine


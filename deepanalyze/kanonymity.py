"""DeepAnalyze v4.0 k-Anonymity & l-Diversity Re-Identification Risk Engine.

Analyzes datasets for Quasi-Identifiers (QIs) that, when combined, can uniquely
re-identify individuals even when direct identifiers (names, IDs) are masked.
Computes equivalence classes, minimum k-anonymity, l-diversity on sensitive columns,
and provides actionable generalization recommendations.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import polars as pl
try:
    import pandas as pd
except ImportError:
    pd = None


COMMON_QI_PATTERNS = [
    re.compile(r".*(age|years?_old).*", re.I),
    re.compile(r"^(gender|sex)$", re.I),
    re.compile(r".*(birth|dob|born).*", re.I),
    re.compile(r".*(zip|postal|postcode).*", re.I),
    re.compile(r".*(city|town|municipality|state|province|county|region|country).*", re.I),
    re.compile(r".*(ethnicity|race|nationality).*", re.I),
    re.compile(r".*(marital|marriage|spouse).*", re.I),
    re.compile(r".*(visit_date|admission|discharge|registration_date).*", re.I),
    re.compile(r".*(department|division|job_title|occupation|rank).*", re.I),
]

COMMON_SENSITIVE_PATTERNS = [
    re.compile(r".*(condition|diagnosis|disease|illness|syndrome|pathology).*", re.I),
    re.compile(r".*(medication|prescription|drug|dosage).*", re.I),
    re.compile(r".*(salary|income|wage|compensation|bonus|net_worth).*", re.I),
    re.compile(r".*(credit_score|debt|balance|loan).*", re.I),
    re.compile(r".*(religion|political|union).*", re.I),
]


@dataclass
class KAnonymityReport:
    """Detailed summary of k-Anonymity and l-Diversity re-identification risk."""
    quasi_identifiers: List[str]
    total_records: int
    equivalence_classes_count: int
    min_k: int
    avg_k: float
    median_k: float
    records_at_risk: int  # count of rows with class size < 3
    risk_percentage: float
    risk_level: str  # "LOW", "MODERATE", "CRITICAL"
    sensitive_column: Optional[str] = None
    min_l_diversity: Optional[int] = None
    sample_outlier_classes: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def detect_quasi_identifiers(df: Union[pl.DataFrame, Any]) -> List[str]:
    """Automatically detects columns that act as Quasi-Identifiers."""
    columns = df.columns if hasattr(df, "columns") else []
    qis = []
    for col in columns:
        col_str = str(col).strip()
        for pat in COMMON_QI_PATTERNS:
            if pat.match(col_str):
                qis.append(col_str)
                break
    return qis


def detect_sensitive_column(df: Union[pl.DataFrame, Any]) -> Optional[str]:
    """Detects primary sensitive attribute column (e.g. Medical Condition, Salary)."""
    columns = df.columns if hasattr(df, "columns") else []
    for col in columns:
        col_str = str(col).strip()
        for pat in COMMON_SENSITIVE_PATTERNS:
            if pat.match(col_str):
                return col_str
    return None


def analyze_kanonymity(
    df: Union[pl.DataFrame, Any],
    quasi_identifiers: Optional[Sequence[str]] = None,
    sensitive_col: Optional[str] = None,
    threshold_k: int = 3
) -> KAnonymityReport:
    """Computes k-Anonymity equivalence classes and l-Diversity for a DataFrame."""
    if hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame):
        try:
            pl_df = pl.from_pandas(df)
        except Exception:
            pl_df = pl.DataFrame(df)
    else:
        pl_df = df

    total_records = len(pl_df)
    if total_records == 0:
        return KAnonymityReport(
            quasi_identifiers=[],
            total_records=0,
            equivalence_classes_count=0,
            min_k=0,
            avg_k=0.0,
            median_k=0.0,
            records_at_risk=0,
            risk_percentage=0.0,
            risk_level="LOW",
            recommendations=["Dataset is empty."]
        )

    # Resolve QIs
    valid_cols = set(pl_df.columns)
    if quasi_identifiers:
        qis = [c for c in quasi_identifiers if c in valid_cols]
    else:
        qis = detect_quasi_identifiers(pl_df)

    # Fallback: if no recognized QIs, take low/medium cardinality columns
    if not qis:
        candidate_qis = []
        for c in pl_df.columns:
            try:
                n_unique = pl_df[c].n_unique()
                if 2 <= n_unique <= min(100, max(2, total_records // 2)):
                    candidate_qis.append(c)
            except Exception:
                pass
        qis = candidate_qis[:3]

    if not qis:
        return KAnonymityReport(
            quasi_identifiers=[],
            total_records=total_records,
            equivalence_classes_count=1,
            min_k=total_records,
            avg_k=float(total_records),
            median_k=float(total_records),
            records_at_risk=0,
            risk_percentage=0.0,
            risk_level="LOW",
            recommendations=["No quasi-identifier combinations detected."]
        )

    # Resolve sensitive column
    if not sensitive_col:
        sensitive_col = detect_sensitive_column(pl_df)
        if sensitive_col in qis:
            sensitive_col = None

    # Group by QIs and count class sizes
    try:
        # Cast all QIs to string representation to avoid type mismatch during aggregation
        exprs = [pl.col(c).cast(pl.Utf8).fill_null("<NULL>") for c in qis]
        grouped = pl_df.with_columns(exprs).group_by(qis).len()
        class_sizes = grouped["len"].to_list()
    except Exception:
        # Fallback using python hashing
        counts: Dict[Tuple, int] = {}
        rows = pl_df.select([pl.col(c).cast(pl.Utf8) for c in qis]).to_dicts()
        for r in rows:
            key = tuple(str(r[c]) for c in qis)
            counts[key] = counts.get(key, 0) + 1
        class_sizes = list(counts.values())

    class_sizes.sort()
    eq_count = len(class_sizes)
    min_k = class_sizes[0] if class_sizes else total_records
    avg_k = sum(class_sizes) / max(eq_count, 1)
    median_k = class_sizes[eq_count // 2] if eq_count > 0 else 0

    records_at_risk = sum(cnt for cnt in class_sizes if cnt < threshold_k)
    risk_percentage = round((records_at_risk / total_records) * 100.0, 2)

    if risk_percentage > 25.0 or min_k == 1:
        risk_level = "CRITICAL"
    elif risk_percentage > 5.0 or min_k < threshold_k:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Compute l-Diversity if sensitive column available
    min_l_diversity = None
    if sensitive_col and sensitive_col in valid_cols:
        try:
            div_grouped = (
                pl_df.select(qis + [sensitive_col])
                .with_columns([pl.col(c).cast(pl.Utf8).fill_null("<NULL>") for c in qis + [sensitive_col]])
                .group_by(qis)
                .agg(pl.col(sensitive_col).n_unique().alias("l_div"))
            )
            min_l_diversity = int(div_grouped["l_div"].min())
        except Exception:
            min_l_diversity = None

    # Sample outlier classes
    sample_outliers = []
    if min_k < threshold_k:
        try:
            low_k_keys = grouped.filter(pl.col("len") < threshold_k).head(3).to_dicts()
            for row in low_k_keys:
                sample_outliers.append(row)
        except Exception:
            pass

    recommendations = []
    if min_k < threshold_k:
        recommendations.append(
            f"Found {records_at_risk} record(s) ({risk_percentage}%) with k < {threshold_k}. "
            "These rows possess unique QI combinations vulnerable to re-identification."
        )
        for qi in qis:
            if "age" in qi.lower():
                recommendations.append(f"Consider generalizing '{qi}' into 10-year bands (e.g. 20-29, 30-39).")
            elif "date" in qi.lower():
                recommendations.append(f"Consider coarsening '{qi}' to Month/Year (YYYY-MM) instead of exact day.")
            elif any(k in qi.lower() for k in ["zip", "postal"]):
                recommendations.append(f"Consider truncating '{qi}' to 3 digits (e.g. 100XX).")

    if min_l_diversity is not None and min_l_diversity < 2:
        recommendations.append(
            f"Sensitive attribute '{sensitive_col}' exhibits l-diversity = {min_l_diversity}. "
            "Homogeneity attack vulnerability detected in small equivalence classes."
        )

    if not recommendations:
        recommendations.append(f"k-Anonymity is strong: Minimum group size k={min_k} across all combinations.")

    return KAnonymityReport(
        quasi_identifiers=qis,
        total_records=total_records,
        equivalence_classes_count=eq_count,
        min_k=min_k,
        avg_k=round(avg_k, 2),
        median_k=round(median_k, 2),
        records_at_risk=records_at_risk,
        risk_percentage=risk_percentage,
        risk_level=risk_level,
        sensitive_column=sensitive_col,
        min_l_diversity=min_l_diversity,
        sample_outlier_classes=sample_outliers,
        recommendations=recommendations
    )


def bin_column_series(series: Sequence[Any], bin_size: int = 10) -> List[str]:
    """Helper to bin numeric values (like age) into standard ranges."""
    binned = []
    for val in series:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            binned.append("<UNKNOWN>")
            continue
        try:
            num = float(val)
            lower = int(num // bin_size) * bin_size
            upper = lower + bin_size - 1
            binned.append(f"{lower}-{upper}")
        except (ValueError, TypeError):
            binned.append(str(val))
    return binned

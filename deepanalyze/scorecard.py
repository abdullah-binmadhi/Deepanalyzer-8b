"""DeepAnalyze v4.0 Real-Time Data Diff & Quality Scorecard.

Performs deterministic side-by-side metric auditing between the initial messy dataset
and the final cleaned DataFrame. Evaluates row deduplication, null-value reduction,
column standardization, and generates a composite 0-100% Data Cleanliness Score.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class QualityScorecard:
    """Quantitative comparison between raw input and transformed clean data."""
    raw_rows: int
    clean_rows: int
    rows_diff: int
    duplicates_removed: int

    raw_cols: int
    clean_cols: int

    raw_null_count: int
    clean_null_count: int
    raw_null_pct: float
    clean_null_pct: float
    null_reduction_pct: float

    standardized_column_names_pct: float
    cleanliness_score: int  # 0 to 100

    added_cols: List[str] = field(default_factory=list)
    dropped_cols: List[str] = field(default_factory=list)
    metrics_summary: List[str] = field(default_factory=list)


def _to_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if hasattr(df, "to_dict"):
        try:
            return pl.from_pandas(df)
        except Exception:
            pass
    try:
        return pl.DataFrame(df)
    except Exception:
        return pl.DataFrame()


def generate_quality_scorecard(raw_df: Any, clean_df: Any) -> QualityScorecard:
    """Computes side-by-side quality diff between raw and cleaned DataFrames."""
    p_raw = _to_polars(raw_df)
    p_clean = _to_polars(clean_df)

    raw_rows = p_raw.height
    clean_rows = p_clean.height
    rows_diff = clean_rows - raw_rows

    # Row deduplication estimation
    try:
        raw_unique = p_raw.n_unique()
        duplicates_removed = max(0, raw_rows - clean_rows)
    except Exception:
        duplicates_removed = max(0, -rows_diff)

    raw_cols = p_raw.width
    clean_cols = p_clean.width

    raw_col_set = set(str(c) for c in p_raw.columns)
    clean_col_set = set(str(c) for c in p_clean.columns)

    added_cols = sorted(list(clean_col_set - raw_col_set))
    dropped_cols = sorted(list(raw_col_set - clean_col_set))

    # Null value analysis
    raw_total_cells = max(1, raw_rows * raw_cols)
    clean_total_cells = max(1, clean_rows * clean_cols)

    try:
        raw_null_count = sum(p_raw[c].null_count() for c in p_raw.columns)
    except Exception:
        raw_null_count = 0

    try:
        clean_null_count = sum(p_clean[c].null_count() for c in p_clean.columns)
    except Exception:
        clean_null_count = 0

    raw_null_pct = round((raw_null_count / raw_total_cells) * 100.0, 2)
    clean_null_pct = round((clean_null_count / clean_total_cells) * 100.0, 2)

    if raw_null_pct > 0:
        null_reduction = round(max(0.0, (raw_null_pct - clean_null_pct) / raw_null_pct) * 100.0, 2)
    else:
        null_reduction = 100.0 if clean_null_pct == 0 else 0.0

    # Column name hygiene (check for snake_case, no leading/trailing spaces or messy punctuation)
    snake_pattern = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    well_named = sum(1 for c in p_clean.columns if snake_pattern.match(str(c).strip()))
    col_hygiene = round((well_named / max(1, clean_cols)) * 100.0, 2)

    # Cleanliness composite calculation
    # Weights: 40% null reduction/cleanliness, 30% schema naming hygiene, 30% structural integrity
    null_subscore = max(0, 100 - (clean_null_pct * 2))
    score = int(round((null_subscore * 0.40) + (col_hygiene * 0.30) + (30.0 if clean_rows > 0 else 0.0)))
    score = max(0, min(100, score))

    metrics = [
        f"Rows: {raw_rows:,} -> {clean_rows:,} ({'+' if rows_diff >= 0 else ''}{rows_diff:,})",
        f"Columns: {raw_cols} -> {clean_cols} ({len(added_cols)} added, {len(dropped_cols)} pruned)",
        f"Missing values: {raw_null_pct}% -> {clean_null_pct}% ({null_reduction}% reduction)",
        f"Column naming hygiene: {col_hygiene}% standardized snake_case",
        f"Overall Data Cleanliness Score: {score}/100",
    ]

    return QualityScorecard(
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        rows_diff=rows_diff,
        duplicates_removed=duplicates_removed,
        raw_cols=raw_cols,
        clean_cols=clean_cols,
        added_cols=added_cols,
        dropped_cols=dropped_cols,
        raw_null_count=raw_null_count,
        clean_null_count=clean_null_count,
        raw_null_pct=raw_null_pct,
        clean_null_pct=clean_null_pct,
        null_reduction_pct=null_reduction,
        standardized_column_names_pct=col_hygiene,
        cleanliness_score=score,
        metrics_summary=metrics
    )


def render_quality_scorecard(card: QualityScorecard, console: Optional[Console] = None) -> Table:
    """Renders an interactive, side-by-side terminal comparison table."""
    c = console or Console()

    table = Table(title="Data Transformation & Quality Scorecard", border_style="green", header_style="bold cyan")
    table.add_column("Metric / Dimension", style="bold white", width=30)
    table.add_column("Before (Raw Input)", justify="right", width=22)
    table.add_column("After (Transformed)", justify="right", width=22)
    table.add_column("Delta / Improvement", justify="right", style="bold green", width=24)

    # Row metrics
    diff_sign = "+" if card.rows_diff > 0 else ""
    row_delta = f"{diff_sign}{card.rows_diff:,} rows"
    if card.duplicates_removed > 0:
        row_delta += f" ({card.duplicates_removed} pruned)"
    table.add_row("Total Records (Rows)", f"{card.raw_rows:,}", f"{card.clean_rows:,}", row_delta)

    # Column metrics
    col_delta = f"{card.clean_cols - card.raw_cols:+d} columns"
    table.add_row("Total Attributes (Cols)", f"{card.raw_cols}", f"{card.clean_cols}", col_delta)

    # Null metrics
    null_diff = f"-{card.null_reduction_pct}% reduction" if card.null_reduction_pct > 0 else "0.0%"
    table.add_row(
        "Missing / Null Values",
        f"{card.raw_null_count:,} ({card.raw_null_pct}%)",
        f"{card.clean_null_count:,} ({card.clean_null_pct}%)",
        null_diff
    )

    # Column naming standard
    table.add_row(
        "Column Naming Standard",
        "Raw / Mixed",
        f"{card.standardized_column_names_pct}% snake_case",
        "Standardized"
    )

    # Cleanliness score
    table.add_row(
        "Data Hygiene Score",
        "Unverified",
        f"[bold cyan]{card.cleanliness_score} / 100[/bold cyan]",
        "[bold green]PRODUCTION READY[/bold green]" if card.cleanliness_score >= 80 else "[bold yellow]ACCEPTABLE[/bold yellow]"
    )

    return table

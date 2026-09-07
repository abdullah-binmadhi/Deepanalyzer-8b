"""DeepAnalyze Dynamic Charting & Visual Analytics Engine.

Provides high-precision terminal Unicode/ASCII visualizations and standalone
Plotly/Matplotlib Python script generators for custom exploratory data analysis.
Operates strictly in volatile RAM on Polars and Pandas DataFrames.
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

try:
    import pandas as pd
except ImportError:
    pd = None


def _to_polars(df: Any) -> pl.DataFrame:
    """Safely converts any input DataFrame to Polars."""
    if hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame):
        try:
            return pl.from_pandas(df)
        except Exception:
            return pl.DataFrame(df)
    return df if isinstance(df, pl.DataFrame) else pl.DataFrame()


def detect_dimensions_and_measures(df: Any) -> Tuple[List[str], List[str]]:
    """Separates columns into Dimensions (Categorical/Date/ID) and Measures (Numerical)."""
    pl_df = _to_polars(df)
    dimensions: List[str] = []
    measures: List[str] = []

    for col in pl_df.columns:
        dtype_str = str(pl_df.schema[col]).lower()
        if any(t in dtype_str for t in ("int", "float", "decimal")):
            # If it is clearly an ID or key column, it acts as a dimension
            if re.search(r"(_?id|_?key|_?num|_?no|_?code|uuid|guid)$", str(col).strip(), re.I):
                dimensions.append(col)
            else:
                measures.append(col)
        else:
            dimensions.append(col)

    return dimensions, measures


# =============================================================================
# 1. HISTOGRAM & DENSITY CURVE
# =============================================================================

def render_histogram(
    df: Any,
    column: str,
    bins: int = 8,
    stat: str = "COUNT"
) -> Tuple[str, Dict[str, Any]]:
    """Computes distribution bins and renders a vertical/horizontal Unicode histogram."""
    pl_df = _to_polars(df)
    if column not in pl_df.columns:
        return f"Column '{column}' not found in dataset.", {}

    series = pl_df[column].drop_nulls()
    # Try numeric conversion
    try:
        vals = [float(v) for v in series.to_list() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    except (ValueError, TypeError):
        vals = []

    if not vals:
        return f"Column '{column}' contains no numeric data for histogram calculation.", {}

    arr = np.array(vals)
    n = len(arr)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr)) if n > 1 else 0.0

    # Skewness
    if std_val > 1e-6 and n > 2:
        skew_val = float(np.mean(((arr - mean_val) / std_val) ** 3))
    else:
        skew_val = 0.0

    counts, bin_edges = np.histogram(arr, bins=max(3, min(bins, 25)))
    max_count = max(counts) if len(counts) > 0 else 1

    lines = [
        f"DISTRIBUTION HISTOGRAM: [{column.upper()}] (N = {n:,})",
        "─" * 65
    ]

    bar_width = 30

    for i in range(len(counts)):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        cnt = counts[i]
        pct = (cnt / max(n, 1)) * 100.0
        fraction = cnt / max(max_count, 1)
        full_blocks = int(fraction * bar_width)
        bar_str = "█" * full_blocks
        lines.append(f" {low:>8.2f} - {high:<8.2f} │ {bar_str:<{bar_width}} │ {cnt:>5} ({pct:>5.1f}%)")

    lines.append("─" * 65)
    lines.append(
        f"STATISTICS: Min={min_val:.2f} │ Mean={mean_val:.2f} │ Median={median_val:.2f} │ "
        f"Max={max_val:.2f} │ Std={std_val:.2f} │ Skew={skew_val:+.2f}"
    )

    stats = {
        "n": n,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "skewness": skew_val,
        "bins": len(counts),
    }

    return "\n".join(lines), stats


# =============================================================================
# 2. BOX PLOT (TUKEY 5-NUMBER SUMMARY)
# =============================================================================

def render_box_plot(
    df: Any,
    column: str,
    group_by: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Calculates Tukey 5-number summary and renders ASCII/Unicode box-and-whisker plots."""
    pl_df = _to_polars(df)
    if column not in pl_df.columns:
        return f"Column '{column}' not found.", {}

    def _calc_stats(s_vals: List[float]) -> Dict[str, Any]:
        if not s_vals:
            return {}
        a = np.array(s_vals)
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        iqr = q3 - q1
        lower_whisker = float(np.min(a[a >= q1 - 1.5 * iqr])) if any(a >= q1 - 1.5 * iqr) else float(np.min(a))
        upper_whisker = float(np.max(a[a <= q3 + 1.5 * iqr])) if any(a <= q3 + 1.5 * iqr) else float(np.max(a))
        outliers = [float(x) for x in a if x < lower_whisker or x > upper_whisker]
        return {
            "min": float(np.min(a)),
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "max": float(np.max(a)),
            "iqr": float(iqr),
            "lower_whisker": lower_whisker,
            "upper_whisker": upper_whisker,
            "outlier_count": len(outliers),
            "outliers": outliers[:5],
            "n": len(a)
        }

    groups: Dict[str, List[float]] = {}
    if group_by and group_by in pl_df.columns:
        for r in pl_df.select([group_by, column]).drop_nulls().to_dicts():
            grp = str(r[group_by])
            try:
                val = float(r[column])
                groups.setdefault(grp, []).append(val)
            except (ValueError, TypeError):
                pass
    else:
        series = pl_df[column].drop_nulls().to_list()
        numeric = []
        for v in series:
            try:
                numeric.append(float(v))
            except (ValueError, TypeError):
                pass
        groups[column] = numeric

    if not groups or all(len(v) == 0 for v in groups.values()):
        return f"No numeric data available for box plot on '{column}'.", {}

    # Find overall min and max for scale
    all_vals = [x for vals in groups.values() for x in vals]
    global_min = min(all_vals)
    global_max = max(all_vals)
    span = max(global_max - global_min, 1e-6)
    scale_width = 45

    def _pos(v: float) -> int:
        return int(((v - global_min) / span) * (scale_width - 1))

    lines = [
        f"BOX & WHISKER PLOT: [{column.upper()}] (SCALE: {global_min:.2f} TO {global_max:.2f})",
        "─" * 70
    ]

    all_stats: Dict[str, Any] = {}

    for grp_name, g_vals in list(groups.items())[:6]:
        st = _calc_stats(g_vals)
        if not st:
            continue
        all_stats[grp_name] = st

        p_low = _pos(st["lower_whisker"])
        p_q1 = _pos(st["q1"])
        p_med = _pos(st["median"])
        p_q3 = _pos(st["q3"])
        p_high = _pos(st["upper_whisker"])

        # Construct visual box line
        buf = [" "] * scale_width
        # Whisker left
        for j in range(p_low, p_q1):
            buf[j] = "─"
        buf[p_low] = "├"

        # Box
        for j in range(p_q1, p_q3 + 1):
            buf[j] = "█"
        buf[p_med] = "│"

        # Whisker right
        for j in range(p_q3 + 1, p_high + 1):
            buf[j] = "─"
        buf[p_high] = "┤"

        # Outliers
        for out in st["outliers"]:
            p_out = _pos(out)
            if 0 <= p_out < scale_width:
                buf[p_out] = "•"

        box_str = "".join(buf)
        label = (grp_name[:16] + "..") if len(grp_name) > 18 else grp_name
        lines.append(f"{label:<18} │{box_str}│ N={st['n']}")
        lines.append(f"                   │ Q1={st['q1']:.1f}  Med={st['median']:.1f}  Q3={st['q3']:.1f}  IQR={st['iqr']:.1f}  Outliers={st['outlier_count']}")

    lines.append("─" * 70)
    return "\n".join(lines), all_stats


# =============================================================================
# 3. CORRELATION & MUTUAL INFORMATION HEATMAP
# =============================================================================

def render_correlation_heatmap(
    df: Any,
    columns: Optional[List[str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """Calculates pairwise correlation matrix and renders a terminal matrix heatmap."""
    pl_df = _to_polars(df)
    cols = columns or []
    if not cols:
        _, measures = detect_dimensions_and_measures(pl_df)
        cols = measures[:6]

    if len(cols) < 2:
        return "At least 2 numeric columns are required for correlation analysis.", {}

    # Extract clean numeric matrices
    data_dict = {}
    for c in cols:
        try:
            data_dict[c] = np.array([float(x) if x is not None else 0.0 for x in pl_df[c].to_list()])
        except Exception:
            pass

    valid_cols = list(data_dict.keys())
    if len(valid_cols) < 2:
        return "Insufficient numeric attributes for correlation matrix.", {}

    matrix = []
    for c1 in valid_cols:
        row = []
        for c2 in valid_cols:
            v1 = data_dict[c1]
            v2 = data_dict[c2]
            std1 = np.std(v1)
            std2 = np.std(v2)
            if std1 > 1e-6 and std2 > 1e-6:
                r = float(np.corrcoef(v1, v2)[0, 1])
                row.append(round(r, 2))
            else:
                row.append(1.0 if c1 == c2 else 0.0)
        matrix.append(row)

    lines = [
        "PAIRWISE CORRELATION HEATMAP (PEARSON COEFFICIENTS)",
        "─" * 65
    ]

    # Header row
    hdr = " " * 16 + "".join(f"{c[:8]:>9}" for c in valid_cols)
    lines.append(hdr)
    lines.append(" " * 16 + "─" * (len(valid_cols) * 9))

    for idx, c1 in enumerate(valid_cols):
        row_str = f"{c1[:14]:<15} │"
        for val in matrix[idx]:
            # Density block based on magnitude
            abs_val = abs(val)
            if abs_val >= 0.8:
                block = "██"
            elif abs_val >= 0.5:
                block = "▓▓"
            elif abs_val >= 0.2:
                block = "▒▒"
            else:
                block = "░░"
            row_str += f" {block}{val:>5.2f}"
        lines.append(row_str)

    lines.append("─" * 65)
    lines.append("DENSITY LEGEND: ██ Strong (|r|>=0.8) │ ▓▓ Moderate (>=0.5) │ ▒▒ Mild (>=0.2) │ ░░ Weak (<0.2)")

    return "\n".join(lines), {"columns": valid_cols, "matrix": matrix}


# =============================================================================
# 4. CATEGORICAL & PARETO BAR CHART
# =============================================================================

def render_categorical_bar(
    df: Any,
    category_col: str,
    measure_col: Optional[str] = None,
    agg: str = "COUNT",
    top_n: int = 8,
    measure: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Calculates categorical aggregation and renders horizontal bars with Pareto analysis."""
    pl_df = _to_polars(df)
    if category_col not in pl_df.columns:
        return f"Category column '{category_col}' not found.", {}

    target_measure = measure_col or measure
    if target_measure and target_measure in pl_df.columns:
        if agg.upper() == "SUM":
            grouped = pl_df.group_by(category_col).agg(pl.col(target_measure).cast(pl.Float64).sum().alias("val"))
        elif agg.upper() in ("AVG", "MEAN"):
            grouped = pl_df.group_by(category_col).agg(pl.col(target_measure).cast(pl.Float64).mean().alias("val"))
        else:
            grouped = pl_df.group_by(category_col).agg(pl.len().alias("val"))
    else:
        grouped = pl_df.group_by(category_col).agg(pl.len().alias("val"))

    sorted_df = grouped.sort("val", descending=True).head(top_n)
    labels = [str(x) for x in sorted_df[category_col].to_list()]
    vals = [float(x) if x is not None else 0.0 for x in sorted_df["val"].to_list()]

    total_sum = sum(vals) if sum(vals) > 0 else 1.0
    max_val = max(vals) if vals else 1.0
    bar_width = 32

    lines = [
        f"CATEGORICAL BREAKDOWN: [{category_col.upper()}] (AGG: {agg.upper()})",
        "─" * 65
    ]

    cum_pct = 0.0
    for lbl, v in zip(labels, vals):
        pct = (v / total_sum) * 100.0
        cum_pct += pct
        frac = v / max_val
        full = int(frac * bar_width)
        bar = "█" * full
        lbl_clean = (lbl[:15] + "..") if len(lbl) > 17 else lbl
        lines.append(f"{lbl_clean:<17} │ {bar:<{bar_width}} │ {v:>8.1f} ({pct:>5.1f}% │ Cum {cum_pct:>5.1f}%)")

    lines.append("─" * 65)
    return "\n".join(lines), {"categories": labels, "values": vals, "total": total_sum}


# =============================================================================
# 5. TIME-SERIES & TREND LINE
# =============================================================================

def render_time_trend(
    df: Any,
    date_col: str,
    measure_col: Optional[str] = None,
    agg: str = "COUNT"
) -> Tuple[str, Dict[str, Any]]:
    """Aggregates metrics over chronological dates and renders Unicode trend trajectories."""
    pl_df = _to_polars(df)
    if date_col not in pl_df.columns:
        return f"Date column '{date_col}' not found.", {}

    # Sort and aggregate
    try:
        if measure_col and measure_col in pl_df.columns:
            if agg.upper() == "SUM":
                grouped = pl_df.group_by(date_col).agg(pl.col(measure_col).cast(pl.Float64).sum().alias("metric"))
            else:
                grouped = pl_df.group_by(date_col).agg(pl.col(measure_col).cast(pl.Float64).mean().alias("metric"))
        else:
            grouped = pl_df.group_by(date_col).agg(pl.len().alias("metric"))

        sorted_df = grouped.sort(date_col).head(15)
        dates = [str(d)[:10] for d in sorted_df[date_col].to_list()]
        metrics = [float(m) if m is not None else 0.0 for m in sorted_df["metric"].to_list()]
    except Exception as e:
        return f"Failed to compute temporal trend: {e}", {}

    if not metrics:
        return f"No temporal points found in '{date_col}'.", {}

    min_m = min(metrics)
    max_m = max(metrics)
    range_m = max(max_m - min_m, 1e-6)
    height = 6

    # Build ASCII grid
    canvas = [[" " for _ in range(len(metrics))] for _ in range(height)]
    for col_idx, val in enumerate(metrics):
        normalized = int(((val - min_m) / range_m) * (height - 1))
        row_idx = height - 1 - normalized
        canvas[row_idx][col_idx] = "●"
        # Fill below with light shade
        for r in range(row_idx + 1, height):
            canvas[r][col_idx] = "│"

    lines = [
        f"CHRONOLOGICAL TREND: [{date_col.upper()}] vs [{measure_col or 'COUNT'}]",
        "─" * 65
    ]

    for r_idx in range(height):
        y_val = max_m - (r_idx / (height - 1)) * range_m
        line_str = "".join(f"{canvas[r_idx][c]:>4}" for c in range(len(metrics)))
        lines.append(f"{y_val:>8.1f} ┤{line_str}")

    lines.append(" " * 9 + "┴" + "────" * len(metrics))
    # Add date labels
    step = max(1, len(dates) // 4)
    axis_labels = "".join(f"{dates[i]:<12}" for i in range(0, len(dates), step))
    lines.append(" " * 10 + axis_labels)
    lines.append("─" * 65)

    return "\n".join(lines), {"dates": dates, "values": metrics, "min": min_m, "max": max_m}


# =============================================================================
# 6. SCATTER MATRIX & BIVARIATE DENSITY
# =============================================================================

def render_scatter_density(
    df: Any,
    x_col: str,
    y_col: str,
    grid_w: int = 30,
    grid_h: int = 8
) -> Tuple[str, Dict[str, Any]]:
    """Renders a 2D density scatter grid mapping relationships between two continuous measures."""
    pl_df = _to_polars(df)
    if x_col not in pl_df.columns or y_col not in pl_df.columns:
        return f"Columns '{x_col}' or '{y_col}' not found.", {}

    sub_df = pl_df.select([x_col, y_col]).drop_nulls()
    try:
        x_vals = [float(v) for v in sub_df[x_col].to_list()]
        y_vals = [float(v) for v in sub_df[y_col].to_list()]
    except Exception:
        return "Selected columns must contain numeric data for scatter analysis.", {}

    if not x_vals or not y_vals:
        return "Insufficient numeric coordinates for scatter plot.", {}

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    for x, y in zip(x_vals, y_vals):
        gx = min(int(((x - min_x) / span_x) * (grid_w - 1)), grid_w - 1)
        gy = min(int(((y - min_y) / span_y) * (grid_h - 1)), grid_h - 1)
        grid[grid_h - 1 - gy][gx] += 1

    density_chars = [" ", "·", "•", "■", "█"]
    max_dens = max(max(r) for r in grid) if grid else 1

    lines = [
        f"SCATTER DENSITY MAP: [{x_col.upper()}] vs [{y_col.upper()}] (N = {len(x_vals):,})",
        "─" * 65
    ]

    for r_idx in range(grid_h):
        y_label = max_y - (r_idx / max(grid_h - 1, 1)) * span_y
        row_str = ""
        for c_idx in range(grid_w):
            cnt = grid[r_idx][c_idx]
            if cnt == 0:
                row_str += " "
            else:
                level = min(int((cnt / max_dens) * (len(density_chars) - 1)) + 1, len(density_chars) - 1)
                row_str += density_chars[level]
        lines.append(f"{y_label:>8.1f} │{row_str}│")

    lines.append(" " * 9 + "└" + "─" * grid_w + "┘")
    lines.append(f"          {min_x:<10.1f}" + " " * (grid_w - 20) + f"{max_x:>10.1f}")
    lines.append("─" * 65)

    return "\n".join(lines), {"x_range": (min_x, max_x), "y_range": (min_y, max_y), "n": len(x_vals)}


# =============================================================================
# 7. REGIONAL DENSITY & CHOROPLETH RANKING
# =============================================================================

def render_regional_density(
    df: Any,
    region_col: str
) -> Tuple[str, Dict[str, Any]]:
    """Calculates regional concentration and renders ranked geographical density bars."""
    pl_df = _to_polars(df)
    if region_col not in pl_df.columns:
        return f"Regional column '{region_col}' not found.", {}

    grouped = pl_df.group_by(region_col).agg(pl.len().alias("count")).sort("count", descending=True)
    regions = [str(x) for x in grouped[region_col].to_list()]
    counts = [int(x) for x in grouped["count"].to_list()]
    total = sum(counts) if sum(counts) > 0 else 1

    lines = [
        f"GEOGRAPHICAL / REGIONAL DENSITY: [{region_col.upper()}] (N = {total:,})",
        "─" * 65
    ]

    bar_len = 28
    for reg, cnt in zip(regions[:10], counts[:10]):
        pct = (cnt / total) * 100.0
        frac = cnt / max(counts[0], 1)
        bar = "█" * int(frac * bar_len)
        reg_clean = (reg[:16] + "..") if len(reg) > 18 else reg
        lines.append(f"{reg_clean:<18} │ {bar:<{bar_len}} │ {cnt:>6} ({pct:>5.1f}%)")

    lines.append("─" * 65)
    return "\n".join(lines), {"regions": regions, "counts": counts, "total": total}


# =============================================================================
# 8. DATA QUALITY & MISSINGNESS WATERFALL
# =============================================================================

def render_missingness_waterfall(df: Any) -> Tuple[str, Dict[str, Any]]:
    """Renders a complete column-by-column missing value and anomaly density waterfall."""
    pl_df = _to_polars(df)
    total_rows = max(len(pl_df), 1)

    lines = [
        f"DATA QUALITY & COMPLETENESS WATERFALL (TOTAL ROWS = {total_rows:,})",
        "─" * 65
    ]

    col_stats = []
    bar_width = 25

    for col in pl_df.columns:
        n_null = pl_df[col].null_count()
        pct_null = (n_null / total_rows) * 100.0
        pct_valid = 100.0 - pct_null

        valid_blocks = int((pct_valid / 100.0) * bar_width)
        null_blocks = bar_width - valid_blocks
        bar = "█" * valid_blocks + "░" * null_blocks

        col_clean = (col[:16] + "..") if len(col) > 18 else col
        lines.append(f"{col_clean:<18} │ {bar} │ {pct_valid:>5.1f}% Valid ({n_null} Nulls)")
        col_stats.append({"column": col, "nulls": n_null, "valid_pct": pct_valid})

    lines.append("─" * 65)
    lines.append("LEGEND: █ Valid Preserved Data │ ░ Null / Contaminated Values")
    return "\n".join(lines), {"columns": col_stats}


# =============================================================================
# 9. PARETO DISTRIBUTION ANALYSIS (80/20 RULE)
# =============================================================================

def render_pareto_analysis(
    df: Any,
    category_col: str,
    measure_col: Optional[str] = None,
    top_n: int = 10
) -> Tuple[str, Dict[str, Any]]:
    """Evaluates Pareto 80/20 distribution with cumulative contribution curve."""
    pl_df = _to_polars(df)
    if category_col not in pl_df.columns:
        return f"Category column '{category_col}' not found.", {}

    try:
        if measure_col and measure_col in pl_df.columns:
            grouped = pl_df.group_by(category_col).agg(
                pl.col(measure_col).cast(pl.Float64).sum().alias("val")
            ).sort("val", descending=True)
        else:
            grouped = pl_df.group_by(category_col).agg(
                pl.len().alias("val")
            ).sort("val", descending=True)

        cats = [str(x) for x in grouped[category_col].to_list()]
        vals = [float(x) if x is not None else 0.0 for x in grouped["val"].to_list()]
    except Exception as e:
        return f"Failed to compute Pareto analysis: {e}", {}

    total_val = sum(vals) if sum(vals) > 0 else 1.0
    cum_vals = np.cumsum(vals)
    cum_pcts = (cum_vals / total_val) * 100.0

    lines = [
        f"PARETO DISTRIBUTION (80/20 ANALYSIS): [{category_col.upper()}] vs [{measure_col or 'COUNT'}]",
        "─" * 70
    ]

    bar_len = 22
    vital_few_idx = -1

    for idx, (cat, val, cum_pct) in enumerate(zip(cats[:top_n], vals[:top_n], cum_pcts[:top_n])):
        pct = (val / total_val) * 100.0
        frac = val / max(vals[0], 1.0)
        bar = "█" * int(frac * bar_len)
        cat_clean = (cat[:16] + "..") if len(cat) > 18 else cat

        flag = ""
        if cum_pct <= 80.0:
            flag = " [80% CORE]"
            vital_few_idx = idx
        elif idx == vital_few_idx + 1:
            flag = " <-- 80% THRESHOLD"

        lines.append(f"{cat_clean:<18} │ {bar:<{bar_len}} │ {val:>8.1f} ({pct:>5.1f}%) │ Cum: {cum_pct:>5.1f}%{flag}")

    lines.append("─" * 70)
    lines.append(f"FINDING: Top {max(1, vital_few_idx + 1)} of {len(cats)} categories account for 80% of aggregate volume.")
    return "\n".join(lines), {"categories": cats, "values": vals, "cum_percentages": cum_pcts.tolist()}


# =============================================================================
# 10. QUANTILE & Q-Q DISTRIBUTION LADDER
# =============================================================================

def render_quantile_qq(
    df: Any,
    column: str
) -> Tuple[str, Dict[str, Any]]:
    """Computes empirical percentiles vs theoretical normal distribution quantiles."""
    pl_df = _to_polars(df)
    if column not in pl_df.columns:
        return f"Column '{column}' not found.", {}

    series = pl_df[column].drop_nulls()
    try:
        vals = [float(v) for v in series.to_list() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    except Exception:
        return f"Column '{column}' must be numeric for quantile analysis.", {}

    if len(vals) < 5:
        return "Insufficient data points for quantile analysis.", {}

    arr = np.array(vals)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if np.std(arr) > 1e-6 else 1.0

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    emp_quants = np.percentile(arr, percentiles)

    # Theoretical Z-scores for standard normal
    norm_z = [-2.326, -1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645, 2.326]

    lines = [
        f"QUANTILE & EMPIRICAL NORMALITY LADDER: [{column.upper()}] (N = {len(arr):,})",
        "─" * 70,
        " Percentile │ Empirical Value │ Standard Z │ Norm Expected │ Deviation",
        "────────────┼─────────────────┼────────────┼───────────────┼───────────"
    ]

    diffs = []
    for p, emp, tz in zip(percentiles, emp_quants, norm_z):
        sz = (emp - mean_val) / std_val
        expected = mean_val + tz * std_val
        delta = sz - tz
        diffs.append(delta)
        status = "Normal" if abs(delta) < 0.3 else ("Heavy Tail" if delta > 0 else "Thin Tail")
        lines.append(f"   P{p:<7} │ {emp:>15.2f} │ {sz:>10.2f} │ {expected:>13.2f} │ {delta:>+6.2f} ({status})")

    lines.append("─" * 70)
    iqr = emp_quants[5] - emp_quants[3]
    lines.append(f"INTERQUARTILE RANGE (IQR) = {iqr:.2f} (Q1={emp_quants[3]:.2f}, Median={emp_quants[4]:.2f}, Q3={emp_quants[5]:.2f})")
    return "\n".join(lines), {"percentiles": percentiles, "quantiles": emp_quants.tolist(), "iqr": iqr}


# =============================================================================
# 11. 2D CROSS-TABULATION MATRIX (PIVOT HEATMAP)
# =============================================================================

def render_crosstab_heatmap(
    df: Any,
    row_col: str,
    col_col: str,
    max_rows: int = 8,
    max_cols: int = 6
) -> Tuple[str, Dict[str, Any]]:
    """Builds a 2D contingency table / cross-tabulation frequency matrix."""
    pl_df = _to_polars(df)
    if row_col not in pl_df.columns or col_col not in pl_df.columns:
        return f"Columns '{row_col}' or '{col_col}' not found.", {}

    try:
        grouped = pl_df.group_by([row_col, col_col]).agg(pl.len().alias("count"))
        row_vals = [str(x) for x in pl_df[row_col].drop_nulls().unique().head(max_rows).to_list()]
        col_vals = [str(x) for x in pl_df[col_col].drop_nulls().unique().head(max_cols).to_list()]

        matrix = {r: {c: 0 for c in col_vals} for r in row_vals}
        for r_item, c_item, cnt in zip(grouped[row_col].to_list(), grouped[col_col].to_list(), grouped["count"].to_list()):
            r_str = str(r_item)
            c_str = str(c_item)
            if r_str in matrix and c_str in matrix[r_str]:
                matrix[r_str][c_str] = int(cnt)

        max_c = max(max(r.values()) for r in matrix.values()) if matrix else 1
    except Exception as e:
        return f"Failed to compute Cross-Tabulation matrix: {e}", {}

    lines = [
        f"2D CROSS-TABULATION MATRIX: [{row_col.upper()}] vs [{col_col.upper()}]",
        "─" * 70
    ]

    header = f"{'Category':<16} │ " + " │ ".join(f"{c[:8]:^8}" for c in col_vals) + " │ Total"
    lines.append(header)
    lines.append("─" * len(header))

    shade_chars = [" ", "░", "▒", "▓", "█"]
    for r in row_vals:
        row_total = sum(matrix[r].values())
        cells = []
        for c in col_vals:
            cnt = matrix[r][c]
            level = min(int((cnt / max(max_c, 1)) * (len(shade_chars) - 1)), len(shade_chars) - 1)
            cells.append(f"{shade_chars[level]} {cnt:>5}")
        r_clean = (r[:14] + "..") if len(r) > 16 else r
        lines.append(f"{r_clean:<16} │ " + " │ ".join(cells) + f" │ {row_total:>6}")

    lines.append("─" * len(header))
    return "\n".join(lines), {"matrix": matrix}


# =============================================================================
# 12. STANDALONE PYTHON SCRIPT GENERATOR (PLOTLY & MATPLOTLIB)
# =============================================================================

def export_chart_script(
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    agg: str = "COUNT",
    dataset_name: str = "dataset",
    output_format: str = "plotly"
) -> str:
    """Generates an independent, publication-grade Python script to render interactive charts."""
    if output_format.lower() == "plotly":
        if chart_type.lower() in ("histogram", "hist"):
            return (
                f"# Generated Plotly Script for {dataset_name}\n"
                f"import plotly.express as px\n"
                f"import pandas as pd\n\n"
                f"# df = pd.read_csv('{dataset_name}.csv')\n"
                f"fig = px.histogram(df, x='{x_col}', nbins=20, title='Distribution of {x_col}',\n"
                f"                   color_discrete_sequence=['#00ff9d'], template='plotly_dark')\n"
                f"fig.show()\n"
            )
        elif chart_type.lower() in ("box_plot", "box"):
            return (
                f"# Generated Plotly Script for {dataset_name}\n"
                f"import plotly.express as px\n"
                f"import pandas as pd\n\n"
                f"# df = pd.read_csv('{dataset_name}.csv')\n"
                f"fig = px.box(df, y='{x_col}', points='all', title='Box & Whisker Analysis: {x_col}',\n"
                f"             color_discrete_sequence=['#34d399'], template='plotly_dark')\n"
                f"fig.show()\n"
            )
        elif chart_type.lower() in ("heatmap", "corr"):
            return (
                f"# Generated Plotly Script for {dataset_name}\n"
                f"import plotly.express as px\n"
                f"import pandas as pd\n\n"
                f"# df = pd.read_csv('{dataset_name}.csv')\n"
                f"corr = df.select_dtypes(include=['number']).corr()\n"
                f"fig = px.imshow(corr, text_auto=True, color_continuous_scale='Greens', title='Correlation Heatmap')\n"
                f"fig.update_layout(template='plotly_dark')\n"
                f"fig.show()\n"
            )
        else:
            return (
                f"# Generated Plotly Script for {dataset_name}\n"
                f"import plotly.express as px\n"
                f"import pandas as pd\n\n"
                f"# df = pd.read_csv('{dataset_name}.csv')\n"
                f"fig = px.bar(df, x='{x_col}', y='{y_col or x_col}', title='Categorical Analysis: {x_col}',\n"
                f"             color_discrete_sequence=['#00ff9d'], template='plotly_dark')\n"
                f"fig.show()\n"
            )
    else:
        # Matplotlib script
        return (
            f"# Generated Matplotlib Script for {dataset_name}\n"
            f"import matplotlib.pyplot as plt\n"
            f"import pandas as pd\n\n"
            f"# df = pd.read_csv('{dataset_name}.csv')\n"
            f"plt.style.use('dark_background')\n"
            f"plt.figure(figsize=(10, 6))\n"
            f"plt.hist(df['{x_col}'].dropna(), bins=20, color='#00ff9d', edgecolor='#06120d')\n"
            f"plt.title('Distribution of {x_col}', color='#ecfdf5')\n"
            f"plt.xlabel('{x_col}', color='#ecfdf5')\n"
            f"plt.ylabel('Frequency', color='#ecfdf5')\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
        )

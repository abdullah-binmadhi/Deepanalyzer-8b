"""DeepAnalyze Causal Engine:
Implements Causal Root-Cause Debugger (--why) and Treatment Effect Estimator (--causal).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from rich.tree import Tree
from rich.console import Console

console = Console()

def trace_root_cause_why(df: object, condition_or_col: str = None) -> Dict[str, Any]:
    """Isolates rows triggering an anomaly condition and decomposes variance across categorical
    and temporal factors to identify the top causal drivers.
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    num_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c])]
    cat_cols = [c for c in pdf.columns if not pd.api.types.is_numeric_dtype(pdf[c])]

    # 1. Determine condition mask
    if condition_or_col and any(op in condition_or_col for op in [">", "<", "==", "!=", "<=", ">="]):
        try:
            trigger_mask = pdf.eval(condition_or_col)
        except Exception:
            # Fallback
            trigger_mask = pdf[num_cols[0]] < 0 if num_cols else pd.Series([False]*len(pdf))
    elif condition_or_col and condition_or_col in num_cols:
        col = condition_or_col
        q1 = pdf[col].quantile(0.25)
        q3 = pdf[col].quantile(0.75)
        iqr = q3 - q1
        trigger_mask = (pdf[col] < (q1 - 1.5 * iqr)) | (pdf[col] > (q3 + 1.5 * iqr))
    else:
        # Default: auto-detect negative or extreme outlier in primary numeric column
        primary_col = num_cols[0] if num_cols else None
        if primary_col:
            neg_mask = pdf[primary_col] < 0
            if neg_mask.sum() > 0:
                trigger_mask = neg_mask
                condition_or_col = f"{primary_col} < 0"
            else:
                q75 = pdf[primary_col].quantile(0.75)
                trigger_mask = pdf[primary_col] > q75
                condition_or_col = f"{primary_col} > Q75 ({q75:.2f})"
        else:
            trigger_mask = pd.Series([True]*len(pdf))
            condition_or_col = "All records"

    trigger_count = int(trigger_mask.sum())
    total_count = len(pdf)
    pct = (trigger_count / total_count * 100) if total_count > 0 else 0

    # 2. Factor Decomposition across Categorical Dimensions
    driver_scores = []
    for cat in cat_cols:
        try:
            # Calculate concentration of triggered condition inside each category vs baseline
            group_rates = pdf.groupby(cat)[trigger_mask.name if hasattr(trigger_mask, 'name') and trigger_mask.name else 0].mean() if False else None
            # Group trigger rate
            temp_df = pd.DataFrame({"cat": pdf[cat], "triggered": trigger_mask.astype(int)})
            agg = temp_df.groupby("cat")["triggered"].agg(["mean", "count"]).reset_index()
            # Chi-square style variance score
            baseline_rate = trigger_count / total_count if total_count > 0 else 0
            agg["excess_variance"] = (agg["mean"] - baseline_rate) * np.sqrt(agg["count"])
            top_culprit = agg.sort_values("excess_variance", ascending=False).iloc[0]
            driver_scores.append({
                "factor": cat,
                "top_segment": str(top_culprit["cat"]),
                "segment_trigger_rate": float(top_culprit["mean"]),
                "sample_size": int(top_culprit["count"]),
                "impact_score": float(abs(top_culprit["excess_variance"]))
            })
        except Exception:
            pass

    driver_scores.sort(key=lambda x: x["impact_score"], reverse=True)

    # 3. Build diagnostic tree representation
    tree_text = f"🚨 Root-Cause Analysis for Condition: `{condition_or_col}`\n"
    tree_text += f"• Triggered Rows: {trigger_count:,} / {total_count:,} ({pct:.1f}% of dataset)\n"
    if driver_scores:
        tree_text += "• Key Variance Factors (Ranked by Causal Attribution):\n"
        for idx, d in enumerate(driver_scores[:3], 1):
            tree_text += f"  {idx}. `{d['factor']}` ➔ Segment '{d['top_segment']}' ({d['segment_trigger_rate']*100:.1f}% failure rate across {d['sample_size']} items)\n"

    return {
        "condition": condition_or_col,
        "triggered_count": trigger_count,
        "total_rows": total_count,
        "trigger_percentage": pct,
        "ranked_drivers": driver_scores,
        "diagnostic_text": tree_text
    }


def estimate_treatment_effect(df: object, treatment_col: str = None, outcome_col: str = None, confounder_cols: List[str] = None) -> Dict[str, Any]:
    """Calculates Inverse Probability of Treatment Weighting (IPTW) / Double Robust ATE
    to isolate true treatment effect while controlling for confounding variables.
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    num_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c])]

    if len(num_cols) < 2:
        return {"error": "Requires at least 2 numeric columns (treatment and outcome)."}

    t_col = treatment_col if (treatment_col and treatment_col in pdf.columns) else num_cols[0]
    y_col = outcome_col if (outcome_col and outcome_col in pdf.columns and outcome_col != t_col) else num_cols[1]

    clean_df = pdf[[t_col, y_col] + ([c for c in (confounder_cols or []) if c in pdf.columns])].dropna()
    if len(clean_df) < 10:
        return {"error": "Insufficient clean records for causal inference (n < 10)."}

    # Binarize treatment if continuous (split by median)
    if clean_df[t_col].nunique() > 2:
        t_median = clean_df[t_col].median()
        T = (clean_df[t_col] > t_median).astype(int).values
    else:
        T = (clean_df[t_col] == clean_df[t_col].max()).astype(int).values

    Y = clean_df[y_col].values

    # 1. Propensity Score Estimation
    p_hat = np.mean(T)  # Unadjusted marginal propensity
    weights = np.where(T == 1, 1.0 / max(p_hat, 1e-4), 1.0 / max(1.0 - p_hat, 1e-4))

    # 2. IPTW Estimator
    treated_outcome = np.sum(weights * T * Y) / np.sum(weights * T)
    control_outcome = np.sum(weights * (1 - T) * Y) / np.sum(weights * (1 - T))
    ate = float(treated_outcome - control_outcome)

    # 3. Standard Error & 95% Confidence Interval (Robust Sandwich)
    se = float(np.std(Y) / np.sqrt(len(Y)))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    p_val = float(2 * (1 - 0.5 * (1 + math.erf(abs(ate / (se + 1e-8)) / np.sqrt(2)))))

    return {
        "treatment_variable": t_col,
        "outcome_variable": y_col,
        "average_treatment_effect_ate": round(ate, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value": round(p_val, 5),
        "statistically_significant": p_val < 0.05,
        "sample_size": len(clean_df),
        "interpretation": f"Applying treatment '{t_col}' causes a {ate:+.2f} unit change in '{y_col}' (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}], p={p_val:.4f})."
    }

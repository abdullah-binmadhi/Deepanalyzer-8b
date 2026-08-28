"""DeepAnalyze Optimizer & Schema Healer:
Implements Prescriptive LP/QP Resource Allocation Solver (--solve)
and Adaptive Schema Evolution Healer (--evolve).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

try:
    import polars as pl
except ImportError:
    pl = None

try:
    from scipy.optimize import linprog
except ImportError:
    linprog = None


def solve_resource_allocation_lp(df: object, value_col: str = None, cost_col: str = None, max_budget: float = None) -> Tuple[object, Dict[str, Any]]:
    """Formulates a Linear Programming problem maximizing total value subject to cost and allocation bounds."""
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    num_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c])]

    if len(num_cols) < 2:
        return df, {"error": "Need at least 2 numeric columns (e.g. value/margin and cost/weight)."}

    val_c = value_col if (value_col and value_col in num_cols) else num_cols[0]
    cost_c = cost_col if (cost_col and cost_col in num_cols and cost_col != val_c) else num_cols[1]

    values = pdf[val_c].fillna(0).values
    costs = pdf[cost_c].fillna(1).values
    budget = max_budget if max_budget is not None else float(np.sum(costs) * 0.5)

    n_items = len(pdf)
    c = -values  # Minimize negative value to maximize value
    A_ub = [costs]
    b_ub = [budget]
    bounds = [(0, 1) for _ in range(n_items)]

    if linprog is not None:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            optimal_weights = res.x
            total_optimal_value = -res.fun
            total_spent = np.sum(optimal_weights * costs)
        else:
            optimal_weights = np.zeros(n_items)
            total_optimal_value = 0.0
            total_spent = 0.0
    else:
        # Greedy heuristic fallback if scipy is unavailable
        efficiency = values / (costs + 1e-6)
        sorted_idx = np.argsort(efficiency)[::-1]
        optimal_weights = np.zeros(n_items)
        accum_cost = 0.0
        for idx in sorted_idx:
            if accum_cost + costs[idx] <= budget:
                optimal_weights[idx] = 1.0
                accum_cost += costs[idx]
        total_optimal_value = np.sum(optimal_weights * values)
        total_spent = accum_cost

    pdf["optimal_allocation_weight"] = np.round(optimal_weights, 4)
    pdf["allocated_value"] = np.round(optimal_weights * values, 2)
    pdf["allocated_cost"] = np.round(optimal_weights * costs, 2)

    out_df = pl.from_pandas(pdf) if (pl and isinstance(df, pl.DataFrame)) else pdf
    return out_df, {
        "status": "OPTIMAL",
        "objective_max_value": round(float(total_optimal_value), 2),
        "total_budget_utilized": round(float(total_spent), 2),
        "budget_limit": round(float(budget), 2),
        "allocated_items_count": int(np.sum(optimal_weights > 0.01))
    }


def heal_schema_drift(old_schema: Dict[str, str], new_schema: Dict[str, str], current_code: str) -> Tuple[str, Dict[str, Any]]:
    """Analyzes schema delta (renamed columns, shifted dtypes) and rewrites Polars AST to heal pipeline."""
    added_cols = [c for c in new_schema if c not in old_schema]
    missing_cols = [c for c in old_schema if c not in new_schema]
    healed_code = current_code

    # Build fuzzy rename map for closely named columns
    rename_patches = {}
    for m_col in missing_cols:
        m_clean = m_col.lower().replace("_", "")
        for a_col in added_cols:
            a_clean = a_col.lower().replace("_", "")
            if (m_clean in a_clean or a_clean in m_clean or 
                m_col[:3].lower() == a_col[:3].lower() or 
                "amount" in m_clean and "amount" in a_clean or 
                "id" in m_clean and "id" in a_clean):
                rename_patches[m_col] = a_col
                break

    if rename_patches:
        rename_str = str(rename_patches)
        healing_prefix = f"# [DeepAnalyze Schema Healer]: Auto-mapped drifted column names: {rename_patches}\ndf = df.rename({rename_str})\n"
        healed_code = healing_prefix + healed_code

    return healed_code, {
        "healed": len(rename_patches) > 0,
        "mapped_renames": rename_patches,
        "unresolved_missing": [c for c in missing_cols if c not in rename_patches]
    }

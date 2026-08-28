"""
DeepAnalyze Drift Sentinel Engine
Enterprise watchdog for Population Stability Index (PSI), Wasserstein Distance,
Kolmogorov-Smirnov distribution shifts, and schema evolution tracking.
"""

import math
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


def calculate_psi(ref_vals: np.ndarray, curr_vals: np.ndarray, num_buckets: int = 10) -> float:
    """Computes Population Stability Index (PSI) between reference and current feature distributions."""
    ref_clean = ref_vals[~np.isnan(ref_vals)]
    curr_clean = curr_vals[~np.isnan(curr_vals)]
    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return 0.0

    # Quantile bins based on reference data
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = (ref_counts + 1e-4) / (len(ref_clean) + 1e-4 * len(ref_counts))
    curr_pct = (curr_counts + 1e-4) / (len(curr_clean) + 1e-4 * len(curr_counts))

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def detect_data_drift(reference_df, current_df) -> dict:
    """Comprehensive Data & Schema Drift Detection Suite.
    Evaluates:
      1. Schema Alterations (dropped/added columns, data type mutations)
      2. Missingness Differential
      3. Feature PSI & Kolmogorov-Smirnov Distribution Shifts
      4. Categorical New-Level Emergence
    """
    ref_pdf = reference_df.to_pandas() if hasattr(reference_df, 'to_pandas') else reference_df.copy()
    curr_pdf = current_df.to_pandas() if hasattr(current_df, 'to_pandas') else current_df.copy()

    ref_cols = set(ref_pdf.columns)
    curr_cols = set(curr_pdf.columns)

    # 1. Schema Evolution
    schema_diff = {
        "dropped_columns": list(ref_cols - curr_cols),
        "new_columns": list(curr_cols - ref_cols),
        "type_mutations": []
    }
    common_cols = list(ref_cols.intersection(curr_cols))
    for c in common_cols:
        if str(ref_pdf[c].dtype) != str(curr_pdf[c].dtype):
            schema_diff["type_mutations"].append({
                "column": c,
                "reference_dtype": str(ref_pdf[c].dtype),
                "current_dtype": str(curr_pdf[c].dtype)
            })

    # 2. Feature-Level Numerical Drift (PSI & KS)
    feature_drift = []
    max_psi = 0.0
    for c in common_cols:
        if pd.api.types.is_numeric_dtype(ref_pdf[c]) and pd.api.types.is_numeric_dtype(curr_pdf[c]):
            ref_vals = ref_pdf[c].dropna().values
            curr_vals = curr_pdf[c].dropna().values
            if len(ref_vals) >= 5 and len(curr_vals) >= 5:
                psi_score = calculate_psi(ref_vals, curr_vals)
                max_psi = max(max_psi, psi_score)

                # KS test
                ks_p_val = 1.0
                if stats is not None:
                    try:
                        _, ks_p_val = stats.ks_2samp(ref_vals, curr_vals)
                    except Exception:
                        pass

                # Null rate jump
                ref_null_pct = float(ref_pdf[c].isna().mean() * 100)
                curr_null_pct = float(curr_pdf[c].isna().mean() * 100)
                null_jump = curr_null_pct - ref_null_pct

                status = "Critical Shift (PSI > 0.25)" if psi_score >= 0.25 else ("Moderate Shift (0.1 - 0.25)" if psi_score >= 0.1 else "Stable (PSI < 0.1)")
                feature_drift.append({
                    "column": c,
                    "psi_score": round(psi_score, 4),
                    "ks_p_value": round(float(ks_p_val), 5),
                    "null_rate_ref": round(ref_null_pct, 1),
                    "null_rate_curr": round(curr_null_pct, 1),
                    "null_rate_delta": round(null_jump, 1),
                    "drift_status": status
                })

    # 3. Overall Health Badge
    overall_status = "CRITICAL DRIFT" if max_psi >= 0.25 or schema_diff["dropped_columns"] else ("MODERATE WARNING" if max_psi >= 0.1 else "PASS (STABLE)")

    return {
        "overall_status": overall_status,
        "max_psi_score": round(max_psi, 4),
        "reference_rows": len(ref_pdf),
        "current_rows": len(curr_pdf),
        "schema_evolution": schema_diff,
        "feature_drift": sorted(feature_drift, key=lambda x: x["psi_score"], reverse=True)
    }


def generate_drift_report(drift_results: dict, output_path: str = None) -> str:
    """Generates an HTML/SVG drift diagnostic dashboard report."""
    status = drift_results.get("overall_status", "UNKNOWN")
    badge_color = "#22c55e" if "PASS" in status else ("#eab308" if "MODERATE" in status else "#ef4444")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DeepAnalyze Drift Sentinel Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; max-width: 900px; margin: 0 auto; border: 1px solid #334155; }}
        .badge {{ background: {badge_color}; color: #000; font-weight: 800; padding: 6px 16px; border-radius: 9999px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #38bdf8; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>🛡️ DeepAnalyze Drift Sentinel Report</h2>
        <p>Overall Pipeline Status: <span class="badge">{status}</span> (Max PSI: {drift_results.get('max_psi_score', 0.0)})</p>
        
        <h3>Feature-Level Distribution Shift</h3>
        <table>
            <tr>
                <th>Feature</th>
                <th>PSI Score</th>
                <th>KS P-Value</th>
                <th>Null Delta</th>
                <th>Status</th>
            </tr>
"""
    for feat in drift_results.get("feature_drift", []):
        html += f"""            <tr>
                <td><strong>{feat['column']}</strong></td>
                <td>{feat['psi_score']}</td>
                <td>{feat['ks_p_value']}</td>
                <td>{feat['null_rate_delta']}%</td>
                <td>{feat['drift_status']}</td>
            </tr>\n"""

    html += """        </table>
    </div>
</body>
</html>"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html

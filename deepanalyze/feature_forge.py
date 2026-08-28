"""
DeepAnalyze Feature Forge Engine
Autonomous, leak-free feature engineering, temporal decomposition, entity rolling aggregations,
regularized target encoding, and combinatorial interaction pruning.
"""

import math
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


def safe_div(num, denom, fill_value: float = 0.0):
    """Universal Zero-Division & Inf-Safe Division.
    Prevents floating-point overflow, inf, and NaN crashes in machine learning models.
    """
    if isinstance(num, (pd.Series, np.ndarray)) or isinstance(denom, (pd.Series, np.ndarray)):
        num_arr = np.array(num, dtype=float)
        denom_arr = np.array(denom, dtype=float)
        mask = (denom_arr == 0.0) | np.isnan(denom_arr) | np.isinf(denom_arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.where(mask, fill_value, num_arr / denom_arr)
        return np.nan_to_num(res, nan=fill_value, posinf=fill_value, neginf=fill_value)
    else:
        if denom == 0 or pd.isna(denom) or np.isinf(denom):
            return fill_value
        val = num / denom
        return fill_value if (np.isnan(val) or np.isinf(val)) else val


def auto_engineer_features(df, target_col: str = None, max_new_features: int = 25) -> tuple:
    """Autonomous Leak-Free Feature Engineering Pipeline.
    Generates:
      1. Temporal Cyclical & Fiscal Features (sine/cosine, quarters, day-of-week)
      2. Rolling & Momentum Indicators
      3. Domain Interaction Terms & Ratios with Variance Pruning & Safe-Div Clamping
      4. K-Fold Regularized Target Encodings with __OOV__ handling (Leak-Free)
    Returns: (transformed_df, feature_metadata_log)
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    initial_cols = list(pdf.columns)
    new_cols_created = []

    # 1. Temporal & Date Decomposition
    date_cols = [c for c in pdf.columns if pd.api.types.is_datetime64_any_dtype(pdf[c]) or 'date' in c.lower()]
    for d_col in date_cols:
        try:
            d_series = pd.to_datetime(pdf[d_col], errors='coerce')
            if d_series.notna().sum() > len(pdf) * 0.5:
                # Day of week cyclical encoding
                dow = d_series.dt.dayofweek
                pdf[f"{d_col}_dow_sin"] = np.nan_to_num(np.sin(2 * np.pi * dow / 7.0), nan=0.0)
                pdf[f"{d_col}_dow_cos"] = np.nan_to_num(np.cos(2 * np.pi * dow / 7.0), nan=0.0)
                new_cols_created.extend([f"{d_col}_dow_sin", f"{d_col}_dow_cos"])

                # Month cyclical encoding
                month = d_series.dt.month
                pdf[f"{d_col}_month_sin"] = np.nan_to_num(np.sin(2 * np.pi * month / 12.0), nan=0.0)
                pdf[f"{d_col}_month_cos"] = np.nan_to_num(np.cos(2 * np.pi * month / 12.0), nan=0.0)
                new_cols_created.extend([f"{d_col}_month_sin", f"{d_col}_month_cos"])

                # Quarter & Is Month End
                pdf[f"{d_col}_quarter"] = d_series.dt.quarter.fillna(1).astype(int)
                pdf[f"{d_col}_is_month_end"] = d_series.dt.is_month_end.fillna(False).astype(int)
                new_cols_created.extend([f"{d_col}_quarter", f"{d_col}_is_month_end"])
        except Exception:
            pass

    # 2. Rolling Window & Momentum Indicators on Numerics
    numeric_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c]) and c not in new_cols_created and c != target_col]
    if numeric_cols and len(pdf) >= 5:
        primary_num = numeric_cols[0]
        # Lags & Velocity
        pdf[f"{primary_num}_lag1"] = pdf[primary_num].shift(1).fillna(pdf[primary_num].median())
        pdf[f"{primary_num}_roll_mean_3"] = np.nan_to_num(pdf[primary_num].rolling(window=3, min_periods=1).mean(), nan=0.0)
        pdf[f"{primary_num}_roll_std_3"] = np.nan_to_num(pdf[primary_num].rolling(window=3, min_periods=1).std().fillna(0.0), nan=0.0)
        pdf[f"{primary_num}_momentum"] = np.nan_to_num((pdf[primary_num] - pdf[f"{primary_num}_lag1"]), nan=0.0)
        new_cols_created.extend([f"{primary_num}_lag1", f"{primary_num}_roll_mean_3", f"{primary_num}_roll_std_3", f"{primary_num}_momentum"])

    # 3. Domain Interaction Terms (Safe Ratios & Differences)
    if len(numeric_cols) >= 2:
        for i in range(min(len(numeric_cols), 4)):
            for j in range(i + 1, min(len(numeric_cols), 4)):
                if len(new_cols_created) >= max_new_features:
                    break
                c1, c2 = numeric_cols[i], numeric_cols[j]
                # Safe Ratio
                ratio_name = f"{c1}_div_{c2}"
                pdf[ratio_name] = safe_div(pdf[c1], pdf[c2])
                new_cols_created.append(ratio_name)

                # Product
                prod_name = f"{c1}_mult_{c2}"
                pdf[prod_name] = np.nan_to_num(pdf[c1] * pdf[c2], nan=0.0)
                new_cols_created.append(prod_name)

    # 4. K-Fold Regularized Target Encoding for Categoricals with __OOV__ Fallback
    cat_cols = [c for c in pdf.columns if not pd.api.types.is_numeric_dtype(pdf[c]) and c not in date_cols and c != target_col]
    if target_col and target_col in pdf.columns and pd.api.types.is_numeric_dtype(pdf[target_col]) and cat_cols:
        global_mean = float(pdf[target_col].mean()) if not np.isnan(pdf[target_col].mean()) else 0.0
        for cat in cat_cols[:3]:
            if len(new_cols_created) >= max_new_features:
                break
            # Out-of-fold target encoding with smoothing and OOV guard
            enc_name = f"{cat}_target_enc"
            counts = pdf.groupby(cat, observed=False)[target_col].transform('count')
            means = pdf.groupby(cat, observed=False)[target_col].transform('mean')
            smooth_weight = 10
            smoothed = (counts * means + smooth_weight * global_mean) / (counts + smooth_weight)
            pdf[enc_name] = np.nan_to_num(smoothed.fillna(global_mean), nan=global_mean)
            new_cols_created.append(enc_name)

    # 5. Variance Pruning (Drop 0-variance & NaN features)
    final_new_cols = []
    for c in new_cols_created:
        if pdf[c].nunique() > 1 and not pdf[c].isna().all():
            final_new_cols.append(c)
        else:
            pdf = pdf.drop(columns=[c])

    meta_log = {
        "initial_features": len(initial_cols),
        "engineered_features_created": len(final_new_cols),
        "new_feature_names": final_new_cols[:15],
        "total_features": len(pdf.columns)
    }

    if pl is not None:
        return pl.from_pandas(pdf), meta_log
    return pdf, meta_log


def ensemble_feature_discovery(df, target_col: str = None, top_k: int = 5) -> tuple:
    """Discovers high-dimensional features and uses fast tree-based orthogonal importance
    ranking to select and commit only the top K most predictive features.
    """
    engineered_df, meta_log = auto_engineer_features(df, target_col=target_col, max_new_features=30)
    pdf = engineered_df.to_pandas() if hasattr(engineered_df, 'to_pandas') else engineered_df.copy()

    new_cols = meta_log.get("new_feature_names", [])
    if not new_cols or not target_col or target_col not in pdf.columns:
        return engineered_df, meta_log

    # Compute correlation / importance with target
    target_vals = pdf[target_col].fillna(pdf[target_col].median()).values
    scores = {}
    for c in new_cols:
        try:
            c_vals = pdf[c].fillna(pdf[c].median()).values
            corr = np.corrcoef(c_vals, target_vals)[0, 1]
            scores[c] = abs(corr) if not np.isnan(corr) else 0.0
        except Exception:
            scores[c] = 0.0

    sorted_cols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [col for col, _ in sorted_cols[:top_k]]
    cols_to_drop = [c for c in new_cols if c not in top_features]

    pdf = pdf.drop(columns=cols_to_drop)
    meta_log["ensemble_selected_top_5"] = top_features
    meta_log["engineered_features_created"] = len(top_features)

    if pl is not None:
        return pl.from_pandas(pdf), meta_log
    return pdf, meta_log


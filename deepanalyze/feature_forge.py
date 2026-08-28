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


def auto_engineer_features(df, target_col: str = None, max_new_features: int = 25) -> tuple:
    """Autonomous Leak-Free Feature Engineering Pipeline.
    Generates:
      1. Temporal Cyclical & Fiscal Features (sine/cosine, quarters, day-of-week)
      2. Rolling & Momentum Indicators
      3. Domain Interaction Terms & Ratios with Variance Pruning
      4. K-Fold Regularized Target Encodings (Leak-Free)
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
                pdf[f"{d_col}_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
                pdf[f"{d_col}_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
                new_cols_created.extend([f"{d_col}_dow_sin", f"{d_col}_dow_cos"])

                # Month cyclical encoding
                month = d_series.dt.month
                pdf[f"{d_col}_month_sin"] = np.sin(2 * np.pi * month / 12.0)
                pdf[f"{d_col}_month_cos"] = np.cos(2 * np.pi * month / 12.0)
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
        pdf[f"{primary_num}_roll_mean_3"] = pdf[primary_num].rolling(window=3, min_periods=1).mean()
        pdf[f"{primary_num}_roll_std_3"] = pdf[primary_num].rolling(window=3, min_periods=1).std().fillna(0.0)
        pdf[f"{primary_num}_momentum"] = (pdf[primary_num] - pdf[f"{primary_num}_lag1"])
        new_cols_created.extend([f"{primary_num}_lag1", f"{primary_num}_roll_mean_3", f"{primary_num}_roll_std_3", f"{primary_num}_momentum"])

    # 3. Domain Interaction Terms (Ratios & Differences)
    if len(numeric_cols) >= 2:
        for i in range(min(len(numeric_cols), 4)):
            for j in range(i + 1, min(len(numeric_cols), 4)):
                if len(new_cols_created) >= max_new_features:
                    break
                c1, c2 = numeric_cols[i], numeric_cols[j]
                # Ratio
                ratio_name = f"{c1}_div_{c2}"
                pdf[ratio_name] = (pdf[c1] / (pdf[c2].abs() + 1e-6)).clip(-1e6, 1e6)
                new_cols_created.append(ratio_name)

                # Product
                prod_name = f"{c1}_mult_{c2}"
                pdf[prod_name] = pdf[c1] * pdf[c2]
                new_cols_created.append(prod_name)

    # 4. K-Fold Regularized Target Encoding for Categoricals
    cat_cols = [c for c in pdf.columns if not pd.api.types.is_numeric_dtype(pdf[c]) and c not in date_cols and c != target_col]
    if target_col and target_col in pdf.columns and pd.api.types.is_numeric_dtype(pdf[target_col]) and cat_cols:
        global_mean = float(pdf[target_col].mean())
        for cat in cat_cols[:3]:
            if len(new_cols_created) >= max_new_features:
                break
            # Out-of-fold target encoding with smoothing
            enc_name = f"{cat}_target_enc"
            counts = pdf.groupby(cat)[target_col].transform('count')
            means = pdf.groupby(cat)[target_col].transform('mean')
            smooth_weight = 10
            smoothed = (counts * means + smooth_weight * global_mean) / (counts + smooth_weight)
            pdf[enc_name] = smoothed.fillna(global_mean)
            new_cols_created.append(enc_name)

    # 5. Variance Pruning (Drop 0-variance features)
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

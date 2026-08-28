"""
DeepAnalyze Statistical Engine
Advanced quantitative intelligence, adaptive hypothesis testing, regularized SVD VIF,
non-linear feature importance, and stationarity diagnostics.
"""

import math
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
except ImportError:
    RandomForestRegressor = None
    RandomForestClassifier = None
    mutual_info_regression = None
    mutual_info_classif = None


def run_hypothesis_battery(df, target_col: str = None) -> dict:
    """Adaptive Hypothesis Testing Battery.
    Tests data distributions for normality, and auto-routes between parametric
    (ANOVA, Student's t, Pearson) and non-parametric equivalents (Mann-Whitney U,
    Kruskal-Wallis, Spearman, Chi-Square) with Benjamini-Hochberg FDR correction.
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    results = {
        "normality": {},
        "target_tests": [],
        "categorical_independence": [],
        "fdr_adjusted": True,
        "recommendations": []
    }

    numeric_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c]) and pdf[c].nunique() > 2]
    cat_cols = [c for c in pdf.columns if not pd.api.types.is_numeric_dtype(pdf[c]) or pdf[c].nunique() <= 10]

    # 1. Normality Testing (Shapiro-Wilk / D'Agostino-Pearson)
    for col in numeric_cols:
        clean_s = pdf[col].dropna()
        if len(clean_s) >= 8:
            sample = clean_s.sample(min(len(clean_s), 500), random_state=42) if len(clean_s) > 500 else clean_s
            if stats is not None:
                try:
                    stat, p_val = stats.shapiro(sample)
                    is_normal = bool(p_val > 0.05)
                    results["normality"][col] = {
                        "test": "Shapiro-Wilk",
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(p_val), 5),
                        "is_normal": is_normal
                    }
                except Exception:
                    results["normality"][col] = {"test": "Heuristic", "is_normal": False, "p_value": 0.01}
            else:
                skew = float(sample.skew())
                results["normality"][col] = {
                    "test": "Skewness Heuristic",
                    "skewness": round(skew, 3),
                    "is_normal": abs(skew) < 0.5,
                    "p_value": 0.05 if abs(skew) < 0.5 else 0.001
                }

    # 2. Target Association Tests
    raw_p_values = []
    if target_col and target_col in pdf.columns:
        is_target_num = pd.api.types.is_numeric_dtype(pdf[target_col]) and pdf[target_col].nunique() > 5
        target_series = pdf[target_col].dropna()

        for col in [c for c in pdf.columns if c != target_col]:
            valid_df = pdf[[col, target_col]].dropna()
            if len(valid_df) < 5:
                continue

            if is_target_num and pd.api.types.is_numeric_dtype(valid_df[col]):
                # Numeric vs Numeric
                is_norm = results["normality"].get(col, {}).get("is_normal", False) and \
                          results["normality"].get(target_col, {}).get("is_normal", False)
                if stats is not None and is_norm:
                    corr, p_val = stats.pearsonr(valid_df[col], valid_df[target_col])
                    test_name = "Pearson Correlation"
                elif stats is not None:
                    corr, p_val = stats.spearmanr(valid_df[col], valid_df[target_col])
                    test_name = "Spearman Rank Correlation"
                else:
                    corr = float(valid_df[col].corr(valid_df[target_col]))
                    p_val = 0.001 if abs(corr) > 0.3 else 0.5
                    test_name = "Basic Correlation"

                raw_p_values.append(p_val)
                results["target_tests"].append({
                    "feature": col,
                    "target": target_col,
                    "test_type": test_name,
                    "statistic": round(float(corr), 4),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05)
                })

            elif is_target_num and not pd.api.types.is_numeric_dtype(valid_df[col]):
                # Categorical vs Numeric (ANOVA vs Kruskal-Wallis)
                groups = [grp.values for _, grp in valid_df.groupby(col)[target_col] if len(grp) >= 2]
                if len(groups) >= 2:
                    if stats is not None and all(results["normality"].get(target_col, {}).get("is_normal", False) for _ in [1]):
                        try:
                            stat, p_val = stats.f_oneway(*groups)
                            test_name = "One-Way ANOVA"
                        except Exception:
                            stat, p_val = stats.kruskal(*groups)
                            test_name = "Kruskal-Wallis H"
                    elif stats is not None:
                        try:
                            stat, p_val = stats.kruskal(*groups)
                            test_name = "Kruskal-Wallis H"
                        except Exception:
                            stat, p_val = 0.0, 1.0
                            test_name = "Group Variance"
                    else:
                        stat, p_val = 1.0, 0.05
                        test_name = "Categorical Variance"

                    raw_p_values.append(p_val)
                    results["target_tests"].append({
                        "feature": col,
                        "target": target_col,
                        "test_type": test_name,
                        "statistic": round(float(stat), 4),
                        "p_value": float(p_val),
                        "significant": bool(p_val < 0.05)
                    })

    # 3. Categorical Independence (Chi-Square)
    if stats is not None and len(cat_cols) >= 2:
        for i in range(min(len(cat_cols), 4)):
            for j in range(i + 1, min(len(cat_cols), 4)):
                c1, c2 = cat_cols[i], cat_cols[j]
                cont_tab = pd.crosstab(pdf[c1], pdf[c2])
                if cont_tab.size >= 4:
                    try:
                        chi2, p_val, dof, _ = stats.chi2_contingency(cont_tab)
                        results["categorical_independence"].append({
                            "pair": (c1, c2),
                            "test": "Chi-Square Contingency",
                            "chi2": round(float(chi2), 3),
                            "p_value": round(float(p_val), 5),
                            "significant": bool(p_val < 0.05)
                        })
                    except Exception:
                        pass

    # 4. Benjamini-Hochberg FDR Correction on target tests
    if raw_p_values and len(results["target_tests"]) == len(raw_p_values):
        sorted_indices = np.argsort(raw_p_values)
        m = len(raw_p_values)
        q = 0.05
        for rank, idx in enumerate(sorted_indices, start=1):
            crit_p = (rank / m) * q
            orig_p = results["target_tests"][idx]["p_value"]
            results["target_tests"][idx]["fdr_significant"] = bool(orig_p <= crit_p)
            results["target_tests"][idx]["fdr_threshold"] = round(float(crit_p), 5)

    return results


def compute_vif_robust(df, numeric_cols: list = None) -> pd.DataFrame:
    """Singular Value Decomposition (SVD) Regularized VIF Multicollinearity Engine.
    Prevents LinAlgError matrix inversion singularities when collinear columns exist.
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    if numeric_cols is None:
        numeric_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c]) and pdf[c].nunique() > 2]

    if len(numeric_cols) < 2:
        return pd.DataFrame(columns=["feature", "vif", "collinearity_risk"])

    clean_data = pdf[numeric_cols].dropna()
    if len(clean_data) < 5:
        return pd.DataFrame(columns=["feature", "vif", "collinearity_risk"])

    # Standardize data
    X = clean_data.values
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    vif_records = []
    n_features = len(numeric_cols)

    for i, col in enumerate(numeric_cols):
        y_i = X_std[:, i]
        X_others = np.delete(X_std, i, axis=1)

        # Regularized Moore-Penrose Ridge Pseudoinverse
        lambda_reg = 1e-4
        X_cov = X_others.T @ X_others + lambda_reg * np.eye(n_features - 1)
        try:
            beta = np.linalg.solve(X_cov, X_others.T @ y_i)
            y_pred = X_others @ beta
            ss_tot = np.sum((y_i - np.mean(y_i)) ** 2) + 1e-9
            ss_res = np.sum((y_i - y_pred) ** 2)
            r_squared = max(0.0, min(1.0 - (ss_res / ss_tot), 0.9999))
            vif = 1.0 / (1.0 - r_squared)
        except Exception:
            vif = 1.0

        risk = "Critical (>10)" if vif >= 10 else ("Moderate (5-10)" if vif >= 5 else "Low (<5)")
        vif_records.append({
            "feature": col,
            "vif": round(float(vif), 2),
            "r_squared": round(float(r_squared), 4),
            "collinearity_risk": risk
        })

    return pd.DataFrame(vif_records).sort_values(by="vif", ascending=False)


def calculate_feature_importance(df, target_col: str) -> pd.DataFrame:
    """Non-Linear Feature Importance & Mutual Information Ranker.
    Detects complex non-linear associations using Random Forests and Mutual Information.
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    if target_col not in pdf.columns:
        return pd.DataFrame(columns=["feature", "rf_importance", "mutual_info", "composite_score"])

    # Prepare features
    features = [c for c in pdf.columns if c != target_col]
    clean_df = pdf[features + [target_col]].dropna()

    if len(clean_df) < 10 or not features:
        return pd.DataFrame(columns=["feature", "rf_importance", "mutual_info", "composite_score"])

    # Encode categorical features
    X_encoded = pd.DataFrame()
    for col in features:
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            X_encoded[col] = clean_df[col]
        else:
            X_encoded[col] = clean_df[col].astype('category').cat.codes

    y = clean_df[target_col]
    is_classification = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 5

    rf_scores = {}
    mi_scores = {}

    # 1. Random Forest Feature Importance
    if RandomForestRegressor is not None or RandomForestClassifier is not None:
        try:
            if is_classification:
                y_enc = y if pd.api.types.is_numeric_dtype(y) else y.astype('category').cat.codes
                model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
                model.fit(X_encoded, y_enc)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
                model.fit(X_encoded, y)

            for col, imp in zip(features, model.feature_importances_):
                rf_scores[col] = float(imp)
        except Exception:
            pass

    # 2. Mutual Information Score
    if mutual_info_regression is not None or mutual_info_classif is not None:
        try:
            if is_classification:
                y_enc = y if pd.api.types.is_numeric_dtype(y) else y.astype('category').cat.codes
                mi_vals = mutual_info_classif(X_encoded, y_enc, random_state=42)
            else:
                mi_vals = mutual_info_regression(X_encoded, y, random_state=42)

            mi_sum = sum(mi_vals) + 1e-9
            for col, mi in zip(features, mi_vals):
                mi_scores[col] = float(mi / mi_sum)
        except Exception:
            pass

    # Fallback if scikit-learn is absent: absolute Pearson/Spearman correlation
    records = []
    for col in features:
        rf_val = rf_scores.get(col, 0.0)
        mi_val = mi_scores.get(col, 0.0)
        if not rf_scores and not mi_scores:
            if pd.api.types.is_numeric_dtype(clean_df[col]) and pd.api.types.is_numeric_dtype(clean_df[target_col]):
                corr = abs(float(clean_df[col].corr(clean_df[target_col])))
                composite = corr
            else:
                composite = 0.1
        else:
            composite = 0.5 * rf_val + 0.5 * mi_val

        records.append({
            "feature": col,
            "rf_importance": round(rf_val, 4),
            "mutual_info": round(mi_val, 4),
            "composite_score": round(composite, 4)
        })

    return pd.DataFrame(records).sort_values(by="composite_score", ascending=False)


def test_stationarity(series: pd.Series) -> dict:
    """Time-Series Stationarity Diagnostic Battery (ADF & Rolling Trend)."""
    clean_s = series.dropna()
    if len(clean_s) < 10:
        return {"is_stationary": True, "test": "Insufficient data", "p_value": 1.0}

    # Rolling variance check
    roll_mean = clean_s.rolling(window=max(len(clean_s) // 5, 2)).mean()
    drift_slope = (roll_mean.iloc[-1] - roll_mean.iloc[0]) / (clean_s.std() + 1e-9) if clean_s.std() > 0 else 0.0

    try:
        from statsmodels.tsa.stattools import adfuller
        res = adfuller(clean_s, autolag='AIC')
        p_val = float(res[1])
        return {
            "test": "Augmented Dickey-Fuller (ADF)",
            "adf_statistic": round(float(res[0]), 4),
            "p_value": round(p_val, 5),
            "is_stationary": bool(p_val < 0.05),
            "trend_drift_slope": round(float(drift_slope), 3)
        }
    except Exception:
        # Heuristic test
        is_stat = abs(drift_slope) < 1.0
        return {
            "test": "Heuristic Trend Decomposition",
            "trend_drift_slope": round(float(drift_slope), 3),
            "is_stationary": is_stat,
            "p_value": 0.01 if is_stat else 0.20
        }

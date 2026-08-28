import argparse
import ast
import builtins
import datetime
import difflib
import gc
import json
import os
import re
import shlex
import sys
import time
import traceback
import urllib.error
import urllib.request
import httpx
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.utils import io as ipy_io
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

FLAGS = [
    "-x", "--exec", "--execute",
    "--target",
    "--audit-only",
    "--privacy",
    "--persona",
    "--context",
    "--critic",
    "--critic-pro",
    "--preview",
    "--diff",
    "--guard",
    "--stress",
    "--meta",
    "--simulate",
    "--spark",
    "--roadmap",
    "--kickstart",
    "--interview",
    "--brainstorm",
    "--radar",
    "--dag",
    "--gui",
    "--history",
    "--next",
    "--auto-clean",
    "--spawn",
    "-u", "--unravel",
    "-s", "--sql",
    "-f", "--feat",
    "-v", "--viz",
    "--save",
    "-p", "--profile",
    "-t", "--stat",
    "-m", "--ml",
    "--validate",
    "--tune",
    "--explain",
    "-i", "--insight",
    "-r", "--repair",
    "--pro",
    "--flash",
    "--think",
    "-d", "--deterministic",
    "--fast",
    "--deep",
    "--ultra",
    "--undo",
    "--toggle",
    "-c", "--continue",
    "--retries",
    "--status"
]

def deepanalyze_completer(self, event):
    """Provides auto-complete suggestions for %deepanalyze flags."""
    return [flag for flag in FLAGS if flag.startswith(event.symbol)]

# --- UNIVERSAL POLYMORPHIC ADAPTER ---
try:
    import polars as pl
    from polars.expr.string import ExprStringNameSpace
    from polars.series.string import StringNameSpace
except ImportError:
    pl = None

# Ensure startup directory is in sys.path for privacy module imports
from .privacy_knife import DeepAnalyzePrivacyKnife, LocalGatekeeper

try:
    import duckdb
    _DUCKDB_CON = duckdb.connect(database=":memory:")
except ImportError:
    duckdb = None
    _DUCKDB_CON = None

_DF_SNAPSHOTS = {}
_DF_SNAPSHOT_METADATA = {}
_ACTIVE_ROADMAP = {"phase": 1, "goal": None, "hypotheses": []}
_INTERCEPTOR_ACTIVE = False
_LAST_GENERATED_CODE = ""
_LAST_USER_PROMPT = ""
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"

__version__ = "2.1.0"



# --- DEEPSEEK CLOUD CONFIGURATION ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

def _get_client(is_deepseek=False):
    """Dynamic Client Router: Local Server vs DeepSeek Cloud"""
    if is_deepseek:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY! Run `import os; os.environ['DEEPSEEK_API_KEY'] = 'sk-...'` in your session.")
        return OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(180.0, connect=10.0))
        )
    return OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="none",
        http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(180.0, connect=10.0))
    )

KNOWN_GLOBAL_SYMBOLS = {
    "pd", "np", "pl", "plt", "sns", "duckdb", "scipy", "stats", "sklearn",
    "math", "os", "sys", "re", "json", "datetime", "warnings", "difflib",
    "con", "_DUCKDB_CON", "True", "False", "None"
} | set(dir(builtins))

TRANSIENT_VARS = {
    "X", "y", "X_train", "X_test", "y_train", "y_test", "y_pred",
    "result", "mean_pp", "group_normal", "group_htn", "stat", "p_value",
    "column_summary", "null_percentage", "executive_summary", "data_health_audit", "strategic_roadmap",
    "records", "item", "last_item", "df_flat", "clean_df"
}

SKILL_RULEBOOKS = {
    "general": (
        "[GENERAL CODING RULES - MAXIMUM RELIABILITY]:\n"
        "1. ENGINE DETECTION: Check active environment context to identify whether the target DataFrame is Pandas or Polars.\n"
        "2. PANDAS IN-PLACE MUTATION & ALIASING:\n"
        "   - Mutate columns directly: `df['col'] = ...` or assign filtered subsets `df = df[df['col'] > 0]`.\n"
        "   - NEVER trigger SettingWithCopyWarning; use `.loc[:, 'col'] = ...` or assign cleanly.\n"
        "   - NEVER use deprecated methods: `.append()` (use `pd.concat`), `.applymap()` (use `.map()` or `.apply()`), `inplace=True` on slices.\n"
        "   - For unflattened/hierarchical reports or dictionaries, assign `df = pd.DataFrame(records)`.\n"
        "3. POLARS IDIOMATIC TRANSFORMATIONS:\n"
        "   - Use explicit Polars expressions and assign back to the target variable: `df = df.with_columns(pl.col('a') * 2)`.\n"
        "   - Use `.group_by(...)` instead of deprecated `.groupby(...)`.\n"
        "4. DEFENSIVE NUMERICAL CASTING & ZERO-DIVISION:\n"
        "   - Pandas: `pd.to_numeric(df['col'].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0.0)`.\n"
        "   - Protect divisions: `np.where(denom == 0, 0.0, num / denom)`.\n"
        "5. STRICT SYNTAX FORMAT: Output ONLY executable Python code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "unravel": (
        "[HIERARCHICAL ERP REPORT FLATTENING RULEBOOK]:\n"
        "You parse messy, unstructured multi-row ERP accounting exports into clean 2D tabular DataFrames.\n"
        "Layout & Defensive Parsing Directives:\n"
        "1. POSITIONAL ACCESS & STRING CASTER:\n"
        "   - Inside `for _, row in df.iterrows():`, define `c` strictly on `row`:\n"
        "     `c = lambda idx: str(row.iloc[idx]).strip() if len(row) > idx and pd.notna(row.iloc[idx]) and str(row.iloc[idx]).lower() != 'nan' else ''`\n"
        "   - NEVER use `df.iloc[idx]` inside `c()`.\n"
        "2. HORIZONTAL HEADER PARSING: Header rows contain multiple key-value pairs horizontally across columns:\n"
        "   `if c(0).startswith('Doc. No') or c(0) == 'Doc. No': active_doc = {'doc_no': c(2), 'doc_date': c(4), 'customer': c(6)}`\n"
        "3. DETAIL LINE ITEMS: Detect line items with `if c(0).isdigit() and c(1):`. Merge with `active_doc`, append dictionary to `records`, and set `last_item = item`.\n"
        "4. NUMERIC CASTING: Use `float(re.sub(r'[^0-9.-]', '', c(idx))) if re.sub(r'[^0-9.-]', '', c(idx)) else 0.0`.\n"
        "5. WRAPPED TRAILING TEXT: If `not c(0) and c(2) and last_item is not None and c(2).startswith('-')`, append cleanly to the PRECEDING item:\n"
        "   `last_item['description'] += ' ' + c(2).lstrip('- ').strip()`\n"
        "6. SUMMARY ROW TERMINATION: Stop parsing at totals (`if any(k in c(0).lower() for k in ['grand total', 'total']): break`).\n"
        "7. FINAL VARIABLE ASSIGNMENT: Assign `df = pd.DataFrame(records)` and cast numeric/date columns appropriately.\n"
        "8. Output ONLY executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "feature": (
        "[FEATURE ENGINEERING & WRANGLING RULEBOOK]:\n"
        "1. ENGINE-SPECIFIC TRANSFORMATIONS: Check if the engine is Polars or Pandas and transform directly.\n"
        "2. PANDAS SAFE CLEANING & CASTING:\n"
        "   - Numbers: `pd.to_numeric(df['col'].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0.0)`.\n"
        "   - Dates: `pd.to_datetime(df['date_col'], errors='coerce')` -> extract `.dt.year`, `.dt.month`, `.dt.day_name()`, `.dt.is_weekend`.\n"
        "   - Strings: `df['str_col'] = df['str_col'].astype(str).str.strip().str.lower()`.\n"
        "   - Zero-Division Protection: `np.where(denom == 0, 0.0, num / denom)`.\n"
        "3. POLARS SAFE CLEANING & CASTING:\n"
        "   - Numbers: `pl.col('col').cast(pl.Utf8, strict=False).str.replace_all(r'[^0-9.-]', '').cast(pl.Float64, strict=False).fill_null(0.0)`.\n"
        "   - Zero-Division Protection: `pl.when(pl.col('denom') == 0).then(0.0).otherwise(pl.col('num') / pl.col('denom'))`.\n"
        "   - Strings: `pl.col('str_col').str.strip_chars().str.to_lowercase()`.\n"
        "4. CATEGORICAL & OUTLIER ENCODING:\n"
        "   - Use frequency encoding or one-hot encoding with `pd.get_dummies(..., drop_first=True, dtype=int)`.\n"
        "   - Cap extreme outliers using IQR or percentile clipping (`.clip(lower, upper)`).\n"
        "5. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "sql": (
        "[DUCKDB SQL RULEBOOK]:\n"
        "1. IN-MEMORY QUERYING: Query Pandas and Polars DataFrames directly in DuckDB without opening manual file connections.\n"
        "   Example: `df = duckdb.sql('SELECT customer, SUM(total) AS total_revenue FROM df GROUP BY customer ORDER BY total_revenue DESC').df()`\n"
        "2. PROPER SQL IDENTIFIER & LITERAL QUOTING:\n"
        "   - Columns with spaces/special characters MUST be enclosed in double quotes: `\"Column Name\"`.\n"
        "   - String literals MUST be enclosed in single quotes: `'Completed'`.\n"
        "3. ADVANCED ANALYTICAL SQL: Leverage window functions (`ROW_NUMBER() OVER (...)`, `LAG()`, `LEAD()`), `FILTER (WHERE ...)`, and CTEs (`WITH ... AS (...)`).\n"
        "4. TARGET ASSIGNMENT: Always assign the resulting DataFrame back to the target variable (e.g. `df = ...`).\n"
        "5. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "viz": (
        "[SEABORN / MATPLOTLIB VISUALIZATION RULEBOOK]:\n"
        "1. MODERN STYLING: Set `sns.set_theme(style='whitegrid', font_scale=1.1)` and `fig, ax = plt.subplots(figsize=(10, 6))`.\n"
        "2. PRE-AGGREGATION & DATA PREP: Sort categories and calculate aggregates before passing to plotting functions for clean visual hierarchy.\n"
        "3. COLOR PALETTES: Use curated Seaborn palettes (e.g., `palette='mako'`, `'crest'`, `'viridis'`, or `'blend:#38bdf8,#6366f1'`).\n"
        "4. ANNOTATIONS & LABELS:\n"
        "   - Add descriptive `ax.set_title('...', fontsize=14, fontweight='bold', pad=12)`.\n"
        "   - Add clean axis labels `ax.set_xlabel('...')` and `ax.set_ylabel('...')`.\n"
        "   - Add data labels / value annotations on bars or points when useful.\n"
        "5. LAYOUT & RENDERING: Always call `plt.tight_layout()` before `plt.show()`.\n"
        "6. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "stat": (
        "[SCIPY / STATISTICAL TESTING RULEBOOK]:\n"
        "1. ASSUMPTION TESTING: Test normality (`scipy.stats.shapiro` or `scipy.stats.normaltest`) and variance homogeneity (`scipy.stats.levene`) before choosing parametric tests.\n"
        "2. RIGOROUS HYPOTHESIS TESTING:\n"
        "   - 2-Sample Continuous: `scipy.stats.ttest_ind` (parametric) or `scipy.stats.mannwhitneyu` (non-parametric).\n"
        "   - Multi-Sample Continuous: `scipy.stats.f_oneway` (ANOVA) or `scipy.stats.kruskal` (Kruskal-Wallis).\n"
        "   - Categorical Independence: `scipy.stats.chi2_contingency` with contingency table.\n"
        "   - Correlation: `scipy.stats.pearsonr` (linear) or `scipy.stats.spearmanr` (monotonic / non-normal).\n"
        "3. STRUCTURED OUTPUT: Print test statistic, p-value, effect size, and actionable interpretation (`Significant (p < 0.05)` vs `Not Significant`).\n"
        "4. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "ml": (
        "[SCIKIT-LEARN MACHINE LEARNING RULEBOOK]:\n"
        "1. TARGET & FEATURE SEPARATION: Extract `X = df[feature_cols]` and `y = df[target_col]`.\n"
        "2. LEAK-FREE SPLIT: Split before preprocessing using `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None)`.\n"
        "3. PRODUCTION-GRADE PIPELINE:\n"
        "   - Bundle preprocessing via `ColumnTransformer` (e.g. `StandardScaler` for numeric, `OneHotEncoder(handle_unknown='ignore')` for categorical).\n"
        "   - Wrap transformer and model in a `Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])`.\n"
        "4. FIT, PREDICT & EVALUATE:\n"
        "   - Fit pipeline strictly on training data `pipeline.fit(X_train, y_train)`.\n"
        "   - Predict on test set `y_pred = pipeline.predict(X_test)`.\n"
        "   - Print comprehensive evaluation (`classification_report(y_test, y_pred, zero_division=0)` or `mean_squared_error`, `r2_score`).\n"
        "5. ASSERTIONS: Assert `not pd.isna(y_pred).any(), 'NaN in predictions'` and `len(y_pred) == len(y_test)`.\n"
        "6. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "profile": (
        "[STRATEGIC DATASET PROFILING RULEBOOK]:\n"
        "1. EXECUTIVE NARRATIVE:\n"
        "   - Synthesize Executive Abstract, Data Health Audit, and Prioritized Strategic Action Items.\n"
        "2. FAST IN-MEMORY DIAGNOSTIC CODE:\n"
        "   - Compute total rows, columns, memory usage, duplicate row count.\n"
        "   - Calculate column-by-column null count and null percentage.\n"
        "   - Identify constant / zero-variance columns and high-cardinality ID columns.\n"
        "   - Print summary table using clean formatting.\n"
        "3. Output diagnostic code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "repair": (
        "[AUTONOMOUS STATIC & RUNTIME REPAIR RULEBOOK]:\n"
        "1. ERROR ROOT-CAUSE REFLECTION:\n"
        "   - Analyze the syntax error, NameError, KeyError, TypeError, or ZeroDivisionError from the traceback.\n"
        "   - Verify exact column names, casings, and active variables from the workspace context.\n"
        "2. DEFENSIVE PATCHING: Wrap unsafe operations in `try-except`, replace missing column names with verified matches, and protect divisions.\n"
        "3. Output ONLY corrected, runnable Python code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "validate": (
        "[RIGOROUS STATISTICAL & ML VALIDATION RULEBOOK]:\n"
        "1. CROSS-VALIDATION: Use `StratifiedKFold` (classification) or `KFold` (regression) with 5 splits.\n"
        "2. CROSS-VAL SCORE EVALUATION: Use `cross_val_score(pipeline, X, y, cv=cv, scoring=metric)` and print mean score +/- standard deviation.\n"
        "3. STRUCTURAL CHECKS: Check for class imbalance, multicollinearity (VIF or correlation matrix), and prediction out-of-bounds.\n"
        "4. STRICT INVARIANT: Do NOT assert arbitrary minimum metric scores (e.g. `assert score > 0.90`) unless explicitly requested by the user.\n"
        "5. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "tune": (
        "[LEAK-FREE HYPERPARAMETER TUNING & PIPELINE RULEBOOK]:\n"
        "1. DATA EXTRACTION: Extract `X = df[feature_cols]` and `y = df[target_col]` from the active DataFrame.\n"
        "2. FULL PIPELINE SEARCH: Place the entire pipeline inside `GridSearchCV` or `RandomizedSearchCV` to prevent preprocessing data leakage across folds.\n"
        "3. FIT BEFORE ACCESS: Always execute `grid_search.fit(X, y)` before accessing `grid_search.best_params_` or `grid_search.best_score_`.\n"
        "4. REPORTING: Print `Best Parameters:` and `Best Cross-Validation Score:` formatted clearly.\n"
        "5. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "explain": (
        "[MODEL INTERPRETABILITY & EXPLAINABILITY RULEBOOK]:\n"
        "1. FEATURE IMPORTANCE EXTRACTION:\n"
        "   - Extract feature names from preprocessor using `preprocessor.get_feature_names_out()`.\n"
        "   - Extract weights from model: `model.feature_importances_` (tree models) or `model.coef_` (linear models).\n"
        "2. RANKED SUMMARY DATAFRAME:\n"
        "   - Create `importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)`.\n"
        "   - Print the top 10 most influential features with percentage contributions.\n"
        "3. STRUCTURAL ASSERTION: Assert `len(importance_df) > 0` and `not importance_df['importance'].isna().any()`.\n"
        "4. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "polars": (
        "[POLARS SEQUENTIAL RULEBOOK - CRITICAL]:\n"
        "1. SEQUENTIAL PIPELINE: NEVER chain everything into a single massive unreadable expression. Use separate assignment steps for clarity and instant type reflection.\n"
        "2. STEP-BY-STEP RECIPE:\n"
        "   - Step 1 (String cleanup): `df = df.with_columns(Client_Code=pl.col('Client_Code').str.strip_chars().str.to_uppercase(), Status=pl.col('Status').str.strip_chars().str.to_uppercase())`\n"
        "   - Step 2 (Parse Numbers): `df = df.with_columns(Gross_Amount=pl.col('Gross_Amount').cast(pl.Utf8, strict=False).str.replace_all(r'[\\$,]', '').cast(pl.Float64, strict=False).fill_null(0.0))`\n"
        "   - Step 3 (Parse Percentages): `df = df.with_columns(Discount_Pct=pl.col('Discount_Pct').cast(pl.Utf8, strict=False).str.replace_all('%', '').cast(pl.Float64, strict=False).fill_null(0.0) / 100.0)`\n"
        "   - Step 4 (Calculate Metrics): `df = df.with_columns(Net_Amount=pl.col('Gross_Amount') * (1.0 - pl.col('Discount_Pct')))`\n"
        "   - Step 5 (Filter & Sort): `df = df.filter((pl.col('Status') == 'COMPLETED')).sort('Net_Amount', descending=True)`\n"
        "3. TARGET ASSIGNMENT: Assign the final cleaned DataFrame back to the target variable.\n"
        "4. Output executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
}

INVARIANT_CHECKLIST = (
    "\n[STRICT INVARIANT DIRECTIVE]:\n"
    "1. NEVER redefine, instantiate, or hardcode sample data (e.g. NEVER write `data = [...]` or `df = pd.DataFrame(...)` for existing tables).\n"
    "2. Assume the target dataframe is already loaded in runtime. Transform it directly.\n"
    "3. DO NOT output conversational preamble or explanations outside code tags.\n"
    "4. Enclose code strictly inside <Answer>```python ... ```</Answer>.\n"
)

def check_engine_status(server_url=DEFAULT_SERVER_URL):
    """Probes the llama-server health/props endpoints and kernel interceptor state."""
    print("=" * 60)
    print("🔍 DeepAnalyze System & Engine Status")
    print("=" * 60)

    health_url = f"{server_url}/health"
    server_online = False

    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "DeepAnalyze-Client"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                server_online = True
    except Exception:
        pass

    print(f"{'✅' if server_online else '❌'} Local Server Status : {'Online & Healthy' if server_online else 'Offline / Unreachable'}")
    print(f"☁️ DeepSeek API Auth  : {'Configured' if DEEPSEEK_API_KEY.startswith('sk-') else 'Missing API Key'}")
    
    interceptor_status = "🟢 Enabled (Auto-pilot on plain English cells)" if _INTERCEPTOR_ACTIVE else "⚪ Disabled (Explicit %deepanalyze calls only)"
    print(f"📡 Cell Interceptor   : {interceptor_status}")
    print(f"⚡ Polars Acceleration : {'✅ Active' if pl is not None else '⚪ Not Installed'}")

    ip = get_ipython()
    tracked_dfs = list(_DF_SNAPSHOTS.keys())
    if tracked_dfs:
        print(f"\n💾 State Snapshots    : {len(tracked_dfs)} active DataFrame rollback points")
        for var_name in tracked_dfs:
            shape = getattr(ip.user_ns.get(var_name), "shape", "Unknown shape")
            print(f"   • {var_name} -> {shape}")
    else:
        print("\n💾 State Snapshots    : No active snapshots (clean state)")

    print("=" * 60)

def _apply_polars_compat_shim():
    """Universal runtime shim for legacy Polars & Pandas string methods generated by 8B models."""
    if pl is None:
        return  # Skip shim if Polars is not installed on this system

    try:
        # 1. Alias stripped/cased methods
        for ns in (ExprStringNameSpace, StringNameSpace):
            if not hasattr(ns, "strip") and hasattr(ns, "strip_chars"):
                ns.strip = ns.strip_chars
            if not hasattr(ns, "upper") and hasattr(ns, "to_uppercase"):
                ns.upper = ns.to_uppercase
            if not hasattr(ns, "lower") and hasattr(ns, "to_lowercase"):
                ns.lower = ns.to_lowercase

        # 2. Tolerant .replace() wrapper (handles positional booleans and regex=False)
        _orig_expr_rep = ExprStringNameSpace.replace
        _orig_series_rep = StringNameSpace.replace

        def _universal_expr_replace(self, pattern, value="", *args, **kwargs):
            if args and isinstance(args[0], bool):
                kwargs["literal"] = args[0]
                args = args[1:]
            if kwargs.pop("regex", None) is False:
                kwargs["literal"] = True
            return _orig_expr_rep(self, pattern, value, *args, **kwargs)

        def _universal_series_replace(self, pattern, value="", *args, **kwargs):
            if args and isinstance(args[0], bool):
                kwargs["literal"] = args[0]
                args = args[1:]
            if kwargs.pop("regex", None) is False:
                kwargs["literal"] = True
            return _orig_series_rep(self, pattern, value, *args, **kwargs)

        ExprStringNameSpace.replace = _universal_expr_replace
        StringNameSpace.replace = _universal_series_replace

    except Exception:
        pass


def load_ipython_extension(ipython):
    # 1. Apply Polars compatibility patches
    _apply_polars_compat_shim()
    
    # 2. Register the %deepanalyze magic command
    ipython.register_magic_function(deepanalyze, magic_kind='line')
    
    # 3. Register the auto-pilot interceptor (for --toggle)
    ipython.set_hook('input_prefilter', deepanalyze_interceptor)
    
    # 4. Set auto-complete suggestions for the flags
    ipython.set_hook('complete_command', deepanalyze_completer, str_key='%deepanalyze')

def _take_snapshot(ip, target="df"):
    global _DF_SNAPSHOTS, _DF_SNAPSHOT_METADATA
    if ip and target in ip.user_ns:
        obj = ip.user_ns[target]
        meta_entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "shape": getattr(obj, "shape", (0, 0)),
            "cols": list(obj.columns) if hasattr(obj, "columns") else []
        }
        if isinstance(obj, pd.DataFrame):
            _DF_SNAPSHOTS[target] = obj.copy(deep=True)
            _DF_SNAPSHOT_METADATA.setdefault(target, []).append(meta_entry)
        elif pl is not None and isinstance(obj, pl.DataFrame):
            _DF_SNAPSHOTS[target] = obj.clone()
            _DF_SNAPSHOT_METADATA.setdefault(target, []).append(meta_entry)

def _restore_snapshot(ip, target="df") -> bool:
    global _DF_SNAPSHOTS
    if ip and target in _DF_SNAPSHOTS and _DF_SNAPSHOTS[target] is not None:
        snap = _DF_SNAPSHOTS[target]
        if isinstance(snap, pd.DataFrame):
            ip.user_ns[target] = snap.copy(deep=True)
        elif pl is not None and isinstance(snap, pl.DataFrame):
            ip.user_ns[target] = snap.clone()
        return True
    return False

def _reconcile_target_dataframe(ip, code_str: str, prompt: str, default_target: str = "df"):
    """Parses executed AST to extract the last assigned DataFrame variable
    and automatically binds it to the designated session target variable.
    """
    if not ip or not code_str:
        return
    try:
        tree = ast.parse(code_str)
        assigned_vars = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.append(target.id)
        if assigned_vars:
            last_var = assigned_vars[-1]
            if last_var in ip.user_ns:
                val = ip.user_ns[last_var]
                if isinstance(val, pd.DataFrame) or (pl is not None and isinstance(val, pl.DataFrame)):
                    match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=", prompt)
                    target_var = match.group(1) if match else default_target
                    ip.user_ns[target_var] = val
    except Exception:
        pass


def _sync_duckdb(ip):
    if duckdb is None or not ip:
        return
    for k, v in ip.user_ns.items():
        if not k.startswith("_") and k not in TRANSIENT_VARS:
            if isinstance(v, pd.DataFrame) or (pl is not None and isinstance(v, pl.DataFrame)):
                try:
                    duckdb.register(k, v)
                except Exception:
                    pass

def _fuzzy_match_columns(prompt: str, ip, target="df") -> str:
    if not ip or target not in ip.user_ns:
        return ""
    
    obj = ip.user_ns[target]
    cols = []
    if isinstance(obj, pd.DataFrame):
        cols = [str(c) for c in obj.columns]
    elif pl is not None and isinstance(obj, pl.DataFrame):
        cols = obj.columns
    else:
        return ""

    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", prompt)
    
    matches = []
    for t in tokens:
        t_lower = t.lower()
        if t in cols:
            continue
        close = difflib.get_close_matches(t_lower, [c.lower() for c in cols], n=1, cutoff=0.85)
        if close:
            orig = next(c for c in cols if c.lower() == close[0])
            if orig != t and not t.lower().startswith("new_") and not t.lower().startswith("mean_"):
                matches.append(f"  - Typo '{t}' -> EXACT Column: `{orig}`")

    if matches:
        return "\n--- PRE-FLIGHT COLUMN RESOLUTION ALIASES ---\n" + "\n".join(set(matches)) + "\n---------------------------------------------\n"
    return ""

def _get_deep_workspace_context(ip, target="df", is_cloud=False, privacy_mode="auto") -> tuple[str, set, object]:
    if not ip:
        return "", set(KNOWN_GLOBAL_SYMBOLS), None

    available_vars = set(KNOWN_GLOBAL_SYMBOLS)
    context_lines = []
    knife_instance = None

    for name, obj in ip.user_ns.items():
        if name.startswith("_") or name in ("In", "Out", "exit", "quit", "get_ipython"):
            continue

        available_vars.add(name)
        if name in TRANSIENT_VARS:
            continue

        is_pandas = isinstance(obj, pd.DataFrame)
        is_polars = pl is not None and isinstance(obj, pl.DataFrame)

        if is_pandas or is_polars:
            engine_name = "Polars" if is_polars else "Pandas"

            # APPLY PRIVACY MASKS IF ROUTED TO CLOUD OR FORCED VIA FLAG
            if is_cloud or privacy_mode != "none":
                strategy_override = None
                if privacy_mode == "mask": strategy_override = "ERP_STRUCTURAL_MASK"
                elif privacy_mode == "mock": strategy_override = "PII_DEIDENTIFIED_MOCK"
                elif privacy_mode == "profile": strategy_override = "STANDARD_STATISTICAL_PROFILE"

                safe_payload, knife_instance = LocalGatekeeper.generate_safe_payload(obj, custom_strategy=strategy_override)
                context_lines.append(
                    f"DataFrame `{name}` (Engine: {engine_name}) [PRIVACY-PRESERVED CONTEXT - NO RAW DATA]:\n"
                    f"{json.dumps(safe_payload, indent=2)}"
                )
            else:
                # LOCAL RAW PREVIEW - Universal Adapter Logic
                if is_polars:
                    shape_0, shape_1 = obj.shape
                    if shape_0 < 20:
                        grid_sample = str(obj.head(18))
                        context_lines.append(
                            f"DataFrame `{name}` (Engine: Polars | Raw Grid Matrix, Shape: {shape_0} rows, {shape_1} cols):\n"
                            f"Top 18 Rows Matrix Preview:\n{grid_sample}"
                        )
                    else:
                        col_profiles = []
                        null_counts = obj.null_count().row(0)
                        for idx, col in enumerate(obj.columns):
                            dtype = str(obj.schema[col])
                            null_pct = round((null_counts[idx] / shape_0) * 100, 1) if shape_0 > 0 else 0.0
                            unique_count = obj[col].n_unique()
                            sample = str(obj[col].drop_nulls()[0]) if obj[col].drop_nulls().len() > 0 else "None"
                            col_profiles.append(f"    - '{col}' ({dtype}) | Nulls: {null_pct}% | Unique: {unique_count} | Sample: {sample}")

                        context_lines.append(
                            f"DataFrame `{name}` (Engine: Polars | Shape: {shape_0} rows, {shape_1} cols):\n"
                            f"  Exact Column Names (CASE-SENSITIVE):\n" + "\n".join(col_profiles)
                        )
                elif is_pandas:
                    is_unnamed = any("unnamed" in str(c).lower() or isinstance(c, (int, np.integer)) for c in obj.columns)
                    if is_unnamed or obj.shape[0] < 20:
                        grid_sample = obj.iloc[:18, :12].to_string()
                        context_lines.append(
                            f"DataFrame `{name}` (Engine: Pandas | Raw Grid Matrix, Shape: {obj.shape[0]} rows, {obj.shape[1]} cols):\n"
                            f"Top 18 Rows Matrix Preview:\n{grid_sample}"
                        )
                    else:
                        col_profiles = []
                        for col in obj.columns:
                            col_str = str(col)
                            dtype = str(obj[col].dtype)
                            null_pct = round(obj[col].isna().mean() * 100, 1)
                            unique_count = obj[col].nunique()
                            sample = obj[col].dropna().iloc[0] if not obj[col].dropna().empty else "None"
                            col_profiles.append(f"    - '{col_str}' ({dtype}) | Nulls: {null_pct}% | Unique: {unique_count} | Sample: {sample}")

                        context_lines.append(
                            f"DataFrame `{name}` (Engine: Pandas | Shape: {obj.shape[0]} rows, {obj.shape[1]} cols):\n"
                            f"  Exact Column Names (CASE-SENSITIVE):\n" + "\n".join(col_profiles)
                        )

    context_str = (
        "\n--- ACTIVE RUNTIME ENVIRONMENT CONTEXT ---\n"
        + ("\n\n".join(context_lines) if context_lines else "No custom DataFrames loaded.")
        + "\n-------------------------------------------\n"
    )
    return context_str, available_vars, knife_instance

def _lint_and_format_code(code_str: str, available_vars: set) -> tuple[bool, str, str]:
    if not code_str.strip():
        return False, "", "Empty code block"
    try:
        # 🛡️ AST SECURE SANDBOX ENFORCEMENT
        DeepAnalyzePrivacyKnife.audit_generated_code(code_str)
        tree = ast.parse(code_str)
        normalized_code = ast.unparse(tree)
    except PermissionError as pe:
        return False, code_str, str(pe)
    except (SyntaxError, IndentationError) as e:
        return False, code_str, f"Syntax Error on line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, code_str, f"AST Parsing Error: {str(e)}"

    defined_in_code = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined_in_code.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_in_code.add(node.name)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                defined_in_code.add(arg.arg)
            if node.args.vararg:
                defined_in_code.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_in_code.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                defined_in_code.add(arg.arg)
            if node.args.vararg:
                defined_in_code.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_in_code.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined_in_code.add(node.id)

    undefined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            var_name = node.id
            if var_name not in available_vars and var_name not in defined_in_code:
                undefined.add(var_name)

    if undefined:
        return False, normalized_code, f"Undefined variable(s) referenced: {sorted(list(undefined))}"

    return True, normalized_code, ""

def _sanitize_traceback(tb_str: str, max_lines: int = 25) -> str:
    lines = [line for line in tb_str.splitlines() if not ("FutureWarning:" in line or "UserWarning:" in line)]
    return "\n".join(lines[-max_lines:])

def _extract_deepanalyze_content(text: str) -> tuple[str, str]:
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    raw_blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)(?:```|$)", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    code = raw_blocks[0].strip() if raw_blocks else ""

    if not code:
        answer_match = re.search(r"<Answer>(.*?)(?:</Answer>|$)", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        if answer_match:
            code = answer_match.group(1).strip()
            code = re.sub(r"```(?:python|py)?", "", code).replace("```", "").strip()

    if not code:
        lines = [l for l in cleaned_text.splitlines() if not l.startswith(('Root Cause:', 'Safe Strategy:', '1.', '2.', '3.'))]
        potential_code = "\n".join(lines).strip()
        if any(keyword in potential_code for keyword in ('import ', 'def ', '=', 'pd.', 'pl.', 'plt.', 'Pipeline', 'GridSearchCV')):
            code = potential_code

    narrative = re.sub(r"<Analyze>.*?(?:</Analyze>|$)", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    narrative = re.sub(r"</?(?:Analyze|Answer|Code)>", "", narrative, flags=re.IGNORECASE)
    narrative = re.sub(r"```(?:python|py|sql)?\s*\n?.*?(?:```|$)", "", narrative, flags=re.DOTALL | re.IGNORECASE).strip()

    return code, narrative

def _call_llm(prompt: str, system_prompt: str, temp: float = 0.0, max_tokens: int = 3500, target_model: str = "deepanalyze-8b") -> str:
    is_ds = target_model.startswith("deepseek")
    engine_name = "☁️ DeepSeek Cloud" if is_ds else "💻 Local Engine"
    
    sys.stdout.write(f"\r[1/3] 🔍 Routing ➔ [2/3] ⚡ Generating Code [{engine_name} ({target_model})]...")
    sys.stdout.flush()

    client = _get_client(is_deepseek=is_ds)
    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        stream=True,
    )

    full_text = []
    token_count = 0
    for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            full_text.append(delta)
            token_count += 1
            if token_count % 25 == 0:
                sys.stdout.write(f"\r[1/3] 🔍 Routing ➔ [2/3] ⚡ Streaming ({token_count} tok) ➔ [3/3] 🛡️ Validating...")
                sys.stdout.flush()

    sys.stdout.write("\r" + " " * 85 + "\r")
    sys.stdout.flush()
    return "".join(full_text)


def _classify_intent(prompt: str, target_model: str = "deepanalyze-8b") -> str:
    """Autonomously classifies the user data request into a specialized skill category."""
    sys_prompt = "Classify the user data request into exactly one category: [sql, viz, ml, stat, unravel, general]. Output ONLY the category word."
    try:
        raw_res = _call_llm(prompt, sys_prompt, temp=0.0, max_tokens=20, target_model=target_model)
        cleaned = re.sub(r'<think>.*?</think>', '', raw_res, flags=re.DOTALL | re.IGNORECASE).strip().lower()
        for cat in ["unravel", "sql", "viz", "ml", "stat", "general"]:
            if cat in cleaned:
                return cat
    except Exception:
        pass
    return "general"


def _generate_sparkline(series, bins: int = 8) -> str:
    """Generates an 8-level ASCII sparkline ( ▂▃▄▅▆▇█) representing numeric distribution."""
    ticks = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    try:
        if hasattr(series, "to_numpy"):
            vals = series.to_numpy()
        elif hasattr(series, "to_list"):
            vals = np.array(series.to_list())
        else:
            vals = np.array(series)

        vals = vals[~pd.isna(vals)] if hasattr(pd, "isna") else vals[~np.isnan(vals)]
        vals = vals.astype(float)
        if len(vals) == 0:
            return "—"
        if np.all(vals == vals[0]):
            return "▄▄▄▄▄▄▄▄"

        counts, _ = np.histogram(vals, bins=bins)
        max_c = np.max(counts)
        if max_c == 0:
            return " " * bins

        return "".join(ticks[min(int((c / max_c) * (len(ticks) - 1)), len(ticks) - 1)] for c in counts)
    except Exception:
        return "—"


def _render_sparkline_minimap(df_obj, target_name: str = "df"):
    """Renders a Rich table containing ASCII sparkline distributions for numeric columns."""
    if df_obj is None:
        return

    is_pandas = isinstance(df_obj, pd.DataFrame)
    is_polars = pl is not None and isinstance(df_obj, pl.DataFrame)

    if not is_pandas and not is_polars:
        return

    cols = list(df_obj.columns)
    table = Table(
        title=f"📈 [bold green]DeepAnalyze Sparkline Minimaps: `{target_name}`[/bold green]",
        header_style="bold green",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Column", style="cyan")
    table.add_column("Min", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Null %", justify="right")
    table.add_column("Distribution Minimap", style="bold yellow", justify="center")

    found_numeric = False
    for col in cols:
        s = df_obj[col]
        dtype_str = str(s.dtype if is_pandas else df_obj.schema[col]).lower()
        if any(k in dtype_str for k in ("int", "float", "double", "decimal", "numeric")):
            found_numeric = True
            spark = _generate_sparkline(s)
            
            if is_pandas:
                clean_s = s.dropna()
                c_min = f"{clean_s.min():.2f}" if not clean_s.empty else "N/A"
                c_med = f"{clean_s.median():.2f}" if not clean_s.empty else "N/A"
                c_max = f"{clean_s.max():.2f}" if not clean_s.empty else "N/A"
                null_pct = f"{(s.isna().mean() * 100):.1f}%"
            else:
                clean_s = s.drop_nulls()
                c_min = f"{clean_s.min():.2f}" if len(clean_s) > 0 else "N/A"
                c_med = f"{clean_s.median():.2f}" if len(clean_s) > 0 else "N/A"
                c_max = f"{clean_s.max():.2f}" if len(clean_s) > 0 else "N/A"
                null_pct = f"{((s.null_count() / len(df_obj)) * 100):.1f}%" if len(df_obj) > 0 else "0.0%"

            table.add_row(str(col), c_min, c_med, c_max, null_pct, spark)

    if found_numeric:
        console.print(Panel(table, border_style="green", expand=False))


def _render_state_diff_hud(orig_obj, new_obj, target_name: str = "df"):
    """Renders a Rich side-by-side delta HUD showing row/col changes, dtype mutations, and null drifts."""
    if orig_obj is None or new_obj is None:
        return

    table = Table(
        title=f"📊 [bold cyan]DeepAnalyze State Diff HUD: `{target_name}`[/bold cyan]",
        header_style="bold magenta",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Attribute / Column", style="cyan", justify="left")
    table.add_column("Original State", justify="center")
    table.add_column("Transformed State", justify="center")
    table.add_column("Delta / Mutation", justify="right")

    orig_shape = getattr(orig_obj, "shape", (0, 0))
    new_shape = getattr(new_obj, "shape", (0, 0))
    row_delta = new_shape[0] - orig_shape[0]
    col_delta = new_shape[1] - orig_shape[1]

    row_delta_str = f"[green]+{row_delta}[/green]" if row_delta > 0 else (f"[red]{row_delta}[/red]" if row_delta < 0 else "0")
    col_delta_str = f"[green]+{col_delta}[/green]" if col_delta > 0 else (f"[red]{col_delta}[/red]" if col_delta < 0 else "0")

    table.add_row(
        "Matrix Shape (Rows x Cols)",
        f"{orig_shape[0]:,} x {orig_shape[1]:,}",
        f"{new_shape[0]:,} x {new_shape[1]:,}",
        f"Rows: {row_delta_str}, Cols: {col_delta_str}"
    )

    is_pandas_orig = isinstance(orig_obj, pd.DataFrame)
    is_pandas_new = isinstance(new_obj, pd.DataFrame)
    is_polars_orig = pl is not None and isinstance(orig_obj, pl.DataFrame)
    is_polars_new = pl is not None and isinstance(new_obj, pl.DataFrame)

    orig_cols = list(orig_obj.columns) if hasattr(orig_obj, "columns") else []
    new_cols = list(new_obj.columns) if hasattr(new_obj, "columns") else []

    added_cols = [c for c in new_cols if c not in orig_cols]
    removed_cols = [c for c in orig_cols if c not in new_cols]
    common_cols = [c for c in orig_cols if c in new_cols]

    if added_cols:
        table.add_row("Added Columns", "—", f"{len(added_cols)} cols", f"[green]+ {', '.join(str(c) for c in added_cols[:4])}{'...' if len(added_cols) > 4 else ''}[/green]")
    if removed_cols:
        table.add_row("Removed Columns", f"{len(removed_cols)} cols", "—", f"[red]- {', '.join(str(c) for c in removed_cols[:4])}{'...' if len(removed_cols) > 4 else ''}[/red]")

    for col in common_cols[:8]:
        orig_dtype = str(orig_obj.dtypes[col] if is_pandas_orig else orig_obj.schema[col])
        new_dtype = str(new_obj.dtypes[col] if is_pandas_new else new_obj.schema[col])
        dtype_mut = f"[yellow]{orig_dtype} -> {new_dtype}[/yellow]" if orig_dtype != new_dtype else "[dim]unchanged[/dim]"

        if is_pandas_orig: orig_nulls = int(orig_obj[col].isna().sum())
        elif is_polars_orig: orig_nulls = int(orig_obj[col].null_count())
        else: orig_nulls = 0

        if is_pandas_new: new_nulls = int(new_obj[col].isna().sum())
        elif is_polars_new: new_nulls = int(new_obj[col].null_count())
        else: new_nulls = 0

        null_delta = new_nulls - orig_nulls
        if orig_dtype != new_dtype or null_delta != 0:
            null_delta_str = f"[red]+{null_delta} nulls[/red]" if null_delta > 0 else (f"[green]{null_delta} nulls[/green]" if null_delta < 0 else "0")
            table.add_row(f"Col `{col}`", f"{orig_dtype} ({orig_nulls} nulls)", f"{new_dtype} ({new_nulls} nulls)", f"{dtype_mut} | {null_delta_str}")

    console.print(Panel(table, border_style="cyan", expand=False))


def _render_simulation_hud(orig_obj, sim_obj, scenario: str, target_name: str = "df"):
    """Renders a comparative What-If simulation HUD between baseline and simulated scenario."""
    if orig_obj is None or sim_obj is None:
        return

    table = Table(
        title=f"🔮 [bold magenta]What-If Simulation HUD: `{target_name}`[/bold magenta]\n[dim]Hypothesis: {scenario}[/dim]",
        header_style="bold magenta",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Metric / Column", style="cyan")
    table.add_column("Baseline Value", justify="right")
    table.add_column("Simulated Value", justify="right")
    table.add_column("Impact / Delta", justify="right")

    orig_shape = getattr(orig_obj, "shape", (0, 0))
    sim_shape = getattr(sim_obj, "shape", (0, 0))
    table.add_row("Row Count", f"{orig_shape[0]:,}", f"{sim_shape[0]:,}", f"{sim_shape[0] - orig_shape[0]:+,}")

    is_pandas = isinstance(orig_obj, pd.DataFrame) and isinstance(sim_obj, pd.DataFrame)
    if is_pandas:
        common_num_cols = [c for c in orig_obj.select_dtypes(include=[np.number]).columns if c in sim_obj.columns]
        for c in common_num_cols[:6]:
            base_sum = float(orig_obj[c].sum())
            sim_sum = float(sim_obj[c].sum())
            delta = sim_sum - base_sum
            pct = ((delta / base_sum) * 100) if base_sum != 0 else 0.0
            color = "green" if delta >= 0 else "red"
            table.add_row(
                f"Sum(`{c}`)",
                f"{base_sum:,.2f}",
                f"{sim_sum:,.2f}",
                f"[{color}]{delta:+,.2f} ({pct:+.1f}%)[/{color}]"
            )

    console.print(Panel(table, border_style="magenta", expand=False))


def _evaluate_quality_gate(guard_expr: str, df_obj, var_name: str = "df") -> tuple[bool, str]:
    """Evaluates a quality gate assertion expression against the given DataFrame."""
    if not guard_expr or df_obj is None:
        return True, ""
    try:
        eval_scope = {
            "pd": pd,
            "np": np,
            "df": df_obj,
            var_name: df_obj,
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "all": all,
            "any": any,
        }
        if pl is not None:
            eval_scope["pl"] = pl

        res = eval(guard_expr, {"__builtins__": builtins}, eval_scope)
        if bool(res):
            return True, f"Guard `{guard_expr}` PASSED."
        else:
            return False, f"Quality gate condition `{guard_expr}` evaluated to False."
    except Exception as e:
        return False, f"Quality gate condition `{guard_expr}` threw exception: {type(e).__name__}: {e}"


def _generate_adversarial_df(df_obj):
    """Constructs a 5-row adversarial DataFrame matching the target schema to stress-test generated code."""
    if df_obj is None:
        return None

    is_polars = pl is not None and isinstance(df_obj, pl.DataFrame)
    is_pandas = isinstance(df_obj, pd.DataFrame)

    if not is_pandas and not is_polars:
        return None

    cols = list(df_obj.columns)
    adv_dict = {}

    for c in cols:
        sample_series = df_obj[c]
        dtype_str = str(sample_series.dtype if is_pandas else df_obj.schema[c]).lower()

        if any(k in dtype_str for k in ("int", "float", "double", "decimal", "numeric")):
            adv_dict[c] = [0.0, np.nan, -1.0, 1000000.0, 0.0]
        elif any(k in dtype_str for k in ("date", "time")):
            adv_dict[c] = [None, "1970-01-01", "2099-12-31", None, "2025-01-01"]
        elif "bool" in dtype_str:
            adv_dict[c] = [True, False, None, False, True]
        else:
            adv_dict[c] = ["$0.00", "", np.nan, "   ", "N/A"]

    if is_polars:
        try:
            return pl.DataFrame(adv_dict).cast(df_obj.schema, strict=False)
        except Exception:
            return pl.DataFrame(adv_dict)
    return pd.DataFrame(adv_dict)


def _run_metamorphic_check(code_str: str, df_obj, target_name: str = "df") -> tuple[bool, str]:
    """Validates metamorphic stability by testing code execution against a 2x scaled numeric matrix."""
    if df_obj is None:
        return True, ""
    try:
        is_pandas = isinstance(df_obj, pd.DataFrame)
        is_polars = pl is not None and isinstance(df_obj, pl.DataFrame)

        if not is_pandas and not is_polars:
            return True, ""

        if is_pandas:
            df_base = df_obj.copy(deep=True)
            df_scaled = df_obj.copy(deep=True)
            for c in df_scaled.select_dtypes(include=[np.number]).columns:
                df_scaled[c] = df_scaled[c] * 2.0
        else:
            df_base = df_obj.clone()
            num_cols = [c for c, dtype in df_obj.schema.items() if str(dtype).startswith(("Float", "Int", "UInt", "Decimal"))]
            if num_cols:
                df_scaled = df_base.with_columns([pl.col(c) * 2.0 for c in num_cols])
            else:
                df_scaled = df_base.clone()

        ns_base = {"pd": pd, "np": np, "duckdb": duckdb, target_name: df_base, "df": df_base}
        if pl is not None: ns_base["pl"] = pl
        exec(code_str, ns_base)

        ns_scaled = {"pd": pd, "np": np, "duckdb": duckdb, target_name: df_scaled, "df": df_scaled}
        if pl is not None: ns_scaled["pl"] = pl
        exec(code_str, ns_scaled)

        del ns_base, ns_scaled, df_base, df_scaled
        gc.collect()

        return True, "Metamorphic invariant passed."
    except Exception as e:
        return False, f"Metamorphic validation failed under 2x perturbation: {type(e).__name__}: {e}"


def _render_roadmap():
    """Renders an interactive project roadmap tracking workflow across 4 phases."""
    phase = _ACTIVE_ROADMAP.get("phase", 1)
    goal = _ACTIVE_ROADMAP.get("goal", None)
    hypotheses = _ACTIVE_ROADMAP.get("hypotheses", [])

    phases_info = [
        (1, "Profiling & Cleaning", "%deepanalyze --kickstart  OR  %deepanalyze --auto-clean"),
        (2, "Goal Interview", "%deepanalyze --interview (align optimization goals)"),
        (3, "Execution & Radar", "%deepanalyze --brainstorm  OR  %deepanalyze -x --radar <task>"),
        (4, "Synthesis & Reporting", "%deepanalyze -x -i --persona exec <task>  OR  --spawn"),
    ]

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold magenta")
    table.add_column("Phase", style="cyan", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Next Recommended Action", style="yellow", justify="left")

    for p_num, p_name, next_cmd in phases_info:
        if p_num < phase:
            status = "[green]✅ Completed[/green]"
        elif p_num == phase:
            status = "[bold yellow]⚡ Active[/bold yellow]"
        else:
            status = "[dim]⚪ Pending[/dim]"
        table.add_row(f"[{p_num}] {p_name}", status, next_cmd if p_num == phase else "[dim]—[/dim]")

    console.print(Panel(table, title=f"🗺️ [bold magenta]DeepAnalyze Autonomous Project Roadmap (Phase {phase}/4)[/bold magenta]", border_style="magenta", expand=False))
    if goal:
        print(f"🎯 [bold cyan]Target Objective & Constraints:[/bold cyan] {goal}")
    if hypotheses:
        print("\n💡 [bold yellow]Active Testable Hypotheses:[/bold yellow]")
        for i, h in enumerate(hypotheses[:3], 1):
            print(f"   {i}. {h}")


def _render_transformation_dag(code_str: str, target_name: str = "df"):
    """Parses AST of generated code and renders an execution lineage DAG tree."""
    from rich.tree import Tree
    try:
        tree_ast = ast.parse(code_str)
    except Exception:
        return

    dag = Tree(f"🌱 [bold green]Source: `{target_name}`[/bold green]")
    step_num = 1

    for node in tree_ast.body:
        if isinstance(node, ast.Assign):
            target_ids = []
            for t in node.targets:
                if isinstance(t, ast.Name): target_ids.append(t.id)
                elif isinstance(t, ast.Attribute): target_ids.append(t.attr)
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name): target_ids.append(f"{t.value.id}[...]")
            target_str = ", ".join(target_ids) if target_ids else "result"
            
            val = node.value
            op_type = "Transform"
            summary = ast.unparse(val) if hasattr(ast, "unparse") else "expr"
            if len(summary) > 42: summary = summary[:39] + "..."

            if isinstance(val, ast.Call):
                if isinstance(val.func, ast.Attribute): op_type = f".{val.func.attr}()"
                elif isinstance(val.func, ast.Name): op_type = f"{val.func.id}()"
            elif isinstance(val, ast.BinOp):
                op_type = "Col Calculation"

            dag.add(f"[bold cyan]Step {step_num}: {target_str}[/bold cyan] ➔ [yellow]{op_type}[/yellow] [dim]({summary})[/dim]")
            step_num += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call_func = node.value.func
            name = ast.unparse(call_func) if hasattr(ast, "unparse") else "call"
            dag.add(f"[bold cyan]Action {step_num}[/bold cyan] ➔ [yellow]{name}()[/yellow]")
            step_num += 1

    dag.add(f"🎯 [bold magenta]Output Target: `{target_name}`[/bold magenta]")
    console.print(Panel(dag, title="🌳 [bold cyan]DeepAnalyze Transformation Flow Graph (AST DAG)[/bold cyan]", border_style="cyan", expand=False))


def _render_gui_explorer(df_obj, target_name: str = "df", max_rows: int = 50):
    """Renders a standalone interactive HTML/JS data table with search, sorting, and sticky headers."""
    if df_obj is None:
        print(f"[DeepAnalyze GUI]: No data loaded in `{target_name}`.")
        return

    is_pandas = isinstance(df_obj, pd.DataFrame)
    is_polars = pl is not None and isinstance(df_obj, pl.DataFrame)

    if not is_pandas and not is_polars:
        print(f"[DeepAnalyze GUI]: Object `{target_name}` is not a tabular DataFrame.")
        return

    sample_df = df_obj.head(max_rows)
    if is_polars:
        sample_df = sample_df.to_pandas()

    columns = list(sample_df.columns)
    dtypes = {c: str(sample_df[c].dtype) for c in columns}
    rows_data = sample_df.fillna("").values.tolist()

    table_id = f"da_table_{int(time.time()*1000)}"
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #18181b; color: #f4f4f5; padding: 16px; border-radius: 10px; border: 1px solid #3f3f46; margin: 12px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 15px; font-weight: bold; color: #38bdf8;">
                📊 DeepAnalyze In-Notebook Explorer: <code>{target_name}</code> <span style="font-size: 12px; color: #a1a1aa;">({len(df_obj):,} rows total, showing top {len(sample_df)})</span>
            </div>
            <input type="text" id="search_{table_id}" onkeyup="filter_{table_id}()" placeholder="🔍 Search records..." 
                   style="background: #27272a; color: #fafafa; border: 1px solid #52525b; border-radius: 6px; padding: 6px 12px; font-size: 13px; outline: none; width: 220px;" />
        </div>
        <div style="max-height: 380px; overflow-y: auto; overflow-x: auto; border: 1px solid #27272a; border-radius: 6px;">
            <table id="{table_id}" style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
                <thead style="position: sticky; top: 0; background: #27272a; z-index: 10;">
                    <tr>
    """
    for i, col in enumerate(columns):
        dtype_badge = dtypes[col]
        html_content += f"""
            <th onclick="sort_{table_id}({i})" style="padding: 10px 14px; border-bottom: 2px solid #52525b; cursor: pointer; user-select: none; white-space: nowrap;">
                {col} <span style="font-size: 10px; background: #3f3f46; color: #38bdf8; padding: 2px 5px; border-radius: 4px; margin-left: 4px;">{dtype_badge}</span> ⬍
            </th>
        """
    html_content += """
                    </tr>
                </thead>
                <tbody>
    """
    for r_idx, row in enumerate(rows_data):
        bg = "#18181b" if r_idx % 2 == 0 else "#202024"
        html_content += f"<tr style='background: {bg}; border-bottom: 1px solid #27272a;'>"
        for val in row:
            html_content += f"<td style='padding: 8px 14px; white-space: nowrap;'>{str(val)}</td>"
        html_content += "</tr>"

    html_content += f"""
                </tbody>
            </table>
        </div>
        <script>
            function filter_{table_id}() {{
                var input = document.getElementById("search_{table_id}").value.toUpperCase();
                var rows = document.getElementById("{table_id}").getElementsByTagName("tbody")[0].getElementsByTagName("tr");
                for (var i = 0; i < rows.length; i++) {{
                    var text = rows[i].textContent || rows[i].innerText;
                    rows[i].style.display = text.toUpperCase().indexOf(input) > -1 ? "" : "none";
                }}
            }}
            function sort_{table_id}(n) {{
                var table = document.getElementById("{table_id}");
                var rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                switching = true; dir = "asc";
                while (switching) {{
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];
                        var xVal = isNaN(x.innerHTML) ? x.innerHTML.toLowerCase() : parseFloat(x.innerHTML);
                        var yVal = isNaN(y.innerHTML) ? y.innerHTML.toLowerCase() : parseFloat(y.innerHTML);
                        if (dir == "asc") {{
                            if (xVal > yVal) {{ shouldSwitch = true; break; }}
                        }} else if (dir == "desc") {{
                            if (xVal < yVal) {{ shouldSwitch = true; break; }}
                        }}
                    }}
                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true; switchcount ++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{ dir = "desc"; switching = true; }}
                    }}
                }}
            }}
        </script>
    </div>
    """
    try:
        from IPython.display import display, HTML
        display(HTML(html_content))
    except Exception:
        rich_table = Table(title=f"📊 DeepAnalyze Explorer: `{target_name}`", box=None)
        for c in columns: rich_table.add_column(str(c))
        for row in rows_data[:10]: rich_table.add_row(*[str(v) for v in row])
        console.print(rich_table)


def _render_history_explorer():
    """Renders a visual time-machine table of all cached DataFrame rollback points."""
    global _DF_SNAPSHOTS, _DF_SNAPSHOT_METADATA
    table = Table(
        title="⏳ [bold cyan]DeepAnalyze DataFrame Time-Machine & Rollback Points[/bold cyan]",
        header_style="bold magenta",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Target Variable", style="cyan", justify="left")
    table.add_column("Latest Timestamp", justify="center")
    table.add_column("Dimensions (Rows x Cols)", justify="center")
    table.add_column("Column Sample", justify="left")
    table.add_column("Status", justify="right", style="green")

    if not _DF_SNAPSHOTS:
        console.print(Panel("[dim]No active DataFrame snapshots in memory. Run %deepanalyze -x ... to create states.[/dim]", title="⏳ Snapshot Time-Machine", border_style="cyan"))
        return

    for var_name, df_snap in _DF_SNAPSHOTS.items():
        meta_list = _DF_SNAPSHOT_METADATA.get(var_name, [])
        last_meta = meta_list[-1] if meta_list else {}
        ts = last_meta.get("time", "Current Session")
        shape_str = f"{df_snap.shape[0]:,} x {df_snap.shape[1]:,}" if hasattr(df_snap, "shape") else "Unknown"
        cols_str = ", ".join(str(c) for c in list(df_snap.columns)[:4]) + ("..." if len(df_snap.columns) > 4 else "")
        table.add_row(f"`{var_name}`", ts, shape_str, f"[dim]{cols_str}[/dim]", "💾 Active Snapshot")

    console.print(Panel(table, border_style="cyan", expand=False))
    print("💡 [Tip]: Restore any target state with `%deepanalyze --undo [--target <var_name>]`.")


def _scan_for_anomalies(orig_df, new_df, target_name: str = "df") -> list[str]:
    """Lightweight proactive anomaly radar scanning for extreme outliers, null drifts, and distribution shifts."""
    anomalies = []
    if orig_df is None or new_df is None:
        return anomalies

    is_pandas_orig = isinstance(orig_df, pd.DataFrame)
    is_pandas_new = isinstance(new_df, pd.DataFrame)

    if not is_pandas_orig or not is_pandas_new:
        return anomalies

    for c in new_df.columns:
        if c in orig_df.columns:
            orig_null_pct = orig_df[c].isna().mean() * 100
            new_null_pct = new_df[c].isna().mean() * 100
            if (new_null_pct - orig_null_pct) >= 20.0:
                anomalies.append(f"Null surge in column `{c}`: +{new_null_pct - orig_null_pct:.1f}% new nulls ({orig_null_pct:.1f}% ➔ {new_null_pct:.1f}%)")

    num_cols = [c for c in new_df.select_dtypes(include=[np.number]).columns if c in orig_df.columns]
    for c in num_cols:
        orig_mean = orig_df[c].dropna().mean()
        new_mean = new_df[c].dropna().mean()
        if pd.notna(orig_mean) and pd.notna(new_mean) and abs(orig_mean) > 1e-5:
            pct_shift = ((new_mean - orig_mean) / abs(orig_mean)) * 100
            if abs(pct_shift) >= 35.0:
                anomalies.append(f"Significant mean shift in `{c}`: {pct_shift:+.1f}% ({orig_mean:.2f} ➔ {new_mean:.2f})")

    for c in num_cols:
        orig_s = orig_df[c].dropna()
        new_s = new_df[c].dropna()
        if not orig_s.empty and (orig_s >= 0).all() and not new_s.empty and (new_s < 0).any():
            neg_count = int((new_s < 0).sum())
            anomalies.append(f"Negative values introduced into previously non-negative column `{c}`: {neg_count} negative entries")

    return anomalies


def _recommend_next_actions(env_context: str, last_prompt: str, target_model: str = "deepanalyze-8b"):
    """Predicts 3 logical follow-up exploratory actions based on workspace context."""
    next_sys = "You are a predictive data science copilot. Suggest exactly 3 logical, high-value next-step analytical actions based on the recent analysis. For each action, output a concise 1-line bullet point followed immediately by the exact executable '%deepanalyze -x ...' command on the next line."
    next_prompt = f"Recent User Action: {last_prompt}\n\nWorkspace Context:\n{env_context}\n\nSuggest 3 next actions."
    try:
        raw_res = _call_llm(next_prompt, next_sys, temp=0.3, max_tokens=800, target_model=target_model)
        clean_res = re.sub(r'<think>.*?</think>', '', raw_res, flags=re.DOTALL | re.IGNORECASE).strip()
        console.print(Panel(clean_res, title="💡 [bold yellow]Suggested Next Actions[/bold yellow]", border_style="yellow", expand=False))
    except Exception:
        pass


def _display_metrics(duration_sec: float, token_count: int, engine_type: str, model_name: str):
    """Renders runtime telemetry and throughput using Rich."""
    tok_per_sec = (token_count / duration_sec) if duration_sec > 0 else 0.0
    
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Latency", justify="center")
    table.add_column("Tokens", justify="center")
    table.add_column("Throughput", justify="center")
    table.add_column("Engine / Model", justify="center")
    
    table.add_row(
        f"{duration_sec * 1000:.1f} ms",
        f"{token_count} tok",
        f"{tok_per_sec:.1f} tok/s",
        f"[green]{engine_type}[/green] ({model_name})"
    )
    
    console.print(Panel(table, title="[bold magenta]DeepAnalyze Telemetry[/bold magenta]", border_style="magenta", expand=False))


def deepanalyze(line, cell=None):
    global _LAST_GENERATED_CODE, _LAST_USER_PROMPT, _INTERCEPTOR_ACTIVE
    raw_input = f"{line}\n{cell}" if cell else line
    ip = get_ipython()

    parser = argparse.ArgumentParser(prog="%deepanalyze", description="Agentic LLM Execution Engine", add_help=False)
    parser.add_argument("--toggle", action="store_true", help="Toggle global cell interceptor on/off")
    parser.add_argument("--status", action="store_true", help="Display server health and security status")
    parser.add_argument("-x", "--exec", "--execute", dest="execute_code", action="store_true")
    parser.add_argument("-c", "--continue", dest="is_continuation", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--ultra", action="store_true")
    
    # CLOUD ROUTING FLAGS
    parser.add_argument("--pro", action="store_true", help="Route prompt to DeepSeek-V4-Pro (deepseek-chat)")
    parser.add_argument("--flash", action="store_true", help="Route prompt to DeepSeek-V4-Flash (deepseek-chat)")
    parser.add_argument("--think", action="store_true", help="Route prompt to DeepSeek-Reasoner (deepseek-reasoner)")
    
    # NEW PRIVACY ENGINE FLAGS
    parser.add_argument("--privacy", type=str, default="auto", choices=["auto", "mask", "profile", "mock", "none"], help="Privacy mode")
    parser.add_argument("--audit-only", action="store_true", help="Display privacy payload without executing LLM")

    # ADVANCED ANALYTICAL UPGRADE FLAGS
    parser.add_argument("--persona", type=str, default="default", choices=["default", "exec", "dev"], help="Insight synthesis persona mode")
    parser.add_argument("--context", type=str, default=None, help="Filepath to business logic schema/dictionary (Markdown or JSON)")
    parser.add_argument("--critic", action="store_true", help="Enable local logical critic loop (deepanalyze-8b)")
    parser.add_argument("--critic-pro", action="store_true", help="Enable cloud logical critic loop (deepseek-reasoner)")

    # ADVANCED VALIDATION, SANDBOXING & UI FLAGS
    parser.add_argument("--preview", action="store_true", help="Ghost execution with state diff HUD and interactive commit")
    parser.add_argument("--diff", action="store_true", help="Render visual state diff HUD showing row/col and schema deltas")
    parser.add_argument("--guard", type=str, default=None, help="Automated quality gate constraint expression")
    parser.add_argument("--stress", action="store_true", help="Adversarial edge-case fuzzer for NaN, empty string, zero-division")
    parser.add_argument("--meta", action="store_true", help="Metamorphic logic validator verifying 2x scaling invariance")
    parser.add_argument("--simulate", type=str, default=None, help="Sandboxed What-If simulation scenario without global state mutation")
    parser.add_argument("--spark", action="store_true", help="Display inline ASCII sparkline minimaps for numeric column distributions")

    # WORKFLOW ORCHESTRATION & NOTEBOOK AUTOMATION FLAGS
    parser.add_argument("--roadmap", action="store_true", help="Display global multi-phase project orchestrator roadmap")
    parser.add_argument("--kickstart", action="store_true", help="Zero-prompt analysis kickstart inferring domain & action plan")
    parser.add_argument("--interview", action="store_true", help="Stakeholder goal & constraint alignment interview")
    parser.add_argument("--brainstorm", action="store_true", help="Autonomous hypothesis generator with executable commands")
    parser.add_argument("--radar", action="store_true", help="Proactive anomaly radar scanning for outliers and drift")
    parser.add_argument("--dag", action="store_true", help="Render AST transformation flow lineage graph")
    parser.add_argument("--gui", action="store_true", help="Interactive in-notebook searchable/sortable data explorer")
    parser.add_argument("--history", action="store_true", help="Visual time-machine displaying snapshot history")
    parser.add_argument("--next", action="store_true", help="Predictive next-action recommender")
    parser.add_argument("--auto-clean", action="store_true", help="Autonomous data sanitizer with interactive preview diff")
    parser.add_argument("--spawn", action="store_true", help="Spawn Markdown narrative and Code cells into notebook")

    # DATA SCIENCE SKILL FLAGS
    parser.add_argument("--validate", action="store_true", help="Enable rigorous ML validation mode")
    parser.add_argument("--tune", action="store_true", help="Enable leak-free hyperparameter tuning mode")
    parser.add_argument("--explain", action="store_true", help="Enable model interpretability mode")
    parser.add_argument("-i", "--insight", action="store_true", help="Generate business insights from output")
    parser.add_argument("-u", "--unravel", action="store_true")
    parser.add_argument("-p", "--profile", action="store_true")
    parser.add_argument("-v", "--viz", action="store_true")
    parser.add_argument("-s", "--sql", action="store_true")
    parser.add_argument("-f", "--feat", action="store_true")
    parser.add_argument("-t", "--stat", action="store_true")
    parser.add_argument("-m", "--ml", action="store_true")
    parser.add_argument("-r", "--repair", action="store_true")
    
    parser.add_argument("-d", "--deterministic", action="store_true")
    parser.add_argument("--target", type=str, default="df")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--undo", action="store_true")

    try:
        if cell is not None:
            try:
                parsed_args, remaining_words = parser.parse_known_args(shlex.split(line))
            except Exception:
                parsed_args, remaining_words = parser.parse_known_args(line.split())
            prompt = ((" ".join(remaining_words) + "\n") if remaining_words else "") + cell.strip()
        else:
            try:
                tokens = shlex.split(line)
            except Exception:
                tokens = line.split()
            parsed_args, remaining_words = parser.parse_known_args(tokens)
            prompt = " ".join(remaining_words).strip()
    except SystemExit:
        return

    if parsed_args.toggle:
        _INTERCEPTOR_ACTIVE = not _INTERCEPTOR_ACTIVE
        state_str = "🟢 ENABLED (Auto-pilot active)" if _INTERCEPTOR_ACTIVE else "⚪ DISABLED (Explicit mode)"
        print(f"🔄 DeepAnalyze Interceptor toggled: {state_str}")
        return

    if parsed_args.status:
        check_engine_status()
        return

    if parsed_args.roadmap:
        _render_roadmap()
        return

    if parsed_args.history:
        _render_history_explorer()
        return

    if parsed_args.gui:
        target_df = ip.user_ns.get(parsed_args.target) if ip else None
        _render_gui_explorer(target_df, target_name=parsed_args.target)
        return

    if parsed_args.undo:
        if _restore_snapshot(ip, target=parsed_args.target):
            df_restored = ip.user_ns[parsed_args.target]
            print(f"[DeepAnalyze Undo]: Restored `{parsed_args.target}` from snapshot. Shape: {df_restored.shape[0]} rows, {df_restored.shape[1]} columns.")
        else:
            print(f"[DeepAnalyze Undo]: No previous snapshot found in memory for `{parsed_args.target}`.")
        return

    active_model = "deepanalyze-8b"
    is_cloud_call = False
    if parsed_args.pro or parsed_args.flash:
        active_model = "deepseek-chat"
        is_cloud_call = True
    elif parsed_args.think:
        active_model = "deepseek-reasoner"
        is_cloud_call = True

    primary_skill = "general"
    if parsed_args.unravel: primary_skill = "unravel"
    elif parsed_args.profile: primary_skill = "profile"
    elif parsed_args.viz: primary_skill = "viz"
    elif parsed_args.sql: primary_skill = "sql"
    elif parsed_args.feat: primary_skill = "feature"
    elif parsed_args.stat: primary_skill = "stat"
    elif parsed_args.ml: primary_skill = "ml"
    elif parsed_args.repair: primary_skill = "repair"
    elif parsed_args.validate: primary_skill = "validate"
    elif parsed_args.tune: primary_skill = "tune"
    elif parsed_args.explain: primary_skill = "explain"

    # Ensemble Intent Routing (Zero-Flag Mode)
    if primary_skill == "general" and prompt and not parsed_args.is_continuation:
        classified_skill = _classify_intent(prompt, target_model=active_model)
        if classified_skill in SKILL_RULEBOOKS:
            primary_skill = classified_skill

    if not prompt and primary_skill != "profile" and not parsed_args.is_continuation and not parsed_args.kickstart and not parsed_args.interview and not parsed_args.brainstorm and not parsed_args.auto_clean:
        print("Usage: %deepanalyze [-x] [--target df] [--think|--pro] [--privacy auto|mask|profile] <task description>")
        return

    _sync_duckdb(ip)

    temp, max_tokens = (0.0, 3500) if parsed_args.deterministic else (0.7, 3500)
    if parsed_args.fast: temp, max_tokens = 0.0, 1000
    elif parsed_args.ultra: temp, max_tokens = 0.05, 4096

    env_context, available_vars, knife = _get_deep_workspace_context(
        ip, target=parsed_args.target, is_cloud=is_cloud_call, privacy_mode=parsed_args.privacy
    )

    # Zero-Prompt Kickstart (--kickstart)
    if parsed_args.kickstart:
        kick_sys = "You are a Principal Data Scientist and AI Architect. Ingest the dataset context, infer the primary business domain, identify core target KPIs, and output a prioritized 3-step action plan in clean markdown bullet points."
        kick_prompt = f"Active Environment Context:\n{env_context}\n\nInfer business domain and outline prioritized 3-step action plan."
        raw_plan = _call_llm(kick_prompt, kick_sys, temp=0.2, max_tokens=1200, target_model=active_model)
        clean_plan = re.sub(r'<think>.*?</think>', '', raw_plan, flags=re.DOTALL | re.IGNORECASE).strip()
        console.print(Panel(clean_plan, title="🚀 [bold cyan]DeepAnalyze Zero-Prompt Kickstart[/bold cyan]", border_style="cyan"))
        _ACTIVE_ROADMAP["phase"] = max(_ACTIVE_ROADMAP.get("phase", 1), 2)
        print("\n👉 [Next Step]: Run `%deepanalyze --interview` to align analytical constraints.")
        return

    # Stakeholder Goal Interview (--interview)
    if parsed_args.interview:
        int_sys = "You are an expert AI Data Architect interviewing a stakeholder. Generate exactly 3 targeted multiple-choice questions regarding analytical constraints (e.g., [1] Optimization metric, [2] Interpretability vs Accuracy, [3] Missing value strategy). Provide clear labeled options (A, B, C) for each."
        int_prompt = f"Dataset Context:\n{env_context}\n\nGenerate the 3 multiple-choice interview questions."
        raw_q = _call_llm(int_prompt, int_sys, temp=0.2, max_tokens=1000, target_model=active_model)
        clean_q = re.sub(r'<think>.*?</think>', '', raw_q, flags=re.DOTALL | re.IGNORECASE).strip()
        console.print(Panel(clean_q, title="🎯 [bold magenta]DeepAnalyze Stakeholder Goal Interview[/bold magenta]", border_style="magenta"))
        try:
            user_ans = input("\n🎯 Enter your preferred choices / constraints (e.g., 1A, 2B, 3C): ").strip()
        except (KeyboardInterrupt, EOFError):
            user_ans = ""
        if user_ans:
            _ACTIVE_ROADMAP["goal"] = user_ans
            _ACTIVE_ROADMAP["phase"] = max(_ACTIVE_ROADMAP.get("phase", 1), 3)
            print(f"✅ Goal recorded: '{user_ans}'. Roadmap advanced to Phase 3: Execution & Radar. Run `%deepanalyze --brainstorm` next.")
        return

    # Autonomous Hypothesis Generator (--brainstorm)
    if parsed_args.brainstorm:
        goal_str = _ACTIVE_ROADMAP.get("goal") or "General exploratory, wrangling, and predictive analysis"
        bs_sys = "You are an elite quantitative analyst and business strategist. Based on the dataset context and goal, generate 3-5 specific, testable business hypotheses. Under each hypothesis, provide the exact executable '%deepanalyze -x ...' command required to test it."
        bs_prompt = f"Goal: {goal_str}\n\nContext:\n{env_context}\n\nGenerate hypotheses with executable %deepanalyze commands."
        raw_bs = _call_llm(bs_prompt, bs_sys, temp=0.3, max_tokens=1500, target_model=active_model)
        clean_bs = re.sub(r'<think>.*?</think>', '', raw_bs, flags=re.DOTALL | re.IGNORECASE).strip()
        console.print(Panel(clean_bs, title="💡 [bold yellow]DeepAnalyze Autonomous Hypotheses & Action Plan[/bold yellow]", border_style="yellow"))
        _ACTIVE_ROADMAP["hypotheses"] = [line.strip() for line in clean_bs.splitlines() if line.strip().startswith(("%deepanalyze", "1.", "2.", "3.", "4.", "5.", "H1:", "H2:", "H3:"))]
        _ACTIVE_ROADMAP["phase"] = max(_ACTIVE_ROADMAP.get("phase", 1), 3)
        return

    # Semantic Auto-Sanitizer (--auto-clean)
    if parsed_args.auto_clean:
        parsed_args.preview = True
        auto_sys = (
            "You are an autonomous data cleaning agent. Detect all formatting anomalies in the active DataFrames "
            "(whitespace, currency '$', '%', commas in numbers, dirty strings, nulls, wrong datatypes). "
            "Generate a clean, robust cleaning script using Polars/Pandas. Output ONLY executable code inside <Answer>```python ... ```</Answer>."
        )
        auto_prompt = f"Inspect active dataframe context and output complete cleaning script:\n{env_context}"
        raw_clean = _call_llm(auto_prompt, auto_sys, temp=0.0, max_tokens=2500, target_model=active_model)
        code, narrative = _extract_deepanalyze_content(raw_clean)
        if not prompt: prompt = "Autonomous data sanitization"

    # Semantic Schema Dictionaries (RAG)
    if parsed_args.context:
        context_path = os.path.expanduser(parsed_args.context)
        if os.path.exists(context_path):
            try:
                with open(context_path, "r", encoding="utf-8") as f:
                    rag_content = f.read()
                env_context += f"\n--- BUSINESS LOGIC & RAG CONTEXT ---\n{rag_content}\n------------------------------------\n"
            except Exception as e:
                print(f"[DeepAnalyze Warning]: Failed to read context file '{parsed_args.context}': {e}")
        else:
            print(f"[DeepAnalyze Warning]: Context file '{parsed_args.context}' not found.")
    
    if parsed_args.audit_only:
        print("🛡️ [DeepAnalyze Privacy Audit - Safe Payload to be Transmitted]:")
        print(env_context)
        return

    fuzzy_aliases = _fuzzy_match_columns(prompt, ip, target=parsed_args.target)
    rulebook = SKILL_RULEBOOKS.get(primary_skill, SKILL_RULEBOOKS["general"])
    
    # Dynamically force the strict Polars template if operating on a Polars target
    if primary_skill == "general" and "Engine: Polars" in env_context:
        rulebook = SKILL_RULEBOOKS.get("polars", rulebook)
        
    save_directive = ""
    if parsed_args.save:
        os.makedirs("charts", exist_ok=True)
        save_directive = (
            "\n[AUTO-SAVE FIGURE DIRECTIVE]:\n"
            "Ensure the directory 'charts' exists. "
            "Before calling `plt.show()`, call `plt.savefig('charts/<meaningful_slug>.png', dpi=300, bbox_inches='tight')` "
            "and print the saved filepath.\n"
        )

    system_prompt = f"{rulebook}\n{INVARIANT_CHECKLIST}\n{save_directive}\n{env_context}\n{fuzzy_aliases}"

    if parsed_args.is_continuation and _LAST_GENERATED_CODE:
        full_prompt = (
            f"PREVIOUS EXECUTED CODE:\n```python\n{_LAST_GENERATED_CODE}\n```\n\n"
            f"USER REFINEMENT REQUEST: {prompt}\n"
            f"Refine and update the previous code cleanly to satisfy the user refinement."
        )
    else:
        full_prompt = f"User Task: {prompt}" if prompt else "Analyze active dataset schema and provide executive summary."

    if primary_skill == "profile" and not parsed_args.fast and not parsed_args.deterministic:
        temp = max(temp, 0.2)

    try:
        # Start timer for telemetry
        start_time = time.time()
        
        if not parsed_args.auto_clean:
            raw_output = _call_llm(full_prompt, system_prompt, temp=temp, max_tokens=max_tokens, target_model=active_model)
            code, narrative = _extract_deepanalyze_content(raw_output)
            token_count = len(raw_output) // 4
        else:
            token_count = len(raw_clean) // 4
        
        duration = time.time() - start_time
        engine_used = "Polars ⚡" if (pl is not None and "polars" in env_context.lower()) else "Pandas 🐼"
        
        _display_metrics(duration, token_count, engine_used, active_model)

        if narrative and primary_skill == "profile":
            print(f"\n[DeepAnalyze Strategic Overview]:\n{narrative}\n")

        if not code and not narrative:
            print("[DeepAnalyze Raw Output]:\n" + raw_output)
            return

        if code:
            is_valid, clean_code, lint_error = _lint_and_format_code(code, available_vars)

            if not is_valid and clean_code:
                print(f"[DeepAnalyze Pre-Flight Catch]: {lint_error}. Auto-repairing...")
                repair_prompt = (
                    f"The generated code failed static validation with error: {lint_error}\n"
                    f"```python\n{code}\n```\n"
                    f"Fix this code to strictly match available variables and exact column casings:\n{env_context}"
                )
                fixed_raw = _call_llm(repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens, target_model=active_model)
                fixed_code, _ = _extract_deepanalyze_content(fixed_raw)
                is_valid_repair, rep_code, _ = _lint_and_format_code(fixed_code, available_vars)
                if is_valid_repair and rep_code:
                    clean_code = rep_code
                    is_valid = True

            # Hybrid Logical Critic Loop (Self-Healing 2.0)
            if clean_code and is_valid and (parsed_args.critic or parsed_args.critic_pro):
                critic_model = "deepseek-reasoner" if parsed_args.critic_pro else "deepanalyze-8b"
                critic_sys = (
                    "You are a rigorous data logic critic. Review Python data analysis code for subtle semantic fallacies, "
                    "business logic bugs, zero-division, incorrect join keys, negative revenue/aggregations, or invalid indexing."
                )
                critic_prompt = (
                    f"Review this generated code for logical data fallacies (e.g., calculating negative revenue, improper joins, zero-division). "
                    f"If the logic is safe, output EXACTLY the word 'SAFE'. If flawed, output the corrected Python code inside <Answer>```python ... ```</Answer>.\n\n"
                    f"Context:\n{env_context}\n\n"
                    f"Generated Code:\n```python\n{clean_code}\n```"
                )
                critic_raw = _call_llm(critic_prompt, critic_sys, temp=0.0, max_tokens=max_tokens, target_model=critic_model)
                critic_text = re.sub(r'<think>.*?</think>', '', critic_raw, flags=re.DOTALL | re.IGNORECASE).strip()
                critic_extracted, _ = _extract_deepanalyze_content(critic_raw)
                if critic_extracted and critic_text.upper() != "SAFE" and "SAFE" not in critic_text.split():
                    is_critic_valid, rep_critic_code, _ = _lint_and_format_code(critic_extracted, available_vars)
                    if is_critic_valid and rep_critic_code:
                        clean_code = rep_critic_code
                        print("🛡️ [Critic Loop]: Logical flaw intercepted and repaired prior to execution.")

            # Adversarial Edge-Case Fuzzer (--stress)
            if clean_code and is_valid and parsed_args.stress:
                target_df = ip.user_ns.get(parsed_args.target) if ip else None
                adv_df = _generate_adversarial_df(target_df)
                if adv_df is not None:
                    stress_ns = {"pd": pd, "np": np, "duckdb": duckdb, parsed_args.target: adv_df, "df": adv_df}
                    if pl is not None: stress_ns["pl"] = pl
                    try:
                        exec(clean_code, stress_ns)
                        print("🧪 [Stress Fuzzer]: PASSED (Survives NaNs, empty strings, and zero-denominators).")
                    except Exception as s_err:
                        print(f"⚠️ [Stress Fuzzer Alert]: Code failed on edge cases ({type(s_err).__name__}: {s_err}). Auto-repairing...")
                        s_raw_tb = traceback.format_exc()
                        s_clean_tb = _sanitize_traceback(s_raw_tb)
                        stress_repair_prompt = (
                            f"[ADVERSARIAL STRESS FAILURE REFLECTION]\n"
                            f"The generated code crashed on an edge-case matrix (containing NaNs, zero values, '$0.00' strings, or empty strings):\n"
                            f"{s_clean_tb}\n\n"
                            f"Make the code defensive against division by zero, null values, and string formatting issues.\n"
                            f"Context:\n{env_context}\n"
                            f"Output ONLY corrected Python code inside <Answer>```python ... ```</Answer>."
                        )
                        s_fixed_raw = _call_llm(stress_repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens, target_model=active_model)
                        s_fixed_code, _ = _extract_deepanalyze_content(s_fixed_raw)
                        s_valid, s_rep_code, _ = _lint_and_format_code(s_fixed_code, available_vars)
                        if s_valid and s_rep_code:
                            clean_code = s_rep_code
                            print("🛡️ [Stress Fuzzer]: Defensively patched and repaired.")
                    finally:
                        del adv_df, stress_ns
                        gc.collect()

            # Metamorphic Logic Validator (--meta)
            if clean_code and is_valid and parsed_args.meta:
                target_df = ip.user_ns.get(parsed_args.target) if ip else None
                meta_passed, meta_msg = _run_metamorphic_check(clean_code, target_df, target_name=parsed_args.target)
                if not meta_passed:
                    print(f"⚠️ [Metamorphic Alert]: {meta_msg}. Auto-repairing...")
                    meta_repair_prompt = (
                        f"[METAMORPHIC VALIDATION FAILURE REFLECTION]\n"
                        f"When tested on a 2x scaled numerical perturbation, the code failed:\n"
                        f"{meta_msg}\n\n"
                        f"Ensure the calculation handles scaling dynamically without hardcoded constants.\n"
                        f"Context:\n{env_context}\n"
                        f"Output ONLY corrected Python code inside <Answer>```python ... ```</Answer>."
                    )
                    m_fixed_raw = _call_llm(meta_repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens, target_model=active_model)
                    m_fixed_code, _ = _extract_deepanalyze_content(m_fixed_raw)
                    m_valid, m_rep_code, _ = _lint_and_format_code(m_fixed_code, available_vars)
                    if m_valid and m_rep_code:
                        clean_code = m_rep_code
                        print("🛡️ [Metamorphic Validator]: Dynamically patched and repaired.")
                else:
                    print("🔬 [Metamorphic Validator]: PASSED (Linear invariance confirmed under 2x perturbation).")

            if clean_code:
                _LAST_GENERATED_CODE = clean_code
                _LAST_USER_PROMPT = prompt

                # Sandboxed "What-If" Simulator (--simulate)
                if parsed_args.simulate and is_valid:
                    target_obj = ip.user_ns.get(parsed_args.target) if ip else None
                    if target_obj is not None:
                        sim_orig = target_obj.copy(deep=True) if isinstance(target_obj, pd.DataFrame) else target_obj.clone()
                        sim_ns = {k: v for k, v in (ip.user_ns.items() if ip else [("pd", pd), ("np", np)]) if not k.startswith("__")}
                        sim_ns[parsed_args.target] = sim_orig.copy(deep=True) if isinstance(sim_orig, pd.DataFrame) else sim_orig.clone()
                        if "df" not in sim_ns:
                            sim_ns["df"] = sim_ns[parsed_args.target]
                        try:
                            exec(clean_code, sim_ns)
                            sim_res = sim_ns.get(parsed_args.target, sim_ns.get("df", None))
                            _render_simulation_hud(sim_orig, sim_res, parsed_args.simulate, target_name=parsed_args.target)
                            if parsed_args.diff:
                                _render_state_diff_hud(sim_orig, sim_res, target_name=parsed_args.target)
                            if parsed_args.spark:
                                _render_sparkline_minimap(sim_res, target_name=parsed_args.target)
                        except Exception as sim_err:
                            print(f"❌ [Simulation Error]: {sim_err}")
                        finally:
                            del sim_orig, sim_ns
                            gc.collect()
                        return

                # Interactive Ghost Execution (--preview)
                if parsed_args.preview and is_valid:
                    target_obj = ip.user_ns.get(parsed_args.target) if ip else None
                    if target_obj is not None:
                        shadow_orig = target_obj.copy(deep=True) if isinstance(target_obj, pd.DataFrame) else target_obj.clone()
                        shadow_ns = {k: v for k, v in (ip.user_ns.items() if ip else [("pd", pd), ("np", np)]) if not k.startswith("__")}
                        shadow_ns[parsed_args.target] = shadow_orig.copy(deep=True) if isinstance(shadow_orig, pd.DataFrame) else shadow_orig.clone()
                        if "df" not in shadow_ns:
                            shadow_ns["df"] = shadow_ns[parsed_args.target]

                        try:
                            exec(clean_code, shadow_ns)
                            shadow_res = shadow_ns.get(parsed_args.target, shadow_ns.get("df", None))

                            guard_passed = True
                            if parsed_args.guard:
                                guard_passed, guard_msg = _evaluate_quality_gate(parsed_args.guard, shadow_res, var_name=parsed_args.target)
                                if not guard_passed:
                                    console.print(Panel(f"[bold red]❌ Quality Gate Violation: {guard_msg}[/bold red]", border_style="red"))
                                else:
                                    print(f"🛡️ [Quality Gate]: PASSED (Constraint: `{parsed_args.guard}`)")

                            _render_state_diff_hud(shadow_orig, shadow_res, target_name=parsed_args.target)

                            if parsed_args.spark:
                                _render_sparkline_minimap(shadow_res, target_name=parsed_args.target)

                            if guard_passed:
                                try:
                                    choice = input("\n[DeepAnalyze Preview]: Commit changes to session memory? [Enter/y = Commit, Esc/n = Discard]: ").strip().lower()
                                    normally_committed = choice in ("", "y", "yes", "commit")
                                except (KeyboardInterrupt, EOFError):
                                    normally_committed = False

                                if normally_committed and ip:
                                    ip.user_ns[parsed_args.target] = shadow_res
                                    _reconcile_target_dataframe(ip, clean_code, prompt, parsed_args.target)
                                    print(f"✅ Changes committed to `{parsed_args.target}` in session memory.")
                                else:
                                    print("🛑 Changes discarded. Target DataFrame unaltered.")
                            else:
                                print("🛑 Commit blocked due to quality gate violation.")
                        except Exception as prev_err:
                            print(f"❌ [Preview Execution Error]: {prev_err}")
                        finally:
                            del shadow_orig, shadow_ns
                            gc.collect()
                        return

                if parsed_args.execute_code and is_valid:
                    _take_snapshot(ip, target=parsed_args.target)
                    original_row_count = ip.user_ns[parsed_args.target].shape[0] if parsed_args.target in ip.user_ns else None

                    print(f"[{active_model} Executing]:\n" + clean_code + "\n" + "-" * 40)
                    
                    captured_output = None
                    if parsed_args.insight:
                        with ipy_io.capture_output() as captured:
                            result = ip.run_cell(clean_code)
                        captured_output = captured
                        if captured.stdout: sys.stdout.write(captured.stdout)
                        if captured.stderr: sys.stderr.write(captured.stderr)
                    else:
                        result = ip.run_cell(clean_code)

                    _reconcile_target_dataframe(ip, clean_code, prompt, parsed_args.target)

                    # Automated Quality Gate verification
                    if parsed_args.guard and not result.error_in_exec:
                        curr_df = ip.user_ns.get(parsed_args.target)
                        guard_passed, guard_msg = _evaluate_quality_gate(parsed_args.guard, curr_df, var_name=parsed_args.target)
                        if not guard_passed:
                            console.print(Panel(f"[bold red]❌ Quality Gate Violation: {guard_msg}[/bold red]", border_style="red"))
                            _restore_snapshot(ip, target=parsed_args.target)
                            if parsed_args.retries > 0:
                                print("\n🔄 Quality gate violated. Escalating to repair...")
                                repair_prompt = (
                                    f"[QUALITY GATE VIOLATION REFLECTION]\n"
                                    f"The generated code violated the required quality gate constraint:\n"
                                    f"Constraint: {parsed_args.guard}\n"
                                    f"Failure Details: {guard_msg}\n\n"
                                    f"Previous Code:\n```python\n{clean_code}\n```\n\n"
                                    f"Fix the code so that it strictly satisfies the constraint:\n{prompt}\n"
                                    f"Context:\n{env_context}\n"
                                    f"Output ONLY corrected Python code inside <Answer>```python ... ```</Answer>."
                                )
                                fixed_raw = _call_llm(repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens, target_model=active_model)
                                fixed_code, _ = _extract_deepanalyze_content(fixed_raw)
                                is_valid_repair, final_code, rep_err = _lint_and_format_code(fixed_code, available_vars)
                                if is_valid_repair and final_code:
                                    _LAST_GENERATED_CODE = final_code
                                    print(f"\n[{active_model} Guard-Repaired Re-Executing]:\n" + final_code + "\n" + "-" * 40)
                                    result = ip.run_cell(final_code)
                                    if not result.error_in_exec:
                                        _reconcile_target_dataframe(ip, final_code, prompt, parsed_args.target)
                                        print("✅ Guard auto-repair succeeded!")
                        else:
                            print(f"🛡️ [Quality Gate]: PASSED (Constraint: `{parsed_args.guard}`)")

                    # State Diff HUD
                    if parsed_args.diff and not result.error_in_exec:
                        _render_state_diff_hud(_DF_SNAPSHOTS.get(parsed_args.target), ip.user_ns.get(parsed_args.target), target_name=parsed_args.target)

                    # Sparklines minimap
                    if parsed_args.spark and not result.error_in_exec:
                        _render_sparkline_minimap(ip.user_ns.get(parsed_args.target), target_name=parsed_args.target)

                    # Proactive Anomaly Radar (--radar)
                    if not result.error_in_exec and (parsed_args.radar or _DF_SNAPSHOTS.get(parsed_args.target) is not None):
                        anomalies = _scan_for_anomalies(_DF_SNAPSHOTS.get(parsed_args.target), ip.user_ns.get(parsed_args.target), target_name=parsed_args.target)
                        if anomalies:
                            alert_msg = "\n".join(f"  • {a}" for a in anomalies)
                            console.print(Panel(f"[bold red]🚨 Radar Alert: Data Anomalies Detected[/bold red]\n{alert_msg}", border_style="red", expand=False))
                        elif parsed_args.radar:
                            print("📡 [Radar]: Clean signal. No distribution anomalies detected.")

                    # Live Transformation Flow Graph (--dag)
                    if parsed_args.dag and not result.error_in_exec:
                        _render_transformation_dag(clean_code, target_name=parsed_args.target)

                    # Advance Roadmap to Phase 4 upon successful execution
                    if not result.error_in_exec:
                        _ACTIVE_ROADMAP["phase"] = max(_ACTIVE_ROADMAP.get("phase", 1), 4)

                    # Predictive Next-Action Recommender (--next)
                    if parsed_args.next and not result.error_in_exec:
                        _recommend_next_actions(env_context, prompt, target_model=active_model)

                    # Notebook Artifact Spawner (--spawn)
                    if parsed_args.spawn and ip and not result.error_in_exec:
                        if narrative:
                            ip.set_next_input(f"### 📊 DeepAnalyze Executive Insights\n\n{narrative}", replace=False)
                        ip.set_next_input(clean_code, replace=False)
                        print("✨ [Artifact Spawner]: Spawned Markdown narrative and Code cells into notebook below.")

                    if result.error_in_exec and parsed_args.retries > 0:
                        err = result.error_in_exec
                        for attempt in range(1, parsed_args.retries + 1):
                            print(f"\n⚠️ [Runtime Crash]: Caught {type(err).__name__}: {err}")
                            
                            try:
                                user_choice = input(
                                    "How would you like to resolve this error?\n"
                                    "  [1] Retry locally with DeepAnalyze-8B (Free / Local)\n"
                                    "  [2] Escalate to DeepSeek Cloud (High-Reasoning Fix)\n"
                                    "  [3] Abort & Cancel Repair\n"
                                    "Select [1/2/3] (default: 1): "
                                ).strip()
                            except (KeyboardInterrupt, EOFError):
                                user_choice = "3"

                            if user_choice == "2":
                                repair_model = "deepseek-reasoner"
                                print(f"\n🚀 Escalating repair traceback to Cloud [{repair_model}]...")
                            elif user_choice == "3":
                                print("\n🛑 Auto-repair aborted. Restore state with `%deepanalyze --undo` if needed.")
                                break
                            else:
                                repair_model = "deepanalyze-8b"
                                print(f"\n🔄 Retrying repair locally with [{repair_model}]...")

                            if hasattr(err, '__traceback__') and err.__traceback__ is not None:
                                raw_tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                            else:
                                raw_tb = f"{type(err).__name__}: {err}"
                            
                            clean_tb = _sanitize_traceback(raw_tb)

                            repair_prompt = (
                                f"[EXECUTION FAILURE REFLECTION]\n"
                                f"The code you generated failed with the following traceback:\n"
                                f"{clean_tb}\n\n"
                                f"Follow this exact structure to fix it:\n"
                                f"1. Root Cause: (Explain in 1 sentence why this failed based on column types or shapes)\n"
                                f"2. Safe Strategy: (Identify which defensive pandas/duckdb method prevents this)\n"
                                f"3. Output ONLY the final corrected code inside <Answer>```python ... ```</Answer>.\n"
                                f"Context:\n{env_context}"
                            )
                            
                            fixed_raw = _call_llm(repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens, target_model=repair_model)
                            fixed_code, _ = _extract_deepanalyze_content(fixed_raw)
                            is_valid_repair, final_code, rep_err = _lint_and_format_code(fixed_code, available_vars)
                            
                            if is_valid_repair and final_code:
                                _LAST_GENERATED_CODE = final_code
                                print(f"\n[{repair_model} Re-Executing]:\n" + final_code + "\n" + "-" * 40)
                                
                                if parsed_args.insight:
                                    with ipy_io.capture_output() as captured:
                                        result = ip.run_cell(final_code)
                                    captured_output = captured
                                    if captured.stdout: sys.stdout.write(captured.stdout)
                                    if captured.stderr: sys.stderr.write(captured.stderr)
                                else:
                                    result = ip.run_cell(final_code)
                                
                                if not result.error_in_exec:
                                    _reconcile_target_dataframe(ip, final_code, prompt, parsed_args.target)
                                    print("✅ Auto-repair succeeded!")
                                    break
                                err = result.error_in_exec
                            else:
                                print(f"❌ Failed to parse valid python code during repair.")
                                break

                    if parsed_args.insight and not result.error_in_exec:
                        out_str = captured_output.stdout.strip() if (captured_output and captured_output.stdout) else "Execution succeeded but produced no console output."
                        print(f"\n🔍 [{active_model} Insights Synthesis]:")
                        if parsed_args.persona == "exec":
                            insight_sys = "You are a Chief Data & Analytics Officer / Executive Strategist. Synthesize high-level business impact, ROI, and actionable strategic directives based on this data output in concise executive bullet points."
                        elif parsed_args.persona == "dev":
                            insight_sys = "You are a Lead Data Engineer / ML Infrastructure Architect. Detail raw statistical distributions, schema anomalies, and potential data pipeline edge cases based on this data output in concise technical bullet points."
                        else:
                            insight_sys = "You are a senior data analyst. Provide 2-3 concise, actionable business bullet points based on this data output."
                        insight_prompt = f"User Request: {prompt}\n\nExecution Output:\n{out_str}\n\nProvide the insights."
                        raw_insights = _call_llm(insight_prompt, insight_sys, temp=0.3, max_tokens=1000, target_model=active_model)
                        clean_insights = re.sub(r'<think>.*?</think>', '', raw_insights, flags=re.DOTALL | re.IGNORECASE).strip()
                        print(clean_insights)
                        print("\n" + "-" * 40)

                else:
                    ip.set_next_input(clean_code)
                    if parsed_args.spark and ip and parsed_args.target in ip.user_ns:
                        _render_sparkline_minimap(ip.user_ns.get(parsed_args.target), target_name=parsed_args.target)
                    if not is_valid:
                        print(f"[DeepAnalyze Warning]: Static validation issue: {lint_error}. Code placed below.")
                    else:
                        print("[DeepAnalyze]: Verified code placed below. Press Enter to run.")

    except Exception as e:
        sys.stdout.write("\r" + " " * 55 + "\r")
        print(f"[DeepAnalyze Error]: Request failed. ({e})")

def deepanalyze_interceptor(lines):
    global _INTERCEPTOR_ACTIVE
    if not lines:
        return lines

    first_line = lines[0].strip()
    if _INTERCEPTOR_ACTIVE:
        if (first_line.startswith("%") or 
            first_line.startswith("!") or 
            first_line.startswith("import ") or 
            first_line.startswith("from ") or 
            "=" in first_line or
            not first_line):
            return lines
        
        full_cell = "".join(lines).strip()
        return [f"""get_ipython().run_line_magic('deepanalyze', '-x {full_cell}')\n"""]

    return lines


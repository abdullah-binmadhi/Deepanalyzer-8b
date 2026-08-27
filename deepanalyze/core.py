import argparse
import ast
import builtins
import datetime
import difflib
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
        "[GENERAL CODING RULES]:\n"
        "1. Check the active environment context to determine if the target DataFrame is Pandas or Polars.\n"
        "2. If Pandas: modify `df` in-place. For unflattened/hierarchical reports, assign `df = pd.DataFrame(records)`.\n"
        "3. If Polars: use strict Polars expressions and assign back to the target variable (e.g., `df = df.with_columns(pl.col(...))`).\n"
        "4. NEVER use deprecated pandas methods (.append(), .applymap()). Use pd.concat() and .map().\n"
        "5. Output executable Python code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "unravel": (
        "[HIERARCHICAL ERP REPORT FLATTENING RULEBOOK]:\n"
        "You parse messy, unstructured multi-row ERP accounting exports into clean 2D tabular DataFrames.\n"
        "Layout & Defensive Parsing Directives:\n"
        "1. POSITIONAL ACCESS & STRING CASTER: Inside `for _, row in df.iterrows():`, define `c` strictly on `row`: `c = lambda idx: str(row.iloc[idx]).strip() if pd.notna(row.iloc[idx]) else ''`. NEVER use `df.iloc[idx]` inside `c()`.\n"
        "2. HORIZONTAL HEADER PARSING: Header rows contain multiple key-value pairs horizontally across columns:\n"
        "   `if c(0) == 'Doc. No': active_doc = {'doc_no': c(2), 'doc_date': c(4), 'customer': c(6)}`\n"
        "3. DETAIL LINE ITEMS: Detect line items with `if c(0).isdigit() and c(1):`. Append item dictionary to `records` and set `last_item = item`.\n"
        "4. NUMERIC CASTING: Use `float(c(idx).replace(',', '')) if c(idx).replace('.', '', 1).isdigit() else 0.0`\n"
        "5. WRAPPED TRAILING TEXT: If `not c(0) and c(2) and last_item is not None and c(2).startswith('-')`, append cleanly to the PRECEDING item:\n"
        "   `last_item['description'] += ' ' + c(2).lstrip('- ').strip()`\n"
        "6. VARIABLE NAME INVARIANT: Pass the exact target variable name to helper functions (e.g., `df_erp = clean_erp_data(df_erp)`).\n"
        "7. TERMINATION & OUTPUT: Stop at summary rows (`if c(0) == 'Grand Total': break`). Assign final output to the target DataFrame variable.\n"
        "8. Output ONLY executable code inside <Answer>```python\n...\n```</Answer>.\n"
    ),
    "feature": (
        "[FEATURE ENGINEERING & WRANGLING RULEBOOK]:\n"
        "1. Check if the engine is Polars or Pandas from the context and transform directly.\n"
        "2. PANDAS SAFE NUMERIC CASTING: ALWAYS use `pd.to_numeric(df['col'].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0)` instead of `.astype(float)`.\n"
        "3. POLARS SAFE NUMERIC CASTING: Use `pl.col('col').str.replace_all(r'[^0-9.-]', '').cast(pl.Float64, strict=False).fill_null(0)`.\n"
        "4. Protect against division by zero: Pandas (`np.where(denom == 0, np.nan, num / denom)`), Polars (`pl.when(denom == 0).then(None).otherwise(num / denom)`).\n"
        "5. Output executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "sql": (
        "[DUCKDB SQL RULEBOOK]:\n"
        "1. Query Pandas and Polars DataFrames directly using DuckDB without opening new connections (e.g., `df_erp = duckdb.sql('SELECT customer, SUM(total) AS total_revenue FROM df_erp GROUP BY customer').df()`).\n"
        "2. Keep SQL clean and standard. Reference columns directly by their exact names (e.g., `customer`, `total`).\n"
        "3. Column names must NEVER be enclosed in single quotes (single quotes are string literals in SQL).\n"
        "4. Always assign the resulting DataFrame back to the target variable (e.g., `df_erp = ...`).\n"
        "5. Output executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "viz": (
        "[SEABORN / MATPLOTLIB VISUALIZATION RULEBOOK]:\n"
        "1. Always set fig, ax = plt.subplots(figsize=(10, 6)) and sns.set_theme(style='whitegrid').\n"
        "2. Always call plt.tight_layout() before plt.show().\n"
        "3. Output code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "stat": (
        "[SCIPY / STATISTICAL TESTING RULEBOOK]:\n"
        "1. Verify assumptions and use appropriate parametric or non-parametric tests.\n"
        "2. Print test statistic, p-value, and conclusion.\n"
        "3. Output code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "ml": (
        "[SCIKIT-LEARN MACHINE LEARNING RULEBOOK]:\n"
        "1. Bundle preprocessing and estimators using Pipeline and ColumnTransformer.\n"
        "2. Verify: assert not pd.isna(y_pred).any(), 'Nulls in predictions'\n"
        "3. Print: print(classification_report(y_test, y_pred, zero_division=0))\n"
        "4. Output code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "profile": (
        "[STRATEGIC DATASET PROFILING RULEBOOK]:\n"
        "1. Provide a markdown summary: Executive Abstract, Data Health Audit, and Strategic Recommendations.\n"
        "2. In the Python block: Safely sample non-null values with df[c].dropna().iloc[:3].\n"
        "3. Output clean diagnostic code in <Answer>```python ... ```</Answer>.\n"
    ),
    "repair": (
        "[AUTONOMOUS STATIC & RUNTIME REPAIR RULEBOOK]:\n"
        "1. Fix the error/traceback and output ONLY pure, runnable Python code in <Answer>```python ... ```</Answer>.\n"
    ),
    "validate": (
        "[RIGOROUS STATISTICAL & ML VALIDATION RULEBOOK]:\n"
        "1. For ML tasks, always implement cross-validation or stratified holdout splits.\n"
        "2. Print comprehensive evaluation metrics (Confusion Matrix, Classification Report, RMSE, or R2 depending on task).\n"
        "3. Include structural assertions (e.g., shape matching, target bounds, no NaN in predictions). DO NOT assert arbitrary metric threshold minimums unless explicitly requested.\n"
        "4. Output executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "tune": (
        "[LEAK-FREE HYPERPARAMETER TUNING & PIPELINE RULEBOOK]:\n"
        "1. DATA EXTRACTION: Always extract feature matrix `X = df[feature_cols]` and target vector `y = df[target_col]` from the active DataFrame.\n"
        "2. ZERO LEAKAGE PIPELINE: Wrap all scalers, imputers, and estimators inside a `Pipeline` or `ColumnTransformer`.\n"
        "3. FIT BEFORE ACCESS: Always call `grid.fit(X, y)` before accessing `grid.best_params_` or `grid.best_score_`.\n"
        "4. Output pure, executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "explain": (
        "[MODEL INTERPRETABILITY & EXPLAINABILITY RULEBOOK]:\n"
        "1. After training any ML model, extract and print feature importances (e.g., `model.feature_importances_` for tree-based models or coefficients for linear models).\n"
        "2. Rank features from highest to lowest impact and print them clearly.\n"
        "3. Include an assertion verifying that feature importance weights sum to expected bounds or that all features are accounted for.\n"
        "4. Output executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "polars": (
        "[POLARS SEQUENTIAL RULEBOOK - CRITICAL]:\n"
        "1. NEVER chain everything into a single massive .with_columns() block.\n"
        "2. Perform transformations step-by-step using separate assignment lines so casts take effect immediately:\n"
        "   - Step 1 (String cleanup): `df = df.with_columns(Client_Code=pl.col('Client_Code').str.strip_chars().str.to_uppercase(), Status=pl.col('Status').str.strip_chars().str.to_uppercase())`\n"
        "   - Step 2 (Parse Gross_Amount): `df = df.with_columns(Gross_Amount=pl.col('Gross_Amount').str.replace_all(r'[\\$,]', '').cast(pl.Float64, strict=False))`\n"
        "   - Step 3 (Parse Discount_Pct): `df = df.with_columns(Discount_Pct=pl.col('Discount_Pct').str.replace_all('%', '').cast(pl.Float64, strict=False).fill_null(0.0) / 100.0)`\n"
        "   - Step 4 (Compute Net_Amount): `df = df.with_columns(Net_Amount=pl.col('Gross_Amount') * (1.0 - pl.col('Discount_Pct')))`\n"
        "   - Step 5 (Filter & Sort): `df = df.filter((pl.col('Category').is_in(['Hardware', 'Consulting'])) & (pl.col('Status') == 'COMPLETED')).sort('Net_Amount', descending=True)`\n"
        "3. Assign final output to `summary_pldf = df`.\n"
        "4. Output executable code inside <Answer>```python ... ```</Answer>.\n"
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
    global _DF_SNAPSHOTS
    if ip and target in ip.user_ns:
        obj = ip.user_ns[target]
        if isinstance(obj, pd.DataFrame):
            _DF_SNAPSHOTS[target] = obj.copy(deep=True)
        elif pl is not None and isinstance(obj, pl.DataFrame):
            _DF_SNAPSHOTS[target] = obj.clone()

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
    
    sys.stdout.write(f"\r[{engine_name} ({target_model})]: Processing request...")
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
            if token_count % 20 == 0:
                sys.stdout.write(f"\r[{engine_name}]: Generating... ({token_count} tokens)")
                sys.stdout.flush()

    sys.stdout.write("\r" + " " * 75 + "\r")
    sys.stdout.flush()
    return "".join(full_text)


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

    if not prompt and primary_skill != "profile" and not parsed_args.is_continuation:
        print("Usage: %deepanalyze [-x] [--target df] [--think|--pro] [--privacy auto|mask|profile] <task description>")
        return

    _sync_duckdb(ip)

    temp, max_tokens = (0.0, 3500) if parsed_args.deterministic else (0.7, 3500)
    if parsed_args.fast: temp, max_tokens = 0.0, 1000
    elif parsed_args.ultra: temp, max_tokens = 0.05, 4096

    env_context, available_vars, knife = _get_deep_workspace_context(
        ip, target=parsed_args.target, is_cloud=is_cloud_call, privacy_mode=parsed_args.privacy
    )
    
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
        
        raw_output = _call_llm(full_prompt, system_prompt, temp=temp, max_tokens=max_tokens, target_model=active_model)
        code, narrative = _extract_deepanalyze_content(raw_output)
        
        # Calculate duration and rough token count, then display metrics
        duration = time.time() - start_time
        token_count = len(raw_output) // 4  # Standard heuristic: ~4 chars per token
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

            if clean_code:
                _LAST_GENERATED_CODE = clean_code
                _LAST_USER_PROMPT = prompt

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
                        insight_sys = "You are a senior data analyst. Provide 2-3 concise, actionable business bullet points based on this data output."
                        insight_prompt = f"User Request: {prompt}\n\nExecution Output:\n{out_str}\n\nProvide the insights."
                        raw_insights = _call_llm(insight_prompt, insight_sys, temp=0.3, max_tokens=1000, target_model=active_model)
                        clean_insights = re.sub(r'<think>.*?</think>', '', raw_insights, flags=re.DOTALL | re.IGNORECASE).strip()
                        print(clean_insights)
                        print("\n" + "-" * 40)

                else:
                    ip.set_next_input(clean_code)
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


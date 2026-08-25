import ast
import builtins
import datetime
import difflib
import json
import os
import re
import sys
import httpx
import shlex
import argparse
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from IPython.core.magic import register_line_cell_magic
from IPython import get_ipython
from IPython.utils import io as ipy_io
from openai import OpenAI

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

KNOWN_GLOBAL_SYMBOLS = {
    "pd", "np", "plt", "sns", "duckdb", "scipy", "stats", "sklearn",
    "math", "os", "sys", "re", "json", "datetime", "warnings", "difflib",
    "con", "_DUCKDB_CON", "True", "False", "None"
} | set(dir(builtins))

TRANSIENT_VARS = {
    "X", "y", "X_train", "X_test", "y_train", "y_test", "y_pred",
    "result", "mean_pp", "group_normal", "group_htn", "stat", "p_value",
    "column_summary", "null_percentage", "executive_summary", "data_health_audit", "strategic_roadmap",
    "records", "item", "last_item", "df_flat", "clean_df"
}

def _get_client():
    return OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="none",
        http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(180.0, connect=10.0))
    )

SKILL_RULEBOOKS = {
    "general": (
        "[GENERAL CODING RULES]:\n"
        "1. For clean tabular data, modify `df` in-place. For unflattened/hierarchical reports, assign `df = pd.DataFrame(records)`.\n"
        "2. NEVER use deprecated pandas methods (.append(), .applymap()). Use pd.concat() and .map().\n"
        "3. Output executable Python code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "unravel": (
        "[HIERARCHICAL ERP REPORT FLATTENING RULEBOOK]:\n"
        "You parse messy, unstructured multi-row ERP accounting exports into clean 2D tabular DataFrames.\n"
        "Layout & Defensive Parsing Directives:\n"
        "1. Positional Access: Access raw cells with `row.iloc[col_index]`.\n"
        "2. TYPE-SAFETY INVARIANT: Sub-header rows contain string text (e.g. 'Quantity', 'Unit Price'). NEVER convert float(row.iloc[col]) at the loop header. Cast to float ONLY inside line-item conditions (`if c0.isdigit():`) using safe conversion:\n"
        "   `qty = float(row.iloc[10]) if pd.notna(row.iloc[10]) and str(row.iloc[10]).replace('.', '', 1).isdigit() else 0.0`\n"
        "3. Output Variable Invariant: Assign the parsed records directly to `df = pd.DataFrame(records)`. DO NOT use `df_flat` or `clean_df`.\n"
        "4. Parsing Pattern:\n"
        "   - Identify invoice headers (e.g. `c0.startswith('IV-')`) and update active invoice state.\n"
        "   - Identify detail rows (e.g. `c0.isdigit()` and `c1 != ''`) and append records.\n"
        "   - Identify wrapped multiline text (e.g. `not c0 and c3 and last_item is not None`) and append to `last_item['description']`.\n"
        "   - Stop processing when reaching summary sections ('grand total', 'account summary', 'item code summary').\n"
        "5. Output ONLY executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "feature": (
        "[FEATURE ENGINEERING & WRANGLING RULEBOOK]:\n"
        "1. Transform target DataFrame directly (e.g. `sales_data['col'] = ...`). DO NOT create `df_clean` or copy variables unless requested.\n"
        "2. SAFE NUMERIC CASTING: ALWAYS use `pd.to_numeric(df['col'].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0)` instead of `.astype(float)`.\n"
        "3. Protect against division by zero: `np.where(denom == 0, np.nan, num / denom)`.\n"
        "4. Assertions: assert len(df) > 0, 'DataFrame is empty'\n"
        "5. Output executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "sql": (
        "[DUCKDB SQL RULEBOOK]:\n"
        "1. Query the target DataFrame directly using DuckDB in-memory session (e.g. `duckdb.query('SELECT ... FROM df').df()`).\n"
        "2. Keep SQL clean and standard. Reference columns directly without quotes or string slicing (e.g. `AVG(temperature_c)`).\n"
        "3. Column names must NEVER be enclosed in single quotes (single quotes are string literals in SQL).\n"
        "4. Output executable code inside <Answer>```python ... ```</Answer>.\n"
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
    )
}

INVARIANT_CHECKLIST = (
    "\n[STRICT INVARIANT DIRECTIVE]:\n"
    "1. DO NOT output conversational preamble or explanations outside code tags.\n"
    "2. Enclose code strictly inside <Answer>```python ... ```</Answer>.\n"
)

def check_engine_status(server_url=DEFAULT_SERVER_URL):
    """Probes the llama-server health/props endpoints and kernel interceptor state."""
    print("=" * 60)
    print("🔍 DeepAnalyze-8B System & Engine Status")
    print("=" * 60)

    health_url = f"{server_url}/health"
    props_url = f"{server_url}/props"
    server_online = False
    server_props = {}

    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "DeepAnalyze-Client"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                server_online = True
    except Exception:
        server_online = False

    if not server_online:
        print("❌ Server Status     : Offline / Unreachable")
        print(f"   Endpoint          : {server_url}")
        print("   Troubleshoot      : Ensure `llama-server` is running on port 8080.")
    else:
        print("✅ Server Status     : Online & Healthy")
        print(f"   Endpoint          : {server_url}")
        try:
            req_props = urllib.request.Request(props_url, headers={"User-Agent": "DeepAnalyze-Client"})
            with urllib.request.urlopen(req_props, timeout=2) as resp:
                server_props = json.loads(resp.read().decode("utf-8"))
        except Exception:
            server_props = {}

        if server_props:
            default_gen = server_props.get("default_generation_settings", {})
            n_ctx = default_gen.get("n_ctx", "Unknown")
            model_path = server_props.get("model_path", "Loaded GGUF")
            print(f"   Active Model      : {model_path.split('/')[-1]}")
            print(f"   Context Allocation: {n_ctx} tokens")
        else:
            print("   Context Allocation: Active (16K configured)")

    interceptor_status = "🟢 Enabled (Auto-pilot on plain English cells)" if _INTERCEPTOR_ACTIVE else "⚪ Disabled (Explicit %deepanalyze calls only)"
    print(f"\n📡 Cell Interceptor  : {interceptor_status}")

    ip = get_ipython()
    tracked_dfs = list(_DF_SNAPSHOTS.keys())
    if tracked_dfs:
        print(f"\n💾 State Snapshots   : {len(tracked_dfs)} active DataFrame rollback points")
        for var_name in tracked_dfs:
            shape = getattr(ip.user_ns.get(var_name), "shape", "Unknown shape")
            print(f"   • {var_name} -> {shape}")
    else:
        print("\n💾 State Snapshots   : No active snapshots (clean state)")

    print("=" * 60)

def _take_snapshot(ip, target="df"):
    global _DF_SNAPSHOTS
    if ip and target in ip.user_ns and isinstance(ip.user_ns[target], pd.DataFrame):
        _DF_SNAPSHOTS[target] = ip.user_ns[target].copy(deep=True)

def _restore_snapshot(ip, target="df") -> bool:
    global _DF_SNAPSHOTS
    if ip and target in _DF_SNAPSHOTS and _DF_SNAPSHOTS[target] is not None:
        ip.user_ns[target] = _DF_SNAPSHOTS[target].copy(deep=True)
        return True
    return False

def _sync_duckdb(ip):
    if not _DUCKDB_CON or not ip:
        return
    for k, v in ip.user_ns.items():
        if isinstance(v, pd.DataFrame) and not k.startswith("_") and k not in TRANSIENT_VARS:
            try:
                _DUCKDB_CON.register(k, v)
            except Exception:
                pass

def _fuzzy_match_columns(prompt: str, ip, target="df") -> str:
    if not ip or target not in ip.user_ns:
        return ""
    df = ip.user_ns[target]
    if not isinstance(df, pd.DataFrame):
        return ""

    cols = [str(c) for c in df.columns]
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

def _get_deep_workspace_context(ip) -> tuple[str, set]:
    if not ip:
        return "", set(KNOWN_GLOBAL_SYMBOLS)

    available_vars = set(KNOWN_GLOBAL_SYMBOLS)
    context_lines = []

    for name, obj in ip.user_ns.items():
        if name.startswith("_") or name in ("In", "Out", "exit", "quit", "get_ipython"):
            continue

        available_vars.add(name)

        if name in TRANSIENT_VARS:
            continue

        if isinstance(obj, pd.DataFrame):
            is_unnamed = any("unnamed" in str(c).lower() or isinstance(c, (int, np.integer)) for c in obj.columns)
            
            if is_unnamed or obj.shape[0] < 20:
                grid_sample = obj.iloc[:18, :12].to_string()
                context_lines.append(
                    f"DataFrame `{name}` (Raw Grid Matrix, Shape: {obj.shape[0]} rows, {obj.shape[1]} cols):\n"
                    f"Top 18 Rows Matrix Preview:\n{grid_sample}"
                )
            else:
                col_profiles = []
                for col in obj.columns:
                    col_str = str(col)
                    dtype = str(obj[col].dtype)
                    null_pct = round(obj[col].isna().mean() * 100, 1)
                    # RICH DATA PROFILING: Added unique cardinality count to prevent coercion errors
                    unique_count = obj[col].nunique()
                    sample = obj[col].dropna().iloc[0] if not obj[col].dropna().empty else "None"
                    col_profiles.append(f"    - '{col_str}' ({dtype}) | Nulls: {null_pct}% | Unique: {unique_count} | Sample: {sample}")

                context_lines.append(
                    f"DataFrame `{name}` (Shape: {obj.shape[0]} rows, {obj.shape[1]} cols):\n"
                    f"  Exact Column Names (CASE-SENSITIVE):\n" + "\n".join(col_profiles)
                )

    context_str = (
        "\n--- ACTIVE RUNTIME ENVIRONMENT CONTEXT ---\n"
        + ("\n\n".join(context_lines) if context_lines else "No custom DataFrames loaded.")
        + "\n-------------------------------------------\n"
    )
    return context_str, available_vars

def _lint_and_format_code(code_str: str, available_vars: set) -> tuple[bool, str, str]:
    if not code_str.strip():
        return False, "", "Empty code block"
    try:
        tree = ast.parse(code_str)
        normalized_code = ast.unparse(tree)
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

def _extract_deepanalyze_content(text: str) -> tuple[str, str]:
    raw_blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)(?:```|$)", text, flags=re.DOTALL | re.IGNORECASE)
    code = raw_blocks[0].strip() if raw_blocks else ""

    if not code:
        answer_match = re.search(r"<Answer>(.*?)(?:</Answer>|$)", text, flags=re.DOTALL | re.IGNORECASE)
        if answer_match:
            code = answer_match.group(1).strip()
            code = re.sub(r"```(?:python|py)?", "", code).replace("```", "").strip()

    narrative = re.sub(r"<Analyze>.*?(?:</Analyze>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    narrative = re.sub(r"<think>.*?(?:</think>|$)", "", narrative, flags=re.DOTALL | re.IGNORECASE)
    narrative = re.sub(r"</?(?:Analyze|think|Answer|Code)>", "", narrative, flags=re.IGNORECASE)
    narrative = re.sub(r"```(?:python|py|sql)?\s*\n?.*?(?:```|$)", "", narrative, flags=re.DOTALL | re.IGNORECASE).strip()

    return code, narrative

def _call_llm(prompt: str, system_prompt: str, temp: float = 0.0, max_tokens: int = 3500) -> str:
    sys.stdout.write("[DeepAnalyze]: Analyzing report topology & invariants...")
    sys.stdout.flush()

    client = _get_client()
    response = client.chat.completions.create(
        model="deepanalyze-8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        stop=["</Answer>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"],
        stream=True,
    )

    full_text = []
    token_count = 0
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full_text.append(delta)
            token_count += 1
            if token_count % 20 == 0:
                sys.stdout.write(f"\r[DeepAnalyze]: Generating text... ({token_count} tokens)")
                sys.stdout.flush()

    sys.stdout.write("\r" + " " * 55 + "\r")
    sys.stdout.flush()
    return "".join(full_text)

@register_line_cell_magic
def deepanalyze(line, cell=None):
    global _LAST_GENERATED_CODE, _LAST_USER_PROMPT, _INTERCEPTOR_ACTIVE
    raw_input = f"{line}\n{cell}" if cell else line
    ip = get_ipython()

    parser = argparse.ArgumentParser(prog="%deepanalyze", description="Agentic LLM Execution Engine", add_help=False)
    parser.add_argument("--toggle", action="store_true", help="Toggle global cell interceptor on/off")
    parser.add_argument("--status", action="store_true", help="Display server health, context size, and interceptor status")
    parser.add_argument("-x", "--exec", "--execute", dest="execute_code", action="store_true")
    parser.add_argument("-c", "--continue", dest="is_continuation", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--ultra", action="store_true")
    
    # NEW INSIGHT FLAG
    parser.add_argument("-i", "--insight", action="store_true", help="Generate business insights from execution output")
    
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
        parsed_args, remaining_words = parser.parse_known_args(shlex.split(raw_input))
        prompt = " ".join(remaining_words).strip()
    except SystemExit:
        return

    # Handle Toggle Flag
    if parsed_args.toggle:
        _INTERCEPTOR_ACTIVE = not _INTERCEPTOR_ACTIVE
        state_str = "🟢 ENABLED (Auto-pilot active)" if _INTERCEPTOR_ACTIVE else "⚪ DISABLED (Explicit mode)"
        print(f"🔄 DeepAnalyze Interceptor toggled: {state_str}")
        return

    # Handle Status Flag
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

    primary_skill = "general"
    if parsed_args.unravel: primary_skill = "unravel"
    elif parsed_args.profile: primary_skill = "profile"
    elif parsed_args.viz: primary_skill = "viz"
    elif parsed_args.sql: primary_skill = "sql"
    elif parsed_args.feat: primary_skill = "feature"
    elif parsed_args.stat: primary_skill = "stat"
    elif parsed_args.ml: primary_skill = "ml"
    elif parsed_args.repair: primary_skill = "repair"

    if not prompt and primary_skill != "profile" and not parsed_args.is_continuation:
        print("Usage: %deepanalyze [-x] [--target df] [-u|-p|-v|-s|-f|-t|-m|-r|-i|--undo|--toggle|--status] <task description>")
        return

    _sync_duckdb(ip)

    temp, max_tokens = (0.0, 3500) if parsed_args.deterministic else (0.7, 3500)
    if parsed_args.fast: temp, max_tokens = 0.0, 1000
    elif parsed_args.ultra: temp, max_tokens = 0.05, 4096

    env_context, available_vars = _get_deep_workspace_context(ip)
    fuzzy_aliases = _fuzzy_match_columns(prompt, ip, target=parsed_args.target)
    rulebook = SKILL_RULEBOOKS.get(primary_skill, SKILL_RULEBOOKS["general"])

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
        raw_output = _call_llm(full_prompt, system_prompt, temp=temp, max_tokens=max_tokens)
        code, narrative = _extract_deepanalyze_content(raw_output)

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
                fixed_raw = _call_llm(repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens)
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

                    print("[DeepAnalyze Executing]:\n" + clean_code + "\n" + "-" * 40)
                    
                    # Execution Block with optional Insight Capture
                    captured_output = None
                    if parsed_args.insight:
                        with ipy_io.capture_output() as captured:
                            result = ip.run_cell(clean_code)
                        captured_output = captured
                        # Print captured stdout/stderr so the user still sees the output
                        if captured.stdout: sys.stdout.write(captured.stdout)
                        if captured.stderr: sys.stderr.write(captured.stderr)
                    else:
                        result = ip.run_cell(clean_code)

                    for alias in ("df_flat", "clean_df", "df_clean", "records_df"):
                        if alias in ip.user_ns and isinstance(ip.user_ns[alias], pd.DataFrame):
                            if original_row_count and parsed_args.target in ip.user_ns and ip.user_ns[parsed_args.target].shape[0] == original_row_count:
                                ip.user_ns[parsed_args.target] = ip.user_ns[alias]
                            break

                    if result.error_in_exec and parsed_args.retries > 0:
                        err = result.error_in_exec
                        for attempt in range(1, parsed_args.retries + 1):
                            print(f"\n[DeepAnalyze Runtime Auto-Repair {attempt}/{parsed_args.retries}]: Caught {type(err).__name__}: {err}")
                            
                            # STRUCTURED ERROR REFLECTION: Forces Root Cause Analysis
                            repair_prompt = (
                                f"[EXECUTION FAILURE REFLECTION]\n"
                                f"The code you generated failed with the following traceback:\n"
                                f"{type(err).__name__}: {err}\n\n"
                                f"Follow this exact structure to fix it:\n"
                                f"1. Root Cause: (Explain in 1 sentence why this failed based on column types or shapes)\n"
                                f"2. Safe Strategy: (Identify which defensive pandas/duckdb method prevents this)\n"
                                f"3. Output ONLY the final corrected code inside <Answer>```python ... ```</Answer>.\n"
                                f"Context:\n{env_context}"
                            )
                            
                            fixed_raw = _call_llm(repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens)
                            fixed_code, _ = _extract_deepanalyze_content(fixed_raw)
                            is_valid_repair, final_code, rep_err = _lint_and_format_code(fixed_code, available_vars)
                            
                            if is_valid_repair and final_code:
                                _LAST_GENERATED_CODE = final_code
                                print("[DeepAnalyze Re-Executing]:\n" + final_code + "\n" + "-" * 40)
                                
                                if parsed_args.insight:
                                    with ipy_io.capture_output() as captured:
                                        result = ip.run_cell(final_code)
                                    captured_output = captured
                                    if captured.stdout: sys.stdout.write(captured.stdout)
                                    if captured.stderr: sys.stderr.write(captured.stderr)
                                else:
                                    result = ip.run_cell(final_code)
                                
                                if not result.error_in_exec:
                                    for alias in ("df_flat", "clean_df", "df_clean", "records_df"):
                                        if alias in ip.user_ns and isinstance(ip.user_ns[alias], pd.DataFrame):
                                            ip.user_ns[parsed_args.target] = ip.user_ns[alias]
                                            break
                                    break
                                err = result.error_in_exec
                            else:
                                break

                    # INSIGHT SYNTHESIS POST-EXECUTION
                    if parsed_args.insight and not result.error_in_exec:
                        out_str = captured_output.stdout.strip() if (captured_output and captured_output.stdout) else "Execution succeeded but produced no console output."
                        print("\n🔍 [DeepAnalyze Insights Synthesis]:")
                        insight_sys = "You are a senior data analyst. Provide 2-3 concise, actionable business bullet points based on this data output."
                        insight_prompt = f"User Request: {prompt}\n\nExecution Output:\n{out_str}\n\nProvide the insights."
                        _call_llm(insight_prompt, insight_sys, temp=0.3, max_tokens=1000)
                        print("\n" + "-" * 40)

                else:
                    ip.set_next_input(clean_code)
                    if not is_valid:
                        print(f"[DeepAnalyze Warning]: Static validation could not auto-resolve: {lint_error}. Code placed below.")
                    else:
                        print("[DeepAnalyze]: Verified code placed below. Press Enter to run.")

    except Exception as e:
        sys.stdout.write("\r" + " " * 55 + "\r")
        print(f"[DeepAnalyze Error]: Request failed. ({e})")

def deepanalyze_interceptor(lines):
    """Intercepts plain-text lines when auto-pilot is enabled."""
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
        # Use triple quotes (''' ... ''') so any quotes inside the prompt are fully safe
        return [f"""get_ipython().run_line_magic('deepanalyze', '-x {full_cell}')\n"""]

    return lines

ip = get_ipython()
if ip and deepanalyze_interceptor not in ip.input_transformers_cleanup:
    ip.input_transformers_cleanup.append(deepanalyze_interceptor)
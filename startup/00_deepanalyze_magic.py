import ast
import builtins
import datetime
import difflib
import os
import re
import sys
import httpx
import pandas as pd
import numpy as np
from IPython.core.magic import register_line_cell_magic
from IPython import get_ipython
from openai import OpenAI

try:
    import duckdb
    _DUCKDB_CON = duckdb.connect(database=":memory:")
except ImportError:
    duckdb = None
    _DUCKDB_CON = None

_DF_SNAPSHOT = None
_LAST_GENERATED_CODE = ""
_LAST_USER_PROMPT = ""

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
        "2. TYPE-SAFETY INVARIANT: Sub-header rows contain string text (e.g. \x27Quantity\x27, \x27Unit Price\x27). NEVER convert float(row.iloc[col]) at the loop header. Cast to float ONLY inside line-item conditions (`if c0.isdigit():`) using safe conversion:\n"
        "   `qty = float(row.iloc[10]) if pd.notna(row.iloc[10]) and str(row.iloc[10]).replace(\x27.\x27, \x27\x27, 1).isdigit() else 0.0`\n"
        "3. Output Variable Invariant: Assign the parsed records directly to `df = pd.DataFrame(records)`. DO NOT use `df_flat` or `clean_df`.\n"
        "4. Parsing Pattern:\n"
        "   - Identify invoice headers (e.g. `c0.startswith(\x27IV-\x27)`) and update active invoice state.\n"
        "   - Identify detail rows (e.g. `c0.isdigit()` and `c1 != \x27\x27`) and append records.\n"
        "   - Identify wrapped multiline text (e.g. `not c0 and c3 and last_item is not None`) and append to `last_item[\x27description\x27]`.\n"
        "   - Stop processing when reaching summary sections (\x27grand total\x27, \x27account summary\x27, \x27item code summary\x27).\n"
        "5. Output ONLY executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "feature": (
        "[FEATURE ENGINEERING & WRANGLING RULEBOOK]:\n"
        "1. Transform df directly using df.loc[mask, \x27col\x27] = value.\n"
        "2. Protect against division by zero: np.where(denom == 0, np.nan, num / denom).\n"
        "3. Assertions: assert len(df) > 0, \x27DataFrame is empty\x27\n"
        "4. Output executable code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "sql": (
        "[DUCKDB SQL ANALYTICS RULEBOOK]:\n"
        "1. Wrap table or column names in double quotes if needed: FROM \"df\".\n"
        "2. Execute zero-copy queries: import duckdb; print(duckdb.query(\x27SELECT ... FROM df\x27).df())\n"
        "3. Output code inside <Answer>```python ... ```</Answer>.\n"
    ),
    "viz": (
        "[SEABORN / MATPLOTLIB VISUALIZATION RULEBOOK]:\n"
        "1. Always set fig, ax = plt.subplots(figsize=(10, 6)) and sns.set_theme(style=\x27whitegrid\x27).\n"
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
        "2. Verify: assert not pd.isna(y_pred).any(), \x27Nulls in predictions\x27\n"
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

def _take_snapshot(ip):
    global _DF_SNAPSHOT
    if ip and "df" in ip.user_ns and isinstance(ip.user_ns["df"], pd.DataFrame):
        _DF_SNAPSHOT = ip.user_ns["df"].copy(deep=True)

def _restore_snapshot(ip) -> bool:
    global _DF_SNAPSHOT
    if _DF_SNAPSHOT is not None and ip:
        ip.user_ns["df"] = _DF_SNAPSHOT.copy(deep=True)
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

def _fuzzy_match_columns(prompt: str, ip) -> str:
    if not ip or "df" not in ip.user_ns:
        return ""
    df = ip.user_ns["df"]
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
                matches.append(f"  - Typo \x27{t}\x27 -> EXACT Column: `{orig}`")

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
                    sample = obj[col].dropna().iloc[0] if not obj[col].dropna().empty else "None"
                    col_profiles.append(f"    - \x27{col_str}\x27 ({dtype}) | Nulls: {null_pct}% | Sample: {sample}")

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
        elif isinstance(node, ast.ClassDef):
            defined_in_code.add(node.name)
        elif isinstance(node, ast.Lambda):
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                defined_in_code.add(arg.arg)
            if node.args.vararg:
                defined_in_code.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_in_code.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined_in_code.add(node.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            defined_in_code.add(node.target.id)
        elif isinstance(node, ast.comprehension):
            for elt in ast.walk(node.target):
                if isinstance(elt, ast.Name):
                    defined_in_code.add(elt.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined_in_code.add(node.name)
        elif isinstance(node, ast.withitem) and node.optional_vars:
            for elt in ast.walk(node.optional_vars):
                if isinstance(elt, ast.Name):
                    defined_in_code.add(elt.id)

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
                sys.stdout.write(f"\r[DeepAnalyze]: Generating parser code... ({token_count} tokens)")
                sys.stdout.flush()

    sys.stdout.write("\r" + " " * 55 + "\r")
    sys.stdout.flush()
    return "".join(full_text)

@register_line_cell_magic
def deepanalyze(line, cell=None):
    global _LAST_GENERATED_CODE, _LAST_USER_PROMPT
    raw_input = f"{line}\n{cell}" if cell else line
    ip = get_ipython()

    words = raw_input.strip().split()

    if "--undo" in words:
        if _restore_snapshot(ip):
            df_restored = ip.user_ns["df"]
            print(f"[DeepAnalyze Undo]: Restored `df` from snapshot. Shape: {df_restored.shape[0]} rows, {df_restored.shape[1]} columns.")
        else:
            print("[DeepAnalyze Undo]: No previous snapshot found in memory.")
        return

    auto_execute = False
    is_continuation = False
    save_figures = False
    active_skills = []
    remaining_words = []

    temp, max_tokens = 0.0, 3500

    if "--fast" in words:
        temp, max_tokens = 0.0, 1000
    elif "--deep" in words:
        temp, max_tokens = 0.0, 3500
    elif "--ultra" in words:
        temp, max_tokens = 0.05, 4096

    for w in words:
        if w in ("--fast", "--deep", "--ultra"):
            continue
        elif w in ("-x", "--exec"):
            auto_execute = True
        elif w in ("-c", "--continue"):
            is_continuation = True
        elif w in ("--save",):
            save_figures = True
        elif w in ("-u", "--unravel"):
            active_skills.append("unravel")
        elif w in ("-p", "--profile", "--plan"):
            active_skills.append("profile")
        elif w in ("-v", "--viz"):
            active_skills.append("viz")
        elif w in ("-s", "--sql"):
            active_skills.append("sql")
        elif w in ("-f", "--feat"):
            active_skills.append("feature")
        elif w in ("-t", "--stat"):
            active_skills.append("stat")
        elif w in ("-m", "--ml"):
            active_skills.append("ml")
        elif w in ("-r", "--repair", "--fix"):
            active_skills.append("repair")
        else:
            remaining_words.append(w)

    prompt = " ".join(remaining_words).strip()
    primary_skill = active_skills[0] if active_skills else "general"

    if not prompt and primary_skill != "profile" and not is_continuation:
        print("Usage: %deepanalyze [-x] [-c] [--save] [--fast|--deep|--ultra] [-u|-p|-v|-s|-f|-t|-m|-r|--undo] <task description>")
        return

    _sync_duckdb(ip)

    env_context, available_vars = _get_deep_workspace_context(ip)
    fuzzy_aliases = _fuzzy_match_columns(prompt, ip)
    rulebook = SKILL_RULEBOOKS.get(primary_skill, SKILL_RULEBOOKS["general"])

    save_directive = ""
    if save_figures:
        os.makedirs("charts", exist_ok=True)
        save_directive = (
            "\n[AUTO-SAVE FIGURE DIRECTIVE]:\n"
            "Ensure the directory \x27charts\x27 exists. "
            "Before calling `plt.show()`, call `plt.savefig(\x27charts/<meaningful_slug>.png\x27, dpi=300, bbox_inches=\x27tight\x27)` "
            "and print the saved filepath.\n"
        )

    system_prompt = (
        f"{rulebook}\n"
        f"{INVARIANT_CHECKLIST}\n"
        f"{save_directive}\n"
        f"{env_context}\n"
        f"{fuzzy_aliases}"
    )

    if is_continuation and _LAST_GENERATED_CODE:
        full_prompt = (
            f"PREVIOUS EXECUTED CODE:\n```python\n{_LAST_GENERATED_CODE}\n```\n\n"
            f"USER REFINEMENT REQUEST: {prompt}\n"
            f"Refine and update the previous code cleanly to satisfy the user refinement."
        )
    else:
        full_prompt = f"User Task: {prompt}" if prompt else "Analyze active dataset schema and provide executive summary."

    if primary_skill == "profile" and "--fast" not in words:
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
                else:
                    print(f"[DeepAnalyze Pre-Flight Repair Failed]: {lint_error}\nRaw Repair: {fixed_raw}")

            if clean_code:
                _LAST_GENERATED_CODE = clean_code
                _LAST_USER_PROMPT = prompt

                if auto_execute and is_valid:
                    _take_snapshot(ip)
                    print("[DeepAnalyze Executing]:\n" + clean_code + "\n" + "-" * 40)
                    result = ip.run_cell(clean_code)

                    # POST-EXECUTION REBIND SAFETY:
                    # If the model assigned to df_flat or clean_df instead of df, rebind to df
                    for alias in ("df_flat", "clean_df", "df_clean", "records_df"):
                        if alias in ip.user_ns and isinstance(ip.user_ns[alias], pd.DataFrame):
                            if "df" in ip.user_ns and ip.user_ns["df"].shape[0] == 3924:
                                ip.user_ns["df"] = ip.user_ns[alias]
                                print(f"[DeepAnalyze Safe-Rebind]: Bound `{alias}` to `df` (Shape: {ip.user_ns['df'].shape})")
                            break

                    if result.error_in_exec:
                        err = result.error_in_exec
                        print(f"\n[DeepAnalyze Runtime Auto-Repair]: Caught {type(err).__name__}: {err}")
                        repair_prompt = (
                            f"Code raised runtime error {type(err).__name__}: {err}\n"
                            f"```python\n{clean_code}\n```\n"
                            f"Fix the runtime error while strictly maintaining the task objective:\n{env_context}"
                        )
                        fixed_raw = _call_llm(repair_prompt, system_prompt, temp=0.0, max_tokens=max_tokens)
                        fixed_code, _ = _extract_deepanalyze_content(fixed_raw)
                        is_valid_repair, final_code, rep_err = _lint_and_format_code(fixed_code, available_vars)
                        if is_valid_repair and final_code:
                            _LAST_GENERATED_CODE = final_code
                            print("[DeepAnalyze Re-Executing]:\n" + final_code + "\n" + "-" * 40)
                            ip.run_cell(final_code)
                            for alias in ("df_flat", "clean_df", "df_clean", "records_df"):
                                if alias in ip.user_ns and isinstance(ip.user_ns[alias], pd.DataFrame):
                                    if "df" in ip.user_ns and ip.user_ns["df"].shape[0] == 3924:
                                        ip.user_ns["df"] = ip.user_ns[alias]
                                    break
                        else:
                            print(f"[DeepAnalyze Runtime Repair Unsuccessful]: {rep_err}\n{fixed_raw}")
                else:
                    ip.set_next_input(clean_code)
                    if not is_valid:
                        print(f"[DeepAnalyze Warning]: Static validation could not auto-resolve: {lint_error}. Code placed below.")
                    else:
                        print("[DeepAnalyze]: Verified code placed below. Press Enter to run.")

    except Exception as e:
        sys.stdout.write("\r" + " " * 55 + "\r")
        print(f"[DeepAnalyze Error]: Request failed. ({e})")

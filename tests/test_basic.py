import os
import argparse
import pytest
import pandas as pd
import numpy as np
import deepanalyze
from deepanalyze.privacy_knife import DeepAnalyzePrivacyKnife, LocalGatekeeper
from deepanalyze.core import _extract_deepanalyze_content, _reconcile_target_dataframe

def test_version():
    assert hasattr(deepanalyze, "__version__")
    assert deepanalyze.__version__ == "3.0.0"

def test_ast_sandbox_allowed_code():
    safe_code = """
import pandas as pd
import numpy as np
df['cleaned_revenue'] = pd.to_numeric(df['gross_revenue'].str.replace('$', ''), errors='coerce').fillna(0)
"""
    assert DeepAnalyzePrivacyKnife.audit_generated_code(safe_code) is True

def test_ast_sandbox_forbidden_imports():
    forbidden_snippets = [
        "import socket",
        "import requests",
        "import urllib.request",
        "import httpx",
        "import subprocess",
        "import shutil",
        "from socket import gethostname",
        "from subprocess import Popen",
    ]
    for snippet in forbidden_snippets:
        with pytest.raises(PermissionError):
            DeepAnalyzePrivacyKnife.audit_generated_code(snippet)

def test_ast_sandbox_forbidden_calls():
    forbidden_calls = [
        "eval('2 + 2')",
        "exec('a = 10')",
        "__import__('os')",
        "compile('x = 1', '<string>', 'exec')",
    ]
    for snippet in forbidden_calls:
        with pytest.raises(PermissionError):
            DeepAnalyzePrivacyKnife.audit_generated_code(snippet)

def test_ast_sandbox_forbidden_os_operations():
    forbidden_os = [
        "import os; os.system('echo pwned')",
        "import os; os.popen('whoami')",
        "import os; os.remove('data.csv')",
        "import os; os.unlink('temp.txt')",
        "import os; os.rmdir('dir')",
    ]
    for snippet in forbidden_os:
        with pytest.raises(PermissionError):
            DeepAnalyzePrivacyKnife.audit_generated_code(snippet)

def test_privacy_knife_structural_erp_masking():
    df = pd.DataFrame([
        ["Doc. No", " : ", "IV-88201", "Doc. Date", "2025-08-10", "Customer", "Apex Global"],
        ["Seq", "Item Code", "Description", "Qty", "UOM", "Unit Price", "Total"],
        ["1000", "SRV-901", "Cloud Infrastructure Setup", "2.0", "UNIT", "1,500.00", "3,000.00"],
    ])
    knife = DeepAnalyzePrivacyKnife(df)
    masked = knife.mask_structural_erp()
    
    # Structural keywords must be preserved
    assert masked.iloc[0, 0] == "Doc. No"
    assert masked.iloc[0, 1] == " : "
    assert masked.iloc[0, 3] == "Doc. Date"
    assert masked.iloc[0, 5] == "Customer"
    assert masked.iloc[1, 0] == "Seq"
    assert masked.iloc[1, 6] == "Total"
    
    # Sensitive alphanumeric values must be masked
    assert masked.iloc[0, 2] == "XX-99999"

def test_privacy_knife_structural_erp_masking_polars_sparse_nulls():
    import polars as pl
    # Simulate a Polars dataframe where early rows are null and later rows have strings/numbers
    data = {
        "doc_no": ["IV-11319"] + [None] * 150 + ["IV-99999"],
        "customer_name": [None] * 150 + ["Acme Inc"] + [None],
        "quantity": [10.0] + [None] * 150 + [5.0]
    }
    df = pl.DataFrame(data)
    knife = DeepAnalyzePrivacyKnife(df)
    masked = knife.mask_structural_erp()
    assert masked.height == 152
    assert masked["doc_no"][0] == "XX-99999"
    assert masked["customer_name"][150] == "Xxxx Xxx"
    assert masked["quantity"][0] == "99.9"

def test_privacy_knife_pii_tokenization():
    df = pd.DataFrame({
        "customer_name": ["Alice Corp", "Bob LLC"],
        "email": ["alice@corp.com", "bob@llc.com"],
        "revenue": [100.0, 200.0]
    })
    knife = DeepAnalyzePrivacyKnife(df)
    tokenized = knife.tokenize_pii_columns(["customer_name", "email"])
    
    assert tokenized["customer_name"].iloc[0] == "[CUSTOMER_NAME_1]"
    assert tokenized["customer_name"].iloc[1] == "[CUSTOMER_NAME_2]"
    assert tokenized["email"].iloc[0] == "[EMAIL_1]"
    assert tokenized["revenue"].iloc[0] == 100.0

def test_privacy_knife_data_profile():
    df = pd.DataFrame({
        "col_a": [1, 2, np.nan, 4],
        "col_b": ["x", "y", "z", "x"]
    })
    knife = DeepAnalyzePrivacyKnife(df)
    profile = knife.get_data_profile()
    
    assert profile["shape"]["rows"] == 4
    assert profile["shape"]["columns"] == 2
    assert profile["columns"]["col_a"]["null_count"] == 1
    assert profile["columns"]["col_b"]["unique_count"] == 3

def test_local_gatekeeper_inspection():
    # ERP matrix
    df_erp = pd.DataFrame({
        "Unnamed: 0": ["Doc. No", "1000"],
        "Unnamed: 1": [":", "Item A"]
    })
    res_erp = LocalGatekeeper.inspect(df_erp)
    assert res_erp["strategy"] == "ERP_STRUCTURAL_MASK"

    # PII table
    df_pii = pd.DataFrame({
        "patient_name": ["John", "Jane"],
        "contact_phone": ["123", "456"]
    })
    res_pii = LocalGatekeeper.inspect(df_pii)
    assert res_pii["strategy"] == "PII_DEIDENTIFIED_MOCK"
    assert "patient_name" in res_pii["pii_columns"]

    # Standard clean table
    df_clean = pd.DataFrame({
        "product_id": [1, 2, 3],
        "price": [10.5, 20.0, 15.0]
    })
    res_clean = LocalGatekeeper.inspect(df_clean)
    assert res_clean["strategy"] == "STANDARD_STATISTICAL_PROFILE"

def test_extract_deepanalyze_content():
    # Model output with <Answer>```python ... ```</Answer>
    text_answer = """
Here is the cleaning code:
<Answer>
```python
df['cleaned'] = df['raw'].str.strip()
```
</Answer>
Done.
"""
    code, narrative = _extract_deepanalyze_content(text_answer)
    assert "df['cleaned'] = df['raw'].str.strip()" in code

    # Model output with markdown code fences
    text_markdown = """
```python
import duckdb
res = duckdb.query("SELECT * FROM df").df()
```
"""
    code_md, _ = _extract_deepanalyze_content(text_markdown)
    assert "import duckdb" in code_md

def test_reconcile_target_dataframe():
    class DummyIP:
        def __init__(self):
            self.user_ns = {}

    ip = DummyIP()
    df_result = pd.DataFrame({"a": [1, 2, 3]})
    ip.user_ns["clean_df"] = df_result

    # Automatic detection of assigned variable
    _reconcile_target_dataframe(ip, "clean_df = df.dropna()", "Clean the dataframe", default_target="sales_data")
    assert "sales_data" in ip.user_ns
    assert ip.user_ns["sales_data"] is df_result

    # Explicit target variable in prompt
    df_custom = pd.DataFrame({"b": [4, 5, 6]})
    ip.user_ns["df_out"] = df_custom
    _reconcile_target_dataframe(ip, "df_out = df.copy()", "custom_target = Clean data", default_target="sales_data")
    assert "custom_target" in ip.user_ns
    assert ip.user_ns["custom_target"] is df_custom

def test_flags_autocomplete():
    from deepanalyze.core import FLAGS
    assert "--persona" in FLAGS
    assert "--context" in FLAGS
    assert "--critic" in FLAGS
    assert "--critic-pro" in FLAGS

def test_classify_intent(monkeypatch):
    from deepanalyze.core import _classify_intent

    # Test sql classification
    monkeypatch.setattr("deepanalyze.core._call_llm", lambda prompt, sys, temp=0.0, max_tokens=20, target_model="deepanalyze-8b": "sql")
    assert _classify_intent("Query total revenue grouped by customer") == "sql"

    # Test viz classification with think tags
    monkeypatch.setattr("deepanalyze.core._call_llm", lambda prompt, sys, temp=0.0, max_tokens=20, target_model="deepanalyze-8b": "<think>plotting request</think> viz")
    assert _classify_intent("Plot sales distribution") == "viz"

    # Test ml classification
    monkeypatch.setattr("deepanalyze.core._call_llm", lambda prompt, sys, temp=0.0, max_tokens=20, target_model="deepanalyze-8b": "ml")
    assert _classify_intent("Train random forest classifier") == "ml"

    # Test fallback on exception or general
    monkeypatch.setattr("deepanalyze.core._call_llm", lambda prompt, sys, temp=0.0, max_tokens=20, target_model="deepanalyze-8b": "unknown_cat")
    assert _classify_intent("Some random request") == "general"

def test_schema_rag_context(tmp_path, monkeypatch):
    from deepanalyze.core import deepanalyze
    
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"table": "orders", "pk": "order_id", "business_rules": "No negative revenue"}', encoding="utf-8")
    
    captured_payloads = []
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        captured_payloads.append(system_prompt)
        return "<Answer>```python\ndf['safe'] = 1\n```</Answer>"
    
    class MockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"safe": [0]})}
        def set_next_input(self, code):
            self.last_input = code

    mock_ip = MockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: mock_ip)
    
    deepanalyze(f"--context {schema_file} Clean data")
    assert len(captured_payloads) > 0
    assert any("--- BUSINESS LOGIC & RAG CONTEXT ---" in payload for payload in captured_payloads)
    assert any("No negative revenue" in payload for payload in captured_payloads)


def test_critic_loop_interception(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze

    calls = []
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        calls.append({"prompt": prompt, "model": target_model, "system": system_prompt})
        if "Review this generated code" in prompt:
            # Critic identifies flaw and repairs it
            return "<Answer>```python\ndf['net_rev'] = df['rev'].clip(lower=0)\n```</Answer>"
        return "<Answer>```python\ndf['net_rev'] = df['rev']\n```</Answer>"

    class MockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"rev": [-10, 20, 30]})}
        def set_next_input(self, code):
            self.last_input = code

    mock_ip = MockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: mock_ip)

    deepanalyze("--critic Compute net revenue")
    captured = capsys.readouterr()
    assert "🛡️ [Critic Loop]: Logical flaw intercepted and repaired prior to execution." in captured.out
    assert "df['net_rev'] = df['rev'].clip(lower=0)" in mock_ip.last_input

def test_persona_modes(monkeypatch):
    from deepanalyze.core import deepanalyze

    recorded_systems = []
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        recorded_systems.append(system_prompt)
        if "Execution Output" in prompt:
            return "Key takeaway: Performance increased by 20%."
        return "<Answer>```python\nprint('Done')\n```</Answer>"

    class ExecMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"a": [1]})}
        def run_cell(self, code):
            class CellResult:
                error_in_exec = None
            return CellResult()

    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: ExecMockIP())

    # Test exec persona
    recorded_systems.clear()
    deepanalyze("-x -i --persona exec Summarize performance")
    assert any("Chief Data & Analytics Officer" in s or "ROI" in s for s in recorded_systems)

    # Test dev persona
    recorded_systems.clear()
    deepanalyze("-x -i --persona dev Summarize performance")
    assert any("Lead Data Engineer" in s or "pipeline edge cases" in s for s in recorded_systems)

    # Test default persona
    recorded_systems.clear()
    deepanalyze("-x -i Summarize performance")
    assert any("senior data analyst" in s for s in recorded_systems)

def test_critic_pro_and_safe_output(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze

    recorded_calls = []
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        recorded_calls.append({"prompt": prompt, "model": target_model})
        if "Review this generated code" in prompt:
            return "SAFE"
        return "<Answer>```python\ndf['net'] = df['gross']\n```</Answer>"

    class MockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"gross": [100, 200]})}
        def set_next_input(self, code):
            self.last_input = code

    mock_ip = MockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: mock_ip)

    # When SAFE, critic-pro routes to deepseek-reasoner and clean_code is unchanged
    deepanalyze("--critic-pro Calculate net")
    captured = capsys.readouterr()
    assert "Logical flaw intercepted" not in captured.out
    assert any(call["model"] == "deepseek-reasoner" for call in recorded_calls)
    assert "df['net'] = df['gross']" in mock_ip.last_input

def test_zero_flag_intent_routing_integration(monkeypatch):
    from deepanalyze.core import deepanalyze

    recorded_systems = []
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        if "Classify the user data request" in system_prompt:
            return "sql"
        recorded_systems.append(system_prompt)
        return "<Answer>```python\nimport duckdb\nres = duckdb.query('SELECT * FROM df').df()\n```</Answer>"

    class MockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"val": [1, 2]})}
        def set_next_input(self, code):
            self.last_input = code

    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: MockIP())

    deepanalyze("Query sales from orders")
    assert any("[DUCKDB SQL RULEBOOK]" in s for s in recorded_systems)

def test_new_flags_autocomplete():
    from deepanalyze.core import FLAGS
    for flag in ["--preview", "--diff", "--guard", "--stress", "--meta", "--simulate", "--spark"]:
        assert flag in FLAGS

def test_sparkline_generator():
    from deepanalyze.core import _generate_sparkline
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    spark = _generate_sparkline(s)
    assert len(spark) == 8
    assert spark != "—"

    # Empty series
    assert _generate_sparkline(pd.Series([])) == "—"

    # Constant series
    assert _generate_sparkline(pd.Series([5.0, 5.0, 5.0])) == "▄▄▄▄▄▄▄▄"

def test_evaluate_quality_gate():
    from deepanalyze.core import _evaluate_quality_gate
    df = pd.DataFrame({"revenue": [100, 200, 300], "cost": [50, 60, 70]})

    # Passing guard
    passed, msg = _evaluate_quality_gate("len(df) == 3 and df['revenue'].min() > 0", df)
    assert passed is True
    assert "PASSED" in msg

    # Failing guard
    passed_fail, msg_fail = _evaluate_quality_gate("df['revenue'].max() > 1000", df)
    assert passed_fail is False
    assert "evaluated to False" in msg_fail

    # Exception guard
    passed_err, msg_err = _evaluate_quality_gate("df['non_existent'].sum() > 0", df)
    assert passed_err is False
    assert "threw exception" in msg_err

def test_adversarial_df_generator():
    from deepanalyze.core import _generate_adversarial_df
    df = pd.DataFrame({
        "num": [10.5, 20.3],
        "txt": ["Alpha", "Beta"],
        "flag": [True, False]
    })
    adv = _generate_adversarial_df(df)
    assert len(adv) == 5
    assert 0.0 in adv["num"].values
    assert "$0.00" in adv["txt"].values or "" in adv["txt"].values

def test_metamorphic_check():
    from deepanalyze.core import _run_metamorphic_check
    df = pd.DataFrame({"sales": [10.0, 20.0, 30.0]})

    # Code that works linearly
    code_linear = "df['double_sales'] = df['sales'] * 2"
    passed, _ = _run_metamorphic_check(code_linear, df, target_name="df")
    assert passed is True

    # Code that crashes on execution
    code_bad = "df['fail'] = df['sales'] + undefined_variable"
    passed_bad, err = _run_metamorphic_check(code_bad, df, target_name="df")
    assert passed_bad is False

def test_preview_ghost_execution_commit_and_discard(monkeypatch):
    from deepanalyze.core import deepanalyze

    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        return "<Answer>```python\ndf['bonus'] = df['salary'] * 0.1\n```</Answer>"

    class PreviewMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"salary": [5000, 6000]})}

    # Test Commit ('y')
    ip1 = PreviewMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: ip1)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    deepanalyze("--preview Calculate bonus")
    assert "bonus" in ip1.user_ns["df"].columns

    # Test Discard ('n')
    ip2 = PreviewMockIP()
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: ip2)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    deepanalyze("--preview Calculate bonus")
    assert "bonus" not in ip2.user_ns["df"].columns

def test_what_if_simulator_no_global_mutation(monkeypatch):
    from deepanalyze.core import deepanalyze

    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        return "<Answer>```python\ndf['price'] = df['price'] * 1.2\n```</Answer>"

    class SimMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"price": [10.0, 20.0]})}

    sim_ip = SimMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: sim_ip)

    deepanalyze('--simulate "20% price surge" Compute surge')
    # Target in user namespace should remain untouched
    assert sim_ip.user_ns["df"]["price"].iloc[0] == 10.0

def test_guard_enforcement_and_repair(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze

    call_count = [0]
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        call_count[0] += 1
        if "QUALITY GATE VIOLATION REFLECTION" in prompt:
            return "<Answer>```python\ndf = df[df['val'] > 0]\n```</Answer>"
        return "<Answer>```python\ndf = df[df['val'] < 0]\n```</Answer>"

    class GuardMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"val": [-10, 20, 30]})}
        def run_cell(self, code):
            class CellRes:
                error_in_exec = None
            exec(code, self.user_ns)
            return CellRes()

    guard_ip = GuardMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: guard_ip)

    deepanalyze('-x --guard "len(df) > 0 and df[\'val\'].min() > 0" Filter positive values')
    captured = capsys.readouterr()
    assert "Quality Gate Violation" in captured.out
    assert "Guard auto-repair succeeded!" in captured.out
    assert (guard_ip.user_ns["df"]["val"] > 0).all()

def test_stress_fuzzer_interception(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze

    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        if "ADVERSARIAL STRESS FAILURE REFLECTION" in prompt:
            return "<Answer>```python\ndf['ratio'] = [n / d if d != 0 else 0 for n, d in zip(df['num'], df['denom'])]\n```</Answer>"
        # First attempt: unsafe division loop that crashes on zero denominator
        return "<Answer>```python\ndf['ratio'] = [n / d for n, d in zip(df['num'], df['denom'])]\n```</Answer>"

    class StressMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"num": [10.0, 20.0], "denom": [2.0, 5.0]})}
        def set_next_input(self, code):
            self.last_input = code

    stress_ip = StressMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: stress_ip)

    deepanalyze('--stress Calculate ratio')
    captured = capsys.readouterr()
    assert "Stress Fuzzer Alert" in captured.out
    assert "Defensively patched and repaired" in captured.out


def test_diff_and_spark_rendering(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze

    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        return "<Answer>```python\ndf['score_pct'] = df['score'] / 100.0\n```</Answer>"

    class ExecMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"score": [80.0, 95.0, 60.0, 75.0]})}
        def run_cell(self, code):
            class CellRes:
                error_in_exec = None
            exec(code, self.user_ns)
            return CellRes()

    exec_ip = ExecMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: exec_ip)

    deepanalyze('-x --diff --spark Add score percentage')
    captured = capsys.readouterr()
    assert "State Diff HUD" in captured.out
    assert "Sparkline Minimaps" in captured.out

def test_workflow_flags_autocomplete():
    from deepanalyze.core import FLAGS
    for flag in ["--roadmap", "--kickstart", "--interview", "--brainstorm", "--radar", "--dag", "--gui", "--history", "--next", "--auto-clean", "--spawn"]:
        assert flag in FLAGS

def test_roadmap_orchestrator(capsys):
    from deepanalyze.core import deepanalyze, _ACTIVE_ROADMAP, _render_roadmap
    _ACTIVE_ROADMAP["phase"] = 1
    _ACTIVE_ROADMAP["goal"] = "Predict customer churn"
    
    deepanalyze("--roadmap")
    captured = capsys.readouterr()
    assert "Autonomous Project Roadmap" in captured.out
    assert "Predict customer churn" in captured.out

def test_zero_prompt_kickstart(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze, _ACTIVE_ROADMAP

    def mock_call_llm(prompt, system_prompt, temp=0.2, max_tokens=1200, target_model="deepanalyze-8b"):
        return "### Prioritized Kickstart Action Plan\n1. Profile missing customer IDs\n2. Group by revenue\n3. Train baseline model"

    class KickMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"cust_id": [1, 2], "revenue": [100, 200]})}

    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: KickMockIP())

    deepanalyze("--kickstart")
    captured = capsys.readouterr()
    assert "Prioritized Kickstart Action Plan" in captured.out
    assert _ACTIVE_ROADMAP["phase"] >= 2

def test_reverse_interview(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze, _ACTIVE_ROADMAP

    def mock_call_llm(prompt, system_prompt, temp=0.2, max_tokens=1000, target_model="deepanalyze-8b"):
        return "1. Optimize for Interpretability or Accuracy?\n2. Latency constraints?"

    class IntMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"a": [1, 2]})}

    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: IntMockIP())
    monkeypatch.setattr("builtins.input", lambda prompt="": "1A, 2B")

    deepanalyze("--interview")
    captured = capsys.readouterr()
    assert "Goal Interview" in captured.out
    assert _ACTIVE_ROADMAP["goal"] == "1A, 2B"
    assert _ACTIVE_ROADMAP["phase"] >= 3

def test_autonomous_hypothesis_generator(monkeypatch, capsys):
    from deepanalyze.core import deepanalyze, _ACTIVE_ROADMAP

    def mock_call_llm(prompt, system_prompt, temp=0.3, max_tokens=1500, target_model="deepanalyze-8b"):
        return "1. H1: Higher discounts drive lower margins\n%deepanalyze -x -v Plot discount vs margin\n2. H2: High tenure customers have lower churn\n%deepanalyze -x -t Churn vs tenure"

    class BsMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"discount": [0.1, 0.2], "margin": [50, 40]})}

    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: BsMockIP())

    deepanalyze("--brainstorm")
    captured = capsys.readouterr()
    assert "Autonomous Hypotheses" in captured.out
    assert len(_ACTIVE_ROADMAP["hypotheses"]) > 0

def test_proactive_anomaly_radar():
    from deepanalyze.core import _scan_for_anomalies
    orig_df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0], "val": [1, 2, 3, 4]})
    
    # 1. Negative value introduction anomaly
    new_df_neg = pd.DataFrame({"a": [10.0, -20.0, 30.0, 40.0], "val": [1, 2, 3, 4]})
    anomalies_neg = _scan_for_anomalies(orig_df, new_df_neg)
    assert any("Negative values" in a for a in anomalies_neg)

    # 2. Null surge anomaly
    new_df_null = pd.DataFrame({"a": [np.nan, np.nan, 30.0, 40.0], "val": [1, 2, 3, 4]})
    anomalies_null = _scan_for_anomalies(orig_df, new_df_null)
    assert any("Null surge" in a for a in anomalies_null)

def test_transformation_dag(capsys):
    from deepanalyze.core import _render_transformation_dag
    code = """
df['clean_price'] = df['price'].str.replace('$', '').astype(float)
df = df.filter(df['clean_price'] > 0)
result = df.groupby('category').agg({'clean_price': 'sum'})
"""
    _render_transformation_dag(code, target_name="df")
    captured = capsys.readouterr()
    assert "Transformation Flow Graph" in captured.out
    assert "clean_price" in captured.out

def test_gui_explorer(capsys):
    from deepanalyze.core import _render_gui_explorer
    df = pd.DataFrame({"product": ["A", "B", "C"], "sales": [100.0, 200.0, 300.0]})
    
    _render_gui_explorer(df, target_name="df")
    captured = capsys.readouterr()
    # Should render HTML or rich fallback without error

def test_history_explorer(capsys):
    from deepanalyze.core import _render_history_explorer, _DF_SNAPSHOTS
    _DF_SNAPSHOTS["df_sales"] = pd.DataFrame({"x": [1, 2, 3]})
    _render_history_explorer()
    captured = capsys.readouterr()
    assert "DataFrame Time-Machine" in captured.out
    assert "df_sales" in captured.out

def test_next_action_recommender(monkeypatch, capsys):
    from deepanalyze.core import _recommend_next_actions

    def mock_call_llm(prompt, system_prompt, temp=0.3, max_tokens=800, target_model="deepanalyze-8b"):
        return "1. Run correlation matrix\n%deepanalyze -x -t Compute correlations\n2. Plot feature importances"

    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    _recommend_next_actions("Active df with 5 cols", "Cleaned data")
    captured = capsys.readouterr()
    assert "Suggested Next Actions" in captured.out

def test_auto_clean_flow(monkeypatch):
    from deepanalyze.core import deepanalyze

    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=2500, target_model="deepanalyze-8b"):
        return "<Answer>```python\ndf['val'] = df['raw_val'].str.replace('$', '').astype(float)\n```</Answer>"

    class AutoCleanMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"raw_val": ["$10", "$20"]})}

    ip = AutoCleanMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: ip)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    deepanalyze("--auto-clean")
    assert "val" in ip.user_ns["df"].columns

def test_artifact_spawner(monkeypatch):
    from deepanalyze.core import deepanalyze

    injected_cells = []
    def mock_call_llm(prompt, system_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        return "Executive summary of data.\n<Answer>```python\ndf['spawned'] = True\n```</Answer>"

    class SpawnMockIP:
        def __init__(self):
            self.user_ns = {"df": pd.DataFrame({"a": [1]})}
        def run_cell(self, code):
            class CellRes:
                error_in_exec = None
            exec(code, self.user_ns)
            return CellRes()
        def set_next_input(self, text, replace=False):
            injected_cells.append(text)

    mock_ip = SpawnMockIP()
    monkeypatch.setattr("deepanalyze.core._call_llm", mock_call_llm)
    monkeypatch.setattr("deepanalyze.core.get_ipython", lambda: mock_ip)

    deepanalyze("-x --spawn Add spawned flag")
    assert len(injected_cells) >= 1
    assert any("spawned" in c for c in injected_cells)


# ==================== IMPORT / EXPORT ENGINE TESTS ====================

from deepanalyze.core import _sanitize_var_name, _estimate_memory_mb, _handle_import, _handle_export

def test_sanitize_var_name_basic():
    """Tests that filenames are sanitized into valid Python identifiers with _df suffix."""
    assert _sanitize_var_name("data/Sales 2026-Q1.csv") == "sales_2026_q1_df"
    assert _sanitize_var_name("raw_export.parquet") == "raw_export_df"
    assert _sanitize_var_name("123_file.csv") == "df_123_file_df"
    assert _sanitize_var_name("my_data_df.csv") == "my_data_df"
    assert _sanitize_var_name("hello world!.tsv") == "hello_world_df"


def test_sanitize_var_name_edge_cases():
    """Tests edge cases for the variable name sanitizer."""
    assert _sanitize_var_name("!!!.csv") == "df__df"
    assert _sanitize_var_name("a.csv") == "a_df"
    assert _sanitize_var_name("already_df.xlsx") == "already_df"


def test_estimate_memory_mb_pandas():
    """Tests memory estimation for a Pandas DataFrame."""
    df = pd.DataFrame({"a": range(1000), "b": [f"x{i}" for i in range(1000)]})
    mem = _estimate_memory_mb(df)
    assert mem > 0


def test_estimate_memory_mb_polars():
    """Tests memory estimation for a Polars DataFrame."""
    try:
        import polars as pl
        df = pl.DataFrame({"a": range(1000), "b": [f"x{i}" for i in range(1000)]})
        mem = _estimate_memory_mb(df)
        assert mem > 0
    except ImportError:
        pytest.skip("Polars not installed")


def test_import_csv_roundtrip(tmp_path, monkeypatch):
    """Tests full CSV import -> session binding -> export roundtrip."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    # Create a test CSV
    csv_path = tmp_path / "test_import.csv"
    test_df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.3, None]
    })
    test_df.write_csv(str(csv_path))

    # Mock IPython
    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=str(csv_path),
        target="df",
        sheet=None,
        lazy=False
    )

    _handle_import(mock_ip, args)

    # Verify variable was bound
    assert "test_import_df" in mock_ip.user_ns
    imported_df = mock_ip.user_ns["test_import_df"]
    assert isinstance(imported_df, pl.DataFrame)
    assert imported_df.shape == (3, 3)
    assert list(imported_df.columns) == ["id", "name", "value"]


def test_import_with_target_override(tmp_path, monkeypatch):
    """Tests that --target overrides the auto-generated variable name."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"x": [1, 2]}).write_csv(str(csv_path))

    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=str(csv_path),
        target="my_custom_name",
        sheet=None,
        lazy=False
    )

    _handle_import(mock_ip, args)
    assert "my_custom_name" in mock_ip.user_ns


def test_import_file_not_found(tmp_path, capsys):
    """Tests that import handles missing files gracefully."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=str(tmp_path / "nonexistent.csv"),
        target="df",
        sheet=None,
        lazy=False
    )

    _handle_import(mock_ip, args)
    captured = capsys.readouterr()
    assert "File not found" in captured.out


def test_import_unsupported_extension(tmp_path, capsys):
    """Tests that import rejects unsupported file extensions."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    weird_file = tmp_path / "data.weird"
    weird_file.write_text("test")

    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=str(weird_file),
        target="df",
        sheet=None,
        lazy=False
    )

    _handle_import(mock_ip, args)
    captured = capsys.readouterr()
    assert "Unsupported file extension" in captured.out


def test_export_csv(tmp_path):
    """Tests exporting a Polars DataFrame to CSV."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    class MockIP:
        user_ns = {"my_df": pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})}
    mock_ip = MockIP()

    dest = str(tmp_path / "output.csv")
    import argparse
    args = argparse.Namespace(
        export="my_df",
        to=dest
    )

    _handle_export(mock_ip, args)
    assert os.path.exists(dest)
    reloaded = pl.read_csv(dest)
    assert reloaded.shape == (3, 2)


def test_export_parquet_default_path(tmp_path, monkeypatch):
    """Tests that export defaults to ./<target>.parquet when --to is omitted."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    monkeypatch.chdir(tmp_path)

    class MockIP:
        user_ns = {"sales": pl.DataFrame({"revenue": [100, 200]})}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        export="sales",
        to=None
    )

    _handle_export(mock_ip, args)
    expected_path = tmp_path / "sales.parquet"
    assert expected_path.exists()


def test_export_creates_directories(tmp_path):
    """Tests that export auto-creates parent directories."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    class MockIP:
        user_ns = {"df": pl.DataFrame({"x": [1]})}
    mock_ip = MockIP()

    dest = str(tmp_path / "deep" / "nested" / "dir" / "output.csv")
    import argparse
    args = argparse.Namespace(
        export="df",
        to=dest
    )

    _handle_export(mock_ip, args)
    assert os.path.exists(dest)


def test_export_variable_not_found(capsys):
    """Tests that export handles missing variables gracefully."""
    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        export="nonexistent_var",
        to="/tmp/out.csv"
    )

    _handle_export(mock_ip, args)
    captured = capsys.readouterr()
    assert "not found in session namespace" in captured.out


def test_export_pandas_to_parquet(tmp_path):
    """Tests that Pandas DataFrames are auto-converted to Polars before export."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    class MockIP:
        user_ns = {"pd_df": pd.DataFrame({"col1": [10, 20], "col2": ["a", "b"]})}
    mock_ip = MockIP()

    dest = str(tmp_path / "pandas_output.parquet")
    import argparse
    args = argparse.Namespace(
        export="pd_df",
        to=dest
    )

    _handle_export(mock_ip, args)
    assert os.path.exists(dest)
    reloaded = pl.read_parquet(dest)
    assert reloaded.shape == (2, 2)


def test_import_parquet_roundtrip(tmp_path):
    """Tests import/export roundtrip via Parquet format."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    # Write a parquet file first
    original = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    pq_path = str(tmp_path / "data.parquet")
    original.write_parquet(pq_path)

    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=pq_path,
        target="df",
        sheet=None,
        lazy=False
    )

    _handle_import(mock_ip, args)
    assert "data_df" in mock_ip.user_ns
    assert mock_ip.user_ns["data_df"].shape == (3, 2)


def test_import_lazy_csv(tmp_path):
    """Tests that --lazy creates a LazyFrame for CSV files."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    csv_path = tmp_path / "lazy_test.csv"
    pl.DataFrame({"a": range(10)}).write_csv(str(csv_path))

    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=str(csv_path),
        target="df",
        sheet=None,
        lazy=True
    )

    _handle_import(mock_ip, args)
    assert "lazy_test_df" in mock_ip.user_ns
    assert isinstance(mock_ip.user_ns["lazy_test_df"], pl.LazyFrame)

def test_import_excel_basic(tmp_path):
    """Tests Excel import with default sheet and session/roadmap state binding."""
    try:
        import polars as pl
        import openpyxl
    except ImportError:
        pytest.skip("Polars or openpyxl not installed")

    excel_path = tmp_path / "inv_listing_31082025.xlsx"
    pd_df = pd.DataFrame({"item_id": [101, 102], "price": [45.0, 99.5]})
    pd_df.to_excel(str(excel_path), index=False, engine="openpyxl")

    from deepanalyze import core
    core._ACTIVE_ROADMAP = {"phase": 1, "goal": None, "hypotheses": []}

    class MockIP:
        user_ns = {}
    mock_ip = MockIP()

    import argparse
    args = argparse.Namespace(
        import_path=str(excel_path),
        target="df",
        sheet=None,
        lazy=False
    )

    core._handle_import(mock_ip, args)

    # 1. Target variable check
    assert "inv_listing_31082025_df" in mock_ip.user_ns
    imported = mock_ip.user_ns["inv_listing_31082025_df"]
    assert imported.shape == (2, 2)
    assert "item_id" in imported.columns
    assert "price" in imported.columns

    # 2. Snapshot check
    assert "0_import_inv_listing_31082025_df" in core._DF_SNAPSHOTS

    # 3. Active Roadmap check
    assert core._ACTIVE_ROADMAP.get("target_df") == "inv_listing_31082025_df"


def test_import_excel_sheet_by_index_and_name(tmp_path):
    """Tests Excel import specifying sheet by digit string and by string name."""
    try:
        import polars as pl
        import openpyxl
    except ImportError:
        pytest.skip("Polars or openpyxl not installed")

    excel_path = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
        pd.DataFrame({"s1_col": [1, 2]}).to_excel(writer, sheet_name="FirstSheet", index=False)
        pd.DataFrame({"s2_col": [10, 20]}).to_excel(writer, sheet_name="SecondSheet", index=False)

    from deepanalyze import core
    class MockIP:
        user_ns = {}

    # Import by sheet index string "2"
    mock_ip = MockIP()
    args_idx = argparse.Namespace(
        import_path=str(excel_path),
        target="sheet2_df",
        sheet="2",
        lazy=False
    )
    core._handle_import(mock_ip, args_idx)
    assert "sheet2_df" in mock_ip.user_ns
    assert "s2_col" in mock_ip.user_ns["sheet2_df"].columns

    # Import by sheet name string "FirstSheet"
    mock_ip2 = MockIP()
    args_name = argparse.Namespace(
        import_path=str(excel_path),
        target="sheet1_df",
        sheet="FirstSheet",
        lazy=False
    )
    core._handle_import(mock_ip2, args_name)
    assert "sheet1_df" in mock_ip2.user_ns
    assert "s1_col" in mock_ip2.user_ns["sheet1_df"].columns


def test_sanitize_var_name_invoice_listing():
    """Verify exact stem sanitization for inv_listing_31082025.xlsx."""
    from deepanalyze.core import _sanitize_var_name
    assert _sanitize_var_name("inv_listing_31082025.xlsx") == "inv_listing_31082025_df"
    assert _sanitize_var_name("/path/to/inv_listing_31082025.xlsx") == "inv_listing_31082025_df"


def test_privacy_token_vault_and_detokenization():
    """Verify bidirectional in-memory tokenization and detokenization in Polars and Pandas."""
    import polars as pl
    from deepanalyze.privacy_knife import DeepAnalyzePrivacyKnife, get_token_vault, clear_token_vault

    clear_token_vault()
    pldf = pl.DataFrame({
        "customer_name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "email": ["alice@corp.com", "bob@corp.com", "charlie@corp.com"],
        "revenue": [100.0, 250.5, 300.0]
    })

    knife = DeepAnalyzePrivacyKnife(pldf, dataset_id="test_sales")
    tokenized_df = knife.tokenize_pii_columns(["customer_name", "email"])
    
    # Verify tokenized columns contain tags
    assert tokenized_df["customer_name"][0] == "[CUSTOMER_NAME_1]"
    assert tokenized_df["email"][0] == "[EMAIL_1]"
    
    # Verify vault mapping
    vault = get_token_vault()
    assert "test_sales" in vault
    assert vault["test_sales"]["[CUSTOMER_NAME_1]"] == "Alice Smith"
    assert vault["test_sales"]["[EMAIL_1]"] == "alice@corp.com"

    # Verify detokenization restores original values
    restored_df = DeepAnalyzePrivacyKnife.detokenize_dataframe(tokenized_df, dataset_id="test_sales")
    assert restored_df["customer_name"][0] == "Alice Smith"
    assert restored_df["email"][0] == "alice@corp.com"

    # Verify text detokenization
    sample_text = "Top customer [CUSTOMER_NAME_1] spent $100. Contact: [EMAIL_1]."
    restored_text = DeepAnalyzePrivacyKnife.detokenize_text(sample_text, dataset_id="test_sales")
    assert "Top customer Alice Smith spent $100. Contact: alice@corp.com." == restored_text


def test_local_gatekeeper_inspect_folder(tmp_path):
    """Verify folder inspection classifying multiple files."""
    import polars as pl
    from deepanalyze.privacy_knife import LocalGatekeeper

    f1 = tmp_path / "patients.csv"
    f1.write_text("patient_name,age,diagnosis\nJohn Doe,45,Hypertension\nJane Roe,50,Diabetes\n")

    f2 = tmp_path / "sales.parquet"
    pl.DataFrame({"product_id": [1, 2], "qty": [10, 20]}).write_parquet(str(f2))

    res = LocalGatekeeper.inspect_folder(str(tmp_path))
    assert "patients.csv" in res
    assert res["patients.csv"]["strategy"] == "PII_DEIDENTIFIED_MOCK"
    assert "sales.parquet" in res
    assert res["sales.parquet"]["strategy"] == "STANDARD_STATISTICAL_PROFILE"


def test_eda_lifecycle_pipeline(monkeypatch, tmp_path):
    """Verify end-to-end 10-stage --EDA lifecycle execution with Polars."""
    import polars as pl
    from deepanalyze import core

    # Mock _call_llm so test runs self-contained without needing live network
    def mock_call_llm(prompt, sys_prompt, temp=0.0, max_tokens=3500, target_model="deepanalyze-8b"):
        if "idiomatic Polars cleaning" in prompt or "POLARS DATA CLEANING" in sys_prompt:
            return "<Answer>```python\ntest_df = test_df.with_columns(pl.col('amount').fill_null(0.0))\n```</Answer>"
        return "<Answer>```text\nInferred business domain: Retail E-Commerce Revenue Analysis.\n```</Answer>"

    monkeypatch.setattr(core, "_call_llm", mock_call_llm)

    class MockIP:
        user_ns = {
            "test_df": pl.DataFrame({
                "customer_name": ["Alice", "Bob", "Charlie"],
                "amount": [50.0, None, 150.0],
                "category": ["Electronics", "Clothing", "Electronics"]
            })
        }

    mock_ip = MockIP()
    monkeypatch.setattr(core, "get_ipython", lambda: mock_ip)

    # Run --EDA lifecycle (10 Stages)
    core.deepanalyze("--EDA --target test_df --goal 'Optimize Q3 Sales'")

    # Check snapshots
    assert "0_raw_test_df" in core._DF_SNAPSHOTS
    assert "1_cleaned_test_df" in core._DF_SNAPSHOTS

    # Check roadmap phase reached 4
    assert core._ACTIVE_ROADMAP["phase"] == 4
    assert core._ACTIVE_ROADMAP["goal"] == "Optimize Q3 Sales"

    # Check multi-modal deliverables created
    assert os.path.exists("./charts/eda_test_df_dashboard.html")
    assert os.path.exists("./charts/eda_test_df_briefing.md")
    assert os.path.exists("./charts/eda_test_df_slides.html")
    assert os.path.exists("./charts/eda_test_df_slides.md")
    assert os.path.exists("./charts/eda_test_df_schema.sql")
    assert os.path.exists("./pipeline.py")
    assert os.path.exists("./eda_quality_monitor.py")


def test_local_gatekeeper_detect_header_offset():
    """Verify detection of true header row index in messy multi-row metadata files."""
    from deepanalyze.privacy_knife import LocalGatekeeper

    messy_lines = [
        "Company: ACME Global Corp",
        "Report: Financial Ledger 2026",
        "Generated Date: 2026-08-28",
        "========================================",
        "Transaction_ID,Posting_Date,Customer_Name,Gross_Amount,Net_Margin",
        "TXN_001,2026-01-01,Alpha Ltd,1000.50,250.00",
        "TXN_002,2026-01-02,Beta LLC,2400.00,600.00"
    ]

    offset = LocalGatekeeper.detect_header_offset(messy_lines, sep=",")
    assert offset == 4  # The 5th line (index 4) is the real header row


def test_sniff_tabular_file_and_import_messy_erp(tmp_path):
    """Verify smart sniffing of semicolon delimiter and leading title rows."""
    import polars as pl
    from deepanalyze import core

    dirty_csv = tmp_path / "messy_ledger.csv"
    dirty_content = (
        "ACME ERP System v4.2\n"
        "Department: Enterprise Sales\n"
        "Period: 2026-Q1\n"
        "\n"
        "Invoice_No;Invoice_Date;Client_Name;Amount_USD;Status\n"
        "INV-101;2026-01-15;Globex Corp;(1,500.00);Completed\n"
        "INV-102;9999-99-99 99:99:99;Initech; $4,250.75 ;Completed\n"
        "INV-103;0000-00-00 00:00:00;Umbrella Corp;(250.00);Pending\n"
    )
    dirty_csv.write_text(dirty_content, encoding="utf-8")

    # 1. Test Sniffer
    sniff_info = core._sniff_tabular_file(str(dirty_csv))
    assert sniff_info["separator"] == ";"
    assert sniff_info["skip_rows"] == 4

    # 2. Test Import Engine
    class MockIP:
        user_ns = {}

    mock_ip = MockIP()
    args = argparse.Namespace(
        import_path=str(dirty_csv),
        target="ledger_df",
        sheet=None,
        lazy=False
    )
    core._handle_import(mock_ip, args)

    assert "ledger_df" in mock_ip.user_ns
    df = mock_ip.user_ns["ledger_df"]
    assert isinstance(df, pl.DataFrame)
    assert "Invoice_No" in df.columns
    assert "Amount_USD" in df.columns
    assert df.height == 3


def test_cleaners_ftfy_unicode_and_mojibake():
    """Verify repair of Mojibake, zero-width spaces, and HTML entities."""
    import polars as pl
    from deepanalyze import cleaners

    df = pl.DataFrame({
        "text": ["CafÃ© &amp; Bistro\u200b", "â€˜Specialâ€™ Quote\xa0", "Clean Name"]
    })
    res = cleaners.sanitize_unicode_and_mojibake(df)
    assert res["text"].to_list() == ["Café & Bistro", "'Special' Quote", "Clean Name"]


def test_cleaners_fuzzy_harmonize_categories():
    """Verify clustering and harmonizing of messy category typos."""
    import polars as pl
    from deepanalyze import cleaners

    df = pl.DataFrame({
        "country": ["United States", "United States", "USA", "United States", "U.S.A.", "Germany", "Germany"]
    })
    res = cleaners.fuzzy_harmonize_categories(df, threshold=0.8)
    vals = res["country"].to_list()
    assert vals[2] == "United States" or vals[4] == "United States"
    assert "Germany" in vals


def test_cleaners_explode_nested_json():
    """Verify flattening of embedded JSON string columns."""
    import polars as pl
    from deepanalyze import cleaners

    df = pl.DataFrame({
        "id": [1, 2],
        "metadata": ['{"device": "iOS", "os_ver": 17}', '{"device": "Android", "os_ver": 14}']
    })
    res = cleaners.explode_nested_json(df)
    assert "metadata_device" in res.columns
    assert "metadata_os_ver" in res.columns
    assert "metadata" not in res.columns
    assert res["metadata_device"].to_list() == ["iOS", "Android"]


def test_cleaners_unpivot_temporal_matrix():
    """Verify reshaping wide monthly reports into tidy tabular rows."""
    import polars as pl
    from deepanalyze import cleaners

    df = pl.DataFrame({
        "Department": ["Sales", "Engineering"],
        "Jan_2025": [100, 200],
        "Feb_2025": [110, 210],
        "Mar_2025": [120, 220]
    })
    res = cleaners.unpivot_temporal_matrix(df)
    assert "period" in res.columns
    assert "value" in res.columns
    assert res.height == 6  # 2 depts * 3 months


def test_cleaners_normalize_units_and_currencies():
    """Verify parsing and standardization of accounting negatives and mixed units."""
    import polars as pl
    from deepanalyze import cleaners

    df = pl.DataFrame({
        "weight": ["5 kg", "5000 g", "10 kg"],
        "balance": ["(1,250.00)", "$3,400.50", "200.00"]
    })
    res = cleaners.normalize_units_and_currencies(df)
    assert res["weight"].to_list() == [5.0, 5.0, 10.0]
    assert res["balance"].to_list() == [-1250.0, 3400.5, 200.0]


def test_cleaners_winsorize_numeric_outliers():
    """Verify clipping of extreme human data entry typos."""
    import polars as pl
    from deepanalyze import cleaners

    # Create dataset with 90 normal values and 2 extreme outliers (999 and -5000)
    ages = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 999]
    df = pl.DataFrame({"age": ages})
    res = cleaners.winsorize_numeric_outliers(df, lower_p=0.05, upper_p=0.90)
    # The max value must be clipped below 999
    assert res["age"].max() < 999


def test_cleaners_auto_cast_data_types():
    """Verify safe coercion of boolean strings and numeric strings."""
    import polars as pl
    from deepanalyze import cleaners

    df = pl.DataFrame({
        "is_active": ["true", "false", "yes", "no"],
        "num_str": ["1,200", "3,450.5", "500", "25"]
    })
    res = cleaners.auto_cast_data_types(df)
    assert res.schema["is_active"] == pl.Boolean
    assert res["is_active"].to_list() == [True, False, True, False]
    assert res.schema["num_str"] == pl.Float64
    assert res["num_str"].to_list() == [1200.0, 3450.5, 500.0, 25.0]


def test_cleaners_auto_stitch_dataframes():
    """Verify relational foreign-key linking across multiple session DataFrames."""
    import polars as pl
    from deepanalyze import cleaners

    customers_df = pl.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    orders_df = pl.DataFrame({
        "order_id": [101, 102],
        "customer_id": [1, 2],
        "total": [250.0, 400.0]
    })
    stitched, log = cleaners.auto_stitch_dataframes({"customers": customers_df, "orders": orders_df})
    assert "name" in stitched.columns
    assert "total" in stitched.columns
    assert stitched.height == 3


def test_eda_dashboard_standalone(tmp_path):
    """Verify generation of interactive Chart.js HTML executive dashboard."""
    import polars as pl
    from deepanalyze import dashboard

    df = pl.DataFrame({
        "revenue": [100.0, 250.0, 300.0, 150.0, 400.0],
        "category": ["A", "B", "A", "C", "B"],
        "margin": [20.0, 50.0, 60.0, 30.0, 80.0]
    })
    out_file = str(tmp_path / "test_dashboard.html")
    dash_path = dashboard.generate_eda_dashboard(
        df,
        target_name="sales_df",
        goal="Maximize Q3 Margin",
        num_cols=["revenue", "margin"],
        cat_cols=["category"],
        corr_highlights=[("revenue", "margin", 0.98)],
        exec_narrative="Strong linear revenue-margin alignment observed.",
        recommendations=["Invest in Category B", "Monitor Category C"],
        output_path=out_file
    )

    assert os.path.exists(dash_path)
    content = open(dash_path, encoding="utf-8").read()
    assert "Executive Analytics Dashboard: sales_df" in content
    assert "Chart.js" in content
    assert "chartDistribution" in content
    assert "Maximize Q3 Margin" in content
    assert "Strong linear revenue-margin alignment observed." in content


def test_cleaners_unravel_hierarchical_erp_report():
    """Verify autonomous unravelling of hierarchical multi-row ERP invoice listings."""
    from deepanalyze import cleaners
    excel_path = "/Users/abdullahbinmadhi/Desktop/deepanalyze/INV LISTING 31082025.xlsx"
    if os.path.exists(excel_path):
        res = cleaners.unravel_hierarchical_erp_report(excel_path)
        assert res.shape[0] >= 1800
        assert "Sequence" in res.columns
        assert "GL-Code" in res.columns
        assert "Quantity" in res.columns
        assert "Unit Price" in res.columns
        assert "Item Amount" in res.columns
        assert "doc_no" in res.columns
        assert "doc_date" in res.columns
        assert "customer_code" in res.columns
        assert "customer_name" in res.columns
        assert "Full_Description" in res.columns
        assert round(float(res["Item Amount"].sum()), 2) == 995261.44
        # Verify 0 nulls across parent document hierarchy
        if hasattr(res, "null_count"):
            assert res["doc_no"].null_count() == 0
            assert res["customer_code"].null_count() == 0
            assert res["customer_name"].null_count() == 0
        else:
            assert res["doc_no"].isna().sum() == 0
            assert res["customer_code"].isna().sum() == 0
            assert res["customer_name"].isna().sum() == 0
        # Verify multi-line description continuation stitching
        desc_list = res["Full_Description"].to_list() if hasattr(res["Full_Description"], "to_list") else res["Full_Description"].tolist()
        assert any("X 30 X 2" in str(d) for d in desc_list)


def test_cleaners_unravel_general_ledger():
    """Verify universal unravelling of General Ledger reports with inline key-value pairs and wrapped notes."""
    import pandas as pd
    from deepanalyze import cleaners

    gl_raw_data = pd.DataFrame([
        ['Account No: 1000-00', 'Account Name: CASH IN HAND', None, None, None],
        ['Date', 'Ref No', 'Particulars', 'Debit', 'Credit'],
        ['2025-01-01', 'OB-001', 'Opening Balance', 5000.0, 0.0],
        ['2025-01-05', 'PV-101', 'Office Supplies Expense', 0.0, 350.0],
        [None, None, 'Paper and ink cartridge replenishment', None, None],
        ['2025-01-10', 'REC-202', 'Cash Sales Received', 1200.0, 0.0],
        ['Account No: 2000-00', 'Account Name: ACCOUNTS PAYABLE', None, None, None],
        ['Date', 'Ref No', 'Particulars', 'Debit', 'Credit'],
        ['2025-01-02', 'BILL-501', 'Supplier Inv ABC Corp', 0.0, 4500.0],
        ['2025-01-15', 'PV-102', 'Payment to ABC Corp', 2000.0, 0.0]
    ])

    res = cleaners.unravel_hierarchical_erp_report(gl_raw_data)
    assert res.shape[0] == 5
    assert "account_no" in res.columns
    assert "account_name" in res.columns
    assert "doc_no" in res.columns
    assert "Debit" in res.columns
    assert "Credit" in res.columns
    assert res["Debit"].sum() == 8200.0
    assert res["Credit"].sum() == 4850.0


def test_cleaners_unravel_purchase_orders():
    """Verify universal unravelling of Purchase Orders with vendor headers and item totals."""
    import pandas as pd
    from deepanalyze import cleaners

    po_raw_data = pd.DataFrame([
        ['PO No: PO-99881', 'PO Date: 2025-03-01', 'Vendor: Global Tech Ltd', 'Currency: USD', None],
        ['Line No', 'SKU Code', 'Item Description', 'Qty Ordered', 'Unit Rate', 'Total Cost'],
        [1, 'TECH-001', 'Dell UltraSharp Monitor 27"', 10, 350.0, 3500.0],
        [2, 'TECH-002', 'Logitech MX Master 3S Mouse', 25, 99.0, 2475.0],
        ['PO No: PO-99882', 'PO Date: 2025-03-05', 'Vendor: Office Supplies Inc', 'Currency: USD', None],
        ['Line No', 'SKU Code', 'Item Description', 'Qty Ordered', 'Unit Rate', 'Total Cost'],
        [1, 'OFF-101', 'Ergonomic Mesh Chair', 5, 180.0, 900.0]
    ])

    res = cleaners.unravel_hierarchical_erp_report(po_raw_data)
    assert res.shape[0] == 3
    assert "doc_no" in res.columns
    assert "customer_code" in res.columns  # Vendor mapped to entity
    assert "Quantity" in res.columns
    assert "Item Amount" in res.columns
    assert res["Item Amount"].sum() == 6875.0


def test_statistical_engine_hypothesis_and_vif():
    """Verify statistical hypothesis battery, SVD VIF, and feature importance."""
    import pandas as pd
    from deepanalyze import statistical_engine

    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "discount": [0.0, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0],
        "region": ["East", "West", "East", "West", "East", "West", "East", "West", "East", "West"]
    })

    hyp_res = statistical_engine.run_hypothesis_battery(df, target_col="revenue")
    assert "normality" in hyp_res
    assert "revenue" in hyp_res["normality"]
    assert len(hyp_res["target_tests"]) >= 1

    vif_df = statistical_engine.compute_vif_robust(df)
    assert not vif_df.empty
    assert "feature" in vif_df.columns
    assert "vif" in vif_df.columns

    feat_imp = statistical_engine.calculate_feature_importance(df, target_col="revenue")
    assert not feat_imp.empty
    assert "composite_score" in feat_imp.columns


def test_storyteller_pyramid_memo_and_exports(tmp_path):
    """Verify McKinsey Pyramid Principle executive briefing and multi-format exports."""
    import pandas as pd
    from deepanalyze import storyteller

    df = pd.DataFrame({
        "sales": [1000, 2500, 3200, 4100, 5000],
        "profit": [200, 450, 600, 820, 1100],
        "category": ["A", "B", "A", "B", "A"]
    })

    memo = storyteller.generate_executive_memo(df, target_col="sales")
    assert "headline" in memo
    assert len(memo["pillars"]) == 3
    assert len(memo["action_plan"]) == 3

    html_path = str(tmp_path / "memo.html")
    md_path = str(tmp_path / "memo.md")

    html_out = storyteller.export_briefing(memo, output_format="html", output_path=html_path)
    md_out = storyteller.export_briefing(memo, output_format="markdown", output_path=md_path)

    assert "Executive Strategic Briefing" in html_out
    assert os.path.exists(html_path)
    assert os.path.exists(md_path)


def test_feature_forge_leak_free_pipeline():
    """Verify automated feature engineering with temporal decomposition and rolling stats."""
    import pandas as pd
    from deepanalyze import feature_forge

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="D"),
        "sales": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280],
        "costs": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        "segment": ["Retail", "Wholesale", "Retail", "Wholesale", "Retail", "Wholesale", "Retail", "Wholesale", "Retail", "Wholesale"]
    })

    fe_df, log = feature_forge.auto_engineer_features(df, target_col="sales")
    assert log["engineered_features_created"] > 0
    assert "costs_lag1" in fe_df.columns
    assert "date_dow_sin" in fe_df.columns


def test_forecaster_cadence_and_conformal_bands():
    """Verify autonomous time-series forecasting and 80%/95% conformal bounds."""
    import pandas as pd
    from deepanalyze import forecaster

    df = pd.DataFrame({
        "txn_date": pd.date_range("2025-01-01", periods=20, freq="D"),
        "revenue": np.linspace(100, 300, 20) + np.random.normal(0, 5, 20)
    })

    res = forecaster.auto_forecast_series(df, horizon=7)
    assert "error" not in res
    assert res["horizon"] == 7
    assert len(res["forecast_table"]) == 7
    assert "lower_80" in res["forecast_table"][0]
    assert "upper_95" in res["forecast_table"][0]


def test_drift_sentinel_psi_and_schema_tracking():
    """Verify PSI, distribution shift detection, and schema evolution tracking."""
    import numpy as np
    import pandas as pd
    from deepanalyze import drift_sentinel

    ref_df = pd.DataFrame({
        "amount": np.random.normal(100, 15, size=100),
        "status": ["active"] * 100
    })
    # Shifted current distribution
    curr_df = pd.DataFrame({
        "amount": np.random.normal(180, 25, size=100),
        "status": ["active"] * 80 + ["churned"] * 20
    })

    drift_res = drift_sentinel.detect_data_drift(ref_df, curr_df)
    assert "overall_status" in drift_res
    assert drift_res["max_psi_score"] > 0.0
    assert len(drift_res["feature_drift"]) >= 1


def test_schema_synthesizer_duckdb_and_dbt():
    """Verify SQL DDL transpilation and dbt schema.yml synthesis."""
    import pandas as pd
    from deepanalyze import schema_synthesizer

    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "price": [10.5, 20.0, 30.5, 40.0, 50.5]
    })

    ddl = schema_synthesizer.infer_sql_schema(df, table_name="products", dialect="duckdb")
    assert "CREATE TABLE IF NOT EXISTS products" in ddl
    assert "PRIMARY KEY" in ddl or "BIGINT" in ddl

    dbt_yml = schema_synthesizer.generate_dbt_models(df, table_name="products")
    assert "version: 2" in dbt_yml
    assert "unique" in dbt_yml

    er = schema_synthesizer.generate_er_diagram(df, table_name="Products")
    assert "erDiagram" in er


def test_synthetic_data_copula_and_fidelity_audit():
    """Verify Gaussian Copula synthetic data cloning and fidelity evaluation."""
    import numpy as np
    import pandas as pd
    from deepanalyze import synthetic_data

    real_df = pd.DataFrame({
        "units": [10, 20, 30, 40, 50, 60, 70, 80],
        "price": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "tier": ["Gold", "Silver", "Gold", "Bronze", "Silver", "Gold", "Bronze", "Silver"]
    })

    synth_df = synthetic_data.generate_synthetic_clone(real_df, num_rows=15)
    assert synth_df.shape[0] == 15
    assert synth_df.shape[1] == 3

    audit = synthetic_data.audit_synthetic_fidelity(real_df, synth_df)
    assert "fidelity_score_pct" in audit
    assert audit["fidelity_score_pct"] >= 70.0


# =============================================================================
# V3.0 REVOLUTIONARY ANALYTICAL CAPABILITIES UNIT TESTS
# =============================================================================

def test_why_causal_debugger():
    """Verify Causal Root-Cause Debugger decomposes variance across categorical factors."""
    import pandas as pd
    from deepanalyze import causal_engine

    df = pd.DataFrame({
        "revenue": [100, 200, -50, 400, -80, 500, 600, -120],
        "category": ["A", "B", "C", "A", "C", "B", "A", "C"],
        "region": ["East", "West", "East", "West", "East", "West", "East", "East"]
    })

    res = causal_engine.trace_root_cause_why(df, condition_or_col="revenue < 0")
    assert res["triggered_count"] == 3
    assert len(res["ranked_drivers"]) >= 1
    assert "Root-Cause Analysis" in res["diagnostic_text"]


def test_distill_rule_memory(tmp_path):
    """Verify autonomous rule distillation and memory persistence."""
    from deepanalyze import brain

    b = brain.BiomimeticBrain(storage_path=str(tmp_path / "memory.json"))
    prompts = [
        "Please standardize date format.",
        "Always filter out negative revenues and null customer_ids.",
        "Must be non-null sequence id."
    ]
    rules = b.distill_rules_from_history(prompts)
    assert len(rules) == 3
    assert any("Always filter out negative revenues" in r for r in rules)


def test_turbo_ast_vectorizer():
    """Verify AST transpilation of row-wise lambdas to vectorized Polars SIMD expressions."""
    from deepanalyze import turbo_compiler

    code = "df = df.with_columns(pl.col('amount').map_elements(lambda x: 100 if x > 50 else 0))"
    optimized_code, log = turbo_compiler.compile_to_turbo_simd(code)
    assert log["optimized"] is True
    assert "pl.when(pl.col('amount') > 50).then(100).otherwise(0)" in optimized_code


def test_debate_dialectical_split():
    """Verify Dialectical Persona Split generates Growth Bull and Risk Auditor perspectives."""
    import pandas as pd
    from deepanalyze import debate_router

    df = pd.DataFrame({
        "sales": [1000, 2500, 5000, 7500],
        "margin": [0.35, 0.40, 0.20, -0.05]
    })
    res = debate_router.generate_debate_analysis(df, goal="Assess Quarterly Unit Economics")
    assert "growth_bull" in res
    assert "risk_auditor" in res
    assert "synthesis" in res


def test_falsify_skeptic_battery():
    """Verify analytical skeptic counter-investigation detects outlier fragility."""
    import pandas as pd
    from deepanalyze import debate_router

    # Highly concentrated dataset (>80% in single row)
    fragile_df = pd.DataFrame({
        "revenue": [10, 15, 20, 15, 1000]
    })
    fals_res = debate_router.run_falsification_battery(fragile_df, target_col="revenue")
    assert fals_res["is_fragile"] is True
    assert len(fals_res["warnings"]) >= 1


def test_pipeline_etl_compiler(tmp_path):
    """Verify compilation of session history into standalone production pipeline script."""
    from deepanalyze import pipeline_compiler

    out_script = str(tmp_path / "prod_pipeline.py")
    res = pipeline_compiler.compile_production_pipeline_script(
        history_metadata=[{"action": "clean_nulls"}, {"action": "unravel_erp"}],
        target_name="sales_data",
        output_path=out_script
    )
    assert os.path.exists(res)
    with open(res, "r", encoding="utf-8") as f:
        content = f.read()
    assert "def run_pipeline" in content
    assert "def main" in content


def test_report_html_brief(tmp_path):
    """Verify generation of self-contained dark-mode executive HTML report."""
    import pandas as pd
    from deepanalyze import pipeline_compiler

    df = pd.DataFrame({
        "revenue": [100, 250, 400, 600],
        "profit": [20, 50, 80, 120]
    })
    out_html = str(tmp_path / "report.html")
    res = pipeline_compiler.generate_self_contained_html_report(df, charts_dir=str(tmp_path), output_path=out_html)
    assert os.path.exists(res)
    with open(res, "r", encoding="utf-8") as f:
        content = f.read()
    assert "<!DOCTYPE html>" in content
    assert "DeepAnalyze Executive Report" in content


def test_enrich_async_fetcher():
    """Verify autonomous data enrichment with standard taxonomy & SIC codes."""
    import pandas as pd
    from deepanalyze import enricher

    df = pd.DataFrame({
        "company": ["Apple Software", "Google Cloud Consulting", "Walmart Retail Store"],
        "revenue": [5000, 4000, 3000]
    })
    en_df, log = enricher.enrich_dataset_async(df)
    assert "enriched_sector" in en_df.columns
    assert "enriched_sic_code" in en_df.columns
    assert log["records_matched"] >= 2


def test_semantic_vector_filter():
    """Verify natural language semantic vector search without regex."""
    import pandas as pd
    from deepanalyze import enricher

    df = pd.DataFrame({
        "ticket_id": [1, 2, 3, 4],
        "complaint": [
            "The device battery is overheating and melting",
            "I want to update my billing address and phone",
            "Screen cracked and power supply unit exploded",
            "Subscription renewal discount question"
        ]
    })
    filtered = enricher.filter_by_semantic_meaning(df, query="hardware failure broken overheating defect", text_col="complaint", top_k=2)
    assert len(filtered) == 2
    assert 1 in filtered["ticket_id"].values or 3 in filtered["ticket_id"].values


def test_causal_treatment_effect():
    """Verify Treatment Effect Engine computes ATE with confidence intervals."""
    import numpy as np
    import pandas as pd
    from deepanalyze import causal_engine

    np.random.seed(42)
    n = 100
    discount = np.random.binomial(1, 0.5, size=n)
    sales = 50 + 25 * discount + np.random.normal(0, 5, size=n)
    df = pd.DataFrame({"discount": discount, "sales": sales})

    ate_res = causal_engine.estimate_treatment_effect(df, treatment_col="discount", outcome_col="sales")
    assert "average_treatment_effect_ate" in ate_res
    assert ate_res["average_treatment_effect_ate"] > 15.0
    assert ate_res["statistically_significant"] is True


def test_auto_feat_ensemble():
    """Verify feature discovery factory selects top-5 orthogonal predictive features."""
    import pandas as pd
    from deepanalyze import feature_forge

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=15, freq="D"),
        "spend": np.linspace(10, 100, 15),
        "conversions": np.linspace(2, 20, 15) + np.random.normal(0, 0.5, 15),
        "channel": ["Search", "Social", "Email"] * 5
    })
    ef_df, log = feature_forge.ensemble_feature_discovery(df, target_col="conversions", top_k=5)
    assert log["engineered_features_created"] <= 5
    assert len(ef_df.columns) <= len(df.columns) + 5


def test_twin_adversarial():
    """Verify Adversarial Digital Twin generates 20% shifted synthetic stress dataset."""
    import pandas as pd
    from deepanalyze import synthetic_data

    real_df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        "units": [10, 20, 30, 40, 50]
    })
    twin_df = synthetic_data.generate_adversarial_digital_twin(real_df, shift_factor=0.20)
    assert twin_df.shape[0] == real_df.shape[0]
    assert twin_df["revenue"].mean() != real_df["revenue"].mean()


def test_weave_cross_lingual_join():
    """Verify cross-lingual fuzzy semantic joining."""
    import pandas as pd
    from deepanalyze import enricher

    df_en = pd.DataFrame({
        "product": ["Ultra Pro Laptop 15", "Wireless Gaming Mouse", "Mechanical Keyboard"],
        "price_usd": [1200, 80, 150]
    })
    df_ar = pd.DataFrame({
        "item_name": ["Laptop Ultra Pro 15 inch", "Gaming Mouse Wireless", "Keypad Mechanical"],
        "stock": [50, 120, 40]
    })
    woven = enricher.cross_lingual_semantic_join(df_en, df_ar, left_on="product", right_on="item_name")
    assert woven.shape[0] == 3
    assert "right_stock" in woven.columns
    assert woven["_weave_similarity"].mean() > 0.3


def test_solve_prescriptive_optimizer():
    """Verify LP Prescriptive Optimization solver for resource allocation."""
    import pandas as pd
    from deepanalyze import optimizer

    df = pd.DataFrame({
        "project": ["Project A", "Project B", "Project C", "Project D"],
        "expected_roi": [50000, 80000, 30000, 120000],
        "cost": [10000, 25000, 5000, 40000]
    })
    opt_df, opt_log = optimizer.solve_resource_allocation_lp(df, value_col="expected_roi", cost_col="cost", max_budget=40000)
    assert "optimal_allocation_weight" in opt_df.columns
    assert opt_log["total_budget_utilized"] <= 40000
    assert opt_log["objective_max_value"] > 0


def test_evolve_schema_healing():
    """Verify adaptive schema healing rewrites Polars AST when columns drift."""
    from deepanalyze import optimizer

    old_schema = {"invoice_id": "Int64", "doc_amount": "Float64"}
    new_schema = {"inv_id": "Int64", "document_amount": "Float64"}
    code = "df = df.filter(pl.col('doc_amount') > 100)"

    healed_code, log = optimizer.heal_schema_drift(old_schema, new_schema, code)
    assert log["healed"] is True
    assert "rename" in healed_code


def test_brain_biomimetic_memory(tmp_path):
    """Verify Biomimetic RAG Brain 3 lifecycles: Phase A hash, Phase B context, Phase C prune."""
    import pandas as pd
    from deepanalyze import brain

    b = brain.BiomimeticBrain(storage_path=str(tmp_path / "brain_memory.json"))
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # Phase A: Passive Ingestion
    geo_hash = b.compute_geometry_hash(df)
    assert len(geo_hash) == 16
    b.log_execution_delta(df, "df = df.select(['a'])", success=True, duration_ms=25.0)

    # Phase B: Context Injection
    ctx = b.get_context_injection(df)
    assert ctx["geometry_hash"] == geo_hash

    # Phase C: Pruning & Consolidation
    b.consolidate_and_prune()
    assert os.path.exists(str(tmp_path / "brain_memory.json"))


def test_storyteller_interactive_html_and_marp_presentation(tmp_path):
    """Verify interactive HTML presentation and Marp markdown generation."""
    import pandas as pd
    from deepanalyze import storyteller

    df = pd.DataFrame({
        "revenue": [1000, 2000, 3000, 4000],
        "profit": [200, 400, 600, 800]
    })
    memo = storyteller.generate_executive_memo(df)

    # 1. Interactive HTML slide deck
    html_out = str(tmp_path / "deck.html")
    html_res = storyteller.generate_interactive_slide_deck_html(memo, output_path=html_out)
    assert os.path.exists(html_out)
    assert "class=\"slide active\"" in html_res
    assert "Executive Strategic Presentation" in html_res

    # 2. Marp presentation
    marp_out = str(tmp_path / "deck.marp.md")
    marp_res = storyteller.generate_marp_presentation_md(memo, output_path=marp_out)
    assert os.path.exists(marp_out)
    assert "marp: true" in marp_res
    assert "Pillar 1:" in marp_res


def test_schema_synthesizer_multi_dialect_ddl_and_partitions():
    """Verify Snowflake cluster keys, BigQuery date partitions, and dbt tests."""
    import pandas as pd
    from deepanalyze import schema_synthesizer

    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "order_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        "status": ["PAID", "PENDING", "PAID", "REFUNDED"],
        "amount": [150.0, 250.0, 350.0, 450.0]
    })

    # BigQuery Partitioning
    bq_ddl = schema_synthesizer.infer_sql_schema(df, table_name="orders", dialect="bigquery")
    assert "PARTITION BY DATE(order_date)" in bq_ddl
    assert "INT64" in bq_ddl

    # Snowflake Cluster
    sf_ddl = schema_synthesizer.infer_sql_schema(df, table_name="orders", dialect="snowflake")
    assert "CLUSTER BY (order_id)" in sf_ddl
    assert "NUMBER(38,0)" in sf_ddl

    # Postgres Index
    pg_ddl = schema_synthesizer.infer_sql_schema(df, table_name="orders", dialect="postgres")
    assert "CREATE INDEX" in pg_ddl

    # dbt Accepted Values Test
    dbt_yml = schema_synthesizer.generate_dbt_models(df, table_name="orders")
    assert "accepted_values" in dbt_yml
    assert "'PAID'" in dbt_yml


def test_cleaners_unravel_with_page_breaks_and_subtotals():
    """Verify ERP unravelling strips repetitive page breaks and subtotals without double-counting."""
    import pandas as pd
    from deepanalyze import cleaners

    messy_report = pd.DataFrame([
        ["Page 1 of 3", None, None, "Printed On: 2026-08-28"],
        ["Customer : All", "Date : From 01/01/2026", None, None],
        ["Invoice No", "Date", "Customer", "Amount"],
        ["INV-001", "2026-01-10", "Acme Corp", "1500.00"],
        ["Page 2 of 3", None, None, "Printed On: 2026-08-28"],
        ["INV-002", "2026-01-12", "Beta LLC", "2500.00"],
        ["Subtotal for North Region", None, None, "4000.00"],
        ["Grand Total", None, None, "4000.00"]
    ])

    unravelled = cleaners.unravel_hierarchical_erp_report(messy_report)
    pdf = unravelled.to_pandas() if hasattr(unravelled, "to_pandas") else unravelled
    # Should only contain 2 clean transaction line items (INV-001 and INV-002)
    assert len(pdf) == 2
    assert "Acme Corp" in pdf.to_string()
    assert "Beta LLC" in pdf.to_string()


def test_core_grammar_constrained_ast_linter():
    """Verify grammar auto-patching of invalid LLM method calls."""
    from deepanalyze.core import _lint_and_format_code

    # Simulated 8B LLM syntax slip: .str_slice(0, 5) and .groupby('id')
    hallucinated_code = """
import pandas as pd
import polars as pl
df = df.with_columns(pl.col('name').str_slice(0, 3))
"""
    is_valid, clean_code, err = _lint_and_format_code(hallucinated_code, {"df", "pl", "pd"})
    assert is_valid is True
    assert ".str.slice(0, 3)" in clean_code


def test_feature_forge_safe_div_and_inf_clamping():
    """Verify universal safe_div prevents zero-division and inf crashes."""
    import numpy as np
    import pandas as pd
    from deepanalyze.feature_forge import safe_div, auto_engineer_features

    # Scalar
    assert safe_div(100, 0) == 0.0
    assert safe_div(100, np.nan) == 0.0

    # Vector
    s1 = pd.Series([10.0, 20.0, 30.0])
    s2 = pd.Series([0.0, 5.0, 0.0])
    res = safe_div(s1, s2)
    assert res[0] == 0.0
    assert res[1] == 4.0
    assert res[2] == 0.0
    assert not np.isinf(res).any()

    # Feature Engineering Pipeline on 0-denominator matrix
    df = pd.DataFrame({"rev": [0, 0, 100, 200], "units": [0, 0, 0, 10]})
    fe_df, log = auto_engineer_features(df)
    pdf = fe_df.to_pandas() if hasattr(fe_df, "to_pandas") else fe_df
    assert not np.isinf(pdf.select_dtypes(include=[np.number]).values).any()


def test_cleaners_dirty_currency_and_mixed_datetime_sanitizers():
    """Verify sanitization of dirty currencies, percentages, and mixed datetimes."""
    import pandas as pd
    from deepanalyze.cleaners import sanitize_dirty_numeric_series, auto_cast_data_types

    assert sanitize_dirty_numeric_series("$1,250.50") == 1250.50
    assert sanitize_dirty_numeric_series("(500.00)") == -500.00
    assert sanitize_dirty_numeric_series("15.5%") == 15.5
    assert sanitize_dirty_numeric_series("SAR 3,400.00") == 3400.00

    dirty_df = pd.DataFrame({
        "revenue": ["$1,000.00", "$2,500.50", "(100.00)", "N/A"],
        "rate": ["5%", "10%", "15%", "20%"],
        "date": ["2026-01-15", "16/01/2026", "2026-02-01", "2026-02-15"]
    })
    cleaned_df = auto_cast_data_types(dirty_df)
    pdf = cleaned_df.to_pandas() if hasattr(cleaned_df, "to_pandas") else cleaned_df
    assert pd.api.types.is_numeric_dtype(pdf["revenue"])
    assert pd.api.types.is_numeric_dtype(pdf["rate"])
    assert pd.api.types.is_datetime64_any_dtype(pdf["date"])


def test_core_atomic_execution_gate_rollback():
    """Verify _AtomicExecutionGate preserves state on runtime exceptions."""
    import pandas as pd
    from deepanalyze.core import _AtomicExecutionGate

    class MockIPython:
        def __init__(self):
            self.user_ns = {"target_df": pd.DataFrame({"a": [1, 2, 3]})}

    mock_ip = MockIPython()
    try:
        with _AtomicExecutionGate(mock_ip, "target_df"):
            mock_ip.user_ns["target_df"] = pd.DataFrame({"a": [999]})
            raise RuntimeError("Simulated crash")
    except RuntimeError:
        pass

    # Target DF must have been rolled back to original [1, 2, 3]
    assert mock_ip.user_ns["target_df"]["a"].tolist() == [1, 2, 3]


def test_core_threadpool_clamp_environment():
    """Verify Apple Silicon threadpool affinity clamps are configured."""
    assert "POLARS_MAX_THREADS" in os.environ
    assert "OMP_NUM_THREADS" in os.environ
    assert int(os.environ["POLARS_MAX_THREADS"]) >= 1


def test_core_wide_table_schema_pruning():
    """Verify schema RAG context on >30 column tables is cleanly capped at top 25."""
    import pandas as pd
    from deepanalyze.core import _get_deep_workspace_context

    class MockIP:
        def __init__(self):
            data = {f"col_{i}": list(range(30)) for i in range(50)}
            self.user_ns = {"wide_df": pd.DataFrame(data)}

    mock_ip = MockIP()
    # 1. Raw Preview
    ctx_raw, _, _ = _get_deep_workspace_context(mock_ip, is_cloud=False, privacy_mode="none")
    assert "wide_df" in ctx_raw
    assert "... and 25 other continuous/categorical dimensions" in ctx_raw

    # 2. Privacy Mode
    ctx_priv, _, _ = _get_deep_workspace_context(mock_ip, is_cloud=False, privacy_mode="profile")
    assert "wide_df" in ctx_priv
    assert "... and 25 other continuous/categorical dimensions" in ctx_priv


def test_core_atomic_export_swap(tmp_path):
    """Verify atomic file export swap writes clean output and leaves no tmp files."""
    import argparse
    import pandas as pd
    from deepanalyze.core import _handle_export

    out_file = str(tmp_path / "clean_output.parquet")
    df = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})

    class MockIP:
        def __init__(self):
            self.user_ns = {"my_df": df}

    mock_ip = MockIP()
    args = argparse.Namespace(export="my_df", to=out_file)
    _handle_export(mock_ip, args)

    assert os.path.exists(out_file)
    # Ensure no lingering .tmp files
    tmp_files = [f for f in os.listdir(str(tmp_path)) if ".tmp" in f]
    assert len(tmp_files) == 0












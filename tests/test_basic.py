import pytest
import pandas as pd
import numpy as np
import deepanalyze
from deepanalyze.privacy_knife import DeepAnalyzePrivacyKnife, LocalGatekeeper
from deepanalyze.core import _extract_deepanalyze_content, _reconcile_target_dataframe

def test_version():
    assert hasattr(deepanalyze, "__version__")
    assert deepanalyze.__version__ == "2.1.0"

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


import os





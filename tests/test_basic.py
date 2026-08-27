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

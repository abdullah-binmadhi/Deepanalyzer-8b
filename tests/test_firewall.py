"""Test 8.3: AST Security Firewall Enforcement Test."""

from deepanalyze.firewall import audit_code, execute_code_safely, ASTSecurityViolation


def test_firewall_blocks_forbidden_imports():
    forbidden_snippets = [
        "import socket\ns = socket.socket()",
        "import requests\nresp = requests.get('https://example.com')",
        "import urllib.request\nurllib.request.urlopen('http://evil.com')",
        "import httpx\nclient = httpx.Client()",
        "import subprocess\nsubprocess.run(['ls', '-la'])",
        "from os import system\nsystem('whoami')",
        "from os import environ\nk = environ['SECRET']",
        "import os\nx = os.environ.get('AWS_KEY')",
        "import os\nos.remove('/etc/hosts')",
        "eval('__import__(\"os\").system(\"ls\")')",
        "exec('import socket')",
        "x = ().__class__.__base__.__subclasses__()"
    ]

    for snippet in forbidden_snippets:
        blocked = False
        try:
            audit_code(snippet)
        except ASTSecurityViolation as e:
            blocked = True
            assert "AST Security Violation" in str(e)
        assert blocked, f"Failed to block dangerous snippet: {snippet}"


def test_firewall_allows_safe_polars_code():
    safe_snippets = [
        "import polars as pl\ndf = df.with_columns(pl.col('x') * 2)",
        "import datetime\nd = datetime.date(2026, 1, 1)",
        "import math\ns = math.sqrt(144)",
        "import re\np = re.compile(r'\\d+')"
    ]

    for snippet in safe_snippets:
        assert audit_code(snippet) is True


def test_execute_code_safely_main_block():
    """Validates that __name__ == '__main__' blocks execute properly."""
    code = """
executed = False
if __name__ == "__main__":
    executed = True
"""
    scope = {}
    execute_code_safely(code, scope)
    assert scope.get("executed") is True


def test_resolve_transformed_dataframe_all_modes(tmp_path):
    """Validates resolve_transformed_dataframe across output files, variables, and functions."""
    import pandas as pd
    from deepanalyze.firewall import resolve_transformed_dataframe

    orig_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # 1. Output file mode
    out_file = str(tmp_path / "cleaned.csv")
    cleaned_df = pd.DataFrame({"a": [10, 20, 30], "b": ["X", "Y", "Z"]})
    cleaned_df.to_csv(out_file, index=False)
    scope1 = {"OUTPUT_FILE": out_file}
    resolved, source = resolve_transformed_dataframe(scope1, orig_df, "df")
    assert "output file" in source
    assert resolved.shape == (3, 2)
    assert resolved["a"].tolist() == [10, 20, 30]

    # 2. Transformed variable mode
    scope2 = {"df_cleaned": cleaned_df}
    resolved2, source2 = resolve_transformed_dataframe(scope2, orig_df, "df")
    assert "df_cleaned" in source2
    assert resolved2["a"].tolist() == [10, 20, 30]

    # 3. Callable function mode
    def clean_records(df):
        df = df.copy()
        df["c"] = [100, 200, 300]
        return df

    scope3 = {"clean_records": clean_records}
    resolved3, source3 = resolve_transformed_dataframe(scope3, orig_df, "df")
    assert "clean_records(df)" in source3
    assert "c" in resolved3.columns
    assert resolved3["c"].tolist() == [100, 200, 300]


def test_push_snapshot_resilience():
    """Validates that push_snapshot handles None, custom objects, and dicts without NameError on copy."""
    from deepanalyze.firewall import push_snapshot, pop_snapshot

    # None should be safely ignored
    push_snapshot("test_none", None)
    assert pop_snapshot("test_none") is None

    # Arbitrary object should be copied with copy.deepcopy without error
    custom_obj = {"sample": [1, 2, 3]}
    push_snapshot("test_dict", custom_obj)
    restored = pop_snapshot("test_dict")
    assert restored == custom_obj
    assert restored is not custom_obj


def test_execute_code_safely_preinjected_copy_and_re():
    """Validates that copy and re are pre-injected into execution scope."""
    code = """
matched = bool(re.search(r"\\d+", "Model 123"))
cloned = copy.deepcopy([1, 2, 3])
"""
    scope = {}
    execute_code_safely(code, scope)
    assert scope["matched"] is True
    assert scope["cloned"] == [1, 2, 3]


def test_audit_transformation_safety_blocks_empty_dataframe():
    """Validates that audit_transformation_safety fatal-blocks empty DataFrames."""
    import pandas as pd
    import pytest
    from deepanalyze.firewall import audit_transformation_safety, OverCleaningViolation

    raw_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "val": [10, 20, 30, 40, 50, 60]})
    empty_df = pd.DataFrame(columns=["id", "val"])

    with pytest.raises(OverCleaningViolation, match="completely empty"):
        audit_transformation_safety(raw_df, empty_df)


def test_audit_transformation_safety_clean_tabular_retention():
    """Validates that audit_transformation_safety blocks excessive row drop in clean tabular data."""
    import pandas as pd
    import pytest
    from deepanalyze.firewall import audit_transformation_safety, OverCleaningViolation

    raw_df = pd.DataFrame({"id": list(range(100)), "amount": [10.0] * 100})

    # 1. Normal mild cleaning (dropping 5% of rows) -> Passes
    cleaned_normal = raw_df.iloc[:95]
    ok, msg = audit_transformation_safety(raw_df, cleaned_normal)
    assert ok is True

    # 2. Destructive drop (dropping 50% of rows via blind dropna) -> Blocks
    cleaned_destructive = raw_df.iloc[:50]
    with pytest.raises(OverCleaningViolation, match="lost 50.0% of records"):
        audit_transformation_safety(raw_df, cleaned_destructive)


def test_audit_transformation_safety_financial_conservation():
    """Validates that audit_transformation_safety flags zeroed out financial sums."""
    import pandas as pd
    import pytest
    from deepanalyze.firewall import audit_transformation_safety, OverCleaningViolation

    raw_df = pd.DataFrame({"item": ["A", "B", "C", "D", "E", "F"], "amount": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]})
    corrupted_df = pd.DataFrame({"item": ["A", "B", "C", "D", "E", "F"], "amount": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    with pytest.raises(OverCleaningViolation, match="Financial Conservation Violation"):
        audit_transformation_safety(raw_df, corrupted_df)


def test_audit_transformation_safety_ragged_erp_line_retention():
    """Validates that audit_transformation_safety checks expected line item retention on ERP data."""
    import pandas as pd
    import pytest
    from deepanalyze.firewall import audit_transformation_safety, OverCleaningViolation

    # ERP with 10 line items
    rows = [
        {"Date": "Doc. No", "Value": "IV-101"},
        {"Date": "1000", "Value": "Item 1"},
        {"Date": "1001", "Value": "Item 2"},
        {"Date": "1002", "Value": "Item 3"},
        {"Date": "1003", "Value": "Item 4"},
        {"Date": "1004", "Value": "Item 5"},
        {"Date": "1005", "Value": "Item 6"},
        {"Date": "1006", "Value": "Item 7"},
        {"Date": "1007", "Value": "Item 8"},
        {"Date": "1008", "Value": "Item 9"},
        {"Date": "1009", "Value": "Item 10"},
    ]
    raw_erp = pd.DataFrame(rows)

    # 1. Properly flattened 10 line items -> Passes
    flattened_good = pd.DataFrame({"doc_no": ["IV-101"] * 10, "seq": list(range(1000, 1010)), "amount": [100.0] * 10})
    ok, _ = audit_transformation_safety(raw_erp, flattened_good, arch_key="ERP_RAGGED")
    assert ok is True

    # 2. Overcleaned: only 2 line items retained (< 40%) -> Blocks
    flattened_bad = pd.DataFrame({"doc_no": ["IV-101"] * 2, "seq": [1000, 1001], "amount": [100.0] * 2})
    with pytest.raises(OverCleaningViolation, match="Over-Cleaning Violation"):
        audit_transformation_safety(raw_erp, flattened_bad, arch_key="ERP_RAGGED")


if __name__ == "__main__":
    test_firewall_blocks_forbidden_imports()
    test_firewall_allows_safe_polars_code()
    test_execute_code_safely_main_block()
    test_push_snapshot_resilience()
    test_execute_code_safely_preinjected_copy_and_re()
    test_audit_transformation_safety_blocks_empty_dataframe()
    test_audit_transformation_safety_clean_tabular_retention()
    test_audit_transformation_safety_financial_conservation()
    test_audit_transformation_safety_ragged_erp_line_retention()
    print("test_firewall.py passed!")

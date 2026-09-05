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


if __name__ == "__main__":
    test_firewall_blocks_forbidden_imports()
    test_firewall_allows_safe_polars_code()
    test_execute_code_safely_main_block()
    print("test_firewall.py passed!")

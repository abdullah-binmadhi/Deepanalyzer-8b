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


if __name__ == "__main__":
    test_firewall_blocks_forbidden_imports()
    test_firewall_allows_safe_polars_code()
    print("✅ test_firewall.py passed!")

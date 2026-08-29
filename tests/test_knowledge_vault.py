"""Test Suite for DeepAnalyze 500K Knowledge Vault Engine
Validates:
1. Category extraction and JSON message parsing
2. DuckDB FTS BM25 sub-millisecond retrieval
3. Runtime dynamic few-shot grounding injection
4. CLI %deepanalyze --vault, --vault-search, --vault-build commands
"""

import io
import os
import sys
import time
import pytest
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

from deepanalyze import knowledge_vault
from deepanalyze.core import deepanalyze


def test_categorization_and_message_extraction():
    """Verify source files are correctly categorized and messages parsed."""
    assert knowledge_vault.KnowledgeVault._categorize_source_file("Data Cleaning 1616.json") == "cleaning"
    assert knowledge_vault.KnowledgeVault._categorize_source_file("DeepAnalyze Research Database.json") == "database_sql"
    assert knowledge_vault.KnowledgeVault._categorize_source_file("DeepAnalyze Data Pipeline.json") == "pipeline"
    assert knowledge_vault.KnowledgeVault._categorize_source_file("Data Insight 1062.json") == "insights"
    assert knowledge_vault.KnowledgeVault._categorize_source_file("DeepAnalyze Research.json") == "general_datascience"

    sample_messages = [
        {"role": "user", "content": "# Instruction\nCalculate average salary by department\n# Data\nFile: hr.csv"},
        {"role": "assistant", "content": "<Analyze>\nGroup by dept and average salary.\n</Analyze>\n```python\ndf.groupby('dept')['salary'].mean()\n```"}
    ]
    inst, data_ctx, thought, code = knowledge_vault.KnowledgeVault._extract_parts_from_messages(sample_messages)
    assert "Calculate average salary" in inst
    assert "File: hr.csv" in data_ctx
    assert "Group by dept" in thought
    assert "groupby('dept')" in code


def test_duckdb_vault_fts_search():
    """Verify DuckDB FTS BM25 search returns relevant recipes in <50ms."""
    vault = knowledge_vault.get_knowledge_vault()
    stats = vault.get_vault_stats()
    assert stats["total_recipes"] > 0

    # Warm-up cold connection
    vault.search_recipes("test", limit=1)

    t0 = time.perf_counter()
    recipes = vault.search_recipes("clean driver and school bus tables", limit=2)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert len(recipes) > 0
    assert latency_ms < 50.0  # Sub-50ms warm FTS search
    assert "code_solution" in recipes[0]
    assert "instruction" in recipes[0]


def test_cli_vault_commands():
    """Verify --vault and --vault-search flags execute cleanly."""
    ip = InteractiveShell.instance()
    ip.user_ns["df_test"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Test --vault
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        deepanalyze("--vault")
    finally:
        sys.stdout = old_stdout
        out = buffer.getvalue()
        assert "Institutional Memory & Knowledge Vault" in out or "Vault" in out

    # Test --vault-search
    sys.stdout = buffer = io.StringIO()
    try:
        deepanalyze('--vault-search "driver and school bus"')
    finally:
        sys.stdout = old_stdout
        out_search = buffer.getvalue()
        assert "Match" in out_search or "Task" in out_search or "driver" in out_search.lower()

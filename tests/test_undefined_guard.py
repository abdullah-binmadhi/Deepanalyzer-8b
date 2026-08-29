import ast
import polars as pl
import pandas as pd
import pytest
from deepanalyze.core import (
    _resolve_target_dataframe,
    _auto_seed_session_namespace,
    _inject_required_imports,
    _pre_sanitize_code_string,
    _lint_and_format_code,
    _display_metrics,
    _resolve_cloud_provider_info
)
from deepanalyze.cleaners import (
    split_compound_slash_columns,
    auto_cast_data_types
)
from deepanalyze.feature_forge import auto_engineer_features


class MockIPython:
    def __init__(self, user_ns=None):
        self.user_ns = user_ns if user_ns is not None else {}
        self.next_input = None

    def set_next_input(self, code):
        self.next_input = code


def test_target_auto_discovery_single_df():
    df = pl.DataFrame({"a": [1, 2, 3]})
    ip = MockIPython({"health_pl": df})
    
    resolved = _resolve_target_dataframe(ip, requested_target="df", prompt="clean all column types")
    assert resolved == "health_pl"
    assert "df" in ip.user_ns
    assert ip.user_ns["df"] is df


def test_target_auto_discovery_multiple_dfs():
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6]})
    ip = MockIPython({"health_pl": df1, "patients_df": df2})

    # Mention patients_df in prompt
    resolved = _resolve_target_dataframe(ip, requested_target="df", prompt="summarize patients_df")
    assert resolved == "patients_df"


def test_auto_seed_session_namespace():
    ip = MockIPython({})
    _auto_seed_session_namespace(ip)
    assert "pd" in ip.user_ns
    assert "np" in ip.user_ns
    assert "pl" in ip.user_ns
    assert "duckdb" in ip.user_ns
    assert "re" in ip.user_ns
    assert "json" in ip.user_ns
    assert "datetime" in ip.user_ns


def test_inject_required_imports():
    raw_code = "con = duckdb.sql('SELECT * FROM df')\nres = pl.from_arrow(con.arrow())"
    injected = _inject_required_imports(raw_code)
    assert "import duckdb" in injected
    assert "import polars as pl" in injected


def test_pre_sanitize_spaced_assignment():
    raw_code = "Blood Pressure = None\ndf = df.drop(['Age'])"
    sanitized = _pre_sanitize_code_string(raw_code, target_name="health_pl")
    assert "Blood Pressure = None" not in sanitized
    assert "health_pl.drop(['Blood Pressure'])" in sanitized


def test_pre_sanitize_unwrap_function_with_toy_data():
    raw_code = """
def clean_health_data(df):
    toy_data = [{'Patient Name': 'John Doe', 'Blood Pressure': '120/80'}]
    df = df.with_columns(Patient_Name=pl.col('Patient Name').str.to_uppercase())
    return df
clean_health_data(df)
"""
    sanitized = _pre_sanitize_code_string(raw_code, target_name="health_pl")
    assert "def clean_health_data" not in sanitized
    assert "Patient_Name=pl.col('Patient Name').str.to_uppercase()" in sanitized


def test_ast_target_rewriter_and_linter():
    # health_pl is available, df is NOT
    available_vars = {"health_pl", "pl", "pd", "np"}
    raw_code = "df = df.with_columns(Clean_Age=pl.col('Age').cast(pl.Int64, strict=False))"
    
    is_valid, clean_code, err = _lint_and_format_code(raw_code, available_vars, target_name="health_pl")
    assert is_valid
    assert "health_pl = health_pl.with_columns" in clean_code


def test_polars_list_index_grammar_patch():
    available_vars = {"health_pl", "pl"}
    raw_code = "health_pl = health_pl.with_columns(Systolic=pl.col('Blood Pressure').str.split('/').str[0])"
    
    is_valid, clean_code, err = _lint_and_format_code(raw_code, available_vars, target_name="health_pl")
    assert is_valid
    assert ".list.get(0)" in clean_code


def test_split_compound_slash_columns_polars():
    df = pl.DataFrame({
        "Patient Name": ["John Doe", "Jane Smith"],
        "Blood Pressure": ["120/80", "140/90"]
    })
    transformed = split_compound_slash_columns(df)
    assert "Blood_Pressure_Systolic" in transformed.columns
    assert "Blood_Pressure_Diastolic" in transformed.columns
    assert transformed["Blood_Pressure_Systolic"].to_list() == [120, 140]
    assert transformed["Blood_Pressure_Diastolic"].to_list() == [80, 90]


def test_auto_cast_mixed_dates_polars():
    df = pl.DataFrame({
        "Visit Date": ["10/24/2023", "2023-11-05", "May 12, 2023"],
        "Age": ["45", "30", "62"]
    })
    transformed = auto_cast_data_types(df)
    assert transformed["Visit Date"].dtype in (pl.Date, pl.Datetime)
    assert transformed["Age"].dtype == pl.Int64


def test_auto_engineer_features_polars():
    df = pl.DataFrame({
        "Blood Pressure": ["120/80", "135/85"],
        "Visit Date": ["10/24/2023", "11/05/2023"]
    })
    fe_df, fe_log = auto_engineer_features(df)
    assert "Blood_Pressure_Systolic" in fe_df.columns
    assert "Blood_Pressure_Diastolic" in fe_df.columns
    assert fe_log["engineered_features_created"] > 0


def test_questions_generator_flags(monkeypatch):
    from deepanalyze.core import FLAGS, deepanalyze
    from IPython.core.interactiveshell import InteractiveShell

    assert "--questions" in FLAGS
    assert "-q" in FLAGS
    assert "--ask" in FLAGS

    ip = InteractiveShell.instance()
    ip.user_ns["sample_sales_df"] = pl.DataFrame({
        "Revenue": [100.0, 200.0, 300.0],
        "Customer_ID": [1, 2, 3],
        "Region": ["North", "South", "East"]
    })

    # Mock _call_llm
    monkeypatch.setattr("deepanalyze.core._call_llm", lambda prompt, system_prompt, **kwargs: (
        "### 1. Revenue Distribution by Region\n"
        "Assess which geographic territories drive disproportionate top-line revenue.\n"
        "%deepanalyze -x group by Region and calculate sum and mean of Revenue\n\n"
        "### 2. High-Value Customer Concentration\n"
        "Identify the top decile of customers by total spend.\n"
        "%deepanalyze -x sort by Revenue descending limit 5"
    ))

    deepanalyze("--questions --target sample_sales_df")
    deepanalyze("--ask --target sample_sales_df")

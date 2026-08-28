"""
DeepAnalyze Schema Synthesizer Engine
Bridges Python DataFrames to modern SQL data fabrics, DuckDB, dbt models,
and Entity-Relationship (ER) lineage diagrams.
"""

import pandas as pd
import numpy as np


def infer_sql_schema(df, table_name: str = "analytics_table", dialect: str = "duckdb") -> str:
    """Infers optimal SQL DDL with constraint discovery (Primary Keys, NOT NULL).
    Supported Dialects: 'duckdb', 'postgres', 'snowflake', 'bigquery', 'sqlite'.
    """
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    dialect = dialect.lower()

    # Type Mapping Matrix
    type_map = {
        "duckdb": {"int64": "BIGINT", "float64": "DOUBLE", "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP", "object": "VARCHAR"},
        "postgres": {"int64": "BIGINT", "float64": "DOUBLE PRECISION", "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP", "object": "TEXT"},
        "snowflake": {"int64": "NUMBER(38,0)", "float64": "FLOAT", "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP_NTZ", "object": "VARCHAR"},
        "bigquery": {"int64": "INT64", "float64": "FLOAT64", "bool": "BOOL", "datetime64[ns]": "TIMESTAMP", "object": "STRING"},
        "sqlite": {"int64": "INTEGER", "float64": "REAL", "bool": "INTEGER", "datetime64[ns]": "TEXT", "object": "TEXT"}
    }
    mapping = type_map.get(dialect, type_map["duckdb"])

    # 1. Discover Candidate Primary Key (Unique & Non-null)
    pk_candidate = None
    for col in pdf.columns:
        if pdf[col].is_unique and pdf[col].notna().all():
            pk_candidate = col
            break

    # 2. Build Column Definitions
    col_defs = []
    for col in pdf.columns:
        dtype_str = str(pdf[col].dtype)
        sql_type = mapping.get(dtype_str, "VARCHAR" if dialect != "bigquery" else "STRING")
        if "int" in dtype_str: sql_type = mapping["int64"]
        elif "float" in dtype_str: sql_type = mapping["float64"]
        elif "datetime" in dtype_str: sql_type = mapping["datetime64[ns]"]

        constraints = []
        if col == pk_candidate:
            constraints.append("PRIMARY KEY")
        elif pdf[col].notna().all():
            constraints.append("NOT NULL")

        const_str = (" " + " ".join(constraints)) if constraints else ""
        col_defs.append(f"    {col} {sql_type}{const_str}")

    ddl = f"CREATE TABLE IF NOT EXISTS {table_name} (\n" + ",\n".join(col_defs) + "\n);"
    return ddl


def generate_dbt_models(df, table_name: str = "analytics_table") -> str:
    """Generates standard dbt schema.yml with automated documentation and test suites."""
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()

    yml = f"""version: 2

models:
  - name: {table_name}
    description: "DeepAnalyze automated staging model synthesized from verified in-memory dataset."
    columns:
"""
    for col in pdf.columns:
        tests = []
        if pdf[col].is_unique:
            tests.append("unique")
        if pdf[col].notna().all():
            tests.append("not_null")

        yml += f"      - name: {col}\n"
        yml += f"        description: \"Auto-inferred column: {col} ({str(pdf[col].dtype)})\"\n"
        if tests:
            yml += "        tests:\n"
            for t in tests:
                yml += f"          - {t}\n"

    return yml


def generate_er_diagram(df, table_name: str = "AnalyticsTable") -> str:
    """Generates Mermaid Entity-Relationship (ER) diagram markup."""
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df.copy()
    mermaid = f"erDiagram\n    {table_name} {{\n"
    for col in pdf.columns:
        dt = str(pdf[col].dtype).replace("[", "_").replace("]", "").replace("<", "").replace(">", "")
        pk_marker = " PK" if pdf[col].is_unique and pdf[col].notna().all() else ""
        mermaid += f"        {dt} {col}{pk_marker}\n"
    mermaid += "    }\n"
    return mermaid

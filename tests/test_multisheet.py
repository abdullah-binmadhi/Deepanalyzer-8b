"""Unit tests for Multi-Sheet Workbook Topology Discovery, Synchronized Tokenization & Execution Airlock."""

import os
import tempfile
import pandas as pd
import polars as pl
import pytest
from deepanalyze.profiler import (
    SheetRole,
    WorkbookTopology,
    detect_foreign_keys,
    profile_workbook,
    generate_engineering_briefing,
)
from deepanalyze.vault import TokenVault
from deepanalyze.wizard import generate_airgap_payload
from deepanalyze.firewall import execute_code_safely, resolve_transformed_dataframe
from deepanalyze.powerquery import generate_powerquery_m_code


@pytest.fixture
def multi_sheet_excel_path():
    """Creates a temporary multi-sheet Excel file matching enterprise topologies:
    - Sheet 1 (Transactions): 10 rows, primary key tx_id, foreign key cust_id, dirty amount, mixed dates
    - Sheet 2 (Monthly_Pivot): Pivot table with department, jan, feb, mar, and a Total row
    - Sheet 3 (Dim_Customer): Dimension table with cust_id, customer_name, region
    """
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        df_tx = pd.DataFrame({
            "tx_id": [f"TX{i:03d}" for i in range(1, 11)],
            "cust_id": ["C101", "C102", "C103", "C101", "C102", "C104", "C105", "C103", "C101", "C102"],
            "date_val": ["2025-01-01", "02/01/2025", "2025-01-03", "04-Jan-2025", "2025-01-05",
                         "06/01/2025", "2025-01-07", "08-Jan-2025", "2025-01-09", "10/01/2025"],
            "amount_raw": ["$1,000.00", "(250.00)", "SAR 500.00", "750.00-", "$1,200.50",
                           "300.00", "(150.00)", "SAR 900.00", "2,100.00", "$450.00"],
            "status": ["COMPLETED", "PENDING", "COMPLETED", "FAILED", "COMPLETED",
                       "COMPLETED", "PENDING", "COMPLETED", "COMPLETED", "FAILED"],
        })
        df_tx.to_excel(writer, sheet_name="Transactions", index=False)

        df_pivot = pd.DataFrame({
            "Department": ["Sales", "Engineering", "Marketing", "Total"],
            "jan": [15000, 25000, 12000, 52000],
            "feb": [18000, 27000, 14000, 59000],
            "mar": [21000, 29000, 16000, 66000],
        })
        df_pivot.to_excel(writer, sheet_name="Monthly_Pivot", index=False)

        df_dim = pd.DataFrame({
            "cust_id": ["C101", "C102", "C103", "C104", "C105"],
            "customer_name": ["Acme Corp", "Global Logistics", "Apex Tech", "Nexus Energy", "Horizon Media"],
            "region": ["Riyadh", "Jeddah", "Dammam", "Khobar", "Riyadh"],
        })
        df_dim.to_excel(writer, sheet_name="Dim_Customer", index=False)

    yield tmp_path

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


def test_profile_workbook(multi_sheet_excel_path):
    topology = profile_workbook(multi_sheet_excel_path)

    assert len(topology.sheets) == 3
    assert "Transactions" in topology.sheets
    assert "Monthly_Pivot" in topology.sheets
    assert "Dim_Customer" in topology.sheets

    # Verify sheet roles
    assert topology.sheets["Transactions"].role == SheetRole.TRANSACTION_LEDGER
    assert topology.sheets["Monthly_Pivot"].role == SheetRole.PIVOT_TABLE
    assert topology.sheets["Dim_Customer"].role == SheetRole.LOOKUP_DIMENSION

    # Verify primary sheet detection
    assert topology.primary_sheet == "Transactions"

    # Verify subtotal row in pivot sheet
    assert len(topology.sheets["Monthly_Pivot"].subtotal_rows) > 0

    # Verify foreign key candidate detection
    fks = topology.foreign_keys
    assert len(fks) >= 1
    cust_fk = next((fk for fk in fks if "cust_id" in fk.from_col and "cust_id" in fk.to_col), None)
    assert cust_fk is not None
    assert cust_fk.overlap_pct >= 80.0

    # Verify recommended pipeline steps
    assert len(topology.recommended_pipeline_steps) >= 2


def test_synchronized_multi_sheet_tokenization():
    """Verify that tokenizing multiple sheets using a single shared TokenVault
    preserves referential integrity across join keys.
    """
    vault = TokenVault()

    df_tx = pl.DataFrame({
        "tx_id": ["TX01", "TX02", "TX03"],
        "customer_id": ["C101", "C102", "C101"],
        "amount": [100.0, 200.0, 300.0],
    })

    df_dim = pl.DataFrame({
        "customer_id": ["C101", "C102", "C103"],
        "customer_name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
    })

    # Tokenize both sheets with the same vault
    tok_tx = vault.tokenize_dataframe(df_tx)
    tok_dim = vault.tokenize_dataframe(df_dim)

    # Key check: C101 in tok_tx MUST match C101 in tok_dim
    tx_c101_token = tok_tx.filter(pl.col("tx_id") == "TX01")["customer_id"][0]
    dim_c101_token = tok_dim.filter(pl.col("customer_name").str.starts_with("<"))["customer_id"][0]

    assert tx_c101_token == dim_c101_token
    assert tx_c101_token != "C101"
    assert tx_c101_token.startswith("<")

    # Detokenize both sheets and verify 100% fidelity restoration
    detok_tx = vault.detokenize_dataframe(tok_tx)
    detok_dim = vault.detokenize_dataframe(tok_dim)

    assert detok_tx["customer_id"].to_list() == ["C101", "C102", "C101"]
    assert detok_dim["customer_id"].to_list() == ["C101", "C102", "C103"]
    assert detok_dim["customer_name"].to_list() == ["Alice Smith", "Bob Jones", "Charlie Brown"]


def test_multi_sheet_airgap_payload(multi_sheet_excel_path):
    topology = profile_workbook(multi_sheet_excel_path)
    
    # Load sheet DataFrames
    multi_sheets = {}
    import pandas as pd
    for sname in topology.sheets.keys():
        pdf = pd.read_excel(multi_sheet_excel_path, sheet_name=sname)
        multi_sheets[sname] = pl.from_pandas(pdf)

    payload, policy, classified_cols = generate_airgap_payload(
        df=multi_sheets[topology.primary_sheet],
        origin_country="SA",
        target_jurisdiction="US",
        user_prompt="Consolidate customer ledger and unpivot department budget",
        target_df_name="df",
        topology=topology,
        multi_sheets=multi_sheets,
    )

    assert "AUTONOMOUS DATA ENGINEERING & TOPOLOGY BRIEFING" in payload
    assert "MULTI-SHEET EXECUTION CONTEXT" in payload
    assert "Monthly_Pivot" in payload
    assert "Dim_Customer" in payload
    assert "SYNTHETIC SCHEMA MOCK" in payload
    # Verify zero real production customer names leak into mock payload
    assert "Acme Corp" not in payload
    assert "Global Logistics" not in payload


def test_multi_sheet_airlock_execution():
    """Verify that multi-sheet variables (sheets dict, df_transactions, df_dim_customer)
    are accessible in SafeAirlockExecutor and execute safely.
    """
    df_tx = pd.DataFrame({
        "tx_id": ["T1", "T2"],
        "cust_id": ["C1", "C2"],
        "amount": [100, 200]
    })
    df_dim = pd.DataFrame({
        "cust_id": ["C1", "C2"],
        "tier": ["Gold", "Silver"]
    })

    sheets = {
        "Transactions": df_tx,
        "Dim_Customer": df_dim,
    }

    # Python code combining both sheets
    code = """
import pandas as pd

# Merge Dim_Customer into Transactions
tx = sheets['Transactions']
dim = sheets['Dim_Customer']
df = tx.merge(dim, on='cust_id', how='left')
df['total_with_bonus'] = df['amount'] * 1.1
"""

    global_scope = {
        "pd": pd,
        "sheets": sheets,
        "df": df_tx.copy(),
        "df_transactions": df_tx,
        "df_dim_customer": df_dim,
    }

    out_scope = execute_code_safely(code, global_scope=global_scope)
    result_df, origin = resolve_transformed_dataframe(out_scope, original_df=df_tx)
    assert "tier" in result_df.columns
    assert "total_with_bonus" in result_df.columns
    assert len(result_df) == 2
    assert result_df["tier"].iloc[0] == "Gold"


def test_multi_sheet_power_query_generation():
    query = generate_powerquery_m_code(
        file_path="/path/to/enterprise_report.xlsx",
        sheet_name="Transactions"
    )

    assert "let" in query
    assert "Excel.Workbook" in query
    assert "Transactions" in query
    assert "in" in query

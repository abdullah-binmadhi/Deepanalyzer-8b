"""Unit tests for DeepAnalyze 7-Brain Cognitive Resonance Engine."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from deepanalyze.brain import (
    Brain1TopologicalCartographer,
    Brain2MorphologicalTypologist,
    Brain3ForensicPathologist,
    Brain4RelationalCryptographer,
    Brain5MathematicalPhysicist,
    Brain6AutonomousFeatureAlchemist,
    Brain7ExecutiveOrchestrator,
    CognitiveBlackboard,
    DynamicResonanceEngine,
    calculate_entropy,
    normalize_bilingual_cell,
)


def test_calculate_entropy():
    # Constant column: entropy must be 0.0
    s_const = pd.Series(["A", "A", "A", "A"])
    assert calculate_entropy(s_const) == 0.0

    # High variance / distinct values: normalized entropy close to 1.0
    s_distinct = pd.Series([f"val_{i}" for i in range(100)])
    ent_distinct = calculate_entropy(s_distinct)
    assert 0.95 <= ent_distinct <= 1.0

    # Empty / NaN column: entropy must be 0.0
    s_empty = pd.Series([None, np.nan, None])
    assert calculate_entropy(s_empty) == 0.0


def test_brain1_topological_cartographer():
    # Construct a matrix with 2 metadata rows, 1 header row, 10 data rows, and 1 footer row
    matrix = [
        ["Report: Monthly Revenue", None, None, None],
        ["Generated on: 2025-01-01", None, None, None],
        ["Invoice_ID", "Customer_Name", "Quantity", "Amount"],
    ]
    for i in range(1, 11):
        matrix.append([f"INV-{1000 + i}", f"Customer {i}", i * 2, i * 150.0])
    matrix.append(["Grand Total Summary", None, None, 15000.0])

    df = pd.DataFrame(matrix)
    bb = CognitiveBlackboard(shape=df.shape, columns=["c0", "c1", "c2", "c3"])

    brain1 = Brain1TopologicalCartographer()
    brain1.execute(df, bb)

    # Header boundary should skip the first 2 sparse rows and land on index 2
    assert bb.header_row_index == 2
    assert 0 in bb.metadata_rows
    assert 1 in bb.metadata_rows

    # Footer boundary should capture the Grand Total row at the end
    assert bb.footer_start_index == len(matrix) - 1


def test_brain2_morphological_typologist():
    df = pd.DataFrame({
        "uuid_col": ["a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d", "b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e"] * 5,
        "date_col": ["2025-01-15", "2025-02-20", "2025-03-10", "2025-04-05", "2025-05-12"] * 2,
        "composite_col": ["120/80", "130/85", "115/75", "140/90", "125/82"] * 2,
        "currency_col": ["$1,200.00", "$450.50", "$3,100.00", "$99.99", "$520.00"] * 2,
        "narrative_col": [
            "Patient reported severe recurring headaches and recommended comprehensive neurological evaluation.",
            "Client requested urgent expedited shipping for high-priority enterprise deployment order."
        ] * 5,
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    brain2 = Brain2MorphologicalTypologist()
    brain2.execute(df, bb)

    assert bb.column_profiles["uuid_col"]["role"] == "PRIMARY_IDENTIFIER"
    assert bb.column_profiles["date_col"]["role"] == "TEMPORAL"
    assert bb.column_profiles["composite_col"]["role"] == "COMPOSITE_KEY"
    assert bb.column_profiles["currency_col"]["role"] == "CONTINUOUS_NUMERIC"
    assert bb.column_profiles["currency_col"]["is_currency"] is True
    assert bb.column_profiles["narrative_col"]["role"] == "FREE_TEXT_NARRATIVE"


def test_brain3_forensic_pathologist():
    # Column with contaminated types (90% numbers, 10% strings)
    # Column with heavy skewness
    # Composite string column
    clean_nums = list(range(1, 10)) + ["N/A"]
    skewed_nums = [1.0, 1.2, 1.1, 1.3, 1.2, 1.4, 1.1, 1.5, 1.2, 1000.0]

    df = pd.DataFrame({
        "contaminated": clean_nums,
        "skewed": skewed_nums,
        "composite": ["12GB/256GB"] * 10,
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    Brain2MorphologicalTypologist().execute(df, bb)
    Brain3ForensicPathologist().execute(df, bb)

    # Check type contamination found
    assert any(c["col"] == "contaminated" for c in bb.type_contaminations)

    # Check composite found
    assert any(c["col"] == "composite" for c in bb.type_contaminations)

    # Check skewness detected
    assert "skewed" in bb.skewed_columns


def test_brain4_relational_cryptographer():
    # State -> Country hierarchy (USA has CA and NY; UK has ENG and SCT)
    # 100% unique primary key
    df = pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(10)],
        "city": ["San Francisco", "Los Angeles", "San Diego", "New York", "Buffalo", "London", "Manchester", "Edinburgh", "Glasgow", "Aberdeen"],
        "state_or_region": ["California", "California", "California", "New York", "New York", "England", "England", "Scotland", "Scotland", "Scotland"],
        "country": ["USA", "USA", "USA", "USA", "USA", "UK", "UK", "UK", "UK", "UK"],
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    Brain2MorphologicalTypologist().execute(df, bb)
    Brain4RelationalCryptographer().execute(df, bb)

    # Primary key discovered
    assert "order_id" in bb.candidate_primary_keys

    # Functional dependency discovered (state_or_region -> country)
    deps = bb.hierarchical_dependencies
    assert any(dep[0] == "country" and dep[1] == "state_or_region" for dep in deps) or any(dep[0] == "state_or_region" and dep[1] == "city" for dep in deps)


def test_brain5_mathematical_physicist():
    # Multiplicative Invariant: qty * unit_price = item_amount
    qty = [2, 3, 5, 10, 4, 6, 8, 1, 7, 9, 12, 15]
    price = [10.0, 15.0, 20.0, 5.0, 25.0, 12.5, 8.0, 100.0, 14.0, 11.0, 30.0, 2.0]
    amount = [q * p for q, p in zip(qty, price)]

    df = pd.DataFrame({
        "quantity": qty,
        "unit_price": price,
        "item_amount": amount,
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    Brain2MorphologicalTypologist().execute(df, bb)
    Brain5MathematicalPhysicist().execute(df, bb)

    assert len(bb.algebraic_laws) >= 1
    assert "Multiplicative Law" in bb.algebraic_laws[0]
    assert "`quantity`" in bb.algebraic_laws[0] or "`unit_price`" in bb.algebraic_laws[0]


def test_brain6_autonomous_feature_alchemist():
    bb = CognitiveBlackboard(shape=(100, 4), columns=["date_col", "text_col", "skewed_col", "spec_col"])
    bb.column_profiles = {
        "date_col": {"role": "TEMPORAL", "is_composite": False},
        "text_col": {"role": "FREE_TEXT_NARRATIVE", "is_composite": False},
        "skewed_col": {"role": "CONTINUOUS_NUMERIC", "is_composite": False},
        "spec_col": {"role": "COMPOSITE_KEY", "is_composite": True},
    }
    bb.skewed_columns = ["skewed_col"]
    bb.algebraic_laws = ["Multiplicative Law: `quantity` * `unit_price` ≈ `item_amount`"]

    brain6 = Brain6AutonomousFeatureAlchemist()
    brain6.execute(bb)

    features = bb.engineered_features
    assert any("Temporal Deconstruction" in f["feature"] for f in features)
    assert any("Narrative Density" in f["feature"] for f in features)
    assert any("Composite Decomposition" in f["feature"] for f in features)
    assert any("Log1p Transformation" in f["feature"] for f in features)
    assert any("Mathematical Invariant Integrity Flag" in f["feature"] for f in features)


def test_brain7_executive_orchestrator():
    bb = CognitiveBlackboard(
        filename="sales_q3.csv",
        shape=(1000, 5),
        columns=["inv_id", "date", "qty", "price", "amount"]
    )
    bb.header_row_index = 0
    bb.column_profiles = {
        "inv_id": {"role": "PRIMARY_IDENTIFIER", "col_index": 0},
        "date": {"role": "TEMPORAL", "col_index": 1},
        "qty": {"role": "DISCRETE_NUMERIC", "col_index": 2},
        "price": {"role": "CONTINUOUS_NUMERIC", "col_index": 3},
        "amount": {"role": "CONTINUOUS_NUMERIC", "col_index": 4},
    }
    bb.candidate_primary_keys = ["inv_id"]
    bb.algebraic_laws = ["Multiplicative Law: `qty` * `price` ≈ `amount` (Validated across 500 records)"]
    bb.engineered_features = [{"feature": "Log1p", "logic": "Transform skewed fields"}]

    brain7 = Brain7ExecutiveOrchestrator()
    prompt = brain7.execute(bb)

    assert "### SYSTEM ROLE & OBJECTIVE" in prompt
    assert "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)" in prompt
    assert "Topological Cartography: Resolved matrix to 1,000 rows x 5 columns." in prompt
    assert "### 1. DATASET TOPOLOGY & BOUNDARIES" in prompt
    assert "### 2. PATHOLOGY REPAIR PROTOCOLS" in prompt
    assert "### 3. MATHEMATICAL INVARIANTS" in prompt
    assert "Multiplicative Law: `qty` * `price` ≈ `amount`" in prompt
    assert "### 4. ALGORITHMIC FEATURE ENGINEERING" in prompt
    assert "### 5. AST SECURITY FIREWALL CONSTRAINTS" in prompt


def test_dynamic_resonance_engine_e2e_polars():
    # End-to-end run on a Polars DataFrame
    qty = [2, 4, 6, 8, 10, 12, 14, 16]
    price = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    total = [q * p for q, p in zip(qty, price)]

    df = pl.DataFrame({
        "order_num": [f"ORD-{i:04d}" for i in range(1, 9)],
        "tx_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08"],
        "quantity": qty,
        "unit_price": price,
        "total_amount": total,
        "specs": ["8GB/128GB", "12GB/256GB", "8GB/128GB", "16GB/512GB", "8GB/256GB", "12GB/128GB", "16GB/256GB", "12GB/512GB"],
    })

    engine = DynamicResonanceEngine(df, filename="orders_test.csv")
    prompt = engine.think_and_synthesize()

    assert "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)" in prompt
    assert len(engine.bb.internal_monologue) >= 2
    assert "order_num" in engine.bb.candidate_primary_keys
    assert len(engine.bb.algebraic_laws) >= 1
    assert "Multiplicative Law" in engine.bb.algebraic_laws[0]


def test_dynamic_resonance_engine_e2e_pandas():
    # End-to-end run on a Pandas DataFrame
    df = pd.DataFrame({
        "user_id": [f"U{i:03d}" for i in range(10)],
        "created_at": ["2025-03-01", "2025-03-02"] * 5,
        "score": [10.5, 20.1, 15.3, 19.8, 12.4, 18.9, 14.2, 17.6, 13.9, 21.0],
    })

    engine = DynamicResonanceEngine(df, filename="users_test.parquet")
    prompt = engine.think_and_synthesize()

    assert "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)" in prompt
    assert engine.bb.shape == (10, 3)
    assert "user_id" in engine.bb.candidate_primary_keys


def test_bilingual_sanitization_eastern_arabic_digits():
    # 1. Direct normalization of Eastern Arabic digits and BiDi marks
    raw_arabic_indic = "\u200f١٢٣٤٫٥٠\u200e"
    normalized = normalize_bilingual_cell(raw_arabic_indic)
    assert normalized == "1234.50"

    raw_thousands = "١٬٢٥٠٫٧٥"
    assert normalize_bilingual_cell(raw_thousands) == "1250.75"

    # 2. DynamicResonanceEngine ingestion converts Eastern Arabic numerals
    df = pd.DataFrame({
        "invoice_no": ["\u200eINV-١٠١", "\u200eINV-١٠٢", "\u200eINV-١٠٣"],
        "total": ["١٥٠٫٠٠", "٣٠٠٫٥٠", "٤٥٠٫٧٥"],
    })
    engine = DynamicResonanceEngine(df, filename="arabic_sales.csv")
    assert engine.df_raw["total"].tolist() == ["150.00", "300.50", "450.75"]
    assert engine.df_raw["invoice_no"].tolist() == ["INV-101", "INV-102", "INV-103"]


def test_brain1_arabic_structural_anchors():
    matrix = [
        ["تقرير المبيعات السنوي لشركة الأفق", None, None, None],
        ["معايير الفرز: الرياض - جدة", None, None, None],
        ["رقم_الفاتورة", "اسم_العميل", "الكمية", "المبلغ_الإجمالي"],
    ]
    for i in range(1, 11):
        matrix.append([f"فاتورة-{100 + i}", f"العميل_{i}", i * 5, i * 250.0])
    matrix.append(["الإجمالي النهائي للفترة", None, None, 13750.0])

    df = pd.DataFrame(matrix)
    bb = CognitiveBlackboard(shape=df.shape, columns=["c0", "c1", "c2", "c3"])

    brain1 = Brain1TopologicalCartographer()
    brain1.execute(df, bb)

    assert bb.header_row_index == 2
    assert 0 in bb.metadata_rows
    assert 1 in bb.metadata_rows
    assert bb.footer_start_index == len(matrix) - 1


def test_brain2_arabic_unicode_composite_and_regional_ids():
    df = pd.DataFrame({
        "zatca_vat_id": ["300012345678903", "310098765432103", "300055566677703", "300011122233303", "300099988877703"] * 2,
        "saudi_cr": ["1010123456", "2050987654", "4030112233", "1010998877", "2050334455"] * 2,
        "arabic_composite": ["فاتورة-101", "ص-204", "عقد-509", "فاتورة-302", "طلب-801"] * 2,
        "arabic_currency": ["1,500.00 ر.س", "2,350.50 ريال", "750.00 د.إ", "4,200 SAR", "890.00 AED"] * 2,
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    brain2 = Brain2MorphologicalTypologist()
    brain2.execute(df, bb)

    assert bb.column_profiles["zatca_vat_id"]["role"] == "PRIMARY_IDENTIFIER"
    assert bb.column_profiles["zatca_vat_id"].get("regional_id") == "ZATCA_VAT_ID"
    assert bb.column_profiles["saudi_cr"]["role"] == "PRIMARY_IDENTIFIER"
    assert bb.column_profiles["saudi_cr"].get("regional_id") == "SAUDI_CR"
    assert bb.column_profiles["arabic_composite"]["role"] == "COMPOSITE_KEY"
    assert bb.column_profiles["arabic_currency"]["role"] == "CONTINUOUS_NUMERIC"
    assert bb.column_profiles["arabic_currency"]["is_currency"] is True


def test_brain2_hijri_date_detection():
    df = pd.DataFrame({
        "hijri_std": ["1446-08-15", "1446-09-01", "1445-12-29", "1446-01-10", "1446-02-20"] * 2,
        "hijri_text": ["15 رمضان 1445 هـ", "01 شوال 1445 هـ", "10 محرم 1446 هـ", "25 رجب 1445 هـ", "05 شعبان 1446 هـ"] * 2,
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    brain2 = Brain2MorphologicalTypologist()
    brain2.execute(df, bb)

    assert bb.column_profiles["hijri_std"]["role"] == "TEMPORAL_HIJRI"
    assert bb.column_profiles["hijri_text"]["role"] == "TEMPORAL_HIJRI"


def test_brain5_zatca_vat_invariant_discovery():
    net = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    tax = [n * 0.15 for n in net]
    gross = [n * 1.15 for n in net]

    df = pd.DataFrame({
        "net_amount": net,
        "vat_amount": tax,
        "gross_amount": gross,
    })
    bb = CognitiveBlackboard(shape=df.shape, columns=list(df.columns))

    brain2 = Brain2MorphologicalTypologist()
    brain2.execute(df, bb)

    brain5 = Brain5MathematicalPhysicist()
    brain5.execute(df, bb)

    assert len(bb.algebraic_laws) >= 1
    vat_laws = [law for law in bb.algebraic_laws if "VAT" in law or "ZATCA" in law]
    assert len(vat_laws) >= 1
    assert "15%" in vat_laws[0]


def test_dynamic_resonance_engine_e2e_arabic_erp():
    matrix = [
        ["تقرير مبيعات المؤسسة للربع السنوي", None, None, None, None],
        ["معايير الفرز: فرع الرياض الرئيسي", None, None, None, None],
        ["رقم_الفاتورة", "التاريخ_الهجري", "الرقم_الضريبي", "المبلغ_الأساسي", "المبلغ_الإجمالي"],
    ]
    for i in range(1, 11):
        net = i * 1000.0
        gross = net * 1.15
        matrix.append([
            f"\u200fفاتورة-١٠{i}\u200e",
            f"1446-08-{10 + i:02d}",
            "300012345678903",
            f"{net:.2f}",
            f"{gross:.2f}",
        ])
    matrix.append(["الإجمالي النهائي", None, None, None, 63250.0])

    df = pd.DataFrame(matrix)
    engine = DynamicResonanceEngine(df, filename="saudi_erp_invoices.xlsx")
    prompt = engine.think_and_synthesize()

    assert "### SYSTEM ROLE & OBJECTIVE" in prompt
    assert "### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)" in prompt
    assert engine.bb.header_row_index == 2
    assert 0 in engine.bb.metadata_rows
    assert engine.bb.footer_start_index == len(matrix) - 1
    assert any("ZATCA" in law or "VAT" in law for law in engine.bb.algebraic_laws)



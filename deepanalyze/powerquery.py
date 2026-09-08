# deepanalyze/powerquery.py
"""Excel Power Query M-Code & Step-by-Step UI Formula Guide Generator.

Provides non-technical users, financial controllers, and accountants with
ready-to-paste Power Query M-code and an explicit, click-by-click UI walkthrough
to clean unflattened ERP spreadsheets directly in Microsoft Excel dynamically.
"""

import os
from typing import Optional, Union
import pandas as pd
import polars as pl


def generate_powerquery_m_code(
    file_path: str,
    sheet_name: str = "Report",
    df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
) -> str:
    """Generates ready-to-paste Power Query M-code for the Excel Advanced Editor."""
    from .erp_cleaner import sniff_erp_layout, ERPLayoutSchema

    normalized_path = file_path.replace("\\", "/")

    schema = None
    if df is not None:
        schema = sniff_erp_layout(df)
    elif os.path.isfile(file_path):
        try:
            schema = sniff_erp_layout(file_path)
        except Exception:
            schema = ERPLayoutSchema()
    else:
        schema = ERPLayoutSchema()

    col_doc = f"Column{schema.doc_col + 1}"
    col_date = f"Column{schema.date_col + 1}" if schema.date_col is not None else "Column3"
    col_code = f"Column{schema.entity_code_col + 1}" if schema.entity_code_col is not None else "Column5"
    col_name = f"Column{schema.entity_name_col + 1}" if schema.entity_name_col is not None else "Column7"
    col_total = f"Column{schema.total_col + 1}" if schema.total_col is not None else "Column16"
    col_desc = f"Column{schema.desc_col + 1}"
    col_qty = f"Column{schema.qty_col + 1}" if schema.qty_col is not None else "Column11"
    col_uom = f"Column{schema.uom_col + 1}" if schema.uom_col is not None else "Column12"
    col_price = f"Column{schema.price_col + 1}" if schema.price_col is not None else "Column13"
    col_amt = f"Column{schema.amount_col + 1}" if schema.amount_col is not None else "Column14"

    prefix_checks = " or ".join([
        f'Text.StartsWith([{col_doc}], "{p}")' for p in schema.doc_prefixes
    ])

    m_code = f"""let
    // 1. Ingest Excel Workbook
    Source = Excel.Workbook(File.Contents("{normalized_path}"), null, true),
    Navigation = Source{{[Item="{sheet_name}", Kind="Sheet"]}}[Data],

    // 2. Filter out summary grand totals and page headers dynamically without hardcoded row slicing
    #"Changed Type Col1" = Table.TransformColumnTypes(Navigation, {{{{"{col_doc}", type text}}}}),
    #"Filtered Grand Total" = Table.SelectRows(#"Changed Type Col1", each ([{col_doc}] = null or (not Text.Contains([{col_doc}], "Grand Total") and not Text.Contains([{col_doc}], "Page ")))),

    // 3. Extract document-level headers dynamically using null-safe conditional columns
    #"Add {schema.doc_name}" = Table.AddColumn(#"Filtered Grand Total", "{schema.doc_name}", each if [{col_doc}] <> null and ({prefix_checks}) then [{col_doc}] else null),
    #"Add {schema.date_name}" = Table.AddColumn(#"Add {schema.doc_name}", "{schema.date_name}", each if [{schema.doc_name}] <> null then [{col_date}] else null),
    #"Add {schema.entity_code_name}" = Table.AddColumn(#"Add {schema.date_name}", "{schema.entity_code_name}", each if [{schema.doc_name}] <> null then [{col_code}] else null),
    #"Add {schema.entity_name_name}" = Table.AddColumn(#"Add {schema.entity_code_name}", "{schema.entity_name_name}", each if [{schema.doc_name}] <> null then [{col_name}] else null),
    #"Add {schema.total_name}" = Table.AddColumn(#"Add {schema.entity_name_name}", "{schema.total_name}", each if [{schema.doc_name}] <> null then [{col_total}] else null),

    // 4. Forward-fill document headers down to all transaction line items
    #"Filled Down Headers" = Table.FillDown(#"Add {schema.total_name}", {{"{schema.doc_name}", "{schema.date_name}", "{schema.entity_code_name}", "{schema.entity_name_name}", "{schema.total_name}"}}),

    // 5. Extract line items and filter out non-item rows
    #"Type Sequence" = Table.TransformColumnTypes(#"Filled Down Headers", {{{{"{col_doc}", Int64.Type}}}}),
    #"Handled Errors" = Table.ReplaceErrorValues(#"Type Sequence", {{{{"{col_doc}", null}}}}),
    #"Filtered Line Items" = Table.SelectRows(#"Handled Errors", each ([{col_doc}] <> null)),

    // 6. Select and rename business columns
    #"Selected Columns" = Table.SelectColumns(#"Filtered Line Items", {{
        "{col_doc}", "{col_code}", "{col_qty}", "{col_uom}", "{col_price}", "{col_amt}",
        "{schema.doc_name}", "{schema.date_name}", "{schema.entity_code_name}", "{schema.entity_name_name}", "{schema.total_name}", "{col_desc}"
    }}),
    #"Renamed Columns" = Table.RenameColumns(#"Selected Columns", {{
        {{"{col_doc}", "{schema.seq_name}"}},
        {{"{col_code}", "{schema.item_code_name}"}},
        {{"{col_qty}", "{schema.qty_name}"}},
        {{"{col_uom}", "{schema.uom_name}"}},
        {{"{col_price}", "{schema.price_name}"}},
        {{"{col_amt}", "{schema.amount_name}"}},
        {{"{col_desc}", "{schema.desc_name}"}}
    }}),

    // 7. Enforce strict types
    #"Final Types" = Table.TransformColumnTypes(#"Renamed Columns", {{
        {{"{schema.seq_name}", Int64.Type}},
        {{"{schema.item_code_name}", type text}},
        {{"{schema.qty_name}", type number}},
        {{"{schema.uom_name}", type text}},
        {{"{schema.price_name}", type number}},
        {{"{schema.amount_name}", type number}},
        {{"{schema.doc_name}", type text}},
        {{"{schema.date_name}", type date}},
        {{"{schema.entity_code_name}", type text}},
        {{"{schema.entity_name_name}", type text}},
        {{"{schema.total_name}", type number}},
        {{"{schema.desc_name}", type text}}
    }}),

    // 8. Sort descending by Document Total
    #"Sorted Rows" = Table.Sort(#"Final Types", {{{{ "{schema.total_name}", Order.Descending }}}})
in
    #"Sorted Rows"
"""
    return m_code.strip()


def generate_powerquery_step_by_step_guide(
    dataset_name: str,
    file_path: Optional[str] = None,
    df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
) -> str:
    """Generates a complete markdown guide with click-by-click UI steps."""
    from .erp_cleaner import sniff_erp_layout, ERPLayoutSchema

    display_path = file_path or f"/path/to/{dataset_name}"
    is_csv = display_path.lower().endswith(".csv")
    file_type_label = "From Text/CSV" if is_csv else "From Excel Workbook"

    schema = None
    if df is not None:
        schema = sniff_erp_layout(df)
    elif file_path and os.path.isfile(file_path):
        try:
            schema = sniff_erp_layout(file_path)
        except Exception:
            schema = ERPLayoutSchema()
    else:
        schema = ERPLayoutSchema()

    col_doc = f"Column{schema.doc_col + 1}"
    col_date = f"Column{schema.date_col + 1}" if schema.date_col is not None else "Column3"
    col_code = f"Column{schema.entity_code_col + 1}" if schema.entity_code_col is not None else "Column5"
    col_name = f"Column{schema.entity_name_col + 1}" if schema.entity_name_col is not None else "Column7"
    col_total = f"Column{schema.total_col + 1}" if schema.total_col is not None else "Column16"
    col_desc = f"Column{schema.desc_col + 1}"

    guide = f"""# Excel Power Query Step-by-Step Data Cleaning Guide

This guide explains how to flatten and clean **{dataset_name}** directly inside Microsoft Excel using **Power Query** (Get & Transform Data). No programming required!

---

## Method 1: The 60-Second Copy-Paste (Recommended)

1. Open Microsoft Excel.
2. Go to the **Data** tab on the top ribbon.
3. Click **Get Data** (or **New Query**) -> **From File** -> **{file_type_label}**.
4. Select your file: `{dataset_name}`.
5. In the Navigator preview window, select your table/sheet and click **Transform Data** (do *not* click Load).
6. In the Power Query Editor window, go to the **Home** tab and click **Advanced Editor**.
7. Open the accompanying `powerquery_script.m` file, copy all text (`Ctrl+A` / `Cmd+A` then `Ctrl+C` / `Cmd+C`), and paste it into the Advanced Editor, replacing any existing code.
8. Click **Done**.
9. In the Home tab, click **Close & Load**.
10. **Done!** You now have a clean, relational table. Every time you get updated data, simply click **Data -> Refresh All**!

---

## Method 2: Click-by-Click Manual UI Walkthrough

If you want to understand or build the steps manually using Excel buttons:

### Step 1: Remove Non-Data Headers / Filter Noise
* If there are title or filter banner rows before table columns, click **Home** tab -> **Remove Rows** -> **Remove Top Rows** (or use text filtering).
* Avoid excessive row skips so early transactions are preserved.

### Step 2: Filter Out Summary Footers
* Click the dropdown arrow on **{col_doc}**.
* Go to **Text Filters** -> **Does Not Contain...**
* Type `Grand Total` and click OK.

### Step 3: Extract Document Header Data into New Columns
* Click the **Add Column** tab -> **Conditional Column**.
* Create the following columns:
  1. **{schema.doc_name}**:
     * *If* `{col_doc}` *begins with* document prefix (e.g. `{"`, `".join(schema.doc_prefixes[:3])}`)
     * *Then* select column: `{col_doc}`
     * *Else* leave empty (null).
  2. **{schema.date_name}**:
     * *If* `{col_doc}` *begins with* document prefix
     * *Then* select column: `{col_date}`
     * *Else* leave empty (null).
  3. **{schema.entity_code_name}**:
     * *If* `{col_doc}` *begins with* document prefix
     * *Then* select column: `{col_code}`
     * *Else* leave empty (null).
  4. **{schema.entity_name_name}**:
     * *If* `{col_doc}` *begins with* document prefix
     * *Then* select column: `{col_name}`
     * *Else* leave empty (null).
  5. **{schema.total_name}**:
     * *If* `{col_doc}` *begins with* document prefix
     * *Then* select column: `{col_total}`
     * *Else* leave empty (null).

### Step 4: Fill Down Header Data
* Hold `Ctrl` (or `Cmd` on Mac) and select the master columns: `{schema.doc_name}`, `{schema.date_name}`, `{schema.entity_code_name}`, `{schema.entity_name_name}`, `{schema.total_name}`.
* Go to the **Transform** tab -> click **Fill** -> **Down**.
* Notice how the document number and customer/account attributes now appear on every single line item!

### Step 5: Filter for Line Items (Numeric Sequence)
* Select **{col_doc}** -> click **Transform** tab -> **Data Type** -> choose **Whole Number**.
* Any non-numeric rows (like headers or empty cells) will turn into errors.
* Right-click the **{col_doc}** header -> select **Replace Errors** -> type `null`.
* Click the filter dropdown on **{col_doc}** -> uncheck `(null)` so only numbers remain.

### Step 6: Choose and Rename Columns
* Click **Home** tab -> **Choose Columns**.
* Keep business columns and rename them to standard canonical names:
  * `{col_doc}` -> **{schema.seq_name}**
  * `{col_desc}` -> **{schema.desc_name}**
  * `{col_total}` -> **{schema.total_name}**

### Step 7: Sort and Load
* Click the dropdown arrow on **{schema.total_name}** -> select **Sort Descending**.
* Click **Home** tab -> **Close & Load**.
"""
    return guide.strip()

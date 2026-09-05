# deepanalyze/powerquery.py
"""Excel Power Query M-Code & Step-by-Step UI Formula Guide Generator.

Provides non-technical users, financial controllers, and accountants with
ready-to-paste Power Query M-code and an explicit, click-by-click UI walkthrough
to clean unflattened ERP spreadsheets directly in Microsoft Excel.
"""

from typing import Optional


def generate_powerquery_m_code(file_path: str, sheet_name: str = "Report") -> str:
    """Generates ready-to-paste Power Query M-code for the Excel Advanced Editor."""
    normalized_path = file_path.replace("\\", "/")

    m_code = f"""let
    // 1. Ingest Excel Workbook
    Source = Excel.Workbook(File.Contents("{normalized_path}"), null, true),
    Navigation = Source{{[Item="{sheet_name}", Kind="Sheet"]}}[Data],

    // 2. Remove top report metadata rows (headers/filters)
    #"Removed Top Rows" = Table.Skip(Navigation, 18),

    // 3. Ensure column 1 is treated as text for pattern matching (nested list syntax)
    #"Changed Type Col1" = Table.TransformColumnTypes(#"Removed Top Rows", {{{{"Column1", type text}}}}),

    // 4. Exclude summary grand totals (null-safe guard)
    #"Filtered Grand Total" = Table.SelectRows(#"Changed Type Col1", each ([Column1] = null or not Text.Contains([Column1], "Grand Total"))),

    // 5. Extract document-level headers using null-safe conditional columns
    #"Add doc_no" = Table.AddColumn(#"Filtered Grand Total", "doc_no", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column1] else null),
    #"Add doc_date" = Table.AddColumn(#"Add doc_no", "doc_date", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column3] else null),
    #"Add customer_code" = Table.AddColumn(#"Add doc_date", "customer_code", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column5] else null),
    #"Add customer_name" = Table.AddColumn(#"Add customer_code", "customer_name", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column7] else null),
    #"Add invoice_total" = Table.AddColumn(#"Add customer_name", "invoice_total", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column16] else null),

    // 6. Forward-fill document headers down to all transaction line items
    #"Filled Down Headers" = Table.FillDown(#"Add invoice_total", {{"doc_no", "doc_date", "customer_code", "customer_name", "invoice_total"}}),

    // 7. Extract numeric sequence items and filter out non-item rows (nested list syntax)
    #"Type Sequence" = Table.TransformColumnTypes(#"Filled Down Headers", {{{{"Column1", Int64.Type}}}}),
    #"Handled Errors" = Table.ReplaceErrorValues(#"Type Sequence", {{"Column1", null}}),
    #"Filtered Line Items" = Table.SelectRows(#"Handled Errors", each ([Column1] <> null)),

    // 8. Select and rename final 12 business columns
    #"Selected Columns" = Table.SelectColumns(#"Filtered Line Items", {{
        "Column1", "Column2", "Column11", "Column12", "Column13", "Column14",
        "doc_no", "doc_date", "customer_code", "customer_name", "invoice_total", "Column4"
    }}),
    #"Renamed Columns" = Table.RenameColumns(#"Selected Columns", {{
        {{"Column1", "Sequence"}},
        {{"Column2", "GL-Code"}},
        {{"Column11", "Quantity"}},
        {{"Column12", "UOM"}},
        {{"Column13", "Unit Price"}},
        {{"Column14", "Item Amount"}},
        {{"Column4", "Full_Description"}}
    }}),

    // 9. Enforce strict types
    #"Final Types" = Table.TransformColumnTypes(#"Renamed Columns", {{
        {{"Sequence", Int64.Type}},
        {{"GL-Code", type text}},
        {{"Quantity", type number}},
        {{"UOM", type text}},
        {{"Unit Price", type number}},
        {{"Item Amount", type number}},
        {{"doc_no", type text}},
        {{"doc_date", type date}},
        {{"customer_code", type text}},
        {{"customer_name", type text}},
        {{"invoice_total", type number}},
        {{"Full_Description", type text}}
    }}),

    // 10. Sort descending by Invoice Total
    #"Sorted Rows" = Table.Sort(#"Final Types", {{{{ "invoice_total", Order.Descending }}}})
in
    #"Sorted Rows"
"""
    return m_code.strip()


def generate_powerquery_step_by_step_guide(
    dataset_name: str,
    file_path: Optional[str] = None
) -> str:
    """Generates a complete markdown guide with click-by-click UI steps.

    No redundant script is embedded since the M-code is provided directly in powerquery_script.m.
    """
    display_path = file_path or f"/path/to/{dataset_name}"
    is_csv = display_path.lower().endswith(".csv")
    file_type_label = "From Text/CSV" if is_csv else "From Excel Workbook"

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

### Step 1: Remove Top Report Headers
* Click **Home** tab -> **Remove Rows** -> **Remove Top Rows**.
* Enter `18` and click OK.

### Step 2: Filter Out Grand Total
* Click the dropdown arrow on **Column1**.
* Go to **Text Filters** -> **Does Not Contain...**
* Type `Grand Total` and click OK.

### Step 3: Extract Invoice Header Data into New Columns
* Click the **Add Column** tab -> **Conditional Column**.
* Create the following 5 columns one by one:
  1. **doc_no**:
     * *If* `Column1` *begins with* `IV-`
     * *Then* select column: `Column1`
     * *Else* leave empty (null).
  2. **doc_date**:
     * *If* `Column1` *begins with* `IV-`
     * *Then* select column: `Column3`
     * *Else* leave empty (null).
  3. **customer_code**:
     * *If* `Column1` *begins with* `IV-`
     * *Then* select column: `Column5`
     * *Else* leave empty (null).
  4. **customer_name**:
     * *If* `Column1` *begins with* `IV-`
     * *Then* select column: `Column7`
     * *Else* leave empty (null).
  5. **invoice_total**:
     * *If* `Column1` *begins with* `IV-`
     * *Then* select column: `Column16`
     * *Else* leave empty (null).

### Step 4: Fill Down Header Data
* Hold `Ctrl` (or `Cmd` on Mac) and select the 5 new columns: `doc_no`, `doc_date`, `customer_code`, `customer_name`, `invoice_total`.
* Go to the **Transform** tab -> click **Fill** -> **Down**.
* Notice how the invoice number and customer names now appear on every single line item!

### Step 5: Filter for Line Items (Numeric Sequence)
* Select **Column1** -> click **Transform** tab -> **Data Type** -> choose **Whole Number**.
* Any non-numeric rows (like Seq or empty cells) will turn into errors.
* Right-click the **Column1** header -> select **Replace Errors** -> type `null`.
* Click the filter dropdown on **Column1** -> uncheck `(null)` so only numbers (1000, 2000, etc.) remain.

### Step 6: Choose and Rename Columns
* Click **Home** tab -> **Choose Columns**.
* Keep only: `Column1`, `Column2`, `Column11`, `Column12`, `Column13`, `Column14`, `doc_no`, `doc_date`, `customer_code`, `customer_name`, `invoice_total`, `Column4`.
* Double-click each header to rename:
  * `Column1` -> **Sequence**
  * `Column2` -> **GL-Code**
  * `Column11` -> **Quantity**
  * `Column12` -> **UOM**
  * `Column13` -> **Unit Price**
  * `Column14` -> **Item Amount**
  * `Column4` -> **Full_Description**

### Step 7: Sort and Load
* Click the dropdown arrow on **invoice_total** -> select **Sort Descending**.
* Click **Home** tab -> **Close & Load**.
"""
    return guide.strip()

# Excel Power Query Step-by-Step Data Cleaning Guide

This guide explains how to flatten and clean **INV LISTING 31082025 copy.xlsx** directly inside Microsoft Excel using **Power Query** (Get & Transform Data). No programming required!

---

## Method 1: The 60-Second Copy-Paste (Recommended)

### Step-by-Step Instructions:
1. Open **Microsoft Excel**.
2. Go to the **Data** tab on the top ribbon.
3. Click **Get Data** (or **New Query**) -> **From File** -> **From Excel Workbook**.
4. Select your file (e.g. `INV LISTING 31082025 copy.xlsx` or `INV LISTING 31082025 copy 2.xlsx`).
5. In the Navigator preview window, select sheet **Report** and click **Transform Data** (do *not* click Load).
6. In the Power Query Editor window:
   * Go to the **Home** tab and click **Advanced Editor** (or on Mac, ensure the editor is showing the full query script).
   * **Crucial:** Do **NOT** paste into the single-line Formula Bar (`fx`) or the Step Script box.
7. In the Advanced Editor window, select everything (`Cmd + A` on Mac or `Ctrl + A` on Windows), delete it, and paste the following M-code:

```powerquery
let
    // 1. Ingest Excel Workbook
    Source = Excel.Workbook(File.Contents("/Users/abdullahbinmadhi/Desktop/deepanalyze/INV LISTING 31082025 copy.xlsx"), null, true),
    Navigation = Source{[Item="Report", Kind="Sheet"]}[Data],

    // 2. Remove top report metadata rows (headers/filters)
    #"Removed Top Rows" = Table.Skip(Navigation, 18),

    // 3. Ensure column 1 is treated as text for pattern matching (nested list of lists required)
    #"Changed Type Col1" = Table.TransformColumnTypes(#"Removed Top Rows", {{"Column1", type text}}),

    // 4. Exclude summary grand totals (null-safe guard against separator rows)
    #"Filtered Grand Total" = Table.SelectRows(#"Changed Type Col1", each ([Column1] = null or not Text.Contains([Column1], "Grand Total"))),

    // 5. Extract document-level headers using null-safe conditional columns
    #"Add doc_no" = Table.AddColumn(#"Filtered Grand Total", "doc_no", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column1] else null),
    #"Add doc_date" = Table.AddColumn(#"Add doc_no", "doc_date", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column3] else null),
    #"Add customer_code" = Table.AddColumn(#"Add doc_date", "customer_code", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column5] else null),
    #"Add customer_name" = Table.AddColumn(#"Add customer_code", "customer_name", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column7] else null),
    #"Add invoice_total" = Table.AddColumn(#"Add customer_name", "invoice_total", each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then [Column16] else null),

    // 6. Forward-fill document headers down to all transaction line items
    #"Filled Down Headers" = Table.FillDown(#"Add invoice_total", {"doc_no", "doc_date", "customer_code", "customer_name", "invoice_total"}),

    // 7. Extract numeric sequence items and filter out non-item rows (nested list of lists required)
    #"Type Sequence" = Table.TransformColumnTypes(#"Filled Down Headers", {{"Column1", Int64.Type}}),
    #"Handled Errors" = Table.ReplaceErrorValues(#"Type Sequence", {"Column1", null}),
    #"Filtered Line Items" = Table.SelectRows(#"Handled Errors", each ([Column1] <> null)),

    // 8. Select and rename final 12 business columns
    #"Selected Columns" = Table.SelectColumns(#"Filtered Line Items", {
        "Column1", "Column2", "Column11", "Column12", "Column13", "Column14",
        "doc_no", "doc_date", "customer_code", "customer_name", "invoice_total", "Column4"
    }),
    #"Renamed Columns" = Table.RenameColumns(#"Selected Columns", {
        {"Column1", "Sequence"},
        {"Column2", "GL-Code"},
        {"Column11", "Quantity"},
        {"Column12", "UOM"},
        {"Column13", "Unit Price"},
        {"Column14", "Item Amount"},
        {"Column4", "Full_Description"}
    }),

    // 9. Enforce strict types
    #"Final Types" = Table.TransformColumnTypes(#"Renamed Columns", {
        {"Sequence", Int64.Type},
        {"GL-Code", type text},
        {"Quantity", type number},
        {"UOM", type text},
        {"Unit Price", type number},
        {"Item Amount", type number},
        {"doc_no", type text},
        {"doc_date", type date},
        {"customer_code", type text},
        {"customer_name", type text},
        {"invoice_total", type number},
        {"Full_Description", type text}
    }),

    // 10. Sort descending by Invoice Total
    #"Sorted Rows" = Table.Sort(#"Final Types", {{"invoice_total", Order.Descending}})
in
    #"Sorted Rows"
```

8. Click **Done** (or **OK**).
9. In the Home tab, click **Close & Load**.
10. **Done!** You now have a clean, relational table with all 12 columns. Every time you get a new monthly export, just click **Data -> Refresh All**!

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

---

## Troubleshooting & Investigation of Common Errors

### 1. `Expected to find one of the following, but a keyword <'section'> was found instead`
* **Root Cause**: In the Power Query M language, `section` is a reserved system keyword (used in Power Query extensions / SDK files such as `section Section1;`). The M parser expects a value or literal (`hex literal, keyword false, #infinity, #nan, true, numeric literal, null, text`), but encounters the keyword `section`.
* **How this happens**:
  1. **Typing or pasting `section` without quotes**: If the word `section` is typed without double quotes (e.g. `Kind = section` instead of `Kind = "Sheet"`, or `[Column1 = section]`), the parser fails.
  2. **Pasting Section document headers**: Pasting code that begins with `section Section1; shared ...` into Excel's Advanced Editor (which only accepts expression documents `let ... in ...`).
  3. **Pasting into the Step Script or Formula Bar**: Pasting a full `let ... in` script into the single-line formula bar (`fx`) or when the editor scope is set to "Step Script" rather than opening the **Advanced Editor**.
* **Solution**: Always open the full **Advanced Editor** (Home -> Advanced Editor), select all existing text, delete it completely, and paste the clean `let ... in ...` script provided above.

### 2. `Expression.Error: We cannot convert the value null to type Text`
* **Root Cause**: Power Query's `Text.Contains` and `Text.StartsWith` functions require non-null text. In ERP exports, there are blank separator rows between invoices where `[Column1]` is `null`. Passing `null` to `Text.StartsWith` causes a fatal evaluation error.
* **Solution**: All predicates are now guarded with null checks:
  `each if [Column1] <> null and Text.StartsWith([Column1], "IV-") then ...`
  `each ([Column1] = null or not Text.Contains([Column1], "Grand Total"))`

### 3. `Expression.Error: We cannot convert the value "Column1" to type List`
* **Root Cause**: `Table.TransformColumnTypes` requires a **list of pairs** (`{{"Column1", type text}}`), not a single list (`{"Column1", type text}`).
* **Solution**: Fully fixed with correct nested braces `{{"Column1", type text}}` and `{{"Column1", Int64.Type}}`.

### 4. `DataSource.Error: Could not find file...`
* **Root Cause**: The file path in `File.Contents(...)` does not match the exact location or filename on your Mac (e.g. `INV LISTING 31082025 copy.xlsx` vs `copy 2.xlsx`), or macOS sandboxing blocked direct path access.
* **Solution**: Use **The 30-Second Plug-in** method above, where Excel's UI connects to the file and you simply paste the transformation steps below Excel's `Source` line.
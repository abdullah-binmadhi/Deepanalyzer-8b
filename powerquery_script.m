let
    // 1. Ingest Excel Workbook
    Source = Excel.Workbook(File.Contents("/Users/abdullahbinmadhi/Desktop/deepanalyze/INV LISTING 31082025 copy.xlsx"), null, true),
    Navigation = Source{[Item="Report", Kind="Sheet"]}[Data],

    // 2. Remove top report metadata rows (headers/filters)
    #"Removed Top Rows" = Table.Skip(Navigation, 18),

    // 3. Ensure column 1 is treated as text for pattern matching
    #"Changed Type Col1" = Table.TransformColumnTypes(#"Removed Top Rows", {"Column1", type text}),

    // 4. Exclude summary grand totals
    #"Filtered Grand Total" = Table.SelectRows(#"Changed Type Col1", each not Text.Contains([Column1], "Grand Total")),

    // 5. Extract document-level headers using conditional columns
    #"Add doc_no" = Table.AddColumn(#"Filtered Grand Total", "doc_no", each if Text.StartsWith([Column1], "IV-") then [Column1] else null),
    #"Add doc_date" = Table.AddColumn(#"Add doc_no", "doc_date", each if Text.StartsWith([Column1], "IV-") then [Column3] else null),
    #"Add customer_code" = Table.AddColumn(#"Add doc_date", "customer_code", each if Text.StartsWith([Column1], "IV-") then [Column5] else null),
    #"Add customer_name" = Table.AddColumn(#"Add customer_code", "customer_name", each if Text.StartsWith([Column1], "IV-") then [Column7] else null),
    #"Add invoice_total" = Table.AddColumn(#"Add customer_name", "invoice_total", each if Text.StartsWith([Column1], "IV-") then [Column16] else null),

    // 6. Forward-fill document headers down to all transaction line items
    #"Filled Down Headers" = Table.FillDown(#"Add invoice_total", {"doc_no", "doc_date", "customer_code", "customer_name", "invoice_total"}),

    // 7. Extract numeric sequence items and filter out non-item rows
    #"Type Sequence" = Table.TransformColumnTypes(#"Filled Down Headers", {"Column1", Int64.Type}),
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
    #"Sorted Rows" = Table.Sort(#"Final Types", { {"invoice_total", Order.Descending} })
in
    #"Sorted Rows"
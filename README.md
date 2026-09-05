# DeepAnalyze: Zero-Code Compliance Air-Gap Gateway

DeepAnalyze is an open-source, deterministic **Data Leak Prevention (DLP) engine and compliance air-gap gateway** for Jupyter, IPython, and the command line. It empowers financial controllers, data teams, and enterprise analysts to leverage frontier cloud AI models (**ChatGPT, Claude, Cursor**) on **messy, unflattened ERP spreadsheets, invoices, accounting ledgers, and clinical records** without exposing confidential business data, personal identities, or proprietary figures.

The engine enforces statutory anonymization in local volatile RAM, produces zero-risk synthetic payloads or full encrypted duplicate files for cloud LLMs, provides an interactive `.py` / `.ipynb` code execution airlock with automatic error retry, reconciles returned transformations locally with zero data leakage, and automatically generates ready-to-paste Excel Power Query companions for non-programmers.

---

## Table of Contents

1. [The Unflattened ERP Challenge & The DeepAnalyze Solution](#1-the-unflattened-erp-challenge--the-deepanalyze-solution)
2. [Key Capabilities & Architecture](#2-key-capabilities--architecture)
3. [Installation & Environment Setup](#3-installation--environment-setup)
4. [Ways to Run DeepAnalyze](#4-ways-to-run-deepanalyze)
   * [Method 1: Interactive Terminal CLI](#method-1-interactive-terminal-cli-zero-code)
   * [Method 2: Jupyter / IPython Magics](#method-2-jupyter--ipython-interactive-magics)
   * [Method 3: Local Offline Inference Server](#method-3-local-offline-inference-server)
   * [Method 4: Python Programmatic API](#method-4-python-programmatic-api)
5. [The Complete 13-Step Interactive Wizard Walkthrough](#5-the-complete-13-step-interactive-wizard-walkthrough)
6. [Excel Power Query Dual-Track (For Non-Programmers)](#6-excel-power-query-dual-track-for-non-programmers)
7. [Command Reference & Directives Cheat Sheet](#7-command-reference--directives-cheat-sheet)
8. [Local Inference Server & Speculative Acceleration](#8-local-inference-server--speculative-acceleration)
9. [Architecture, Security & Compliance FAQ](#9-architecture-security--compliance-faq)
10. [Module Architecture & Verification Test Suite](#10-module-architecture--verification-test-suite)

---

## 1. The Unflattened ERP Challenge & The DeepAnalyze Solution

### The Real-World Operational Problem
Enterprise accounting ledgers, ERP exports (SAP, Oracle, AS400, Microsoft Dynamics), and healthcare records are rarely clean relational tables. Instead, they are ragged, multi-row, unflattened reports featuring:
* Top report metadata headers (filters, print dates, company addresses across rows 1–18).
* Buried document numbers and customer names nested inside row cells (e.g. `Column1: IV-11325`, `Column5: 300-P0220`).
* Separator rows with missing/null values between transaction blocks.
* Multiple sub-headers (`Doc. No`, `Doc Date`, `Seq`, `GL Code`, `Project`, `:`).

**Standard DLP and PII scanners fail completely on these files.** They inspect column headers looking for labels like `customer_name` or `national_id`. In an unflattened ERP export, customer names appear in data rows beneath `Column1` or `Column7`, so standard scanners miss them entirely. 

Organizations face an impossible dilemma:
1. **Legal Risk:** They cannot upload raw spreadsheets to cloud AI due to strict cross-border statutory penalties (**Saudi PDPL & NDMO**, **GDPR**, **HIPAA**, **UK DPA**, **CCPA**).
2. **Technical Bottleneck:** They cannot easily flatten the complex ragged hierarchy without writing fragile, bespoke code.

### The DeepAnalyze Air-Gap Solution
DeepAnalyze acts as a zero-code local security airlock between your confidential files and cloud AI:
1. **Cell-Level Geometric Masking:** Evaluates the entire file cell-by-cell. Preserves structural layout keywords (`Doc. No`, `Doc Date`, `Seq`, `GL Code`, `:`) so cloud models understand the hierarchical layout, while masking all client names to `XXXX`, invoice numbers to `XX-99999`, and figures to `9,999.00`.
2. **Volatile In-Memory Isolation:** All raw data, bidirectional lookup tables, and token vaults live strictly in RAM. Zero unencrypted intermediate data touches disk.
3. **AST Security Firewall:** Intercepts external AI-generated Python code before execution, blocking network sockets, OS system calls, and environment variable exfiltration.
4. **Dual-Track Delivery:** Provides 1-click execution in RAM (generating `Clean_file.xlsx`), and generates ready-to-paste Excel Power Query M-code (`powerquery_script.m`) with a click-by-click UI guide (`powerquery_guide.md`) so accountants can run and refresh transformations directly inside Microsoft Excel.

---

## 2. Key Capabilities & Architecture

```text
+-----------------------------------------------------------------------------+
|                             DEEPANALYZE GATEWAY                             |
|                                                                             |
|   [Raw ERP / Spreadsheet]                                                   |
|             |                                                               |
|             v                                                               |
|   [Step 1: Multi-Column Ingestion] -----> Preserves 100% of Columns (16+)    |
|             |                                                               |
|             v                                                               |
|   [Step 2-4: Origin & Policy Resolver] -> "Not Sure" Intelligent Detection  |
|             |                                                               |
|             v                                                               |
|   [Step 5-7: Deep Scan & Value Teaching] -> Cell Geometric Masking          |
|             |                                                               |
|             +--------------------------+-----------------------------+      |
|             v                          v                             v      |
|   [Encrypted .xlsx File]      [Clipboard Payload]          [Power Query M]  |
|   (100% Layout, 0% PII)       (5-Row Synthetic Mock)       (Excel Native)   |
|             |                          |                             |      |
|             +-----------+--------------+                             |      |
|                         v                                            |      |
|            [External Cloud AI: ChatGPT/Claude/Cursor]                |      |
|                         |                                            |      |
|                         v (Returns Python/Pandas Script)             |      |
|           [Step 9-11: Code Airlock & AST Firewall]                   |      |
|           * Pre-injects pd, np, pl                                   |      |
|           * Blocks Sockets, Env Vars & Subprocesses                  |      |
|           * Interactive Error Self-Healing Loop                      |      |
|                         |                                            |      |
|                         v                                            |      |
|           [Step 12: Local Detokenization & Export]                   |      |
|           * RAM-Only Reconciliation (100% Fidelity)                  |      |
|           * Outputs: Clean_file.xlsx + compliance_audit.md           |      |
+-----------------------------------------------------------------------------+
```

* **Zero-Code Interactive Wizard:** The user never has to write Python code. The conversational wizard guides data ingestion, compliance resolution, layout masking, and execution.
* **"Not Sure" Auto-Detection:** Automatically detects statutory frameworks (Saudi PDPL, GDPR, HIPAA, UK DPA, CCPA) and recognizes unflattened ERP architectures (ragged headers, colon metadata, unpivot structures).
* **Interactive Value Teaching:** If the scanner encounters an internal business code (e.g. General Ledger Code `500-000` or custom sequence `10000`), simply type the column name and an example. DeepAnalyze infers regex rules and re-masks all occurrences across thousands of rows.
* **Encrypted Duplicate Export:** Generates an encrypted duplicate `.xlsx` (e.g. `[filename]_anonymized.xlsx`) with 100% of rows and all 16+ columns preserved, but zero sensitive entities or numbers. Safe to upload directly to any cloud model.
* **Pandas & NumPy Dual-Engine:** Frontier models overwhelmingly write cleaning code using `pandas` (`pd`) and `numpy` (`np`). DeepAnalyze pre-injects `pd`, `np`, and `pl`, automatically detecting Pandas idioms (`df.iloc`, `df.apply`, `df['col']`, `pd.to_datetime`, `np.where`) and seamlessly converting DataFrame formats in RAM.
* **AST Security Firewall:** Statically analyzes pasted code before running it. Blocks network calls (`requests`, `socket`, `urllib`), environment variable access (`os.environ`), and destructive OS operations.
* **Execution Error Self-Healing:** If cloud AI code raises an exception, DeepAnalyze catches it in memory, displays the error, and prompts you to paste the AI's fix without terminating your session or losing state.
* **Power Query Companion:** Generates production-ready Power Query M-code (`powerquery_script.m`) and a comprehensive UI guide (`powerquery_guide.md`) with exact formulas for Microsoft Excel.

---

## 3. Installation & Environment Setup

### Prerequisites
* **Operating System:** macOS (Apple Silicon Metal supported), Linux (Ubuntu, Debian, RHEL), or Windows 10/11.
* **Python:** Python 3.9, 3.10, 3.11, or 3.12.
* **Package Manager:** `pip` or `conda`.

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/deepanalyze.git
cd deepanalyze

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in editable mode with all dependencies
pip install -e .

# 4. Verify installation by running test suite
pytest
```

### Dependencies Installed Automatically
* `polars`, `pyarrow`: Blazing-fast memory-efficient columnar data engine.
* `pandas`, `numpy`, `openpyxl`: Excel ingestion and cloud AI Pandas/NumPy execution compatibility.
* `rich`: Beautiful interactive terminal tables, syntax highlighting, and progress panels.
* `ipython`: Jupyter notebook magics and interactive shell integration.
* `pyperclip`: Cross-platform clipboard integration for instant payload delivery.
* `orjson`, `httpx`: High-throughput serialization and local inference socket communication.

---

## 4. Ways to Run DeepAnalyze

DeepAnalyze provides 4 flexible operating modes tailored for analysts, developers, and non-technical staff:

### Method 1: Interactive Terminal CLI (Zero-Code)
Launch the wizard directly in your terminal:
```bash
# Launch interactive wizard
deepanalyze wizard

# Or point directly to a file to ingest immediately
deepanalyze wizard "/path/to/INV LISTING 31082025 copy.xlsx"
```
*(Alternative syntax: `python -m deepanalyze.wizard`)*

---

### Method 2: Jupyter / IPython Interactive Magics
DeepAnalyze integrates natively into Jupyter Notebook, JupyterLab, and IPython sessions:

```python
# Cell 1: Load the extension
%load_ext deepanalyze

# Cell 2: Launch the interactive zero-code wizard
%deepanalyze
```

#### Available Line & Cell Directives:
```python
# Direct Anonymization to Clipboard (Bypasses Wizard)
%deepanalyze --airgap --origin "Saudi Arabia" --jurisdiction "PDPL" --target df "Clean and unpivot"

# Audit and Safely Execute Code on In-Memory DataFrame
%%deepanalyze --run --target df
# Paste AI-generated code here:
df = df.dropna(subset=['Column1'])

# Instant State Rollback (Up to 5 snapshot levels)
%deepanalyze --undo --target df

# Export Formal Compliance Certificate
%deepanalyze --audit --out compliance_audit.md
```

---

### Method 3: Local Offline Inference Server
If you have a local GGUF model (e.g. `deepanalyze-8b-q4_k_m.gguf`) and want to generate transformation code 100% locally without sending prompts to third-party clouds:

```bash
# Launch server using shell launcher (auto-detects Metal / CUDA acceleration)
./start_server.sh

# Or launch via deepanalyze CLI
deepanalyze server start -m ./models/deepanalyze-8b-q4_k_m.gguf -p 8080
```

---

### Method 4: Python Programmatic API
For data engineers building automated pipelines:

```python
from deepanalyze.wizard import AirGapWizard
from deepanalyze.vault import tokenize_dataframe, detokenize_dataframe
from deepanalyze.policies import resolve_policy
from deepanalyze.firewall import audit_code, execute_code_safely
import polars as pl

# 1. Launch wizard programmatically
wizard = AirGapWizard()
cleaned_df = wizard.run("my_ragged_file.xlsx")

# 2. Or tokenize manually
policy = resolve_policy(origin="Saudi Arabia", target="PDPL")
masked_df, token_map = tokenize_dataframe(df, policy)

# 3. Audit and execute external code
audit_code(untrusted_code)
transformed_df = execute_code_safely(untrusted_code, masked_df)

# 4. Detokenize back to real values in RAM
final_df = detokenize_dataframe(transformed_df)
```

---

## 5. The Complete 13-Step Interactive Wizard Walkthrough

When you run `%deepanalyze` or `deepanalyze wizard`, the system executes the following 13-step pipeline:

### Step 1: Resilient Ingestion & Multi-Column Excel Preservation
* Prompts for file path (`CSV`, `XLSX`, `TSV`, `Parquet`, `JSON`) or variable name.
* **Auto-Sanitization:** Strips drag-and-drop surrounding quotes (`'`, `"`), unescapes shell spaces (`\ `), and expands home paths (`~`).
* **Multi-Column Preservation:** Uses a specialized non-truncating engine (`header=None`) to preserve all 16+ columns, preventing the loss of columns caused by top metadata rows in unflattened ERP spreadsheets.

### Step 2: Country of Origin (Question 1)
* Asks user's operational location: `[1] Saudi Arabia (KSA)`, `[2] Poland (EU)`, `[3] United States (US)`, `[4] United Kingdom (UK)`, `[5] Universal / Other`.

### Step 3: Statutory Compliance Framework (Question 2)
* Dynamically presents statutes tailored to your country.
* **"Not Sure" (Auto-Detect):** Automatically binds the governing regulation (e.g. Saudi Arabia $->$ Saudi PDPL & NDMO Data Standards; Poland $->$ GDPR & UODO).

### Step 4: Dataset Architecture & Geometry Discovery (Question 3)
* Asks for structure type:
  * `[1] Clean Relational / Tabular (Standard Columns)`
  * `[2] Hierarchical / Ragged ERP Report (Invoices, GL Ledgers, Multi-Row Headers)`
  * `[3] Healthcare EHR / Clinical Notes`
  * `[4] Not Sure (Auto-Detect)`
* **"Not Sure" Inspection:** Analyzes colon frequencies (`:`), empty header counts, and ragged offsets to determine if the file is an unflattened ERP matrix.

### Step 5: Full-File Deep Scan & Pattern Categorization
* Evaluates **every single row and cell** (not just headers).
* Categorizes entities into geometric patterns:
  * Company & Personal Names $->$ `XXXX XXXXXX`
  * Document / Invoice IDs $->$ `XX-99999`
  * General Ledger Codes $->$ `999-999` (e.g. `500-000`)
  * Numeric Sequence Counters $->$ `9999` (e.g. `1000`, `2000`)
  * Currency Balances & Amounts $->$ `9,999.00`
  * Timestamps $->$ `9999-99-99 00:00:00`

### Step 6: Unique Masked Pattern Snippet Display
* Renders a Rich table displaying detected patterns, sample original values, sanitized surrogates, and column associations for visual verification.

### Step 7: Interactive Value Teaching & Disambiguation Loop
* Asks: *"Are there more columns or data elements you want me to encrypt? [y/N]"*
* If **Yes**: Enter column names and an example value (e.g. `Seq` $->$ `10000`, `GL Code` $->$ `500-000`).
* DeepAnalyze infers regex rules on the fly and re-masks all matching values across the entire dataset.

### Step 8: Encrypted Duplicate Export vs. Clipboard Payload
* Asks: *"Do you want to download an encrypted duplicate file to disk? [y/N]"*
  * **Option A (Encrypted File):** Exports `[filename]_anonymized.xlsx` in the same directory. Retains 100% of row and column geometry, but replaces all PII and monetary figures with surrogates. **Safe to upload directly to ChatGPT, Claude, or Cursor.**
  * **Option B (Clipboard Payload):** Copies a sanitized 5-row synthetic schema mock and prompt guidelines directly to your clipboard for quick pasting.

### Step 9: Interactive Code Execution Airlock (.py / .ipynb / .m)
* Asks: *"Will code be provided to clean/transform the data? [y/N]"*
* Choose execution mode:
  * `[1] Single Script (.py)`: Paste the entire Python transformation script generated by cloud AI.
  * `[2] Multiple Blocks (.ipynb)`: Paste and test code cell-by-cell.
  * `[3] Power Query (M-Code)`: Paste Power Query M-code generated from cloud AI for native Microsoft Excel execution.
* **Intelligent Auto-Harvesting:** Automatically binds dataset variables (`df`, `data`, `INPUT_FILE`, `OUTPUT_FILE`), executes `__name__ == '__main__'` blocks, and extracts transformed DataFrames from variables, functions, or saved outputs into local memory.
* **Pandas & NumPy Native:** Automatically pre-injects `import pandas as pd`, `import numpy as np`, and `import polars as pl` into scope.

### Step 10: Syntax Preview & AST Security Check
* Renders the pasted code with syntax highlighting.
* Scans the syntax tree with the **AST Security Firewall**. Blocks network requests, socket connections, environment variables, and unauthorized OS commands.
* User presses Enter to execute against the genuine dataset in RAM.

### Step 11: Execution Error Self-Healing Loop
* If the AI code raises a syntax or runtime error:
  * DeepAnalyze catches the error without terminating your session.
  * Displays the exact traceback.
  * Prompts: *"Would you like to paste the corrected code? [y/N]"*.
  * Allows you to paste the AI's fix and re-run immediately.

### Step 12: Automatic Detokenization & Clean Dataset Export
* Automatically detokenizes surrogate tokens in RAM, restoring genuine names, invoice numbers, and figures with 100.00% character fidelity.
* Prompts for output file name (e.g. `Clean_file.xlsx` or `Clean_file.csv`). Saves clean, relational data to disk directly without requiring manual bash script execution.
* When **Power Query [3]** is chosen, writes the accompanying step-by-step UI guide (`powerquery_guide.md`) and saves the M-script (`powerquery_script.m`).

### Step 13: Statutory Compliance Audit Certificate
* Outputs a verifiable `compliance_audit.md` certificate documenting:
  * Timestamp and SHA-256 session hash.
  * Governing statute enforced.
  * Number of protected tokens held in volatile memory.
  * Verification of zero cross-border plaintext leakage.

---

## 6. Excel Power Query Dual-Track (For Non-Programmers)

In addition to automated Python execution in RAM, DeepAnalyze provides a **Dual-Track Delivery** for finance professionals, accountants, and non-analysts who work exclusively in Microsoft Excel.

When selecting delivery format `[3] Power Query (M-Code)` in Step 9:
1. `powerquery_script.m`: Ready-to-paste Power Query M-code saved directly to disk.
2. `powerquery_guide.md`: A complete, click-by-click UI walkthrough with exact Excel steps (no redundant code dumps).

### The 60-Second Copy-Paste (Recommended):
1. In Excel, go to **Data** $->$ **Get Data** $->$ **From File** $->$ **From Excel Workbook**.
2. Select your file and choose sheet **Report** $->$ click **Transform Data**.
3. In Power Query Editor, go to the **Home** tab and click **Advanced Editor**.
4. Select all (`Cmd+A` / `Ctrl+A`), delete existing text, and paste the code from [`powerquery_script.m`](file:///Users/abdullahbinmadhi/Desktop/deepanalyze/powerquery_script.m):

```powerquery
let
    // 1. Ingest Excel Workbook
    Source = Excel.Workbook(File.Contents("/Users/abdullahbinmadhi/Desktop/deepanalyze/INV LISTING 31082025 copy.xlsx"), null, true),
    Navigation = Source{[Item="Report", Kind="Sheet"]}[Data],

    // 2. Remove top report metadata rows (headers/filters)
    #"Removed Top Rows" = Table.Skip(Navigation, 18),

    // 3. Ensure column 1 is treated as text for pattern matching
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

    // 7. Extract numeric sequence items and filter out non-item rows
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
5. Click **Done** $->$ **Close & Load**.
6. **Monthly Refresh:** Every month you receive a new ERP export, simply click **Data $->$ Refresh All**!

---

## 7. Command Reference & Directives Cheat Sheet

| Directive / CLI Flag | Operating Context | Purpose | Syntax Example |
| :--- | :--- | :--- | :--- |
| `deepanalyze wizard` | Shell / Terminal | Launches full 13-step zero-code airlock wizard | `deepanalyze wizard [optional_file_path]` |
| `deepanalyze server start` | Shell / Terminal | Starts local GGUF inference server | `deepanalyze server start -m model.gguf -p 8080` |
| `%deepanalyze` | Jupyter / IPython | Launches full interactive wizard in notebook | `%deepanalyze` |
| `--airgap` | Jupyter / IPython | Direct anonymization & payload copy to clipboard | `%deepanalyze --airgap --origin "Saudi Arabia" --jurisdiction "PDPL" --target df "Clean dates"` |
| `%%deepanalyze --run` | Jupyter Cell Magic | Audits syntax with AST Firewall and executes in RAM | `%%deepanalyze --run --target df`<br>`df['Total'] = df['Qty'] * df['Price']` |
| `--undo` | Jupyter / IPython | Rolls back DataFrame state (up to 5 history snapshots) | `%deepanalyze --undo --target df` |
| `--audit` | Jupyter / IPython | Exports verifiable compliance certificate | `%deepanalyze --audit --out compliance_audit.md` |

---

## 8. Local Inference Server & Speculative Acceleration

DeepAnalyze includes an integrated, high-throughput local inference manager (`server.py` and `start_server.sh`) powered by `llama-server`.

### Hardware Acceleration Auto-Detection
* **macOS (Apple Silicon M1/M2/M3/M4):** Automatically activates **Apple Metal** unified memory acceleration (`-ngl 99`, `-fa on`, Flash Attention, 16K context).
* **Linux (NVIDIA / AMD):** Automatically binds CUDA or ROCm GPU acceleration (`-ngl 99`).
* **Transport:** Binds to Unix Domain Sockets (`/tmp/llama.sock`) for ultra-low latency local IPC with zero TCP overhead.

### Speculative Decoding (2.5x Generation Speedup)
DeepAnalyze supports pairing an 8B target model with a fast speculative draft model (such as `Qwen2.5-Coder-1.5B`). The draft model speculatively generates token candidates that the 8B model verifies in parallel:

```bash
# Start server with speculative draft acceleration
./start_server.sh

# Or via CLI
deepanalyze server start \
  --model ./models/deepanalyze-8b-q4_k_m.gguf \
  --draft-model ./models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --spec-draft-n-max 8 \
  --ctx 16384
```

---

## 9. Architecture, Security & Compliance FAQ

### Q1: How does DeepAnalyze protect messy ERP reports where columns are missing or names are buried in rows?
**Answer:** Standard PII scanners evaluate column headers (e.g. `customer_name`), which fails completely on unflattened ERP reports where client names and invoice numbers are buried in row cells under headers like `Date : From 1/8/2025`. DeepAnalyze applies **Cell-Level Geometric Masking**: it preserves structural layout keywords (`Doc. No`, `Doc Date`, `Seq`, `GL Code`, `:`) so an external AI can understand the hierarchical geometry, while masking all client names to `XXXX`, invoice numbers to `XX-99999`, and figures to `9,999.00`.

### Q2: What happens if I select "Not Sure" for compliance or dataset type?
**Answer:** DeepAnalyze contains built-in statutory and geometric heuristics. If "Not Sure" is chosen for compliance, it maps your operating country to the governing national statute (e.g. Saudi Arabia $->$ Saudi PDPL & NDMO Standards; Poland $->$ GDPR & UODO). If "Not Sure" is chosen for dataset type, it inspects colon frequencies, multi-level headers, and ragged structures to detect whether the dataset is an unflattened ERP report or clean tabular data.

### Q3: How does the interactive value teaching feature work?
**Answer:** If the scanner misses an internal business code (e.g. General Ledger Code `500-000` or sequence number `10000`), simply type the column name and an example value. DeepAnalyze infers regex patterns and length constraints on the fly, registers a dynamic token rule, and re-masks all matching occurrences across thousands of rows.

### Q4: What is the difference between an encrypted duplicate file and a clipboard payload?
**Answer:**
* **Encrypted Duplicate File (`[name]_anonymized.xlsx`):** A complete duplicate spreadsheet saved to disk where 100% of the row/column structure is retained, but every sensitive entity and dollar amount is replaced with safe surrogate values. You can upload this entire file to cloud AI models.
* **Clipboard Payload:** A lightweight 5-row differential synthetic mock and prompt instructions copied directly to your clipboard for quick paste into chat interfaces.

### Q5: What if the cloud AI generates code with bugs or syntax errors?
**Answer:** DeepAnalyze catches execution exceptions in local RAM. Instead of crashing your session, it displays the exact error message and prompts: `"Would you like to paste the corrected code? [y/N]"`. This allows you to iteratively debug with the cloud AI without losing state.

### Q6: How does DeepAnalyze guarantee zero data leakage?
**Answer:** External cloud models only ever see surrogate tokens (`XXXX`, `XX-99999`). When Python code is pasted back, the AST Security Firewall audits the code, blocking network libraries (`socket`, `requests`, `urllib`), environment variable access (`os.environ`), and system commands. Detokenization back to genuine values happens strictly in local RAM.

### Q7: What if the cloud AI writes code using Pandas and NumPy instead of Polars?
**Answer:** DeepAnalyze features a native Dual-Engine execution layer. Cloud LLMs overwhelmingly write data wrangling code using `pandas` (`pd`) and `numpy` (`np`). DeepAnalyze pre-injects `pandas as pd`, `numpy as np`, and `polars as pl` into the local execution scope, automatically detects Pandas operations (`df.iloc`, `df.apply`, `df['col']`, `pd.to_datetime`, `np.where`), and provides `df` in the expected format.

### Q8: What if I am not a programmer and want to clean the spreadsheet in Excel?
**Answer:** DeepAnalyze generates ready-to-paste Power Query M-code (`powerquery_script.m`) and a comprehensive UI guide (`powerquery_guide.md`). Accountants and business users can paste the M-code into Excel's Advanced Editor and clean the spreadsheet natively in Excel. Future monthly files can be refreshed with a single click (**Data $->$ Refresh All**).

### Q9: Why did Power Query previously give an error about keyword `<'section'>`?
**Answer:** In the Power Query M language, `section` is a reserved keyword. This error occurs if:
1. The word `section` is typed or pasted without double quotes (`"section"`).
2. Code starting with `section Section1; shared ...` is pasted into Excel's Advanced Editor (which only accepts expression documents `let ... in ...`).
3. Code is pasted into the single-line Formula Bar (`fx`) or Step Script box instead of opening the **Advanced Editor** (Home $->$ Advanced Editor).
All scripts generated by DeepAnalyze now use fully validated expression syntax with null-safe guards and correct list-of-lists typing.

---

## 10. Module Architecture & Verification Test Suite

### Source Tree
```text
deepanalyze/
├── __init__.py      # Public API exports & IPython extension lifecycle
├── wizard.py        # Zero-Code Interactive 13-Step Air-Gap Wizard
├── policies.py      # Jurisdictional Compliance Engine & "Not Sure" Statute Resolver
├── sentinel.py      # Full-File Deep Scanner, ERP Geometric Masker & Mock Synthesizer
├── vault.py         # In-Memory Token Vault with Dynamic Pattern Learning
├── firewall.py      # AST Security Firewall, Watchdog Guard & Execution Airlock
├── powerquery.py    # Excel Power Query M-Code & Step-by-Step UI Guide Generator
├── transformer.py   # High-Performance Deterministic ERP Flattening Engines
├── magics.py        # IPython Directives (%deepanalyze, --airgap, --run, --undo, --audit)
└── server.py        # Universal CLI & Local GGUF Inference Manager (Metal/CUDA/Socket)
```

### Pre-Commit Test Suite
Every release is validated against 25 rigorous security and performance tests:
```bash
pytest
```
* `tests/test_vault_speed.py`: Validates 100,000 rows tokenized in < 50 ms.
* `tests/test_leakage.py`: Proves 0% plaintext leakage across international identifiers.
* `tests/test_firewall.py`: Verifies 100% of forbidden calls, env vars, and reflection are blocked.
* `tests/test_reconciliation.py`: Confirms 100.00% character fidelity restored across transformations.
* `tests/test_memory_footprint.py`: Enforces memory overhead remains strictly under 250 MB.
* `tests/test_policies.py`: Tests dynamic country resolution and "Not Sure" auto-detection.
* `tests/test_powerquery_and_ingest.py`: Validates full 16-column Excel preservation, Power Query M-code parsing, and 100% ERP transformation fidelity.

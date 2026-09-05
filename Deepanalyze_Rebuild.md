# REBUILD.MD: DEEPANALYZE v4.0 MASTER SPECIFICATION
## Zero-Code Interactive Data Privacy & Compliance Air-Gap Gateway

**Document Target:** Rebuild.md  
**Role:** Principal Cybersecurity Architect, Systems Software Engineer & Lead Data Platform Architect  
**Platform Scope:** Apple Silicon (M-Series Metal Unified Memory), Linux (POSIX), Windows  
**System Target:** `deepanalyze` (v4.0 Production Architecture)  
**Core Purpose:** Transform `deepanalyze` into an autonomous, zero-code Data Leak Prevention (DLP) and Compliance Air-Gap Gateway. The engine intercepts messy, unflattened enterprise spreadsheets (ERP ledgers, invoices, hospital records, CRM dumps) and standard tabular datasets, enforces dynamic statutory anonymization in local volatile memory, generates zero-risk encrypted duplicates and synthetic payloads for frontier cloud models (ChatGPT, Claude, Cursor), guides interactive code execution with single `.py` or multi-block `.ipynb` pipelines and error self-healing, locally decrypts transformations, and exports final cleaned datasets with verifiable audit attestations.

---

## 1. ARCHITECTURAL MANDATE & THE UNFLATTENED ERP CHALLENGE

### 1.1 The Operational Problem
Real-world enterprise accounting ledgers, ERP exports (SAP, Oracle, AS400, Microsoft Dynamics), and healthcare records are rarely clean, relational tables. They are typically ragged, unflattened, multi-row reports with metadata rows, repeated subheaders (`Doc. No`, `Doc Date`, `Seq`, `GL Code`), colon delimiters (`:`), and missing headers.

Standard column-based PII scanners fail completely on these files because client names, national IDs, and invoice numbers are embedded deep in data cells across rows rather than under clean column headers. Organizations face an impossible dilemma:
1. They cannot legally upload raw ERP files to cloud AI models due to statutory cross-border privacy restrictions (GDPR, Saudi PDPL, HIPAA, CCPA, UK DPA).
2. They cannot easily flatten the ragged layouts themselves without writing complex, fragile parsing scripts.

### 1.2 The DeepAnalyze Air-Gap Solution
DeepAnalyze v4.0 solves this by functioning as a zero-code local security airlock:
1. **Interactive Conversational Flow:** The user writes zero Python code. The wizard guides the user step-by-step.
2. **"Not Sure" Auto-Detection:** For both governing compliance frameworks and dataset architectures, the system offers an intelligent "Not Sure" option that auto-detects the right statute and recognizes unflattened ERP matrices.
3. **Full-File Deep Scan & Pattern Masking:** The entire file is evaluated cell-by-cell. Formats (invoice numbers, GL codes, sequences, registration numbers, amounts) are categorized by digit length and pattern.
4. **Interactive Teaching & Disambiguation:** The user can specify additional columns and provide a single example value (e.g., `Seq` -> `10000`, `GL Code` -> `500-000`). The engine learns the regex pattern on the fly and re-masks matching occurrences across all rows.
5. **Dual Output (Encrypted File vs. Clipboard):**
   * Can export an encrypted duplicate file (e.g. `[filename]_anonymized.xlsx`) with 100% of the ERP layout preserved but 0% real PII/figures.
   * Or copies a sanitized 5-row differential mock schema to the clipboard.
6. **Code Airlock (.py / .ipynb) with Error Recovery:** Accepts cloud-generated code in one large script (`.py`) or cell-by-cell notebook (`.ipynb`), inspects it with an AST Firewall, executes it locally against the real data in RAM, and provides a retry loop if the cloud code throws an error.
7. **Local Detokenization & Export:** Restores real identities in RAM, prompts the user for a clean output filename (e.g., `Clean_file.xlsx`), saves it to disk, and outputs a formal `compliance_audit.md` certificate.

---

## 2. STEP-BY-STEP INTERACTIVE PIPELINE SPECIFICATION

### Step 1: Resilient Ingestion & Multi-Column Excel Preservation
* Prompts user for file path (`CSV`, `XLSX`, `TSV`, `Parquet`, `JSON`) or existing in-memory variable name.
* Sanitizes path: automatically strips surrounding quotes (`'` and `"`), unescapes spaces (`\ `), and expands `~`.
* **Multi-Column Excel Ingestion:** For `.xlsx` and `.xls`, uses `pd.read_excel(clean_path, header=None)` to guarantee 100% of rows and columns (e.g. 3,924 rows $\times$ 16 columns) are preserved, avoiding truncation caused by top metadata rows. Safely converts mixed object columns to strings before Polars schema validation.
* Binds dataset to an automatic variable in session (e.g. `inv_listing_31082025`).


### Step 2: Country of Origin (Question 1)
* Asks user's operating location / data residency origin (e.g., `Saudi Arabia`, `Poland`, `United States`, `United Kingdom`, `Universal`).

### Step 3: Dynamic Compliance Framework (Question 2)
* Dynamically presents relevant frameworks based on Question 1.
* Includes a prominent **"Not Sure" (Auto-Detect)** option.
* If "Not Sure" is chosen, the engine automatically resolves the governing statute and outputs explicit feedback:
  `"[System Analysis] Detected best framework for [Country] is '[Statute Name]'. Enforcing this framework."`

### Step 4: Dataset Architecture & Geometry Discovery (Question 3)
* Asks user for data structure type:
  `[1] Clean Relational / Tabular (Standard Columns)`
  `[2] Hierarchical / Ragged ERP Report (Invoices, GL Ledgers, Multi-Row Headers)`
  `[3] Healthcare EHR / Clinical Notes`
  `[4] Not Sure (Auto-Detect)`
* If "Not Sure" is chosen, the engine inspects column headers, colon delimiters (`:`), unnamed columns, and header offsets.
* If ragged ERP patterns are found, it announces:
  `"[System Analysis] Detected 'Hierarchical / Ragged ERP Report'. Activating Structural Geometric Masking."`

### Step 5: Full-File Deep Scan & Pattern Categorization
* Scans every row and cell across the entire file (not just headers).
* Classifies data elements into distinct geometric and PII categories:
  * Corporate / Personal Names (`XXXX XXXXXX`)
  * Document / Invoice IDs (`XX-99999`)
  * Account & GL Codes (`999-999` / `500-000`)
  * Sequential Counters (`9999` / `1000, 2000`)
  * Monetary Balances & Prices (`9,999.00`)
  * Timestamps & Dates (`9999-99-99 00:00:00`)

### Step 6: Unique Masking Snippet Display
* Renders a comprehensive Rich table summarizing all unique patterns detected, raw examples, masked formats, and inferred column associations.

### Step 7: Interactive Value Teaching & Disambiguation Loop
* Asks: `"Are there more columns or data elements you want me to encrypt? [y/N]"`
* If user selects **No**: Proceeds to Step 8.
* If user selects **Yes**:
  * User enters column names (e.g. `Seq, GL Code, Doc. No`).
  * For each field:
    * `"Do you know an expected or potential value for [Field]? [y/N]"`
    * If yes: User types example (e.g. `10000` for Seq, `500-000` for GL Code).
    * Engine infers regex pattern, registers custom token rule, and re-masks all matching occurrences across the dataset.
  * Re-displays updated masked preview for user confirmation.

### Step 8: Encrypted Duplicate Export vs. Clipboard Payload & Dual-Track Prompting
* Asks: `"Do you want to download an encrypted duplicate file to disk? [y/N]"`
  * **If Yes:** Automatically exports an anonymized file in the exact same directory as the imported file (e.g. `INV LISTING 31082025_anonymized.xlsx`). Retains 100% of the ERP row and column structure (all 16+ columns), but masks all sensitive entities, names, codes, and figures to `XXXX`, `XX-99999`, and `9,999.00`.
  * **If No:** Formats a zero-risk 5-row synthetic schema mock and prompt guidelines, copying them directly to the system clipboard.
* **Dual-Track Cloud Prompting:** Step 8 generates instructions telling the frontier AI (ChatGPT/Claude/Cursor) to output:
  1. `Python/Pandas/NumPy` transformation code for automated execution in RAM.
  2. `Excel Power Query M-Code` and a click-by-click formula guide for business users who want to run the pipeline natively inside Microsoft Excel.

### Step 9: Interactive Code Execution Airlock (.py / .ipynb) [Pandas & NumPy Dual-Engine]
* Asks: `"Will code be provided to clean/transform the data? [y/N]"`
  * If No: Wizard concludes and generates audit report.
  * If Yes:
    * Asks: `"Is the code given in one go (single script) or in multiple code blocks? [1] Single Script (.py) [2] Multiple Blocks (.ipynb)"`
    * Creates the corresponding file in the dataset directory (`clean_pipeline.py` or `clean_pipeline.ipynb`).
    * **Cloud Code Compatibility (Pandas & NumPy Native):** Frontier cloud models (ChatGPT, Claude, Cursor) overwhelmingly generate Python scripts using `pandas` (`pd`) and `numpy` (`np`). The airlock pre-injects `pd`, `np`, and `pl` into the execution scope and automatically provides `df` in the expected representation (converting to `pandas.DataFrame` when Pandas operations like `df.iloc`, `df.apply`, `df['col']`, `pd.to_datetime`, or `np.where` are detected, or `polars.DataFrame` when `pl.` expressions are present).
    * **Single Script Mode (.py):**
      * User pastes the script.
      * Wizard renders the code with syntax highlighting and prompts user to press Enter.
      * Code is inspected by the AST Firewall and executed against the real DataFrame in RAM.
    * **Multiple Blocks Mode (.ipynb):**
      * User pastes Block 1 -> previewed -> executed.
      * Wizard prompts: `"Block executed successfully. Is there another code block? [y/N]"`.
      * Repeated block-by-block until complete. Saves valid Jupyter `.ipynb` notebook format.

### Step 10: Execution Error Self-Healing Loop
* If any code block produces an execution error in IPython:
  * Displays the exact traceback and error message.
  * Prompts: `"Would you like to paste the corrected code? [y/N]"`.
  * If Yes: User pastes the fix and execution retries.
  * If No: Execution aborts safely, preserving session state.

### Step 11: Local Detokenization & Reconciliation
* Once execution succeeds, DeepAnalyze automatically detokenizes any surrogate tokens in local RAM, restoring genuine identities with 100.00% character fidelity.

### Step 12: Clean Dataset Export & Power Query Guide Generation
* Prompts: `"Do you want to export the final cleaned dataset? [y/N]"`
  * If Yes: Prompts for file name (e.g., `Clean_file.xlsx` or `Clean_file.csv`).
  * Saves the file in the requested format in the dataset folder.
  * **Excel Power Query Companion:** Automatically generates `powerquery_guide.md` and `powerquery_script.m` in the dataset directory containing ready-to-paste M-code and a click-by-click UI walkthrough with exact Excel formulas for non-technical users.

### Step 13: Statutory Audit Certificate
* Automatically writes `compliance_audit.md` certifying volatile memory isolation and zero data leakage.

---

## 3. TECHNICAL MODULE ARCHITECTURE

```text
deepanalyze/
├── __init__.py      # Package public API & IPython extension lifecycle
├── policies.py      # Dynamic Jurisdictional Compliance Engine & Statute Resolver
├── vault.py         # In-Memory Token Vault with Dynamic Pattern Learning
├── sentinel.py      # Full-File Deep Scanner, ERP Geometric Masker & Mock Synthesizer
├── firewall.py      # AST Security Firewall, Watchdog Guard & .py/.ipynb Pipeline Runner
├── wizard.py        # Zero-Code Interactive 13-Step Air-Gap Wizard
├── powerquery.py    # Excel Power Query M-Code & Step-by-Step UI Formula Guide Generator
├── transformer.py   # High-Performance Deterministic ERP & Tabular Flattening Engines
├── magics.py        # IPython Line & Cell Directives (%deepanalyze, --airgap, --run, --undo, --audit)
└── server.py        # Local Inference Server Manager (/tmp/llama.sock)
```

---

## 4. PRE-COMMIT TEST MATRIX & VERIFICATION
1. `tests/test_vault_speed.py`: 100,000 rows tokenized in < 50 ms.
2. `tests/test_leakage.py`: 0% plaintext leakage across international identifiers.
3. `tests/test_firewall.py`: 100% of forbidden calls, env vars, and dunder reflection blocked.
4. `tests/test_reconciliation.py`: 100.00% character fidelity restored across transformations.
5. `tests/test_memory_footprint.py`: Memory overhead < 250 MB ceiling.
6. `tests/test_policies.py`: Dynamic country resolution and "Not Sure" auto-detection verified.
7. `tests/test_powerquery_and_ingest.py`: Full-column Excel preservation, Power Query M-code, and 100% 12-column ERP cleaning fidelity.


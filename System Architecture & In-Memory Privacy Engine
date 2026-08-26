# DeepAnalyze: System Architecture & In-Memory Privacy Engine

When analyzing messy accounting ledgers, medical records, or proprietary corporate data in Jupyter, data teams usually face an annoying dilemma: **send raw data to a cloud LLM and risk leaking sensitive information, or spend hours manually writing fragile boilerplate parsing scripts.**

DeepAnalyze eliminates this trade-off by decoupling **layout reasoning** from **local execution**. The language model only sees sanitized geometric masks, statistical summaries, or reversible synthetic tokens. The actual parsing, data manipulation, and calculations happen 100% locally in your machine's RAM.

---

## 1. End-to-End System Lifecycle

```text
 ┌──────────────────────────────────────────────┐
 │      1. INPUT: In-Memory Raw DataFrame       │
 │        (e.g., Healthcare CSV or ERP Excel)   │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │   2. LOCAL GATEKEEPER & INSPECTOR (100% Local)│
 │   • Heuristic Rule Engine (Shape/Headers)    │
 │   • Regex & PII Scanner (Names/Emails/PHI)   │
 │   • Local DeepAnalyze-8B (Ambiguity Check)   │
 └──────────────────────┬───────────────────────┘
                        │
          Decision: Route Required Stack
                        │
      ┌─────────────────┼─────────────────┐
      ▼                 ▼                 ▼
 [Unstructured ERP]  [Sensitive PHI/PII]   [Standard Dirty Table]
 • Structural Mask   • Reversible Tokenizer • Statistical Profiler
 • Geometry Preserved• Synthetic 5-Row Mock • Missing Value Ratios
 • Cell Type Masks   • Local Token Mapping  • Dtypes & Quantiles
      │                 │                 │
      └─────────────────┼─────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │     3. INGRESS: Safe JSON / Code Payload     │
 │   (Zero sensitive row data transmitted)      │
 └──────────────────────┬───────────────────────┘
                        │ HTTPS POST (Encrypted Transit)
                        ▼
 ┌──────────────────────────────────────────────┐
 │           4. CLOUD / LOCAL INFERENCE         │
 │  (DeepAnalyze-8B Local / DeepSeek Cloud)     │
 │  • Synthesizes cleaning & parsing logic      │
 └──────────────────────┬───────────────────────┘
                        │ Python Code Payload
                        ▼
 ┌──────────────────────────────────────────────┐
 │     5. EGRESS: Local AST Safety Sandbox      │
 │  • Blocks unauthorized imports (os, socket)  │
 │  • Strips exfiltration attempts & file writes│
 └──────────────────────┬───────────────────────┘
                        │ Pass AST Audit
                        ▼
 ┌──────────────────────────────────────────────┐
 │     6. LOCAL EXECUTION & RECONCILIATION      │
 │  • Executes script on raw data in local RAM  │
 │  • Restores tokenized PII from local map     │
 │  • Interactive Auto-Repair on Runtime Errors │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │         7. OUTPUT: Cleaned DataFrame         │
 │         (Ready in active IPython memory)     │
 └──────────────────────────────────────────────┘
```

---

## 2. Deep Dive: The 7-Tier Architecture

### Tier 1: In-Memory Raw Input
Data is loaded directly into your Python session memory (e.g., `df_erp = pd.read_excel(...)` or a multi-line ragged matrix). DeepAnalyze never forces you to export staging files or write unencrypted intermediate tables to disk.

### Tier 2: The Local Gatekeeper (`privacy_knife.py`)
Before anything touches a model (local or cloud), `LocalGatekeeper` inspects the structure of your DataFrame on-device using fast deterministic heuristics:

```text
[Raw DataFrame] ──► [Tier 1: Deterministic Heuristics (<5ms)] ──► Auto-Configured Stack
                           │ (If layout is ambiguous)
                           ▼
                    [Tier 2: Local DeepAnalyze-8B Scan (In-Memory)]
```

The Gatekeeper routes your table into one of **three privacy strategies**:

#### Strategy A: Structural Geometry Masking (`ERP_STRUCTURAL_MASK`)
* **Best for:** Messy SAP, Navision, Oracle, or QuickBooks invoice exports with multi-row headers, merged blocks, and sub-item notes.
* **How it works:** Replaces all real alphanumeric values with generic character masks (`X-99999`, `9,999.00`) while preserving literal syntax keywords (`Doc. No`, `Doc. Date`, `Seq`, `Grand Total`) and exact grid coordinates (`row.iloc[i]`).
* **What the LLM sees:**

```json
[
  ["Doc. No", " : ", "XX-99999", "Doc. Date", "9999-99-99", "Customer", "XXXXX XXXXXX"],
  ["Seq", "Item Code", "Description", "Qty", "UOM", "Unit Price", "Total"],
  ["9999", "XXX-999", "XXXXX XXXXXXXXXXXXXX XXXXX", "9.0", "XXXX", "9,999.00", "9,999.00"],
  [null, null, "- XXXXXXXX XXX & XXX XXXXXXXXXXXXX", null, null, null, null],
  ["Grand Total", null, null, null, null, null, "9,999.00"]
]
```

#### Strategy B: Reversible PII/PHI Tokenization (`PII_DEIDENTIFIED_MOCK`)
* **Best for:** Healthcare records, HR compensation sheets, or customer databases containing names, emails, phone numbers, and IDs.
* **How it works:** Replaces sensitive entities with consistent surrogate tokens (e.g., `John Doe` $\rightarrow$ `PATIENT_001`, `john@apex.com` $\rightarrow$ `USER_EMAIL_A`). A 5-row synthetic dataset matching the exact schema is sent to the LLM. An isolated lookup dictionary stays in local RAM.
* **Reconciliation:** Once the generated code is verified, the local engine maps the tokens back to real records in local memory.

#### Strategy C: Statistical Schema Profiling (`STANDARD_STATISTICAL_PROFILE`)
* **Best for:** Standard tabular datasets (e.g., CSV files with dirty numbers, formatting discrepancies, and missing values).
* **How it works:** Extracts non-sensitive metadata: column data types, missing value percentages, unique cardinality counts, and min/max quantiles. No raw data rows leave memory.

### Tier 3: Ingress Safe Payload
The sanitized JSON metadata and your task prompt are bundled together. Even if an attacker intercepted the transit payload or logged the cloud API requests, they would only see generic character masks and structural schema metadata.

### Tier 4: Dual-Brain Inference
DeepAnalyze supports both offline local execution and cloud escalation:
* **Local Engine (`deepanalyze-8b`):** Runs offline using `llama-server` on your GPU/Metal unified memory. Fast, private, and free.
* **Cloud Escalation (`--pro`, `--flash`, `--think`):** Routes the sanitized prompt to DeepSeek models (V3 or R1 Reasoner) for heavy mathematical modeling, complex state machines, or intricate edge cases.

### Tier 5: Egress AST Safety Sandbox
When code returns from an LLM, DeepAnalyze runs an Abstract Syntax Tree (AST) security audit before execution:
* **Blocks Dangerous Calls:** Intercepts attempts to use `eval()`, `exec()`, `__import__`, or destructive OS calls (`os.system`, `subprocess.Popen`).
* **Blocks Data Exfiltration:** Blocks network modules (`socket`, `requests`, `urllib`, `httpx`).
* **Blocks File Mutations:** Prevents writing to disk unless explicitly enabled via flags like `--save`.

### Tier 6: Local Execution, State Rollback & Interactive Repair
The validated code executes against the **original, unmasked DataFrame in local RAM**.
* **Automatic Snapshots (`--undo`):** A deepcopy of the DataFrame is saved before running, so you can revert any accidental transformation instantly.
* **Interactive Auto-Escalator:** If a runtime exception occurs (e.g., a `KeyError` or type mismatch), execution pauses and opens a repair menu:

```text
⚠️ [Runtime Crash]: Caught KeyError: 'gross_revenue'
How would you like to resolve this error?
  [1] Retry locally with DeepAnalyze-8B (Free / Local)
  [2] Escalate to DeepSeek Cloud (High-Reasoning Fix)
  [3] Abort & Cancel Repair
Select [1/2/3] (default: 1):
```

### Tier 7: Output Clean DataFrame
The resulting DataFrame is returned to your active IPython kernel under your target variable (e.g., `df_erp` or `df`), fully structured and ready for downstream analytics or DuckDB queries.

---

## 3. Practical Usage & Prompt Cheatsheet

### 1. Pre-Flight Privacy Audit (`--audit-only`)
Use `--audit-only` to review the exact sanitized payload before running code or calling the cloud.

```python
# Check how the ERP Structural Mask protects an accounting table
%deepanalyze --target df_erp --audit-only --privacy mask Flatten this multi-row invoice.

# Check the statistical profile for standard tabular data
%deepanalyze --target sales_df --audit-only --privacy profile Clean missing numeric values.
```

---

### 2. Flattening Messy ERP Reports (`-u`, `--unravel`)
Parses nested parent headers, detail rows, and multi-line descriptions into a clean 2D table.

```python
%%deepanalyze --target df_erp -u -x
Flatten this multi-row ERP invoice export into a clean 2D tabular DataFrame:
1. Track active Doc. No, Doc. Date, and Customer from horizontal header rows.
2. Extract detail line items (Seq, Item Code, Description, Qty, UOM, Unit Price, Total).
3. Append wrapped sub-notes into the preceding item's Description.
4. Filter summary rows and assign the resulting DataFrame to df_erp.
```

---

### 3. In-Memory DuckDB SQL Queries (`-s`, `--sql`)
Run zero-copy analytical SQL queries directly on DataFrames in RAM.

```python
%%deepanalyze --target df_erp -s -x -i
Calculate total revenue and average unit price per customer from df_erp.
Group by customer, order by total revenue descending, and update df_erp.
```

---

### 4. Cloud Escalation for Complex Workloads (`--pro`, `--think`)
Escalate complex modeling tasks to cloud models while keeping raw data local.

```python
%%deepanalyze --target patient_df --think --validate --tune -x
Build a complete classification pipeline to predict readmission_risk:
1. Handle categorical encoders and scale numeric values inside a scikit-learn Pipeline.
2. Tune hyper-parameters using 5-fold StratifiedKFold cross-validation.
3. Assert no data leakage occurs and print a classification report with ROC-AUC.
```

---

### 5. Transactional Rollbacks (`--undo`)
If a transformation produces unexpected results, roll back your DataFrame to its pre-execution state.

```python
# Revert df_erp to its previous snapshot
%deepanalyze --undo --target df_erp
```

---

## 4. Directive Flags Reference

| Flag | Full Name | Category | Purpose & Behavior |
| :--- | :--- | :--- | :--- |
| `-x`, `--exec` | **Execute** | Execution | Runs the verified AST directly into active kernel memory. |
| `--target <var>` | **Target Binding** | Scope | Binds execution to a specific DataFrame in memory (defaults to `df`). |
| `--audit-only` | **Privacy Audit** | Security | Displays the sanitized context payload without calling an LLM or running code. |
| `--privacy <mode>` | **Privacy Override** | Security | Enforces sanitization mode: `auto`, `mask` (ERP), `mock` (PII), `profile` (Stats), or `none`. |
| `-u`, `--unravel` | **Unravel** | Skill | Injects hierarchical state-machine heuristics for ragged spreadsheet reports. |
| `-s`, `--sql` | **DuckDB SQL** | Skill | Executes analytical SQL queries directly on DataFrames via DuckDB. |
| `-f`, `--feat` | **Feature Engineering** | Skill | Constrains the model to vectorized operations, zero-division guards, and in-place typing. |
| `-v`, `--viz` | **Visualization** | Skill | Generates themed Seaborn/Matplotlib visualization scripts. |
| `--save` | **Save Figure** | Output | Automatically saves generated plots to disk (`charts/<slug>.png`) at 300 DPI. |
| `-p`, `--profile` | **Dataset Profile** | Skill | Generates an executive summary, health audit, and diagnostic sampling. |
| `-t`, `--stat` | **Statistical Test** | Skill | Selects and executes parametric/non-parametric hypothesis tests (ANOVA, t-test, Chi-square). |
| `-m`, `--ml` | **Machine Learning** | Skill | Bundles scikit-learn preprocessing and estimators inside `Pipeline` objects. |
| `--validate` | **ML Validation** | ML Guardrail | Enforces cross-validation splits, confusion matrices, and structural metric assertions. |
| `--tune` | **Hyperparameter Tuning**| ML Guardrail | Encapsulates models in leak-free `Pipeline` and `GridSearchCV` routines. |
| `--explain` | **Interpretability** | ML Guardrail | Extracts, ranks, and asserts feature importance and model coefficient distributions. |
| `-i`, `--insight` | **Insight Synthesis** | Analytics | Captures execution stdout and generates concise business takeaways. |
| `-r`, `--repair` | **Autonomous Repair** | Reliability | Forces a dedicated syntax/logic repair prompt against a broken code snippet. |
| `--pro` | **Cloud Pro** | Routing | Routes prompt to `deepseek-chat` (DeepSeek-V3) for complex workloads. |
| `--flash` | **Cloud Flash** | Routing | Routes prompt to lighter, high-speed cloud reasoning models. |
| `--think` | **Cloud Reasoner** | Routing | Routes prompt to `deepseek-reasoner` (R1) for deep Chain-of-Thought processing. |
| `-d`, `--deterministic`| **Deterministic** | Model Tuning | Clamps generation temperature to `0.0` for exact, repeatable code output. |
| `--fast` | **Fast Profile** | Model Tuning | Sets temperature to `0.0` and caps output at 1,000 tokens for rapid execution. |
| `--ultra` | **Ultra Context** | Model Tuning | Expands generation ceiling to 4,096 tokens for large matrices and multi-step state machines. |
| `--undo` | **State Rollback** | State Manager| Restores the target DataFrame to its exact pre-execution deepcopy snapshot. |
| `--toggle` | **Auto-Pilot Toggle** | Convenience | Toggles cell interceptor on/off for plain-English notebook execution without magic prefixes. |
| `-c`, `--continue` | **Continuation** | Workflow | Iterates on previous generated code using natural language refinement instructions. |
| `--retries <n>` | **Retry Count** | Reliability | Sets the maximum number of automated runtime exception repair loops (default: 1). |
| `--status` | **System Health** | Diagnostic | Checks `llama-server` connectivity, API keys, active snapshots, and interceptor state. |

---

## 5. End-to-End Verification Test Suite

### Benchmark 1: Dirty Financial Ledger Cleaning (`-f`, `-x`)

```python
import pandas as pd
import numpy as np

sales_data = pd.DataFrame({
    'invoice_id': ['INV-1001', 'INV-1002', 'INV-1003', 'INV-1004', 'INV-1005'],
    'customer': ['  Acme Corp ', 'BETA LLC', 'Charlie Co.  ', 'David Inc', 'Eve Ltd'],
    'gross_revenue': ['$1,250.50', '$3,400.00', 'N/A', '$450.75', '   '],
    'tax_rate': ['5%', '10%', '7.5%', 'missing', '5%'],
    'order_date': ['2026-01-15', '16/01/2026', '2026-02-01', 'invalid_date', '2026-02-20']
})

%deepanalyze -x -f -d --target sales_data Clean gross_revenue by stripping symbols and parsing to numeric (fill NaN with 0). Clean customer names and parse order_date defensively.
```

---

### Benchmark 2: Hierarchical ERP Invoice Unravelling (`-u`, `-x`)

```python
raw_erp = pd.DataFrame([
    ["Doc. No", " : ", "IV-88201", "Doc. Date", "2025-08-10", "Customer", "Apex Global"],
    ["Seq", "Item Code", "Description", "Qty", "UOM", "Unit Price", "Total"],
    ["1000", "SRV-901", "Cloud Infrastructure Setup", "2.0", "UNIT", "1,500.00", "3,000.00"],
    [None, None, "- Includes VPC & IAM configuration", None, None, None, None],
    ["2000", "LIC-004", "Enterprise Security License", "5.0", "USER", "220.00", "1,100.00"],
    ["Doc. No", " : ", "IV-88202", "Doc. Date", "2025-08-11", "Customer", "Omni Logistics"],
    ["Seq", "Item Code", "Description", "Qty", "UOM", "Unit Price", "Total"],
    ["1000", "HRD-102", "Managed Edge Router v2", "1.0", "UNIT", "850.00", "850.00"],
    [None, None, "- Hardware serial: SN-8829103", None, None, None, None],
    ["Grand Total", None, None, None, None, None, "4,950.00"]
])

%%deepanalyze --target raw_erp -u -x
Flatten this multi-row ERP invoice export into a clean 2D tabular DataFrame:
1. Track active Doc. No, Doc. Date, and Customer from horizontal header rows.
2. Extract detail line items (Seq, Item Code, Description, Qty, UOM, Unit Price, Total).
3. Append wrapped sub-notes into the preceding item's Description.
4. Filter summary rows and assign the resulting DataFrame to raw_erp.
```

---

### Benchmark 3: Zero-Copy DuckDB SQL Analytics (`-s`, `-x`)

```python
%%deepanalyze --target sales_data -s -x -i
Calculate total revenue and transaction count per customer from sales_data.
Order by total revenue descending.
```

---

### Benchmark 4: Publication-Quality Visualization Export (`-v`, `--save`, `-x`)

```python
np.random.seed(42)
telemetry_df = pd.DataFrame({
    'device_id': np.random.choice(['Node_A', 'Node_B', 'Node_C', 'Node_D'], size=100),
    'temperature_c': np.random.normal(65, 10, size=100).round(2),
    'vibration_hz': np.random.exponential(1.5, size=100).round(3)
})

%deepanalyze -x -v --save --target telemetry_df Create a publication-ready boxplot showing temperature_c distributions grouped by device_id with mean markers.
```

---

### Benchmark 5: Guarded ML Pipeline & Feature Importance (`--tune`, `--validate`, `--explain`, `--pro`)

```python
from sklearn.datasets import make_classification

X_raw, y_raw = make_classification(n_samples=200, n_features=6, random_state=42)
ml_df = pd.DataFrame(X_raw, columns=[f'feat_{i}' for i in range(6)])
ml_df['target'] = y_raw

%%deepanalyze --target ml_df --pro --tune --validate --explain -x
Train a Random Forest classifier to predict target:
1. Wrap preprocessing and classifier inside a Pipeline.
2. Optimize n_estimators and max_depth using 5-fold StratifiedKFold cross-validation.
3. Assert no prediction nulls and output the classification report and ranked feature importances.
```

---

## 6. Troubleshooting & Operational FAQ

* **Issue: `ValueError: No closing quotation` on multi-line cell prompts**
  * *Root Cause:* Standard `shlex.split` fails when a cell prompt contains apostrophes (e.g., `"item's description"`).
  * *Resolution:* Ensure your startup magic script separates argument line parsing from the cell string body using `parser.parse_known_args(shlex.split(line))` rather than running `shlex` on the combined cell.

* **Issue: Static linter flags lambda parameter as undefined (e.g., `Undefined variable: ['idx']`)**
  * *Root Cause:* The AST walker only extracts arguments from `ast.FunctionDef` instead of handling `ast.Lambda` nodes.
  * *Resolution:* Confirm that `_lint_and_format_code` checks `isinstance(node, ast.Lambda)` and registers its `args` into the local symbol table.

* **Issue: Local inference server latency on Apple Silicon**
  * *Root Cause:* Thread saturation or context memory fragmentation on unified memory.
  * *Resolution:* Run `llama-server` with `-fa on` (Flash Attention), `--cache-type-k q8_0`, `--cache-type-v q8_0`, and set `-t` to performance core count (e.g., 4 or 6 on M2).

---

## 7. Attribution & Licensing

* **Base Architecture & Research:** [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* **Training Corpus:** [RUC-DataLab/DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
* **Fine-Tuned Weights:** [aboOod3d/deepanalyze-8b](https://huggingface.co/aboOod3d/deepanalyze-8b)
* **License:** [MIT License](LICENSE)

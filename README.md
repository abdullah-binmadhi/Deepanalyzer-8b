# DeepAnalyze: Agentic In-Memory Data Execution Engine

DeepAnalyze is an autonomous, privacy-first data execution environment for Jupyter and IPython, powered by a fine-tuned 8B large language model. Designed for complex enterprise data operations, it translates natural language into deterministic state-machine logic to parse hierarchical spreadsheets, execute zero-copy DuckDB SQL queries, train guarded ML pipelines, and perform defensive feature engineering.

Unlike standard code-generation assistants, DeepAnalyze operates as a closed-loop agent: it executes code directly in your local memory space, catches runtime exceptions via an AST sandbox, sanitizes data through an in-memory privacy gatekeeper, and autonomously patches its own code before returning the final output.

---

## Key Capabilities

* **Zero-Data-Leakage Privacy Gatekeeper (`privacy_knife`):** Intercepts data before any cloud transmission. Generates reversible PII tokenizations, statistical schema profiles, or structural geometry masks (`ERP_STRUCTURAL_MASK`), ensuring sensitive row records never leave local RAM.
* **Pre-Flight Privacy Auditing (`--audit-only`):** Inspects the exact sanitized JSON schema payload scheduled for transmission before dispatching API requests.
* **Dual-Brain Model Routing:** Runs entirely locally via `llama-server` by default, with on-demand escalation to cloud reasoning models (DeepSeek-V3/R1 via `--pro`, `--flash`, `--think`) for mathematically dense or structurally complex operations.
* **Interactive Auto-Escalator & Self-Repair:** Traps runtime exceptions and syntax crashes in an isolated sandbox, opening a human-in-the-loop menu to retry locally with DeepAnalyze-8B, escalate tracebacks to DeepSeek Reasoner, or abort without mutating session state.
* **Egress AST Security Sandbox:** Audits generated code before execution, blocking unauthorized network sockets (`requests`, `urllib`, `socket`), disk writes, or shell exploits.
* **Hierarchical Unravelling Engine (`-u`, `--unravel`):** Deterministic state machines that parse ragged, multi-line, non-rectangular ERP ledger exports (SAP, Navision, Oracle) into normalized 2D DataFrames.
* **Zero-Copy DuckDB SQL Engine (`-s`, `--sql`):** High-speed analytical SQL execution directly on in-memory pandas DataFrames with automated schema registration.
* **Enterprise ML Guardrails (`--validate`, `--tune`, `--explain`):** Enforces strict scikit-learn `Pipeline` encapsulation, leak-free `GridSearchCV`, metric assertions, and feature importance extractions.
* **Insight Synthesis (`-i`, `--insight`):** Automatically captures execution stdout and runs a secondary analytical pass to generate actionable business takeaways.
* **Transactional State Rollback (`--undo`):** Automated deepcopy snapshotting prior to execution for instant rollback and safe experimentation.
* **BYOK (Bring Your Own Key) Security:** Pulls credentials dynamically from OS environment variables (`DEEPSEEK_API_KEY`) without hardcoding secrets in notebooks.

---

## System Architecture

DeepAnalyze enforces a strict separation between raw session memory and external inference engines through its 7-tier architecture:

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
### The Interactive Auto-Escalator & Self-Repair Cycle

When the model generates a block of code, it does not immediately overwrite user variables. Instead, it enters a sandbox verification loop. If the generated logic throws a runtime exception, the engine pauses execution, **sanitizes the traceback to prevent token-overflow from repetitive warnings**, and opens an interactive human-in-the-loop prompt in your notebook:

[Runtime Crash]: Caught KeyError: 'gross_revenue'
How would you like to resolve this error?
  [1] Retry locally with DeepAnalyze-8B (Free / Local)
  [2] Escalate to DeepSeek Cloud (High-Reasoning Fix)
  [3] Abort & Cancel Repair
Select [1/2/3] (default: 1):

```text
┌────────────────────────┐      ┌────────────────────────┐
│     Code Generation    │─────▶│   AST Linting Engine   │
│  (Extracts <Answer>)   │      │ (Validates structure)  │
└────────────────────────┘      └──────────┬─────────────┘
                    ▲                              │ (Syntax Safe)
        (User Selection)                           ▼ 
┌────────────────────────┐      ┌────────────────────────┐
│  Interactive Escalator │◀─────│  Sandbox Test Runtime  │ (Runtime Exception)
│  [1] Local [2] Cloud   │      │  (Executes in-memory)  │
└────────────────────────┘      └──────────┬─────────────┘
                    ▲                              │
                    │                              ▼ (Success)
                    │                   ┌────────────────────────┐
                    └───────────────────│  Environment Mutation  │
                                        │  (Updates live data)   │
                                        └────────────────────────┘
```

---
## 🔬 Enterprise Machine Learning Guardrails

DeepAnalyze enforces strict data science best practices through targeted skill flags, bridging the gap between raw code generation and production-ready ML:

*   **`--validate` (Rigorous Validation):** For ML tasks, the engine automatically implements cross-validation or stratified holdout splits, prints comprehensive metrics (Classification Reports, Confusion Matrices), and embeds programmatic `assert` statements to guarantee shape matching and mathematical integrity.
*   **`--tune` (Leak-Free Pipelines):** Prevents data leakage by encapsulating all imputers, scalers, and estimators inside strict scikit-learn `Pipeline` or `ColumnTransformer` objects, combined with `GridSearchCV` for isolated out-of-fold hyperparameter tuning.
*   **`--explain` (Model Interpretability):** Extracts feature importances (or coefficients), ranks them to provide transparent insights into model decision-making, and validates weight distributions.

---

## ⚡ Hardware & Syntax Optimizations

DeepAnalyze integrates low-level runtime optimizations for efficient local inference on Apple Silicon (Unified Memory):

* **8-bit KV-Cache Quantization (`--cache-type-k q8_0 --cache-type-v q8_0`):** Compresses the key-value context memory footprint by ~50%, allowing seamless 16K to 32K context windows without risking unified memory exhaustion or OS SSD swap lag.
* **Flash Attention (`--fa on`):** Accelerates memory-bandwidth operations on Metal GPUs during multi-turn notebook sessions.
* **Grammar-Constrained Inference (`grammars/deepanalyze.gbnf`):** Applies formal context-free grammar constraints at the token logit level, mathematically guaranteeing that generated output strictly adheres to `<Execute>` syntax blocks without formatting degradation or parser crashes.

---

## Model Serving

DeepAnalyze requires a local inference server to host the quantized 8B model. The base weights are hosted on Hugging Face: **[aboOod3d/deepanalyze-8b](https://huggingface.co/aboOod3d/deepanalyze-8b)**.


### Context Window & Memory Profiles (16GB Unified Memory)

The native pre-trained context window for DeepAnalyze-8B is **8,192 tokens**. However, `llama-server` supports dynamic context expansion via the `-c` flag. 

When running locally on Apple Silicon (e.g., M2 16GB), allocate context according to your workspace workload:

| Context Length (`-c`) | Total RAM Footprint | Feasibility & Speed (16 GB Unified Memory) | Recommended Workload |
| :--- | :--- | :--- | :--- |
| **8,192 (Stock Default)** | ~6.0 GB total | **Optimal** (Fast Metal execution, 0% SSD swap) | Standard DataFrames, routine feature engineering, SQL tasks. |
| **16,384 (16K)** | ~7.2 GB total | **Very Fast & Safe** (Leaves ~8 GB for OS & Jupyter) | Multi-table joins, detailed correlation matrices, Seaborn plotting. |
| **32,768 (32K)** | ~9.2 GB total | **Recommended Sweet Spot** for deep data analytics | Hierarchical ERP log parsing, multi-step ML pipelines, large raw matrix dumps. |
| **65,536 (64K)** | ~13.0 GB total | **Upper Limit** (High memory pressure; slight swap risk) | Massive multi-page accounting text files and extensive tracebacks. |
| **128K+** | >18.0 GB total | **Not Recommended** (Forces heavy SSD swap; <1 tok/sec) | Exceeds physical 16GB RAM budget. |

#### Launching with Custom Context

To scale up to 32K context for deep unravelling and traceback analysis, run:

```bash
llama-server -m ~/Desktop/deepanalyze-8b.gguf --port 8080 -c 32768 -np 1 -ngl 99
```
---


## Installation & Setup

### 1. Prerequisites

Ensure you have a local instance of `llama-server` or an OpenAI-compatible server running the quantized DeepAnalyze GGUF model:


# Example: Running llama-server locally
```bash
llama-server \
  -m models/deepanalyze-8b-q4_k_m.gguf \
  --port 8080 \
  -c 16384 \
  --host 127.0.0.1
```
# High-Efficiency Startup with KV-Cache Quantization & GBNF Enforcement
```bash
llama-server \
  -m models/deepanalyze-8b-q4_k_m.gguf \
  --port 8080 \
  -c 16384 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --grammar-file grammars/deepanalyze.gbnf
```

### `llama-server` Configuration Reference

| Parameter / Flag | Recommended Value | Purpose & Description |
| :--- | :--- | :--- |
| `-m`, `--model` | `models/deepanalyze-8b-q4_k_m.gguf` | Specifies the path to the quantized model weights file. |
| `--port` | `8080` | Sets the HTTP port for the local OpenAI-compatible API expected by DeepAnalyze. |
| `--host` | `127.0.0.1` | Binds the server to localhost, restricting network access strictly to your local machine. |
| `-c`, `--ctx-size` | `16384` | Context window size in tokens. Use `16384` (16K) for standard data science workflows or `32768` (32K) for large matrices. |
| `-ngl`, `--n-gpu-layers` | `99` | Offloads all transformer layers to GPU/Metal unified memory for hardware acceleration. |
| `--cache-type-k` | `q8_0` | Quantizes Key-cache to 8-bit precision, cutting context memory usage in half with no degradation. |
| `--cache-type-v` | `q8_0` | Quantizes Value-cache to 8-bit precision to maintain low RAM overhead during multi-step runs. |
| `-fa on` | `on` | Enables Flash Attention to speed up memory bandwidth operations on Apple Silicon. |
| `--grammar-file` | `grammars/deepanalyze.gbnf` | *(Optional)* Forces token-level structural compliance to guarantee clean `<Execute>` tags. |

### Quick-Launch Shell Configuration (`start-deepanalyze`)

To avoid manually typing long startup commands while maintaining the ability to override flags on the fly, add a dedicated launcher function to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
# Add to ~/.zshrc
start-deepanalyze() {
  llama-server \
    -m ~/Desktop/deepanalyze/models/deepanalyze-8b-q4_k_m.gguf \
    --port 8080 \
    -c 16384 \
    -fa on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --grammar-file ~/Desktop/deepanalyze/grammars/deepanalyze.gbnf \
    "$@"
}
```
Apply the updated configuration:
```
source ~/.zshrc
```

Default Launch (16K Context + 8-bit KV Quantization + GBNF):
```
start-deepanalyze
```

Dynamic Context Expansion (e.g., Scaling to 32K Context):

```
start-deepanalyze -c 32768
```

Port Reassignment:

```
start-deepanalyze --port 8000
```

Custom Model Path Override:

```
start-deepanalyze -m /path/to/another-model.gguf
```

### 2. Python Dependencies

Install the scientific computing stack:

```bash
pip install pandas numpy duckdb matplotlib seaborn openai
```

### 3. Deploy the IPython Magic Extension

Clone this repository and link the startup script into your default IPython profile:

```bash
# Clone the repository
git clone [https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git](https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git) ~/Desktop/deepanalyze
cd ~/Desktop/deepanalyze

# Create IPython startup directory if it doesn't exist
mkdir -p ~/.ipython/profile_default/startup/

# Copy the startup magic script
cp startup/00_deepanalyze_magic.py ~/.ipython/profile_default/startup/00_deepanalyze_magic.py
```

Launch `ipython` or start a Jupyter notebook. The `%deepanalyze` magic will load automatically into your session.
---

## Usage Guide

DeepAnalyze operates directly on the variables currently loaded in your session. Load your raw data, then call the `%deepanalyze` magic command to trigger the agent.

```python
import pandas as pd

# Load an unstructured, multi-level Excel report
df = pd.read_excel("raw_hierarchical_invoice.xlsx", header=None)

# Autonomous flattening and type-casting
%deepanalyze -x -u Restructure the raw invoice dataframe into normalized tabular records.
```

### Runtime Directives

The execution engine is controlled via CLI flags passed to the magic command, allowing you to define the strictness, target variable, and execution scope of the agent.

| Directive | Flag | Behavior | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Engine Status** | `--status` | Probes `llama-server` health endpoints, context window size, active model parameters, and current interceptor state. | Engine monitoring, health checks, and runtime environment inspection. |
| **Privacy Mode** | `--privacy <mode>` | Enforces privacy strategy: auto, mask (ERP), profile (Stats), mock (PII), or none. | Controlling data sanitization levels prior to model ingestion. |
| **Auto-Pilot Toggle** | `--toggle` | Dynamically flips the global cell interceptor on or off for plain-English auto-pilot execution. | Toggling between explicit magic calls and natural language auto-interception without restarting Jupyter. |
| **Execute** | `-x`, `--exec` | Bypasses dry-run inspection and executes the verified AST directly into the active kernel namespace. | Autonomous pipelines and trusted in-memory transformations. |
| **Target Binding** | `--target <var>` | Dynamically designates the target DataFrame in session memory (defaults to `df`). | Working with named datasets (e.g., `sales_data`, `raw_df`) without variable renaming. |
| **Cloud Pro** | `--pro` | Bypasses the local engine and routes the prompt to `deepseek-chat` (DeepSeek-V4-Pro). | Fast, highly accurate cloud processing for standard heavy analytics. |
| **Cloud Flash** | `--flash` | Routes the prompt to the lighter, high-speed `deepseek-chat` variant. | Rapid cloud execution for simple parsing or lighter workloads. |
| **Deep Reasoner** | `--think` | Routes the prompt to `deepseek-reasoner` (R1) for deep Chain-of-Thought processing. | Highly complex transformations, dense mathematical modeling, or debugging tricky tracebacks. |
| **Insight Synthesis** | `-i`, `--insight` | Captures execution stdout and triggers a secondary LLM pass to explain the metrics. | Translating raw numbers into actionable business insights. |
| **Unravel** | `-u`, `--unravel` | Activates hierarchical state-machine unravelling heuristics and defensive parsing rules. | Normalizing nested, multi-row, non-rectangular ERP ledger exports into flat tables. |
| **Feature** | `-f`, `--feat` | Constrains the model to vectorized operations, safe casting, and in-place transformations. | Defensive feature engineering, missing value imputation, and schema cleaning. |
| **SQL Engine** | `-s`, `--sql` | Routes transformations through DuckDB for zero-copy memory execution. | High-performance SQL queries and complex relational joins directly on DataFrames. |
| **Visualize** | `-v`, `--viz` | Instructs the model to generate styled Matplotlib/Seaborn rendering scripts. | Automated exploratory data analysis (EDA) and publication-quality distributions. |
| **Statistical Test** | `-t`, `--stat` | Automatically selects and runs parametric/non-parametric tests. | Hypothesis testing (ANOVA, t-test, Chi-square). |
| **Machine Learning** | `-m`, `--ml` | Bundles scikit-learn preprocessing and estimators via Pipeline. | Rapid baseline model training and classification reports. |
| **Save Charts** | `--save` | Forces the visualization generator to write figures to disk (`charts/<slug>.png`) at 300 DPI. | Batch artifact export and automated reporting workflows. |
| **Profile** | `-p`, `--profile` | Generates a strategic structural health audit alongside safe diagnostic sampling. | Schema inspection, cardinality profiling, and null distribution checks. |
| **Deterministic** | `-d`, `--deterministic` | Clamps generation temperature to `0.0` for repeatable, exact syntax. | Strict ETL pipelines and reproducible transformations. |
| **Ultra Context** | `--ultra` | Expands token generation limits up to 4,096 tokens. | Large data matrices, multi-step state machines, or extensive AST traceback repairs. |
| **Auto-Repair** | `--retries <n>` | Specifies the maximum number of automated runtime exception retry loops (defaults to 1). | Fault-tolerant execution handling transient syntax or execution exceptions. |
| **Revert State** | `--undo` | Restores the specified target DataFrame to the exact deepcopy snapshot taken prior to execution. | Instant state rollback, safety isolation, and non-destructive experimentation. |
| **Validate** | `--validate` | Forces strict cross-validation, explicit metric reporting, and shape/type assertions. | Ensuring model integrity and preventing evaluation on training data. |
| **Tune** | `--tune` | Encapsulates all estimators/scalers in `Pipeline` and `GridSearchCV`. | Zero-leakage hyperparameter optimization. |
| **Explain** | `--explain` | Extracts and ranks feature importances/coefficients with mathematical invariants. | Model interpretability and auditing. |
---

## Protocol & Prompt Architecture

DeepAnalyze utilizes specific token formatting to differentiate standard conversational queries from agentic execution tasks. Understanding these tokens is required if you intend to build custom API wrappers around the model.

### 1. The Analytic Trigger Token

To force the model into deterministic reasoning mode, the final sequence of the prompt must include the `<Analyze>` token. This prevents the model from generating conversational filler.

```text
<｜User｜> {data context and instructions} <｜Assistant｜><Analyze>
```

### 2. The Execution Sandbox Block

Standard instruct models output Python within Markdown backticks (` ```python `). DeepAnalyze has been fine-tuned to emit code intended for immediate runtime evaluation inside `<Execute>` tags. 

The IPython frontend monitors the output stream, extracts the payload within these tags, validates the Abstract Syntax Tree (AST), and executes it against the host memory.

**Raw Model Output Example:**

```python
Based on the provided dtypes, the dataframe requires grouping by the primary key.

<Execute>
import duckdb

query = """
    SELECT 
        category, 
        SUM(revenue) AS total_revenue
    FROM df 
    WHERE status = 'Paid'
    GROUP BY 1
"""
cleaned_df = duckdb.query(query).df()
</Execute>

Execution complete. The variable `cleaned_df` is now available.
```

---

## Example Test Suite & Benchmarks

### 1. Initialize Benchmark Data

```python
import pandas as pd
import numpy as np

# Financial ledger with dirty strings and missing values
sales_data = pd.DataFrame({
    'invoice_id': ['INV-1001', 'INV-1002', 'INV-1003', 'INV-1004', 'INV-1005', 'INV-1006'],
    'customer_name': ['  Alice Corp ', 'BOB LLC', 'Charlie & Co.  ', 'David Inc', 'Eve Ltd', 'Frank Corp'],
    'gross_revenue': ['$1,250.50', '$3,400.00', 'N/A', '$450.75', '   ', '$9,100.20'],
    'tax_rate': ['5%', '10%', '7.5%', 'missing', '5%', '12%'],
    'order_date': ['2026-01-15', '16/01/2026', '2026-02-01', 'invalid_date', '2026-02-20', '2026-03-05'],
    'status': ['PAID', 'pending', 'Paid', 'REFUNDED', 'paid', 'PENDING']
})

# Hierarchical, ragged ERP export
erp_export = pd.DataFrame({
    'raw_line': [
        'BRANCH: NORTH REGION - 2026',
        'EMP-01 | John Doe | Senior Analyst | 85000',
        'EMP-02 | Jane Smith | Lead Engineer | 110000',
        'BRANCH: SOUTH REGION - 2026',
        'EMP-03 | Mark Brown | Consultant | 72000',
        'EMP-04 | Lucy Liu | Product Manager | 95000',
        'EMP-05 | David Clark | QA Specialist | 68000'
    ]
})

# Telemetry sensor log
np.random.seed(42)
telemetry_df = pd.DataFrame({
    'device_id': np.random.choice(['DEV_A', 'DEV_B', 'DEV_C', 'DEV_D'], size=20),
    'temperature_c': np.random.uniform(20.0, 95.0, size=20).round(2),
    'vibration_hz': np.random.uniform(0.1, 4.5, size=20).round(3),
    'status_code': np.random.choice([200, 200, 200, 500, 503], size=20)
})
```

---

### 2. Auto-Pilot Mode (Cell Interceptor)

```python
# Check backend server health
%deepanalyze --status

# Toggle auto-pilot interceptor on
%deepanalyze --toggle

# Execute natural language directly without %deepanalyze prefix
Print the shape and column types of telemetry_df

# Toggle auto-pilot off
%deepanalyze --toggle
```

---

### 3. Defensive Feature Engineering (`-f`, `-x`)

```python
%deepanalyze -x -f -d --target sales_data Clean gross_revenue by stripping non-numeric characters using pd.to_numeric with errors='coerce' filling NaNs with 0. Standardize status to uppercase and strip customer_name.
```

---

### 4. Hierarchical ERP Unravelling (`-u`, `--ultra`, `-x`)

```python
%deepanalyze -x --ultra -u --target erp_export Parse raw_line into a structured DataFrame named erp_clean with columns: branch, emp_id, emp_name, role, and salary (as integer).
```

---

### 5. Zero-Copy In-Memory DuckDB Engine (`-s`, `-x`)

```python
%deepanalyze -x -s -d --target telemetry_df Find average temperature_c and maximum vibration_hz per device_id where status_code is 200 using DuckDB.
```

---

### 6. Visual Diagnostics & 300 DPI Export (`-v`, `--save`, `-x`)

```python
%deepanalyze -x -v --save --target telemetry_df Create a boxplot of temperature_c grouped by device_id and save the figure to disk.
```

---

### 7. Transactional State Rollback (`--undo`)

```python
# Accidental or experimental destructive operation
%deepanalyze -x -f --target sales_data Keep only the status column in sales_data and drop everything else in place.

# Restore the target DataFrame back to its pre-execution state
%deepanalyze --undo --target sales_data
```
---

### 8. Multi-Model Evaluation & Guardrails (`--tune`, `--validate`, `--pro`)

```python
%deepanalyze -x --tune --validate --pro --target retail_df Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting to predict is_returned. Use GridSearchCV with 5-fold StratifiedKFold. Calculate out-of-fold Mean CV Accuracy, CV Std Dev, and F1-score, displaying results in a consolidated DataFrame.
```
---

## Attribution & Licensing

* **Base Architecture & Research:** [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* **Training Corpus:** [RUC-DataLab/DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
* **License:** [MIT License](LICENSE)

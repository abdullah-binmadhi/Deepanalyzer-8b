# DeepAnalyze: Agentic In-Memory Data Execution Engine

DeepAnalyze is an autonomous execution environment for Jupyter and IPython, powered by a fine-tuned 8B large language model. Designed for complex data operations, it translates natural language into deterministic state-machine logic to parse hierarchical spreadsheets, execute zero-copy DuckDB SQL queries, and perform feature engineering. 

Unlike standard code-generation assistants, DeepAnalyze operates as a closed-loop agent: it executes code directly in your memory space, catches runtime errors via an AST linter, and autonomously patches its own code before returning the final output.

---

## Key Capabilities

* **Zero-Friction In-Memory Analytics:** Directly inspects variable schemas and DataFrame dtypes from active kernel memory.
* **Auto-Pilot Interceptor:** Intercepts plain English instructions in standard code cells without requiring explicit magic prefixes.
* **DuckDB SQL Engine:** Zero-copy querying over in-memory pandas DataFrames using standard SQL dialect.
* **Hierarchical Unravelling:** Deterministic state machines that parse and forward-fill ragged, non-rectangular tabular text reports.
* **State Snapshot & Rollback:** Automated deepcopy snapshotting before execution, allowing instant state rollbacks via `--undo`.
* **Runtime Auto-Repair Loop:** Traps syntax errors and runtime exceptions, feeds AST tracebacks back into the model, and retries execution autonomously.

---
## System Architecture

The core of DeepAnalyze is its ability to extract context directly from the host environment, formulate a solution, and validate it before applying changes.

text
```
 ┌──────────────────────────────────────────────────────────┐
 │               IPython / Jupyter Kernel                   │
 │                                                          │
 │   Plain English Prompt / %deepanalyze Directive          │
 │                            │                             │
 │                            ▼                             │
 │         Input Interceptor / Magic Flag Parser            │
 │                            │                             │
 │                            ▼                             │
 │          Runtime Schema & Dtype Inspection               │
 │                            │                             │
 └────────────────────────────┼─────────────────────────────┘
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │          Local Inference Server (llama-server)           │
 │                                                          │
 │     Prompt + Skill Rulebooks (SQL / Viz / Wrangling)     │
 │                            │                             │
 │                            ▼                             │
 │             DeepAnalyze-8B Reasoning Core                │
 └────────────────────────────┼─────────────────────────────┘
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │                  Execution Engine                        │
 │                                                          │
 │   AST Validation ──► [Auto-Repair on Error (1-3x)]       │
 │   DuckDB Engine  ──► Zero-Copy In-Memory SQL Execution   │
 │   State Manager  ──► Deepcopy Snapshots & `--undo`       │
 └──────────────────────────────────────────────────────────┘
```

### The Autonomous Self-Repair Cycle

When the model generates a block of code, it does not immediately overwrite user variables. Instead, it enters a sandbox verification loop. If the generated logic fails on edge cases (e.g., hidden NaN values, index mismatches), the engine captures the Python traceback and feeds it back to the LLM for autonomous patching.

```text
┌────────────────────────┐      ┌────────────────────────┐
│     Code Generation    │─────▶│   AST Linting Engine   │
│  (Extracts <Execute>)  │      │  (Validates structure) │
└────────────────────────┘      └──────────┬─────────────┘
            ▲                              │ (Syntax Safe)
(Traceback) │                              ▼ 
┌────────────────────────┐      ┌────────────────────────┐
│ LLM Context Injection  │◀─────│  Sandbox Test Runtime  │ (Runtime Exception)
│ (Appends Error Data)   │      │  (Executes in-memory)  │
└────────────────────────┘      └──────────┬─────────────┘
                                           │
                                           ▼ (Success)
                                ┌────────────────────────┐
                                │  Environment Mutation  │
                                │  (Updates live data)   │
                                └────────────────────────┘
```

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

```bash
# Example: Running llama-server locally
llama-server \
  -m models/deepanalyze-8b-q4_k_m.gguf \
  --port 8080 \
  -c 16384 \
  --host 127.0.0.1
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
| **Auto-Pilot Toggle** | `--toggle` | Dynamically flips the global cell interceptor on or off for plain-English auto-pilot execution. | Toggling between explicit magic calls and natural language auto-interception without restarting Jupyter. |
| **Execute** | `-x`, `--exec` | Bypasses dry-run inspection and executes the verified AST directly into the active kernel namespace. | Autonomous pipelines and trusted in-memory transformations. |
| **Target Binding** | `--target <var>` | Dynamically designates the target DataFrame in session memory (defaults to `df`). | Working with named datasets (e.g., `sales_data`, `raw_df`) without variable renaming. |
| **Unravel** | `-u`, `--unravel` | Activates hierarchical state-machine unravelling heuristics and defensive parsing rules. | Normalizing nested, multi-row, non-rectangular ERP ledger exports into flat tables. |
| **Feature** | `-f`, `--feat` | Constrains the model to vectorized operations, safe casting, and in-place transformations. | Defensive feature engineering, missing value imputation, and schema cleaning. |
| **SQL Engine** | `-s`, `--sql` | Routes transformations through DuckDB for zero-copy memory execution. | High-performance SQL queries and complex relational joins directly on DataFrames. |
| **Visualize** | `-v`, `--viz` | Instructs the model to generate styled Matplotlib/Seaborn rendering scripts. | Automated exploratory data analysis (EDA) and publication-quality distributions. |
| **Save Charts** | `--save` | Forces the visualization generator to write figures to disk (`charts/<slug>.png`) at 300 DPI. | Batch artifact export and automated reporting workflows. |
| **Profile** | `-p`, `--profile` | Generates a strategic structural health audit alongside safe diagnostic sampling. | Schema inspection, cardinality profiling, and null distribution checks. |
| **Deterministic** | `-d`, `--deterministic` | Clamps generation temperature to `0.0` for repeatable, exact syntax. | Strict ETL pipelines and reproducible transformations. |
| **Ultra Context** | `--ultra` | Expands token generation limits up to 4,096 tokens. | Large data matrices, multi-step state machines, or extensive AST traceback repairs. |
| **Auto-Repair** | `--retries <n>` | Specifies the maximum number of automated runtime exception retry loops (defaults to 1). | Fault-tolerant execution handling transient syntax or execution exceptions. |
| **Revert State** | `--undo` | Restores the specified target DataFrame to the exact deepcopy snapshot taken prior to execution. | Instant state rollback, safety isolation, and non-destructive experimentation. |

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
## Attribution & Licensing

* **Base Architecture & Research:** [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* **Training Corpus:** [RUC-DataLab/DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
* **License:** [MIT License](LICENSE)

# DeepAnalyze: Agentic In-Memory Data Execution Engine

DeepAnalyze is an autonomous execution environment for Jupyter and IPython, powered by a fine-tuned 8B large language model. Designed for complex data operations, it translates natural language into deterministic state-machine logic to parse hierarchical spreadsheets, execute zero-copy DuckDB SQL queries, and perform feature engineering. 

Unlike standard code-generation assistants, DeepAnalyze operates as a closed-loop agent: it executes code directly in your memory space, catches runtime errors via an AST linter, and autonomously patches its own code before returning the final output.

---

## System Architecture

The core of DeepAnalyze is its ability to extract context directly from the host environment, formulate a solution, and validate it before applying changes.

```text
┌─────────────────────────────────────────────────────────┐
│              IPython / Jupyter Notebook Session         │
│  User: %deepanalyze -x -u "Flatten invoice df"          │
└───────────────────────────┬─────────────────────────────┘
                            │ (Context: df.head + dtypes)
                            ▼
┌─────────────────────────────────────────────────────────┐
│            Local Server (aboOod3d/deepanalyze-8b)       │
│  Outputs structured <Execute> Python/SQL state machine  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│             AST Verification & Self-Repair Loop         │
│  * Parses syntax tree for unsafe/deprecated calls       │
│  * In-memory test execution & error feedback capture    │
│  * Auto-patches code on runtime exceptions              │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               In-Memory Mutation / DuckDB Query         │
│  Mutates `df` in-place with zero-copy execution         │
└─────────────────────────────────────────────────────────┘
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

Initialize the backend via `llama.cpp` or standard `llama-server` distributions:

```bash
llama-server \
  --hf-repo aboOod3d/deepanalyze-8b \
  --hf-file deepanalyze-8b.gguf \
  --port 8080 \
  -c 8192 \
  -ngl 99
```
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

## Installation

Once the backend is running, install the agent frontend into your local IPython profile.

```bash
# 1. Clone the environment integration repository
git clone [https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git](https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git)

# 2. Register the IPython magic extension
mkdir -p ~/.ipython/profile_default/startup/
cp Deepanalyzer-8b/startup/00_deepanalyze_magic.py ~/.ipython/profile_default/startup/

# 3. Install runtime dependencies
pip install -r Deepanalyzer-8b/requirements.txt
```

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
| **Auto-Repair** | `--retries <n>` | Specifies the maximum number of automated runtime exception retry loops (defaults to `1`). | Fault-tolerant execution handling transient syntax or execution exceptions. |
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

## Attribution & Licensing

* **Base Architecture & Research:** [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* **Training Corpus:** [RUC-DataLab/DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
* **License:** [MIT License](LICENSE)

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

The execution engine is controlled via CLI flags passed to the magic command, allowing you to define the strictness and scope of the agent's memory access.

| Directive | Flag | Behavior | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Execute** | `-x` | Bypasses the confirmation prompt, running the generated AST directly in the active session. | Continuous workflows and trusted transformations. |
| **Unravel** | `-u` | Invokes the state-machine parsing logic. | Normalizing complex, non-rectangular ERP exports. |
| **Feature** | `-f` | Constrains the agent to in-place column mutations and vector operations. | Feature engineering and data enrichment. |
| **SQL** | `-s` | Forces the model to utilize DuckDB for in-memory querying instead of pandas APIs. | High-performance aggregations on massive datasets. |
| **Visualize** | `-v` | Directs the model to synthesize Seaborn or Matplotlib rendering code. | Automated Exploratory Data Analysis (EDA). |
| **Expanded** | `--ultra` | Expands the reasoning window allocation up to 4,096 tokens. | Handling massive traceback chains or complex logic. |
| **Revert State**| `--undo` | Restores the in-memory dataframe to the exact snapshot captured prior to execution. | Error recovery and experiment rollback. |

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

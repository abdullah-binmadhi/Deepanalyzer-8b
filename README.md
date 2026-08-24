# DeepAnalyze: Autonomous In-Memory Data Agent & LLM Execution Engine

DeepAnalyze is an IPython magic extension (`%deepanalyze`) integrated with a local fine-tuned 8B LLM. It autonomously parses complex multi-row hierarchical spreadsheets, cleans messy tabular data, executes zero-copy DuckDB SQL queries, and generates visualizations with an AST linting self-repair loop.

---

## Architecture & How It Works

```
┌─────────────────────────────────────────────────────────┐
│              IPython / Jupyter Notebook Session         │
│  User: %deepanalyze -x -u "Flatten invoice df"        │
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
│  Mutates `df` in-place with zero-copy execution        │
└─────────────────────────────────────────────────────────┘
```

---

## Model Serving

The quantized model weights are hosted on Hugging Face: **[aboOod3d/deepanalyze-8b](https://huggingface.co/aboOod3d/deepanalyze-8b)**.

### Run with `llama-server`
```bash
llama-server   --hf-repo aboOod3d/deepanalyze-8b   --hf-file deepanalyze-8b.gguf   --port 8080   -c 8192   -ngl 99
```

---

## Installation & Setup

```bash
# 1. Clone this repository
git clone https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git

# 2. Install extension to IPython startup
mkdir -p ~/.ipython/profile_default/startup/
cp Deepanalyzer-8b/startup/00_deepanalyze_magic.py ~/.ipython/profile_default/startup/

# 3. Install dependencies
pip install -r Deepanalyzer-8b/requirements.txt
```

---

## Usage Example

Launch `ipython` or a Jupyter Notebook:

```python
import pandas as pd
df = pd.read_excel("raw_hierarchical_invoice.xlsx", header=None)

# Autonomous unravelling with self-healing execution
%deepanalyze -x -u Flatten raw invoice listing df into tabular records.
```

### Magic Flags Reference
* `-x`, `--exec`: Auto-executes generated code directly in the active session.
* `-u`, `--unravel`: Hierarchical report state-machine parser.
* `-f`, `--feat`: Feature engineering and in-place transformations.
* `-s`, `--sql`: Zero-copy DuckDB SQL querying.
* `-v`, `--viz`: Seaborn visualization generation.
* `--ultra`: Expands token reasoning window to 4,096 tokens.
* `--undo`: Restores DataFrame state to snapshot before last execution.

---

## Prompt & Chat Format

The model uses specialized analytical tags to trigger code generation mode:

```text
<｜User｜>{prompt}<｜Assistant｜><Analyze>
```

Code blocks intended for the agent execution engine are emitted inside `<Execute>` delimiters:

```python
<Execute>
import duckdb
result = duckdb.query("SELECT category, SUM(revenue) FROM df GROUP BY 1").df()
</Execute>
```

---

## Citation & Attribution

* **Base Weights & Research:** [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* **Dataset:** [RUC-DataLab/DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
* **License:** [MIT License](LICENSE)

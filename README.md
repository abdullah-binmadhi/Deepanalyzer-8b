# DeepAnalyze: Autonomous In-Memory Data Agent & LLM Execution Engine

DeepAnalyze is an IPython magic extension (`%deepanalyze`) integrated with a local fine-tuned 8B LLM. It autonomously parses complex multi-row hierarchical spreadsheets, cleans messy tabular data, executes zero-copy DuckDB SQL queries, and generates visualizations with an AST linting self-repair loop.

---

## Quick Start

### 1. Launch Local Model Server
The fine-tuned model is hosted on Hugging Face: [aboOod3d/deepanalyze-8b](https://huggingface.co/aboOod3d/deepanalyze-8b).

Run via `llama-server`:
```bash
llama-server   --hf-repo aboOod3d/deepanalyze-8b   --hf-file deepanalyze-8b.gguf   --port 8080   -c 8192
```

### 2. Install IPython Magic Script
```bash
git clone https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git
mkdir -p ~/.ipython/profile_default/startup/
cp Deepanalyzer-8b/startup/00_deepanalyze_magic.py ~/.ipython/profile_default/startup/
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

### Flags Reference
* `-x`, `--exec`: Auto-executes generated code in the active session.
* `-u`, `--unravel`: Hierarchical report state-machine parser.
* `-f`, `--feat`: Feature engineering and in-place transformations.
* `-s`, `--sql`: Zero-copy DuckDB SQL querying.
* `-v`, `--viz`: Seaborn visualization generation.
* `--ultra`: Expands token reasoning window to 4,096 tokens.
* `--undo`: Restores DataFrame state to snapshot before last execution.

---

## Attribution & License
* Base model: [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* License: MIT License

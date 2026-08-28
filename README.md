# DeepAnalyze: Agentic In-Memory Data Execution Engine

DeepAnalyze is an autonomous, privacy-first data execution environment for Jupyter and IPython, powered by a fine-tuned 8B large language model. Designed for complex enterprise data operations, it translates natural language into deterministic state-machine logic to parse hierarchical spreadsheets, execute zero-copy DuckDB SQL queries, train guarded ML pipelines, and perform defensive feature engineering seamlessly across both **Pandas and Polars**.

Unlike standard code-generation assistants, DeepAnalyze operates as a closed-loop agent: it executes code directly in your local memory space, catches runtime exceptions via an AST sandbox, sanitizes data through an in-memory privacy gatekeeper, and autonomously patches its own code before returning the final output.

---

## 📚 Essential Documentation & Guides

* 📖 **[Comprehensive Q&A and Knowledge Guide (FAQ)](DEEPANALYZE_FAQ_AND_QA.md):** In-depth, college-level explanations of privacy isolation, zero-hallucination math engines, Apple Silicon hardware optimizations, and ERP unravelling.
* ⚡ **[Master Command Cheat Sheet](DEEPANALYZE_COMMAND_CHEAT_SHEET.md):** Complete practical matrix of all CLI flags, workflows, code examples, and pro tips.
* 🛡️ **[System Architecture & In-Memory Privacy Engine](System%20Architecture%20&%20In-Memory%20Privacy%20Engine.md):** Technical deep-dive into the token vault, AST sandbox, and zero-copy Arrow memory engine.

---

## Key Capabilities

### Core Engine
* **Universal Polymorphic Adapter:** Seamlessly detects and handles both Pandas and Polars DataFrames at runtime. Automatically dispatches high-speed, parallelized Rust expressions for large datasets while maintaining 100% backward compatibility for legacy Pandas text-parsing workflows.
* **Zero-Data-Leakage Privacy Gatekeeper (`privacy_knife`):** Intercepts data before any cloud transmission. Generates reversible PII tokenizations, statistical schema profiles, or structural geometry masks (`ERP_STRUCTURAL_MASK`), ensuring sensitive row records never leave local RAM.
* **Pre-Flight Privacy Auditing (`--audit-only`):** Inspects the exact sanitized JSON schema payload scheduled for transmission before dispatching API requests.
* **Dual-Brain Model Routing:** Runs entirely locally via `llama-server` by default, with on-demand escalation to cloud reasoning models (DeepSeek-V3/R1 via `--pro`, `--flash`, `--think`) for mathematically dense or structurally complex operations.
* **Interactive Auto-Escalator & Self-Repair:** Traps runtime exceptions and syntax crashes in an isolated sandbox, opening a human-in-the-loop menu to retry locally with DeepAnalyze-8B, escalate tracebacks to DeepSeek Reasoner, or abort without mutating session state.
* **Egress AST Security Sandbox:** Audits generated code before execution, blocking unauthorized network sockets (`requests`, `urllib`, `socket`), disk writes, or shell exploits.
* **Streaming Syntax HUD & Step Tracker:** Replaces static progress text with an animated step-by-step indicator (`[1/3] 🔍 Routing ➔ [2/3] ⚡ Streaming ➔ [3/3] 🛡️ Validating`) with real-time token count during LLM inference.
* **Resilient Data Ingestion (`--import`):** High-speed Polars ingestion engine with automatic path resolution, quote sanitization, encoding fallback (UTF-8 ➔ latin-1), Excel calamine/openpyxl fallback, line-delimited NDJSON support, clipboard reading, and automatic target variable name binding.
* **Defensive Polyglot Exporter (`--export`):** Universal export engine supporting `.parquet`, `.csv`, `.tsv`, `.xlsx`, `.json`, `.ndjson`, `.ipc`/`.arrow`, and DuckDB database tables (`db.duckdb:table_name`) with auto directory creation and LazyFrame auto-collection.
* **BYOK (Bring Your Own Key) Security:** Pulls credentials dynamically from OS environment variables (`DEEPSEEK_API_KEY`) without hardcoding secrets in notebooks.

### Universal Dirty Data Cleaning Suite
* **Multilingual Unicode & Mojibake Sanitizer (`--ftfy`):** Fixes double-encoding artifacts (`Ã©` → `é`, `â€™` → `'`), strips invisible zero-width characters (`\u200b`), non-breaking spaces (`\xa0`), and unescapes HTML entities.
* **Entity Resolution & Fuzzy Deduplication (`--fuzzy-clean`):** Rapid in-memory similarity clustering unifies categorical typos and acronyms (`"USA"`, `"United States"`, `"U.S.A."`) into dominant canonical entities.
* **Semi-Structured JSON / Struct Exploder (`--explode`):** Automatically detects and unrolls embedded JSON strings, dictionaries, and lists into top-level typed columns.
* **Wide-to-Long Matrix Unpivoter (`--unpivot`):** Detects wide financial reports (e.g. `[Jan_2025, Feb_2025, Mar_2025]`) and unpivots them into tidy `[Period, Value]` rows.
* **Mixed Units & Currency Normalizer (`--convert-units`):** Converts mixed unit strings (`"5 kg"`, `"11 lbs"`, `"5000 g"`) and currencies (`$`, `€`, `SAR`, `AED`, `£`) into standardized base numbers.
* **Statistical Outlier & Typo Guard (`--winsorize`):** Detects extreme human typos (e.g. Age = `999`, Price = `-$50,000`) and non-destructively clips them using IQR / percentile fences.
* **Automatic Data Type & Boolean Asserter (`--auto-type`):** Coerces boolean strings (`"true"`, `"yes"`, `"1"`), numeric strings, and timestamps with zero data loss.
* **Relational Multi-Table Auto-Linker (`--stitch`):** Detects foreign-key overlap across session DataFrames and performs star-schema joins in DuckDB/Polars.

### Data Science Skills
* **Hierarchical Unravelling Engine (`-u`, `--unravel`):** Deterministic state machines that parse ragged, multi-line, non-rectangular ERP ledger exports (SAP, Navision, Oracle) into normalized 2D DataFrames.
* **Zero-Copy DuckDB SQL Engine (`-s`, `--sql`):** High-speed analytical SQL execution directly on in-memory Pandas and Polars DataFrames via Apache Arrow memory pointers, featuring automated schema registration.
* **Enterprise ML Guardrails (`--validate`, `--tune`, `--explain`):** Enforces strict scikit-learn `Pipeline` encapsulation, leak-free `GridSearchCV`, metric assertions, and feature importance extractions.
* **Insight Synthesis (`-i`, `--insight`):** Automatically captures execution stdout and runs a secondary analytical pass to generate actionable business takeaways.
* **Custom Persona Modes (`--persona exec|dev`):** Switch insight synthesis between Executive Strategist (ROI, business impact) and Lead Data Engineer (schema anomalies, pipeline edge cases) personas.
* **Ensemble Intent Routing:** Automatically classifies zero-flag natural language prompts into specialized skill categories (SQL, visualization, statistics, ML, feature engineering) before code generation.

### Advanced Validation & Sandboxing
* **Logical Critic Verification (`--critic`, `--critic-pro`):** Before execution, a secondary LLM pass scans the generated code for logical flaws (incorrect grouping, wrong aggregation, join key mismatches). `--critic` runs locally; `--critic-pro` escalates to DeepSeek Reasoner.
* **Interactive Ghost Execution (`--preview`) & Visual State Diff HUD (`--diff`):** Clones the target DataFrame into an isolated shadow namespace, executes the generated code, and renders a side-by-side Rich delta HUD showing row/col count changes, data type mutations, null count drifts, and schema additions/removals before prompting interactive commit.
* **Automated Quality Gates (`--guard`):** Evaluates custom Python boolean expressions (e.g., `--guard "len(df) > 0 and df['price'].min() >= 0"`) against the resulting DataFrame. On violation, blocks commit, reverts state, and routes the failure into the auto-repair loop.
* **Adversarial Edge-Case Fuzzer (`--stress`):** Synthesizes a 5-row schema-matched adversarial matrix (NaN, empty strings, `$0.00`, zero-denominators) and pre-tests the generated code to catch crash-prone edge cases before execution.
* **Metamorphic Logic Validator (`--meta`):** Creates a 2x numerically scaled copy of the DataFrame, runs the generated code against both, and verifies proportional linear scaling invariance to detect hardcoded constants.
* **Sandboxed "What-If" Simulator (`--simulate`):** Forks the target DataFrame, injects a hypothesis scenario, renders a comparative Rich table, and immediately garbage-collects all shadow objects without polluting global state.
* **Inline Data Minimaps (`--spark`):** Renders 8-level ASCII sparkline distribution plots (` ▂▃▄▅▆▇█`) for numeric columns alongside Min, Median, Max, and Null % summaries.

### Workflow Orchestration
* **Autonomous 10-Stage Intelligence Lifecycle (`--EDA`):** Executes the full data analysis lifecycle (Ask → Prepare → Process → Profile → Engineer → Reason → Falsify → Project → Publish → Deploy) in a single autonomous Polars-powered run, with local privacy tokenization, causal root-cause backtracing, SVD VIF multicollinearity screening, 14-day conformal forecasting, dialectical debate, interactive HTML dashboards, Marp slide decks, SQL DDL, production `pipeline.py`, and automated continuous monitoring sentinel.
* **Global State Orchestrator (`--roadmap`):** Tracks project progress across 4 phases (Profiling & Cleaning → Goal Interview → Execution & Radar → Synthesis) in a persistent global state dictionary, rendering the next recommended `%deepanalyze` command.
* **Zero-Prompt Kickstart (`--kickstart`):** Sends workspace context to the LLM to autonomously infer business domain, identify target KPIs, and output a prioritized 3-step action plan.
* **Reverse-Prompting Interview (`--interview`):** The LLM generates 3 targeted multiple-choice analytical constraint questions. User choices are recorded as the project goal for downstream hypothesis generation.
* **Autonomous Hypothesis Generator (`--brainstorm`):** Reads the aligned project goal and dataset context to generate 3–5 specific, testable business hypotheses with exact executable `%deepanalyze` commands.
* **Proactive Anomaly Radar (`--radar`):** Runs automatically during execution to detect null surges (>20%), metric mean shifts (>35%), and sign flips in previously non-negative columns, rendering red alert panels.

### UI, Visuals & Notebook Automation
* **Live Transformation Flow Graph (`--dag`):** Parses the generated AST and renders a Rich tree showing step-by-step lineage from source DataFrame through filters, aggregations, and mutations to the final output target.
* **Interactive In-Notebook Data Explorer (`--gui`):** Injects an HTML/JS data table widget via `IPython.display.HTML` with sticky headers, live search, column sorting, and data type badges.
* **Visual Time-Machine Explorer (`--history`):** Displays a Rich table of all cached DataFrame snapshots with timestamps, dimensions, and column samples for easy rollback navigation.
* **Predictive Next-Action Recommender (`--next`):** Suggests 3 logical follow-up analytical actions with executable `%deepanalyze` commands after each execution.
* **Semantic Auto-Sanitizer (`--auto-clean`):** Autonomously detects formatting anomalies (currency symbols, dirty strings, wrong types) and generates a cleaning script, routing it through the `--preview` ghost execution flow for confirmation.
* **Notebook Artifact Spawner (`--spawn`):** Injects formatted Markdown narrative cells and validated Code cells directly below the active cell.
* **Transactional State Rollback (`--undo`):** Automated deepcopy (Pandas) or instant zero-copy clone (Polars) prior to execution, enabling instant state rollback and safe experimentation with zero RAM overhead.
* **Semantic Schema Dictionaries (`--context`):** Inject external business logic schemas (Markdown/JSON) into the LLM context for domain-aware code generation.

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
##  Enterprise Machine Learning Guardrails

DeepAnalyze enforces strict data science best practices through targeted skill flags, bridging the gap between raw code generation and production-ready ML:

*   **`--validate` (Rigorous Validation):** For ML tasks, the engine automatically implements cross-validation or stratified holdout splits, prints comprehensive metrics (Classification Reports, Confusion Matrices), and embeds programmatic `assert` statements to guarantee shape matching and mathematical integrity.
*   **`--tune` (Leak-Free Pipelines):** Prevents data leakage by encapsulating all imputers, scalers, and estimators inside strict scikit-learn `Pipeline` or `ColumnTransformer` objects, combined with `GridSearchCV` for isolated out-of-fold hyperparameter tuning.
*   **`--explain` (Model Interpretability):** Extracts feature importances (or coefficients), ranks them to provide transparent insights into model decision-making, and validates weight distributions.

---

##  Hardware & Syntax Optimizations

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

Install the core engine alongside the modern scientific computing stack:

```bash
pip install pandas polars pyarrow numpy scipy scikit-learn statsmodels duckdb fastexcel openpyxl xlsxwriter matplotlib seaborn openai httpx rich ipython
```

### 3. Install the DeepAnalyze Package
Clone this repository and install it as an editable Python package using pip. This automatically registers the core engine and privacy modules into your environment.

```
# Clone the repository
git clone [https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git](https://github.com/abdullah-binmadhi/Deepanalyzer-8b.git) ~/Desktop/deepanalyze
cd ~/Desktop/deepanalyze
```
# Install the package (Editable mode recommended)
```
pip install -e .
```
Launch ipython or start a Jupyter notebook. Load the extension by running the following command in your first cell:

```
%load_ext deepanalyze
```


---

## Usage Guide

DeepAnalyze operates directly on the variables currently loaded in your session. Load your raw data, then call the `%deepanalyze` magic command to trigger the agent.

```python
import pandas as pd
import polars as pl

# 1. Load the extension
%load_ext deepanalyze

# 2. Load an unstructured, multi-level Excel report (or a Polars DataFrame)
df = pd.read_excel("raw_hierarchical_invoice.xlsx", header=None)

# 3. Autonomous flattening and type-casting
%deepanalyze -x -u Restructure the raw invoice dataframe into normalized tabular records.
```

### Runtime Directives

The execution engine is controlled via CLI flags passed to the magic command, allowing you to define the strictness, target variable, and execution scope of the agent.

#### Core Execution & Routing

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Execute** | `-x`, `--exec` | Runs the verified AST directly into the active kernel namespace. |
| **Target Binding** | `--target <var>` | Designates the target DataFrame in session memory (defaults to `df`). |
| **Cloud Pro** | `--pro` | Routes prompt to `deepseek-chat` (DeepSeek-V4-Pro). |
| **Cloud Flash** | `--flash` | Routes prompt to the lighter, high-speed cloud model. |
| **Deep Reasoner** | `--think` | Routes prompt to `deepseek-reasoner` (R1) for Chain-of-Thought processing. |
| **Deterministic** | `-d`, `--deterministic` | Clamps generation temperature to `0.0` for repeatable output. |
| **Ultra Context** | `--ultra` | Expands token generation limits up to 4,096 tokens. |
| **Fast Profile** | `--fast` | Temperature `0.0`, max 1,000 tokens for rapid execution. |
| **Auto-Repair** | `--retries <n>` | Max automated runtime exception retry loops (default: 1). |
| **Continuation** | `-c`, `--continue` | Iterates on previous generated code using refinement instructions. |

#### Privacy & Security

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Privacy Mode** | `--privacy <mode>` | Enforces sanitization: `auto`, `mask` (ERP), `profile` (Stats), `mock` (PII), or `none`. |
| **Privacy Audit** | `--audit-only` | Displays the sanitized context payload without calling the LLM. |
| **Schema Context** | `--context <path>` | Injects external business logic schema (Markdown/JSON) into LLM context. |

#### Data Science Skills

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Unravel** | `-u`, `--unravel` | Hierarchical state-machine parsing for nested ERP ledger exports. |
| **Feature** | `-f`, `--feat` | Vectorized operations, safe casting, and in-place transformations. |
| **SQL Engine** | `-s`, `--sql` | Zero-copy DuckDB execution on in-memory DataFrames. |
| **Visualize** | `-v`, `--viz` | Themed Matplotlib/Seaborn visualization scripts. |
| **Statistical Test** | `-t`, `--stat` | Parametric/non-parametric hypothesis tests (ANOVA, t-test, Chi-square). |
| **Machine Learning** | `-m`, `--ml` | scikit-learn Pipeline-bundled preprocessing and estimators. |
| **Profile** | `-p`, `--profile` | Executive structural health audit and diagnostic sampling. |
| **Insight Synthesis** | `-i`, `--insight` | Captures execution stdout and generates business takeaways. |
| **Persona Mode** | `--persona <mode>` | Insight persona: `default` (analyst), `exec` (C-suite strategist), `dev` (data engineer). |
| **Save Charts** | `--save` | Saves generated plots to disk (`charts/<slug>.png`) at 300 DPI. |

#### ML Guardrails

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Validate** | `--validate` | Cross-validation, confusion matrices, and structural metric assertions. |
| **Tune** | `--tune` | Leak-free `Pipeline` + `GridSearchCV` hyperparameter optimization. |
| **Explain** | `--explain` | Feature importance extraction, ranking, and weight validation. |

#### Advanced Validation & Sandboxing

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Critic Loop** | `--critic` | Local logical critic verification loop before execution. |
| **Critic Pro** | `--critic-pro` | Cloud critic loop via DeepSeek Reasoner for deep logical verification. |
| **Ghost Preview** | `--preview` | Shadow execution with State Diff HUD and interactive commit/discard. |
| **State Diff HUD** | `--diff` | Renders side-by-side delta showing row/col, dtype, and null changes after execution. |
| **Quality Gate** | `--guard <expr>` | Evaluates boolean constraint; blocks commit and triggers repair on violation. |
| **Stress Fuzzer** | `--stress` | Pre-tests code against a 5-row adversarial edge-case matrix (NaN, zero-division). |
| **Metamorphic Check** | `--meta` | Validates code against 2x numerical perturbation for scaling invariance. |
| **What-If Simulator** | `--simulate <scenario>` | Sandboxed hypothesis simulation with comparative HUD, zero global mutation. |
| **Sparkline Minimaps** | `--spark` | ASCII distribution minimaps (` ▂▃▄▅▆▇█`) for numeric columns. |

#### Workflow Orchestration

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Autonomous Lifecycle** | `--EDA` | Autonomous 6-stage Data Analysis Lifecycle (Polars-native, local privacy, charts, monitoring). |
| **Target Goal** | `--goal <text>` | Explicit domain objective or KPI guidance for `--EDA` or `--roadmap`. |
| **Roadmap** | `--roadmap` | Multi-phase project orchestrator HUD with next-action recommendations. |
| **Kickstart** | `--kickstart` | Zero-prompt domain inference and prioritized 3-step action plan. |
| **Interview** | `--interview` | Stakeholder goal & constraint alignment via multiple-choice questions. |
| **Brainstorm** | `--brainstorm` | Autonomous hypothesis generator with executable `%deepanalyze` commands. |
| **Anomaly Radar** | `--radar` | Proactive anomaly scanning for null surges, metric shifts, and sign flips. |

#### Specialized Intelligence Engines

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Statistical Battery** | `--stats`, `-st` | Adaptive hypothesis testing battery, SVD-regularized VIF & non-linear driver ranking. |
| **Executive Storyteller** | `--story`, `-sm` | Synthesizes McKinsey Pyramid Principle executive briefing memo (`.html`/`.md`). |
| **Feature Forge** | `--engineer`, `-fe` | Autonomous leak-free feature engineering, cyclical temporal lags & interaction pruning. |
| **Autonomous Forecaster** | `--forecast`, `-fc` | Automatic cadence detection, STL decomposition & 80%/95% conformal prediction bands. |
| **Drift Sentinel** | `--drift`, `-dr` | Population Stability Index (PSI), Kolmogorov-Smirnov & schema evolution tracking. |
| **Schema Synthesizer** | `--schema`, `-sc` | Synthesizes DuckDB/PostgreSQL/Snowflake SQL DDL, dbt `schema.yml`, & Mermaid ER diagrams. |
| **Synthetic Data Generator** | `--synthetic`, `-sy` | Generates differentially private Gaussian Copula synthetic clone with zero PII leakage. |

#### V3.0 Revolutionary Analytical Capabilities

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Root-Cause Debugger** | `--why <cond>` | Isolates anomaly rows and runs factor variance decomposition to identify top causal drivers. |
| **Rule Distillation** | `--distill` | Distills invariant data rules from prompt history and persists to `.deepanalyze_memory.json`. |
| **Turbo SIMD Compiler** | `--turbo` | Transpiles row-wise lambdas, applies, and loops to pure Polars SIMD expressions (8.5x–45x speedup). |
| **Dialectical Debate** | `--debate` | Spawns concurrent Growth Bull vs Risk Auditor analysis in a split-screen Rich panel. |
| **Analytical Skeptic** | `--falsify` | Runs a 3-point counter-investigation (outlier concentration, return lag, cohort shift) before finalizing. |
| **Production ETL Compiler** | `--pipeline` | Distills session transformation lineage into a standalone, typed `pipeline.py` executable. |
| **Self-Contained Report** | `--report` | Bundles Base64 charts, KPI metrics, and sortable tables into a dark-mode interactive HTML file. |
| **Autonomous Data Fetcher** | `--enrich` | Vector-appends standard industry taxonomy, SEC SIC codes, and external dimensions. |
| **Semantic Vector Filter** | `--semantic <query>` | Natural language conceptual semantic filtering on text columns without rigid regex. |
| **Treatment Effect Engine**| `--causal` | Inverse Probability of Treatment Weighting (IPTW) estimating true Average Treatment Effect (ATE). |
| **Ensemble Feature Factory**| `--auto-feat ensemble` | High-dimensional feature discovery with orthogonal GBDT importance pruning to top 5 features. |
| **Adversarial Digital Twin**| `--twin adversarial` | Synthesizes 20% shifted adversarial stress-test datasets with zero real PII exposure. |
| **Cross-Lingual Weave** | `--weave <target>` | Fuzzy semantic join across languages and varying naming conventions via cosine similarity. |
| **Prescriptive Optimizer** | `--solve` | Formulates Linear/Quadratic programming optimization and outputs optimal allocation weights. |
| **Adaptive Schema Healing**| `--evolve` | Intercepts schema drift and auto-maps drifted column names in Polars transformation pipelines. |
| **Biomimetic RAG Brain** | `--brain` | Multi-phase institutional memory with geometric hashing, delta logging, and hardware OOM reflexes. |


#### UI, Visuals & Notebook Automation

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **DAG Graph** | `--dag` | Renders AST transformation lineage as a Rich tree. |
| **GUI Explorer** | `--gui` | Interactive in-notebook HTML data table with search, sort, and type badges. |
| **History** | `--history` | Visual time-machine table of DataFrame snapshot rollback points. |
| **Next Actions** | `--next` | Predictive 3-action recommender with executable commands. |
| **Auto-Clean** | `--auto-clean` | Autonomous data sanitizer routed through `--preview` ghost execution. |
| **Spawn Cells** | `--spawn` | Injects Markdown narrative + Code cells into notebook below current cell. |

#### Data Ingestion & Export

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Import Data** | `--import <path\|url\|clip>` | Ingest CSV, TSV, Parquet, IPC, Arrow, Excel, JSON/NDJSON, or clipboard into session DataFrame. |
| **Export Data** | `--export <var>` | Export session DataFrame to file or DuckDB database table. |
| **Export Destination** | `--to <path>` | Destination filepath for `--export` (defaults to `./<var>.parquet`). |
| **Excel Sheet** | `--sheet <name\|idx>` | Specify sheet name or index for Excel workbooks. |
| **Lazy Scan** | `--lazy` | Instantiate a Polars `pl.LazyFrame` instead of eager `pl.DataFrame` (CSV, Parquet, IPC). |

#### State Management & Convenience

| Directive | Flag | Behavior |
| :--- | :--- | :--- |
| **Revert State** | `--undo` | Restores target DataFrame to pre-execution deepcopy/clone snapshot. |
| **Auto-Pilot Toggle** | `--toggle` | Toggles cell interceptor on/off for plain-English auto-pilot execution. |
| **Engine Status** | `--status` | Probes server health, API keys, snapshots, and interceptor state. |
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

### 9. High-Speed Polars Analytics (Universal Adapter)

```python
import polars as pl

# Create large Polars DataFrame
sensor_pl = pl.DataFrame({
    'sensor_id': ['S1', 'S2', 'S1', 'S3', 'S2'],
    'reading': [' 12.5 ', 'N/A', ' 45.2 ', ' 0.0 ', ' 18.9 '],
    'flag': ['ok', 'error', 'ok', 'ok', 'error']
})

# Polars-native transformation
%deepanalyze -x -f --target sensor_pl Clean reading column into numeric float filling nulls with 0 and filter for flag == 'ok'.
```

---

### 10. Autonomous Workflow Orchestration

```python
# Phase 1: Zero-prompt analysis kickstart
%deepanalyze --kickstart

# Phase 2: Stakeholder goal alignment interview
%deepanalyze --interview

# Phase 3: Autonomous hypothesis generation
%deepanalyze --brainstorm

# Check roadmap progress at any time
%deepanalyze --roadmap
```

---

### 11. Ghost Preview, Quality Gates & Stress Testing

```python
# Preview changes before committing (interactive commit/discard)
%deepanalyze --preview --diff --spark -f --target sales_data Clean gross_revenue and standardize status.

# Enforce quality constraints with auto-repair
%deepanalyze -x --guard "len(df) > 0 and df['gross_revenue'].min() >= 0" -f --target sales_data Clean and validate revenue.

# Pre-test code against adversarial edge cases
%deepanalyze --stress -f --target sales_data Strip currency symbols and cast to numeric.

# Metamorphic invariance check
%deepanalyze --meta -f --target sales_data Compute net_revenue as gross_revenue * (1 - tax_rate).
```

---

### 12. Interactive Notebook Explorer & Time-Machine

```python
# Launch interactive in-notebook data table with search and sort
%deepanalyze --gui --target sales_data

# View all DataFrame snapshot rollback points
%deepanalyze --history

# Sandboxed What-If simulation
%deepanalyze --simulate "30% revenue decline" -f --target sales_data Apply revenue shock scenario.
```

---

### 13. Critic Verification & Persona Insights

```python
# Local logical critic loop before execution
%deepanalyze -x --critic -f --target sales_data Group by customer and sum revenue.

# Executive persona insight synthesis
%deepanalyze -x -i --persona exec --target sales_data Summarize top customers by total revenue.

# Dev/engineer persona insight synthesis
%deepanalyze -x -i --persona dev --target sales_data Profile null distributions and type anomalies.
```

---

### 14. Auto-Clean, DAG Visualization & Artifact Spawning

```python
# Autonomous data sanitization with interactive preview
%deepanalyze --auto-clean --target sales_data

# Render transformation lineage DAG after execution
%deepanalyze -x --dag -f --target sales_data Clean all columns and compute net revenue.

# Spawn Markdown narrative + Code cells into notebook
%deepanalyze -x --spawn -i --target sales_data Generate executive summary of cleaned data.

# Predictive next-action recommender
%deepanalyze -x --next -f --target sales_data Strip whitespace from all string columns.
```

---

### 15. Resilient Data Ingestion & Polyglot Exporter (`--import`, `--export`)

```python
# 1. Ingest CSV with auto-sanitized variable name (e.g. sales_2026_q1_df) and Rich telemetry
%deepanalyze --import "data/Sales 2026-Q1.csv"

# 2. Ingest specific sheet from Excel with target override
%deepanalyze --import ~/Downloads/financial_report.xlsx --sheet "Q4 Ledger" --target ledger_df

# 3. Ingest large Parquet file lazily as a Polars LazyFrame
%deepanalyze --import "s3://bucket/huge_telemetry.parquet" --lazy --target telemetry_lazy

# 4. Ingest raw tabular data directly from system clipboard
%deepanalyze --import clip --target clipboard_data

# 5. Export cleaned DataFrame to zstd-compressed Parquet with automatic directory creation
%deepanalyze --export sales_2026_q1_df --to "exports/curated/sales_clean.parquet"

# 6. Export directly to an embedded DuckDB database table
%deepanalyze --export ledger_df --to "analytics.duckdb:quarterly_ledger"
```

---

### 16. Universal Hierarchical Report & ERP Unraveller (`--unravel`, `-u`)

Transforms multi-level hierarchical accounting and ERP reports (Invoice Listings, General Ledgers, Purchase Orders, AR/AP Aging) into clean, flat 2D normalized datasets:

```python
# Autonomous unravelling of messy multi-row ERP report
%deepanalyze --unravel --target inv_listing_df

# Or end-to-end autonomous import, unravel, and 6-stage EDA
%deepanalyze --import "INV LISTING 31082025.xlsx" --EDA --goal "Find sequence, doc_no, item_amount and compute invoice totals"
```

---

### 17. Specialized Intelligence Engines

```python
# 1. Statistical Hypothesis Testing Battery & SVD VIF
%deepanalyze --stats --target sales_data

# 2. Executive Storyteller Memo (McKinsey Pyramid Principle HTML/MD briefing)
%deepanalyze --story --target sales_data

# 3. Autonomous Leak-Free Feature Engineering & Lags
%deepanalyze --engineer --target sales_data

# 4. Autonomous 14-Day Time-Series Forecast with Conformal Bounds
%deepanalyze --forecast --target sales_data

# 5. Data Drift Sentinel & Population Stability Index (PSI)
%deepanalyze --drift --target sales_data

# 6. SQL Schema DDL Synthesizer (DuckDB / PostgreSQL / Snowflake) & dbt Models
%deepanalyze --schema --target sales_data

# 7. Differentially Private Synthetic Data Generator (Gaussian Copula)
%deepanalyze --synthetic --target sales_data
```

## Attribution & Licensing

* **Base Architecture & Research:** [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
* **Training Corpus:** [RUC-DataLab/DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
* **License:** [MIT License](LICENSE)


# DeepAnalyze: Master Command Cheat Sheet & Quick Reference

A complete, practical reference guide for using DeepAnalyze (`%deepanalyze`) across data ingestion, messy data cleaning, autonomous exploratory data analysis, statistical testing, causal reasoning, feature engineering, and executive reporting.

---

## Quick Start Matrix

| Category | Flag / Command | Purpose | Example |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `--import <path>` | Smart ingestion (CSV, Parquet, Excel, Clipboard) | `%deepanalyze --import "sales.xlsx" --sheet "2026"` |
| **Ingestion** | `--lazy` | Out-of-core streaming LazyFrame for 500MB+ files | `%deepanalyze --import "big_data.csv" --lazy` |
| **Ingestion** | `--export <var>` | Atomic disk export (.parquet, .csv, .duckdb) | `%deepanalyze --export df --to "./clean.parquet"` |
| **Autonomous EDA**| `--EDA` | Full 10-stage autonomous intelligence lifecycle | `%deepanalyze --EDA --goal "Analyze margin drivers"` |
| **Interview** | `--interview` | Socratic clarifying questions before code generation | `%deepanalyze --interview "Segment top customers"` |
| **Messy Data** | `--unravel` | Unravel hierarchical ERP reports (SAP/Oracle) | `%deepanalyze --unravel --target erp_raw` |
| **Messy Data** | `--auto-type` | Auto-cast dirty currencies, dates, and numbers | `%deepanalyze --auto-type --target raw_df` |
| **Messy Data** | `--winsorize` | Clip 1st/99th percentile extreme data entry errors | `%deepanalyze --winsorize --target raw_df` |
| **Messy Data** | `--stitch` | Auto-join multi-table relational datasets | `%deepanalyze --stitch` |
| **Root Cause** | `--why <condition>` | AST root-cause debugger isolating variance anomalies | `%deepanalyze --why "gross_margin < 0" --target df` |
| **Causal ATE** | `--causal` | Treatment effect engine with IPTW propensity scores | `%deepanalyze --causal --target campaign_df` |
| **Statistics** | `--stats` | Automated hypothesis testing battery & SVD VIF | `%deepanalyze --stats --target sales_df` |
| **Forecasting** | `--forecast` | 14-day cadence forecast with 95% conformal bounds | `%deepanalyze --forecast --target sales_df` |
| **Debate** | `--debate` | Dialectical split: Growth Bull vs Risk Auditor | `%deepanalyze --debate --target revenue_df` |
| **Skepticism** | `--falsify` | 3-point counter-investigation stress battery | `%deepanalyze --falsify --target kpi_df` |
| **Auto Features** | `--auto-feat ensemble`| Discover & commit top predictive features | `%deepanalyze --auto-feat ensemble --target df` |
| **Synthetic** | `--synthetic` | Generate private Gaussian Copula clone | `%deepanalyze --synthetic --target client_df` |
| **Stress Twin** | `--twin adversarial` | Generate 20% shifted stress-test dataset | `%deepanalyze --twin adversarial --target df` |
| **Enrichment** | `--enrich industry` | Async taxonomy fetcher (SEC SIC, UNSPSC, NAICS) | `%deepanalyze --enrich industry --target companies`|
| **Semantic** | `--semantic <query>`| In-memory cosine vector semantic search | `%deepanalyze --semantic "broken display" --target logs`|
| **Optimization** | `--solve` | Prescriptive LP/QP mathematical resource solver | `%deepanalyze --solve --target budget_df` |
| **Transpile** | `--pipeline` | Transpile notebook transformations to `pipeline.py` | `%deepanalyze --pipeline --target clean_df` |
| **Executive Brief**| `--report` | Standalone interactive HTML executive brief | `%deepanalyze --report --target kpi_df` |
| **Presentations**| `--story` | Marp Markdown slide decks and executive memos | `%deepanalyze --story --target sales_df` |
| **Direct SQL** | `--sql "<query>"` | Zero-copy ANSI SQL query execution on Arrow buffers | `%deepanalyze --sql "SELECT dept, AVG(sal) FROM df GROUP BY dept"` |
| **Custom Model** | `--model <name>` | Explicit cloud model override (Claude, Gemini, GPT, Mythos) | `%deepanalyze --model claude-3-7-sonnet "Analyze churn"` |
| **Reasoning Effort**| `--effort <level>`| Control reasoning depth for thinking models (`low`, `medium`, `high`, `max`) | `%deepanalyze --think --effort high "Prove invariant"` |
| **Assertions** | `--assert` | Auto-generate & verify 2-3 runtime data invariants | `%deepanalyze --assert "Filter inactive users"` |
| **Statistical Drift**| `--diff-stats` | Kolmogorov-Smirnov distribution drift HUD | `%deepanalyze --diff-stats "Winsorize outliers"` |
| **Memory Vault** | `--vault`, `--memory` | Inspect 1,200+ stored schema blueprints & cache stats | `%deepanalyze --vault` |
| **Safety** | `--preview` | Ghost execution without mutating session state | `%deepanalyze --preview "Filter outliers"` |
| **Safety** | `--undo` | 5-level LIFO rollback to previous snapshots | `%deepanalyze --undo --target df` |
| **Server CLI** | `deepanalyze server`| Universal cross-platform server launcher | `deepanalyze server start --port 8080` |

---

## Autonomous One-Liner Execution (The 4-Pillar Gold Standard)

You **never need to memorize or select individual cleaning flags**. DeepAnalyze autonomously profiles, selects, and executes the optimal compiled cleaning routines under `%deepanalyze --EDA`.

```python
# 1. Ingest & Auto-Remedy any raw messy file (ERP invoices, clinical trials, logistics)
%deepanalyze --import "data/raw_report.xlsx" --EDA --target clean_df

# 2. Auto-Remedy any DataFrame currently in session
%deepanalyze --EDA --target raw_dataframe

# 3. Inspect Institutional Schema Memory Vault statistics & stored patterns
%deepanalyze --vault
```

## Detailed Command Workflows & Examples

### 1. Ingestion & Out-of-Core Streaming

#### Smart Ingestion
```python
# Import from local CSV, Parquet, or Excel
%deepanalyze --import "data/raw_invoices.xlsx" --sheet "Q1_2026" --target invoices_df

# Import directly from system clipboard
%deepanalyze --import "clip" --target copied_table

# Stream large 500MB+ dataset via Polars LazyFrame
%deepanalyze --import "data/big_transactions.parquet" --lazy --target stream_df
```

#### Safe Disk Export
```python
# Atomic export to Parquet (Zstandard compressed)
%deepanalyze --export invoices_df --to "exports/clean_invoices.parquet"

# Export to Excel-compatible CSV with UTF-8 BOM
%deepanalyze --export invoices_df --to "exports/report.csv"

# Ingest into local DuckDB table directly
%deepanalyze --export invoices_df --to "analytics.duckdb:invoices"
```

---

### 2. Autonomous Exploratory Data Analysis & Discovery

```python
# 10-Stage Autonomous Intelligence Engine with Causal, Forecasting, Slide Decks & Production Pipeline
%deepanalyze --EDA --goal "Identify quarterly churn patterns and margin drivers" --target customer_df

# Socratic interview mode (asks 3 clarifying questions before executing)
%deepanalyze --interview "Prepare customer data for lifetime value modeling" --target customer_df

# Autonomous hypothesis generator
%deepanalyze --brainstorm --target sales_df
```

---

### 3. Messy Real-World Data & ERP Reconstruction

```python
# Universal ERP Unraveller (strips report titles, print dates, and subtotal lines)
%deepanalyze --unravel --target sap_export_df

# Dirty currency & accounting string sanitizer ($1,250.50, (500.00) -> -500.00)
%deepanalyze --auto-type --target erp_df

# Outlier winsorization (clips extreme 1st/99th percentiles)
%deepanalyze --winsorize --target erp_df

# Relational table auto-stitcher (detects foreign keys and joins session DataFrames)
%deepanalyze --stitch
```

---

### 4. Advanced Statistics, Causal Reasoning & Forecasting

```python
# Causal Root-Cause Debugger: back-traces rows triggering margin erosion
%deepanalyze --why "gross_margin < 0" --target transactions_df

# Average Treatment Effect (ATE) with Propensity Score Weighting (IPTW)
%deepanalyze --causal --target campaign_df

# SVD Moore-Penrose Regularized VIF and Non-Parametric Hypothesis Battery
%deepanalyze --stats --target metrics_df

# 14-Day Cadence Forecast with 95% Conformal Prediction Intervals
%deepanalyze --forecast --target sales_df
```

---

### 5. Dialectical Perspectives & Adversarial Testing

```python
# Dialectical split: Growth Bull vs Risk Auditor
%deepanalyze --debate --target quarterly_kpis

# 3-Point Counter-Investigation (Falsification Battery)
%deepanalyze --falsify --target revenue_model

# Adversarial stress twin (simulates 20% macroeconomic shift)
%deepanalyze --twin adversarial --target financial_df

# Private Gaussian Copula synthetic twin (zero PII exfiltration)
%deepanalyze --synthetic --target client_records
```

---

### 6. Executive Storytelling, Slide Decks & Pipeline Deployment

```python
# Generate Marp Markdown slide decks and executive briefing memos
%deepanalyze --story --target sales_summary

# Compile session transformation DAG into production pipeline.py script
%deepanalyze --pipeline --target final_dataset

# Generate interactive HTML brief with embedded KPI cards and charts
%deepanalyze --report --target executive_kpis

# Generate Snowflake / BigQuery / DuckDB DDL with dbt validation tests
%deepanalyze --schema --target mart_orders
```

---

### 7. Safety, Rollback & Transaction Controls

```python
# Ghost Execution Preview: Inspect output diff without committing changes
%deepanalyze --preview "Drop all rows where revenue is null" --target df

# Atomic Undo: Instantly revert target dataset to previous state
%deepanalyze --undo --target df

# Quality Gate Assertion: Roll back and auto-repair if constraint fails
%deepanalyze --guard "df['revenue'].min() >= 0" "Clean invoice adjustments" --target df

# Inspect transformation DAG lineage and history
%deepanalyze --dag --target df
%deepanalyze --history
```

---

### 8. Direct SQL, Runtime Invariants & Server CLI

```python
# Direct ANSI SQL against active session DataFrames (DuckDB ↔ Polars zero-copy Arrow)
%deepanalyze --sql "SELECT region, SUM(revenue) AS total_rev FROM df GROUP BY region ORDER BY total_rev DESC" --target region_summary

# Self-Generated Runtime Invariant Assertions (verifies row & column invariants)
%deepanalyze --assert "Clean transaction amounts and normalize dates" --target df

# Time-Travel Kolmogorov-Smirnov Statistical Distribution Drift HUD
%deepanalyze --diff-stats "Winsorize extreme outliers" --target df

# Multi-Step Undo: Repeatedly step backward through prior dataset states
%deepanalyze --undo --target df
```

#### Universal Server Launcher (CLI)
```bash
# Start server with automatic hardware acceleration detection (macOS Metal / Linux CUDA / CPU)
deepanalyze server start

# Start server with speculative decoding (auto-detects Qwen-1.5B draft model)
deepanalyze server start --draft-model ./models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Check server health and latency
deepanalyze server status
```

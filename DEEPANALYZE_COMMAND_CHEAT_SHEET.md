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
| **Database DDL** | `--schema` | Multi-dialect SQL (Snowflake, BigQuery, DuckDB) | `%deepanalyze --schema --target data_df` |
| **Safety** | `--preview` | Ghost execution without mutating session state | `%deepanalyze --preview "Filter outliers"` |
| **Safety** | `--undo` | Roll back target DataFrame to previous snapshot | `%deepanalyze --undo --target df` |

---

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

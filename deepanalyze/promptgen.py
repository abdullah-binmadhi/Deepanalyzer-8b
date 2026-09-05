"""DeepAnalyze: Super-Intelligent Prompt Generation, Domain Engineering & Interactive Refinement Engine."""

import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from .brain import CognitiveBlackboard, DynamicResonanceEngine
from .policies import CompliancePolicy, classify_dataframe_columns, resolve_policy
from .profiler import SheetProfile, SheetRole, WorkbookTopology, profile_dataframe
from .sentinel import generate_synthetic_mock


def infer_domain_feature_engineering(
    df: pl.DataFrame,
    topology: Optional[WorkbookTopology] = None,
    arch_key: str = "CLEAN_TABULAR"
) -> List[str]:
    """Inspects column names and values to deduce domain-specific feature engineering rules."""
    features: List[str] = []
    col_names_lower = {c: c.lower() for c in df.columns}

    # 1. Tech / Smartphone / E-Commerce Specs
    if any(k in col_names_lower for k in ["ram", "storage", "rom", "battery", "camera", "processor", "display", "price"]):
        if "ram" in col_names_lower:
            features.append(
                "Memory & Storage Parsing: Use regex on `ram` to extract integer RAM capacity into `ram_gb` "
                "(e.g. r'(\\d+)\\s*GB RAM') and internal storage into `storage_gb` (e.g. r'(\\d+)\\s*GB inbuilt')."
            )
        if "battery" in col_names_lower:
            features.append(
                "Power Architecture: Use regex on `battery` to extract battery capacity into `battery_mah` "
                "(e.g. r'(\\d+)\\s*mAh') and fast charging power into `fast_charging_w` (e.g. r'(\\d+)\\s*W')."
            )
        if "camera" in col_names_lower:
            features.append(
                "Optics Specification: Parse primary rear camera megapixel into `primary_rear_camera_mp` "
                "and front selfie camera into `front_camera_mp`."
            )
        if "display" in col_names_lower:
            features.append(
                "Display Geometry: Extract screen size in inches into `screen_size_inches` "
                "and refresh rate into `refresh_rate_hz` (e.g. r'(\\d+)\\s*Hz')."
            )
        if "price" in col_names_lower:
            features.append(
                "Value Ratios: Strip currency symbols (`₹`, `$`, commas) from `price` and compute "
                "`price_per_gb_ram = price / ram_gb` and `price_per_mah = price / battery_mah`."
            )

    # 2. Healthcare & Clinical EHR
    if any(k in col_names_lower for k in ["blood_pressure", "bp", "cholesterol", "condition", "medication", "patient", "visit_date"]):
        bp_col = next((c for c, l in col_names_lower.items() if "blood" in l or l == "bp"), None)
        if bp_col:
            features.append(
                f"Biometric Splitting: Split `{bp_col}` string ('120/80') into numeric `systolic_bp` and `diastolic_bp`."
            )
        chol_col = next((c for c, l in col_names_lower.items() if "chol" in l), None)
        if chol_col:
            features.append(
                f"Clinical Categorization: Create `cholesterol_risk_tier` (<200: 'Normal', 200-239: 'Borderline', >=240: 'High')."
            )
        age_col = next((c for c, l in col_names_lower.items() if l == "age" or "patient_age" in l), None)
        if age_col:
            features.append(
                f"Demographic Cohort: Discretize `{age_col}` into cohorts `age_group` ('<18', '18-35', '36-50', '51-65', '65+')."
            )

    # 3. ERP & Invoicing & Accounting
    if arch_key == "ERP_RAGGED" or any(k in col_names_lower for k in ["qty", "quantity", "unit_price", "amount", "total", "doc_no", "invoice"]):
        features.append(
            "Financial Integrity Reconciliations: Verify `Calculated_Line_Gross = Quantity * Unit_Price` and "
            "compute `Line_Discount = Calculated_Line_Gross - Amount`. Add boolean flag `Reconciliation_Discrepancy_Flag`."
        )
        features.append(
            "Sequential Position: Add integer rank `Line_Item_Index` denoting item position within parent transaction."
        )

    # 4. Multi-Sheet Relational Topologies
    if topology and len(topology.sheets) > 1:
        for fk in topology.foreign_keys:
            features.append(
                f"Relational Enrichment: Merge sheet '{fk.to_sheet}' into primary sheet '{fk.from_sheet}' "
                f"using foreign key `{fk.from_col}` == `{fk.to_col}` ({fk.overlap_pct}% match)."
            )
        pivot_sheets = [s for s, p in topology.sheets.items() if p.role == SheetRole.PIVOT_TABLE]
        for ps in pivot_sheets:
            features.append(
                f"Dimensional Unpivot: Reshape wide monthly/quarterly columns in pivot sheet '{ps}' into normalized rows."
            )

    # 5. Temporal Features for Date Columns
    for col in df.columns:
        c_lower = col.lower()
        if "date" in c_lower or "time" in c_lower or "created" in c_lower or "visit" in c_lower:
            features.append(
                f"Temporal Intelligence: Parse `{col}` to datetime64[ns] and derive `{col}_day_name`, "
                f"`{col}_is_weekend`, and `{col}_quarter`."
            )
            break

    if not features:
        features.append("Standard Hygiene: Normalize all string columns, trim whitespace, and compute summary descriptive metrics.")

    return features


def build_master_prompt(
    df: pl.DataFrame,
    topology: Optional[WorkbookTopology] = None,
    policy: Optional[CompliancePolicy] = None,
    user_custom_instructions: str = "",
    target_df_name: str = "df",
    multi_sheets: Optional[Dict[str, pl.DataFrame]] = None,
    dataset_name: str = "dataset"
) -> str:
    """Synthesizes the complete industrial-grade prompt incorporating topology, anomalies, DP mocks, and custom user rules."""
    if policy is None:
        policy = resolve_policy()

    if topology is None:
        single_profile = profile_dataframe(df, name=target_df_name)
        topology = WorkbookTopology(
            file_path="",
            sheets={target_df_name: single_profile},
            primary_sheet=target_df_name,
            foreign_keys=[],
            recommended_pipeline_steps=[]
        )

    # Trigger 7-Brain Cognitive Resonance Hive Mind
    try:
        resonance_engine = DynamicResonanceEngine(df, filename=dataset_name)
        resonance_engine.think_and_synthesize()
        bb: Optional[CognitiveBlackboard] = resonance_engine.bb
    except Exception:
        bb = None

    sections: List[str] = []

    # Section 0: System Role & Objective
    sections.append("### SYSTEM ROLE & OBJECTIVE")
    sections.append(
        f"You are an expert Senior Data Engineer and Enterprise Systems Specialist. "
        f"Your objective is to write a deterministic, production-grade Python (Pandas/NumPy) cleaning "
        f"and feature engineering pipeline for the anonymized dataset `{dataset_name}`."
    )
    sections.append("\n---")

    # Hive Mind: Architectural Inspection (Internal Monologue)
    if bb and bb.internal_monologue:
        sections.append("\n### ARCHITECTURAL INSPECTION (INTERNAL MONOLOGUE)")
        for m in bb.internal_monologue:
            sections.append(f"* {m}")
        sections.append("\n---")

    # Section 1: Dataset Geometry & Masking Specification
    sections.append("\n### 1. DATASET GEOMETRY & COMPLIANCE MASKING SPECIFICATION")
    sheet_count = len(topology.sheets)
    if sheet_count > 1:
        sections.append(f"The source dataset is an enterprise multi-sheet workbook containing {sheet_count} distinct sheets:")
        for sname, prof in topology.sheets.items():
            sections.append(f"- Sheet '{sname}': {prof.row_count:,} rows x {prof.col_count} columns (Role: `{prof.role.value}`)")
    else:
        primary_prof = topology.sheets.get(topology.primary_sheet, next(iter(topology.sheets.values())))
        sections.append(
            f"The source dataset contains {primary_prof.row_count:,} rows x {primary_prof.col_count} columns "
            f"(Role: `{primary_prof.role.value}`)."
        )

    sections.append(
        f"\n**Privacy & Statutory Guarantees ({policy.statute_name}):**\n"
        "- Real PII, customer entities, and confidential records have been sanitized in local memory before prompt generation.\n"
        "- Direct identity columns are protected with surrogate tokens (`<NAME_1>`, `<ID_1>`).\n"
        "- Numeric metrics and distributions are perturbed with calibrated Laplace Differential Privacy ($\\epsilon = 1.0$), "
        "retaining genuine orders of magnitude and variance without exposing production numbers."
    )
    sections.append("\n---")

    # Section 2: Structural Report Topology & Column Anomalies
    sections.append("\n### 2. STRUCTURAL REPORT TOPOLOGY & FIELD ANOMALIES")
    if topology.foreign_keys:
        sections.append("**Relational Sheet Linkages:**")
        for fk in topology.foreign_keys:
            name_match_str = "Exact Column Match" if fk.is_name_match else "Inferred by Value Overlap"
            sections.append(f"- Join: `{fk.from_sheet}.{fk.from_col}` <-> `{fk.to_sheet}.{fk.to_col}` ({fk.overlap_pct}% key match, {name_match_str})")

    sections.append("\n**Detected Column Anomalies & Formatting Variances:**")
    anomaly_detected = False
    for sname, profile in topology.sheets.items():
        prefix = f"[{sname}] " if sheet_count > 1 else ""
        if profile.header_row_offset > 0:
            anomaly_detected = True
            sections.append(f"- {prefix}Top Metadata Offset: Skip first {profile.header_row_offset} rows to access true table headers.")
        if profile.subtotal_rows:
            anomaly_detected = True
            sections.append(f"- {prefix}Subtotal Rows: Exclude {len(profile.subtotal_rows)} summary rows at indices {profile.subtotal_rows[:5]}.")

        for col in profile.columns:
            anom_items: List[str] = []
            if len(col.date_formats) > 1:
                anom_items.append(f"Mixed date formats: {', '.join(col.date_formats)}")
            elif len(col.date_formats) == 1:
                anom_items.append(f"Date format: {col.date_formats[0]}")
            if col.has_accounting_negatives:
                anom_items.append("Accounting negative brackets `(1,000.00)`")
            if col.has_dirty_currency:
                anom_items.append("Contains non-standard currency symbols (e.g. `₹`, `$`, `SAR`)")
            if col.has_whitespace_issues:
                anom_items.append("Non-breaking or thin whitespace (`\\u2009`, `\\u00a0`, tabs)")
            if col.null_pct >= 25.0:
                anom_items.append(f"{col.null_pct}% null rate")

            if anom_items:
                anomaly_detected = True
                sections.append(f"- {prefix}`{col.name}` ({col.inferred_role}): {'; '.join(anom_items)}")

    if bb and bb.type_contaminations:
        anomaly_detected = True
        sections.append("\n**Forensic Pathology & Contamination Protocols:**")
        for tc in bb.type_contaminations:
            sections.append(f"- Column `{tc['col']}`: {tc['defect']} -> Action: {tc['action']}")

    if bb and bb.skewed_columns:
        anomaly_detected = True
        sections.append(f"- Heavy Distribution Skew: Columns {', '.join([f'`{c}`' for c in bb.skewed_columns[:5]])} exhibit high skewness (|skew| > 1.5).")

    if not anomaly_detected:
        sections.append("- Clean column structures detected with standard tabular alignment.")

    sections.append("\n---")

    # Section 3: Required Data Cleaning & Transformation Pipeline
    sections.append("\n### 3. REQUIRED DATA CLEANING & RECONCILIATION LOGIC")
    sections.append("Adhere strictly to the following transformation standards:")
    step_num = 1

    primary_prof = topology.sheets.get(topology.primary_sheet, next(iter(topology.sheets.values())))
    if primary_prof.header_row_offset > 0:
        sections.append(f"{step_num}. Skip top {primary_prof.header_row_offset} metadata rows and promote true column names.")
        step_num += 1

    sections.append(f"{step_num}. Standardize all column names to clean, consistent `snake_case`.")
    step_num += 1

    # Check for currency or accounting negatives
    has_curr = any(c.has_dirty_currency for p in topology.sheets.values() for c in p.columns)
    has_neg = any(c.has_accounting_negatives for p in topology.sheets.values() for c in p.columns)
    has_dates = any(len(c.date_formats) > 0 for p in topology.sheets.values() for c in p.columns)

    if has_curr or has_neg:
        neg_note = " and resolve accounting negatives `(X)` to `-X`" if has_neg else ""
        sections.append(f"{step_num}. Strip currency symbols (`₹`, `$`, `€`, `SAR`, commas){neg_note}, casting quantitative fields to `float64`.")
        step_num += 1

    if has_dates:
        sections.append(f"{step_num}. Standardize date columns to ISO format (`YYYY-MM-DD`) using `pd.to_datetime(..., format='mixed')`.")
        step_num += 1

    sections.append(f"{step_num}. Strip leading/trailing whitespaces and normalize thin spaces (`\\u2009`) or non-breaking spaces across all text fields.")
    step_num += 1

    if topology.foreign_keys:
        for fk in topology.foreign_keys:
            sections.append(f"{step_num}. Merge lookup dimension '{fk.to_sheet}' into primary ledger '{fk.from_sheet}' on join key `{fk.from_col}`.")
            step_num += 1

    if bb and bb.algebraic_laws:
        sections.append(f"{step_num}. Enforce mathematical invariant: {bb.algebraic_laws[0]}. Add boolean flag `reconciliation_anomaly_flag` for any non-conforming records.")
        step_num += 1

    sections.append(f"{step_num}. Remove duplicate records, filter out subtotal/grand total summary rows, and return the cleaned DataFrame.")
    sections.append("\n---")

    # Section 4: Domain Feature Engineering Specification
    arch_key = "ERP_RAGGED" if primary_prof.header_row_offset > 0 else "CLEAN_TABULAR"
    eng_features = infer_domain_feature_engineering(df, topology, arch_key)

    if bb and bb.engineered_features:
        for feat in bb.engineered_features:
            feat_text = f"{feat['feature']}: {feat['logic']}"
            if not any(feat['feature'].lower() in ef.lower() for ef in eng_features):
                eng_features.append(feat_text)

    sections.append("\n### 4. DOMAIN FEATURE ENGINEERING (AUTOMATIC SPEC EXTRACTIONS)")
    for idx, feat in enumerate(eng_features, 1):
        sections.append(f"{idx}. **{feat.split(':')[0]}:**{feat.split(':', 1)[1] if ':' in feat else feat}")
    sections.append("\n---")

    # Section 5: User Domain-Specific Business Logic & Custom Specifications
    sections.append("\n### 5. USER DOMAIN-SPECIFIC BUSINESS LOGIC & CUSTOM SPECIFICATIONS")
    if user_custom_instructions.strip():
        sections.append("The user has provided the following explicit domain rules and required calculations:")
        for line in user_custom_instructions.strip().split("\n"):
            line_clean = line.strip()
            if line_clean:
                if not line_clean.startswith(("-", "*", "•")):
                    line_clean = f"- {line_clean}"
                sections.append(line_clean)
    else:
        sections.append("Standard business intelligence rules apply. No supplementary custom constraints were specified.")
    sections.append("\n---")

    # Section 6: Synthetic Schema Mock (Laplace DP)
    sections.append("\n### 6. SYNTHETIC SCHEMA MOCK (Laplace DP, 0% Real Production Records)")
    sections.append("The following synthetic mock illustrates column structures and realistic numeric variance:")
    if multi_sheets and len(multi_sheets) > 1:
        mock_dict = {s: generate_synthetic_mock(sdf, n_rows=5) for s, sdf in multi_sheets.items()}
        mock_json = json.dumps(mock_dict, indent=2, default=str)
    else:
        mock_rows = generate_synthetic_mock(df, n_rows=5)
        mock_json = json.dumps(mock_rows, indent=2, default=str)
    sections.append(f"```json\n{mock_json}\n```")
    sections.append("\n---")

    # Section 7: Code Output & Security Constraints
    multi_sheet_ctx = ""
    if multi_sheets and len(multi_sheets) > 1:
        sheet_vars = [f"`df_{re.sub(r'[^a-zA-Z0-9_]', '_', s.lower()).strip('_')}`" for s in multi_sheets.keys()]
        multi_sheet_ctx = (
            f"Pre-loaded multi-sheet context in local memory:\n"
            f"- `sheets` dictionary: `{{'sheet_name': DataFrame, ...}}`\n"
            f"- Primary DataFrame: `{target_df_name}` (from sheet '{topology.primary_sheet}')\n"
            f"- Individual sheet DataFrames: {', '.join(sheet_vars)}\n"
        )

    sections.append("\n### 7. CODE OUTPUT & SECURITY CONSTRAINTS")
    sections.append(
        "1. Write clean, vectorized, self-contained Python code using **Pandas (`pd`)** and **NumPy (`np`)** (or Polars `pl`).\n"
        f"2. Assign the final cleaned and feature-engineered DataFrame to `{target_df_name}`.\n"
        "3. **AST Firewall Sandbox Restrictions:** Do NOT use network calls (`requests`, `socket`, `urllib`), "
        "system environment calls (`os.environ`), sensitive paths (`/etc/`, `~/.ssh/`), or timing sleep loops (`time.sleep`). "
        "Code violating these will be rejected by the local airlock firewall.\n"
        f"{multi_sheet_ctx}"
        "4. Wrap your executable script inside a single ```python code block."
    )

    return "\n".join(sections)


def save_prompt_to_disk(
    prompt_text: str,
    dataset_dir: str,
    dataset_base_name: str
) -> str:
    """Saves the finalized prompt to a markdown file on disk."""
    prompt_filename = f"{dataset_base_name}_cleaning_prompt.md"
    target_path = os.path.join(dataset_dir, prompt_filename)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return target_path


def enrich_prompt_with_local_model(
    prompt_text: str,
    server_url: str = "http://127.0.0.1:8080"
) -> str:
    """Optionally enriches prompt directives using local offline GGUF inference model if running."""
    import urllib.error
    import urllib.request

    test_url = f"{server_url}/health"
    try:
        req = urllib.request.Request(test_url, headers={"User-Agent": "DeepAnalyze-PromptGen"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status != 200:
                return prompt_text
    except Exception:
        # Local model not running; gracefully skip with zero delay
        return prompt_text

    # Query local model for extra feature engineering recommendations
    try:
        completion_url = f"{server_url}/v1/chat/completions"
        payload_data = json.dumps({
            "model": "deepanalyze-8b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert data engineer. Suggest 2 high-impact feature engineering additions for this dataset."
                },
                {"role": "user", "content": prompt_text[:2000]}
            ],
            "max_tokens": 200,
            "temperature": 0.2
        }).encode("utf-8")
        req = urllib.request.Request(completion_url, data=payload_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            suggestion = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if suggestion:
                prompt_text += f"\n\n### 8. LOCAL AI MODEL ADVISORY RECOMMENDATIONS\n{suggestion.strip()}\n"
    except Exception:
        pass

    return prompt_text


def interactive_prompt_editor(
    prompt_text: str,
    console: Console,
    dataset_name: str = "dataset"
) -> str:
    """Interactive review and multi-turn refinement loop for the synthesized prompt."""
    current_prompt = prompt_text

    while True:
        # 1. Render current prompt in styled syntax box
        console.print(Panel(
            Syntax(current_prompt, "markdown", theme="monokai", line_numbers=False),
            title=f"Synthesized Cloud Engineering Briefing: {dataset_name}",
            border_style="cyan"
        ))

        # 2. Ask if user wants to modify or add more
        modify = Prompt.ask(
            "\n[bold yellow]Would you like to modify or add instructions to this prompt? [y/N][/bold yellow]",
            default="N"
        )
        if not modify.lower().startswith("y"):
            break

        # 3. Present modification choices
        console.print("\n[bold]Select modification mode:[/bold]")
        console.print("  [1] Append custom instructions / business rules")
        console.print("  [2] Open full prompt in text editor ($EDITOR / nano / notepad)")
        console.print("  [3] Replace a specific phrase or section")
        choice = Prompt.ask("Select option [1-3]", default="1")

        if choice.strip() == "1":
            console.print("\n[bold]Enter additional instructions[/bold] (Paste text and type 'EOF' or press Ctrl+D when finished):")
            lines: List[str] = []
            while True:
                try:
                    line = input()
                    if line.strip() in ("EOF", "END"):
                        break
                    lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    break
            new_text = "\n".join(lines).strip()
            if new_text:
                custom_marker = "### 5. USER DOMAIN-SPECIFIC BUSINESS LOGIC & CUSTOM SPECIFICATIONS"
                if custom_marker in current_prompt:
                    parts = current_prompt.split(custom_marker, 1)
                    current_prompt = f"{parts[0]}{custom_marker}\n- {new_text}\n{parts[1]}"
                else:
                    current_prompt += f"\n\n### SUPPLEMENTARY CUSTOM INSTRUCTIONS\n{new_text}\n"
                console.print("[bold green]Added custom instructions to prompt.[/bold green]\n")

        elif choice.strip() == "2":
            # Launch external editor
            editor = os.environ.get("EDITOR") or ("nano" if os.name != "nt" else "notepad")
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(current_prompt)
                tmp_path = tmp.name

            try:
                subprocess.call([editor, tmp_path])
                with open(tmp_path, "r", encoding="utf-8") as f:
                    current_prompt = f.read()
                console.print("[bold green]Loaded modified prompt from editor.[/bold green]\n")
            except Exception as err:
                console.print(f"[bold red]Editor launch failed:[/bold red] {err}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif choice.strip() == "3":
            target_phrase = Prompt.ask("Enter the exact text to replace")
            replacement_phrase = Prompt.ask("Enter the new text")
            if target_phrase in current_prompt:
                current_prompt = current_prompt.replace(target_phrase, replacement_phrase)
                console.print("[bold green]Replaced text successfully.[/bold green]\n")
            else:
                console.print(f"[bold red]Phrase not found in prompt.[/bold red]\n")

    return current_prompt

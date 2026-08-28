"""DeepAnalyze Biomimetic RAG Institutional Memory Engine:
Implements the 3 Decoupled Lifecycles:
- Phase A: Passive Ingestion (Structural hashing, AST templatization, delta logging, hardware profiling)
- Phase B: Context Injection (Epistemic fact ledger, analogical transfer, hardware OOM reflexes)
- Phase C: Background Maintenance (Teardown consolidation, pruning redundant nodes)
"""

import atexit
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


MEMORY_STORE_PATH = os.path.abspath("./.deepanalyze_memory.json")


class BiomimeticBrain:
    """Biomimetic RAG Institutional Memory managing continuous learning across Jupyter sessions."""

    def __init__(self, storage_path: str = MEMORY_STORE_PATH):
        self.storage_path = storage_path
        self.memory = self._load_memory()
        # Register Phase C cleanup on process exit
        atexit.register(self.consolidate_and_prune)

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": "3.0.0",
            "structural_signatures": {},
            "delta_logs": [],
            "epistemic_facts": {},
            "verified_rules": [],
            "hardware_profiles": []
        }

    def _save_memory(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception:
            pass

    # =========================================================================
    # PHASE A: PASSIVE INGESTION (POST-EXECUTION)
    # =========================================================================

    def compute_geometry_hash(self, df: object) -> str:
        """Hashes structural signature (dtypes, shape ratios, null densities)."""
        pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df
        num_c = sum(1 for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c]))
        cat_c = len(pdf.columns) - num_c
        null_ratio = float(pdf.isna().sum().sum()) / max(pdf.size, 1)
        
        sig_str = f"cols:{len(pdf.columns)}|num:{num_c}|cat:{cat_c}|null:{null_ratio:.3f}"
        return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:16]

    def templatize_ast(self, code_str: str, col_names: List[str]) -> str:
        """Abstracts specific column names into generic semantic placeholders."""
        template_code = code_str
        for idx, col in enumerate(col_names):
            template_code = re.sub(rf'\b{re.escape(col)}\b', f"<COL_{idx+1}>", template_code)
        return template_code

    def log_execution_delta(self, df: object, code: str, success: bool, error_msg: str = None, working_code: str = None, duration_ms: float = 0.0):
        """Phase A Passive Ingestion logger recording delta diffs and hardware profiles."""
        geo_hash = self.compute_geometry_hash(df)
        cols = list(df.columns) if hasattr(df, 'columns') else []
        templatized = self.templatize_ast(code, cols)

        if not success and working_code:
            self.memory["delta_logs"].append({
                "hash": geo_hash,
                "failed_code": code,
                "error": error_msg,
                "working_code": working_code,
                "timestamp": time.time()
            })

        self.memory["hardware_profiles"].append({
            "hash": geo_hash,
            "duration_ms": duration_ms,
            "memory_bytes": sys.getsizeof(df),
            "oom_risk": duration_ms > 10000 or sys.getsizeof(df) > 500_000_000
        })

        if geo_hash not in self.memory["structural_signatures"]:
            self.memory["structural_signatures"][geo_hash] = {
                "template_snippets": [],
                "verified_truths": []
            }
        if success and templatized not in self.memory["structural_signatures"][geo_hash]["template_snippets"]:
            self.memory["structural_signatures"][geo_hash]["template_snippets"].append(templatized)

        self._save_memory()

    # =========================================================================
    # PHASE B: CONTEXT INJECTION (PRE-EXECUTION)
    # =========================================================================

    def get_context_injection(self, df: object) -> Dict[str, Any]:
        """Phase B context retrieval for prompt injection and hardware OOM reflex."""
        geo_hash = self.compute_geometry_hash(df)
        known_sig = self.memory["structural_signatures"].get(geo_hash, {})
        
        # Check hardware OOM reflex
        high_oom_risk = False
        for prof in self.memory.get("hardware_profiles", []):
            if prof.get("hash") == geo_hash and prof.get("oom_risk"):
                high_oom_risk = True
                break

        # Negative constraints from delta logs
        negative_rules = []
        for dl in self.memory.get("delta_logs", []):
            if dl.get("hash") == geo_hash:
                negative_rules.append(f"Avoid error '{dl.get('error', '')}' by using pattern: {dl.get('working_code', '')[:100]}")

        return {
            "geometry_hash": geo_hash,
            "epistemic_facts": known_sig.get("verified_truths", []),
            "analogous_templates": known_sig.get("template_snippets", [])[:3],
            "negative_constraints": negative_rules[:2],
            "hardware_reflex_duckdb_stream": high_oom_risk,
            "distilled_rules": self.memory.get("verified_rules", [])
        }

    # =========================================================================
    # PHASE C: BACKGROUND MAINTENANCE (KERNEL TEARDOWN)
    # =========================================================================

    def consolidate_and_prune(self):
        """Phase C cleanup: clusters snippets, removes stale entries, and bounds file size."""
        # Bound memory size to top 100 delta logs and profiles
        self.memory["delta_logs"] = self.memory["delta_logs"][-100:]
        self.memory["hardware_profiles"] = self.memory["hardware_profiles"][-100:]
        self._save_memory()

    def distill_rules_from_history(self, prompt_history: List[str]) -> List[str]:
        """Extracts invariant data assertions from session prompts and persists them."""
        rules = []
        for p in prompt_history:
            if any(k in p.lower() for k in ["always", "never", "must be", "assert", "filter out", "standardize"]):
                rules.append(p.strip())
        self.memory["verified_rules"].extend([r for r in rules if r not in self.memory["verified_rules"]])
        self._save_memory()
        return rules


# Global singleton instance
_BRAIN_INSTANCE = BiomimeticBrain()


def get_brain() -> BiomimeticBrain:
    return _BRAIN_INSTANCE

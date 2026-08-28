"""DeepAnalyze Biomimetic RAG Institutional Memory Engine:
Implements the 3 Decoupled Lifecycles:
- Phase A: Passive Ingestion (Structural hashing, AST templatization, delta logging, hardware profiling)
- Phase B: Context Injection (Epistemic fact ledger, analogical transfer, hardware OOM reflexes)
- Phase C: Background Maintenance (Teardown consolidation, pruning redundant nodes)
"""

import atexit
import hashlib
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

try:
    import orjson
    def _json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY, default=str).decode("utf-8")
    def _json_loads(data: Any) -> Any:
        return orjson.loads(data)
except ImportError:
    import json
    def _json_dumps(obj: Any) -> str:
        return json.dumps(obj, indent=2, default=str)
    def _json_loads(data: Any) -> Any:
        return json.loads(data)

try:
    import polars as pl
except ImportError:
    pl = None


def _resolve_memory_store_path() -> str:
    """Resolves cross-platform memory store path safely."""
    try:
        home_path = Path.home() / ".deepanalyze_memory.json"
        # Test writability
        if not home_path.exists():
            home_path.touch(exist_ok=True)
        return str(home_path)
    except Exception:
        return os.path.abspath("./.deepanalyze_memory.json")


MEMORY_STORE_PATH = _resolve_memory_store_path()


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
                with open(self.storage_path, "rb") as f:
                    content = f.read()
                    if content:
                        data = _json_loads(content)
                        if "query_cache" not in data:
                            data["query_cache"] = {}
                        return data
            except Exception:
                pass
        return {
            "version": "3.0.0",
            "structural_signatures": {},
            "delta_logs": [],
            "epistemic_facts": {},
            "verified_rules": [],
            "hardware_profiles": [],
            "query_cache": {}
        }

    def _save_memory(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                f.write(_json_dumps(self.memory))
        except Exception:
            pass

    # =========================================================================
    # STRUCTURAL QUERY CACHE
    # =========================================================================

    def compute_query_hash(self, prompt: str, df: object) -> str:
        """Computes deterministic hash from clean prompt and dataset geometry."""
        geo_hash = self.compute_geometry_hash(df) if df is not None else "no_df"
        clean_p = re.sub(r"\s+", " ", prompt.strip().lower())
        sig = f"{clean_p}::{geo_hash}"
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:20]

    def get_cached_query(self, prompt: str, df: object) -> Optional[str]:
        """Retrieves cached verified AST code if present."""
        q_hash = self.compute_query_hash(prompt, df)
        cached = self.memory.get("query_cache", {}).get(q_hash)
        if cached and isinstance(cached, dict) and "code" in cached:
            return cached["code"]
        return None

    def cache_verified_query(self, prompt: str, df: object, code: str):
        """Caches verified AST code mapped to prompt and dataset geometry."""
        if not prompt or not code:
            return
        q_hash = self.compute_query_hash(prompt, df)
        if "query_cache" not in self.memory:
            self.memory["query_cache"] = {}
        self.memory["query_cache"][q_hash] = {
            "prompt": prompt.strip(),
            "code": code.strip(),
            "timestamp": time.time()
        }
        # Keep cache bounded to top 200 items
        if len(self.memory["query_cache"]) > 200:
            oldest_key = min(self.memory["query_cache"].keys(), key=lambda k: self.memory["query_cache"][k].get("timestamp", 0))
            self.memory["query_cache"].pop(oldest_key, None)
        self._save_memory()

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

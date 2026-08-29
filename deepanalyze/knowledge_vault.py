"""DeepAnalyze Knowledge Vault Engine
Ingests, indexes, and performs sub-millisecond retrieval on 500,000+ data science
instruction trajectories from the RUC DataScience-Instruct corpus using embedded DuckDB with FTS BM25.
"""

import glob
import os
import re
from typing import Any, Callable

try:
    import orjson
except ImportError:
    import json as orjson

try:
    import duckdb
except ImportError:
    duckdb = None

VAULT_DB_PATH = os.path.abspath(".deepanalyze_vault.duckdb")


class KnowledgeVault:
    """Embedded High-Performance Knowledge Vault with DuckDB BM25 Full-Text Search."""

    def __init__(self, db_path: str = VAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        if duckdb is None:
            return
        con = self.get_connection()
        if con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_recipes (
                    task_id VARCHAR PRIMARY KEY,
                    source_file VARCHAR,
                    category VARCHAR,
                    instruction VARCHAR,
                    data_context VARCHAR,
                    thought_chain VARCHAR,
                    code_solution VARCHAR,
                    tokens_count INTEGER
                )
            """)

    def get_connection(self):
        """Returns a cached DuckDB connection."""
        if duckdb is None:
            return None
        if not hasattr(self, "_con") or self._con is None:
            try:
                self._con = duckdb.connect(self.db_path)
                try:
                    self._con.execute("LOAD fts;")
                except Exception:
                    pass
            except Exception:
                self._con = None
        return self._con

    @staticmethod
    def _categorize_source_file(filename: str) -> str:
        name_lower = filename.lower()
        if "clean" in name_lower:
            return "cleaning"
        elif "xlsx" in name_lower or "excel" in name_lower:
            return "excel_unravel"
        elif "pipeline" in name_lower:
            return "pipeline"
        elif "insight" in name_lower:
            return "insights"
        elif "database" in name_lower or "db" in name_lower or "sql" in name_lower:
            return "database_sql"
        elif "report" in name_lower or "story" in name_lower:
            return "report_generation"
        elif "prep" in name_lower:
            return "preparation"
        elif "tableqa" in name_lower or "tablegpt" in name_lower:
            return "table_qa"
        return "general_datascience"

    @staticmethod
    def _extract_parts_from_messages(messages: list[dict]) -> tuple[str, str, str, str]:
        """Extracts instruction, data context, thought chain, and code solution from messages."""
        instruction = ""
        data_context = ""
        thought_chain = ""
        code_solution = ""

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                if "# Instruction" in content and "# Data" in content:
                    parts = content.split("# Data", 1)
                    instruction = parts[0].replace("# Instruction", "").strip()
                    data_context = parts[1].strip()
                elif "# Instruction" in content:
                    instruction = content.replace("# Instruction", "").strip()
                else:
                    instruction = content.strip()
            elif role == "assistant":
                if "<Analyze>" in content and "</Analyze>" in content:
                    thought_match = re.search(r"<Analyze>(.*?)</Analyze>", content, re.DOTALL)
                    if thought_match:
                        thought_chain = thought_match.group(1).strip()
                
                code_blocks = re.findall(r"```(?:python|sql|bash)?\n(.*?)```", content, re.DOTALL)
                if code_blocks:
                    code_solution = "\n\n".join(code_blocks).strip()
                elif "</Analyze>" in content:
                    code_solution = content.split("</Analyze>", 1)[1].strip()
                else:
                    code_solution = content.strip()

        return instruction, data_context, thought_chain, code_solution

    def build_vault_from_directory(self, directory_path: str, max_files: int = None, chunk_size: int = 1000, progress_cb: Callable = None) -> int:
        """Indexes all JSON files in the specified directory into DuckDB with FTS index."""
        if duckdb is None:
            return 0

        files = sorted(glob.glob(os.path.join(directory_path, "*.json")))
        if max_files:
            files = files[:max_files]

        con = self.get_connection()
        total_indexed = 0

        for file_idx, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            category = self._categorize_source_file(fname)
            if progress_cb:
                progress_cb(f"Processing ({file_idx+1}/{len(files)}): {fname} [{category}]")

            try:
                with open(fpath, "rb") as fp:
                    raw_data = orjson.loads(fp.read())

                if not isinstance(raw_data, list):
                    continue

                batch_rows = []
                for item_idx, item in enumerate(raw_data):
                    if not isinstance(item, dict):
                        continue

                    task_id = f"{fname}_{item.get('id', item_idx)}"
                    messages = item.get("messages", [])
                    if not messages:
                        continue

                    instruction, data_ctx, thought, code = self._extract_parts_from_messages(messages)
                    tokens_cnt = item.get("total_tokens", 0)

                    batch_rows.append((
                        task_id,
                        fname,
                        category,
                        instruction,
                        data_ctx,
                        thought,
                        code,
                        tokens_cnt
                    ))

                    if len(batch_rows) >= chunk_size:
                        con.executemany("""
                            INSERT OR REPLACE INTO knowledge_recipes 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, batch_rows)
                        total_indexed += len(batch_rows)
                        batch_rows = []

                if batch_rows:
                    con.executemany("""
                        INSERT OR REPLACE INTO knowledge_recipes 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch_rows)
                    total_indexed += len(batch_rows)

            except Exception as e:
                if progress_cb:
                    progress_cb(f"⚠ Warning: Failed to parse {fname}: {e}")

        # Build DuckDB FTS Index for instantaneous BM25 retrieval
        try:
            if progress_cb:
                progress_cb("Building DuckDB FTS BM25 Full-Text Index...")
            con.execute("INSTALL fts; LOAD fts;")
            con.execute("PRAGMA create_fts_index('knowledge_recipes', 'task_id', 'instruction', 'thought_chain', overwrite=1);")
        except Exception as e:
            if progress_cb:
                progress_cb(f"⚠ FTS Index Note: {e}")

        con.close()
        return total_indexed

    def search_recipes(self, query_text: str, category: str = None, limit: int = 2) -> list[dict]:
        """Performs sub-millisecond DuckDB BM25 full-text search over 500,000+ data science recipes."""
        if duckdb is None or not os.path.exists(self.db_path):
            return []

        # Sanitize query keywords
        clean_words = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2 and w not in ("the", "and", "for", "with", "this", "from", "dataset", "data", "table")]
        if not clean_words:
            return []

        con = self.get_connection()
        if con is None:
            return []

        recipes = []
        try:
            # 1. Try DuckDB FTS BM25 search
            try:
                fts_query = " ".join(clean_words[:5]).replace("'", "")
                cat_clause = f"AND category = '{category}'" if category else ""
                sql_fts = f"""
                    SELECT task_id, source_file, category, instruction, thought_chain, code_solution, score
                    FROM (
                        SELECT *, fts_main_knowledge_recipes.match_bm25(task_id, '{fts_query}') AS score
                        FROM knowledge_recipes
                    )
                    WHERE score IS NOT NULL {cat_clause}
                    ORDER BY score DESC
                    LIMIT {limit}
                """
                results = con.execute(sql_fts).fetchall()
                for r in results:
                    recipes.append({
                        "task_id": r[0],
                        "source_file": r[1],
                        "category": r[2],
                        "instruction": r[3],
                        "thought_chain": r[4],
                        "code_solution": r[5],
                        "score": r[6]
                    })
                if recipes:
                    return recipes
            except Exception:
                pass

            # 2. Fallback SQL LIKE Priority Search
            like_clauses = " OR ".join([f"LOWER(instruction) LIKE '%{w}%' OR LOWER(thought_chain) LIKE '%{w}%'" for w in clean_words[:5]])
            cat_clause = f"AND category = '{category}'" if category else ""

            query = f"""
                SELECT task_id, source_file, category, instruction, thought_chain, code_solution
                FROM knowledge_recipes
                WHERE ({like_clauses}) {cat_clause}
                ORDER BY (
                    {' + '.join([f"(CASE WHEN LOWER(instruction) LIKE '%{w}%' THEN 3 ELSE 0 END + CASE WHEN LOWER(thought_chain) LIKE '%{w}%' THEN 1 ELSE 0 END)" for w in clean_words[:5]])}
                ) DESC
                LIMIT {limit}
            """

            results = con.execute(query).fetchall()
            for r in results:
                recipes.append({
                    "task_id": r[0],
                    "source_file": r[1],
                    "category": r[2],
                    "instruction": r[3],
                    "thought_chain": r[4],
                    "code_solution": r[5],
                    "score": 1.0
                })
            return recipes
        except Exception:
            return []

    def get_vault_stats(self) -> dict:
        """Returns statistics on the knowledge vault database."""
        if duckdb is None or not os.path.exists(self.db_path):
            return {"total_recipes": 0, "categories": {}, "db_path": self.db_path}

        con = self.get_connection()
        try:
            total = con.execute("SELECT COUNT(*) FROM knowledge_recipes").fetchone()[0]
            cat_breakdown = con.execute("SELECT category, COUNT(*) FROM knowledge_recipes GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
            categories = {c[0]: c[1] for c in cat_breakdown}
            db_size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0.0

            return {
                "total_recipes": total,
                "categories": categories,
                "db_size_mb": db_size_mb,
                "db_path": self.db_path
            }
        except Exception:
            return {"total_recipes": 0, "categories": {}, "db_path": self.db_path}


_KNOWLEDGE_VAULT_INSTANCE = None

def get_knowledge_vault() -> KnowledgeVault:
    global _KNOWLEDGE_VAULT_INSTANCE
    if _KNOWLEDGE_VAULT_INSTANCE is None:
        _KNOWLEDGE_VAULT_INSTANCE = KnowledgeVault()
    return _KNOWLEDGE_VAULT_INSTANCE

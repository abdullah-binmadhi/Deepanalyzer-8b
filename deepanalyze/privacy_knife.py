import ast
import json
import re
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

class DeepAnalyzePrivacyKnife:
    """Core privacy masking, structural synthesis, and AST security sandbox."""

    FORBIDDEN_MODULES = {
        "socket", "urllib", "requests", "httpx", "http", "subprocess",
        "shutil", "ftplib", "smtplib", "telnetlib", "asyncio"
    }
    FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
    FORBIDDEN_OS_ATTRS = {"system", "popen", "spawn", "fork", "kill", "remove", "unlink", "rmdir"}

    def __init__(self, df):
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
            self.engine = "pandas"
        elif pl is not None and isinstance(df, pl.DataFrame):
            self.df = df.clone()
            self.engine = "polars"
        else:
            raise TypeError("DeepAnalyzePrivacyKnife requires a Pandas or Polars DataFrame.")

    def mask_structural_erp(self):
        """Masks ERP text and numbers while preserving structural delimiters and keywords."""
        structural_keywords = {
            "Doc. No", "Doc No", "Doc. Date", "Doc Date", "Customer", "Seq",
            "Item Code", "Description", "Qty", "UOM", "Unit Price", "Total",
            "Grand Total", "Date", "Document", "Company", "GL Code", " : ", ":"
        }

        def _mask_string(s: str) -> str:
            if s.strip() in structural_keywords:
                return s
            res = []
            for char in s:
                if char.isupper(): res.append('X')
                elif char.islower(): res.append('x')
                elif char.isdigit(): res.append('9')
                else: res.append(char)
            return "".join(res)

        if self.engine == "pandas":
            def mask_cell(val):
                if pd.isna(val) or val is None: return None
                return _mask_string(str(val))
            return self.df.map(mask_cell) if hasattr(self.df, "map") else self.df.applymap(mask_cell)
        
        elif self.engine == "polars":
            # Very slow in Polars, but ERP masking runs on <20 rows anyway
            rows = self.df.to_dicts()
            masked_rows = []
            for row_dict in rows:
                new_row = {}
                for k, v in row_dict.items():
                    if v is None: new_row[k] = None
                    else: new_row[k] = _mask_string(str(v))
                masked_rows.append(new_row)
            return pl.DataFrame(masked_rows)

    def tokenize_pii_columns(self, pii_cols: list):
        """Replaces sensitive PII values with de-identified positional tokens."""
        if self.engine == "pandas":
            df_copy = self.df.copy()
            for col in pii_cols:
                if col in df_copy.columns:
                    clean_tag = re.sub(r'[^A-Za-z0-9]', '_', str(col)).upper()
                    df_copy[col] = [
                        f"[{clean_tag}_{i+1}]" if pd.notna(v) else v 
                        for i, v in enumerate(df_copy[col])
                    ]
            return df_copy
            
        elif self.engine == "polars":
            df_copy = self.df.clone()
            for col in pii_cols:
                if col in df_copy.columns:
                    clean_tag = re.sub(r'[^A-Za-z0-9]', '_', str(col)).upper()
                    df_copy = df_copy.with_columns(
                        pl.when(pl.col(col).is_not_null())
                          .then(pl.lit(f"[{clean_tag}_") + pl.int_range(1, pl.len() + 1).cast(pl.String) + pl.lit("]"))
                          .otherwise(pl.col(col))
                          .alias(col)
                    )
            return df_copy

    def generate_synthetic_toy(self, safe_df=None, n_rows: int = 5) -> list:
        """Directly slices target DataFrame to guarantee uniform column lengths."""
        target = safe_df if safe_df is not None else self.df
        
        if self.engine == "pandas":
            sample_len = min(len(target), n_rows)
            if sample_len == 0: return []
            sample_df = target.head(sample_len).copy()
            sample_df = sample_df.where(pd.notna(sample_df), None)
            return sample_df.to_dict(orient='records')
            
        elif self.engine == "polars":
            sample_len = min(target.height, n_rows)
            if sample_len == 0: return []
            return target.head(sample_len).to_dicts()

    def get_data_profile(self) -> dict:
        """Extracts statistical metadata and null percentages."""
        if self.engine == "pandas":
            profile = {
                "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
                "columns": {}
            }
            for col in self.df.columns:
                col_name = str(col)
                series = self.df[col]
                profile["columns"][col_name] = {
                    "dtype": str(series.dtype),
                    "null_count": int(series.isna().sum()),
                    "unique_count": int(series.nunique(dropna=True))
                }
            return profile
            
        elif self.engine == "polars":
            profile = {
                "shape": {"rows": self.df.height, "columns": self.df.width},
                "columns": {}
            }
            null_counts = self.df.null_count().row(0)
            for idx, col in enumerate(self.df.columns):
                profile["columns"][col] = {
                    "dtype": str(self.df.schema[col]),
                    "null_count": int(null_counts[idx]),
                    "unique_count": self.df[col].n_unique()
                }
            return profile

    @staticmethod
    def audit_generated_code(code_str: str) -> bool:
        """Static AST audit to prevent network exfiltration or OS tampering."""
        if not code_str.strip():
            return True
        try:
            tree = ast.parse(code_str)
        except Exception as e:
            raise SyntaxError(f"AST parse error: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split('.')[0]
                    if root_pkg in DeepAnalyzePrivacyKnife.FORBIDDEN_MODULES:
                        raise PermissionError(f"AST Sandbox Security Violation: Forbidden import `{alias.name}`")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split('.')[0]
                    if root_pkg in DeepAnalyzePrivacyKnife.FORBIDDEN_MODULES:
                        raise PermissionError(f"AST Sandbox Security Violation: Forbidden from-import `{node.module}`")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in DeepAnalyzePrivacyKnife.FORBIDDEN_CALLS:
                    raise PermissionError(f"AST Sandbox Security Violation: Forbidden call `{node.func.id}()`")
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                        if node.func.attr in DeepAnalyzePrivacyKnife.FORBIDDEN_OS_ATTRS:
                            raise PermissionError(f"AST Sandbox Security Violation: Forbidden OS operation `os.{node.func.attr}()`")
        return True


class LocalGatekeeper:
    """Heuristic classifier routing data to appropriate privacy strategies."""

    @classmethod
    def inspect(cls, df) -> dict:
        is_pandas = isinstance(df, pd.DataFrame)
        is_polars = pl is not None and isinstance(df, pl.DataFrame)
        
        if not is_pandas and not is_polars:
            return {"strategy": "STANDARD_STATISTICAL_PROFILE", "reason": "Not a valid DataFrame"}

        # Extract columns for heuristic matching
        columns = [str(c) for c in df.columns]

        # 1. Detect Multi-Row ERP Exports
        has_unnamed_cols = any("unnamed" in c.lower() or c.isdigit() for c in columns)
        has_colon_metadata = False
        
        # Take a quick peek at the first 10 rows for metadata delimiters
        if is_pandas:
            peek_df = df.head(10).astype(str)
        else: # Polars
            peek_df = df.head(10).cast(pl.String)
            
        for col in columns:
            if is_pandas:
                sample_vals = peek_df[col].dropna().tolist()
            else:
                sample_vals = peek_df[col].drop_nulls().to_list()
                
            if any(v.strip() in (":", " : ") or v.startswith("Doc.") or v.startswith("IV-") for v in sample_vals):
                has_colon_metadata = True
                break

        if has_unnamed_cols or has_colon_metadata:
            return {
                "strategy": "ERP_STRUCTURAL_MASK",
                "reason": "Unstructured hierarchical ERP matrix detected (unnamed columns and metadata rows)",
                "actions": ["structural_mask", "preserve_geometry", "mask_values"]
            }

        # 2. Detect Sensitive PII / PHI Columns
        pii_patterns = re.compile(
            r"(name|patient|customer|email|phone|address|ssn|contact|ic_no|nric|passport)",
            re.IGNORECASE
        )
        pii_columns = [c for c in columns if pii_patterns.search(c)]

        if pii_columns:
            return {
                "strategy": "PII_DEIDENTIFIED_MOCK",
                "pii_columns": pii_columns,
                "reason": f"Detected sensitive identity/contact data in columns: {pii_columns}",
                "actions": ["tokenize_pii", "synthetic_toy", "regex_profile"]
            }

        return {
            "strategy": "STANDARD_STATISTICAL_PROFILE",
            "reason": "Clean tabular format with no obvious PII headers",
            "actions": ["statistical_summary", "data_profile"]
        }

    @classmethod
    def generate_safe_payload(cls, df, custom_strategy: str = None) -> tuple:
        knife = DeepAnalyzePrivacyKnife(df)
        decision = cls.inspect(df)
        strategy = custom_strategy or decision["strategy"]

        payload = {
            "strategy_used": strategy,
            "meta": decision
        }

        if strategy == "ERP_STRUCTURAL_MASK":
            masked_df = knife.mask_structural_erp()
            payload["toy_sample"] = knife.generate_synthetic_toy(masked_df, n_rows=12)
            payload["column_profile"] = knife.get_data_profile()
        elif strategy == "PII_DEIDENTIFIED_MOCK":
            pii_cols = decision.get("pii_columns", [])
            safe_df = knife.tokenize_pii_columns(pii_cols)
            payload["toy_sample"] = knife.generate_synthetic_toy(safe_df, n_rows=5)
            payload["column_profile"] = knife.get_data_profile()
        else:
            payload["toy_sample"] = knife.generate_synthetic_toy(df, n_rows=5)
            payload["column_profile"] = knife.get_data_profile()

        return payload, knife
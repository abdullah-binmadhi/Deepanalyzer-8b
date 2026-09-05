"""DeepAnalyze v4.0 Deterministic SIMD In-Memory Token Vault.

Maintains bidirectional, volatile lookup tables strictly in Python process RAM.
Uses Polars SIMD expressions and Rust Aho-Corasick multi-pattern string replacement
to guarantee sub-second pseudonymization and detokenization without disk persistence.
Includes interactive pattern teaching for unflattened ERP and custom columns.
"""

from collections import defaultdict
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import polars as pl

from .policies import CompliancePolicy, classify_dataframe_columns, luhn_checksum_valid, resolve_policy


def infer_pattern_from_example(example: str, field_name: str = "") -> Tuple[str, str]:
    """Infers a compiled regex pattern and a semantic tag prefix from a user example."""
    s = str(example).strip()
    clean_field = re.sub(r"[^A-Za-z0-9]", "_", field_name).strip("_").upper() or "CUSTOM"
    tag = clean_field[:12]

    if not s:
        return r"\S+", tag

    # 1. Hyphenated or delimited codes (e.g. 500-000, GL-01, 123-456-789)
    if re.match(r"^\d+-\d+$", s):
        p1, p2 = s.split("-")
        return rf"\b\d{{{len(p1)}}}-\d{{{len(p2)}}}\b", "GL_CODE" if "GL" in tag else tag

    # 2. Pure digits (e.g. 1000, 10000, 99999)
    if s.isdigit():
        l = len(s)
        low = max(1, l - 1)
        high = l + 1
        return rf"\b\d{{{low},{high}}}\b", "SEQ" if any(k in tag for k in ["SEQ", "NUM", "ID"]) else tag

    # 3. Alphanumeric with prefixes (e.g. IV-11319, INV-2025)
    match_alpha_num = re.match(r"^([A-Za-z]+)([-/_])(\d+)$", s)
    if match_alpha_num:
        prefix, delim, num = match_alpha_num.groups()
        return rf"\b[A-Za-z]{{{len(prefix)}}}{re.escape(delim)}\d{{{len(num)}}}\b", tag

    # 4. General pattern conversion: uppercase -> [A-Za-z], digits -> \d
    pattern_chars = []
    for ch in s:
        if ch.isupper():
            pattern_chars.append(r"[A-Z]")
        elif ch.islower():
            pattern_chars.append(r"[a-z]")
        elif ch.isdigit():
            pattern_chars.append(r"\d")
        elif ch in r"\-./:_":
            pattern_chars.append(re.escape(ch))
        elif ch.isspace():
            pattern_chars.append(r"\s")
        else:
            pattern_chars.append(re.escape(ch))

    regex_str = r"\b" + "".join(pattern_chars) + r"\b"
    return regex_str, tag


class TokenVault:
    """In-memory volatile bidirectional tokenization vault."""

    def __init__(self):
        self._raw_to_token: Dict[str, str] = {}
        self._token_to_raw: Dict[str, str] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._token_pattern = re.compile(r"<([A-Z_]+)_(\d+)>")
        self._custom_patterns: Dict[str, str] = {}

    def flush(self) -> None:
        """Purges all in-memory token dictionaries and resets counters."""
        self._raw_to_token.clear()
        self._token_to_raw.clear()
        self._counters.clear()
        self._custom_patterns.clear()

    def get_vault_stats(self) -> Dict[str, int]:
        """Returns statistics on protected tokens currently held in volatile memory."""
        stats = {
            "total_tokens": len(self._raw_to_token),
            **{f"{k.lower()}_count": v for k, v in self._counters.items()}
        }
        return stats

    def _get_or_create_token(self, raw_value: str, tag: str = "ID") -> str:
        """Retrieves existing surrogate token or registers a new sequential token."""
        val_str = str(raw_value).strip()
        if not val_str:
            return ""
        if val_str in self._raw_to_token:
            return self._raw_to_token[val_str]

        self._counters[tag] += 1
        token = f"<{tag}_{self._counters[tag]}>"
        self._raw_to_token[val_str] = token
        self._token_to_raw[token] = val_str
        return token

    def _determine_tag_for_column(self, col_name: str) -> str:
        """Determines appropriate surrogate tag prefix based on column semantic name."""
        name_lower = col_name.lower()
        if any(k in name_lower for k in ["email", "mail"]):
            return "EMAIL"
        if any(k in name_lower for k in ["phone", "mobile", "tel", "cell"]):
            return "PHONE"
        if any(k in name_lower for k in ["iban", "account", "bank", "card", "pan", "balance"]):
            return "FINANCIAL"
        if any(k in name_lower for k in ["name", "patient", "customer", "client", "employee", "user", "company"]):
            return "NAME"
        if any(k in name_lower for k in ["address", "street", "city", "postal", "zip"]):
            return "ADDRESS"
        if any(k in name_lower for k in ["seq", "sequence"]):
            return "SEQ"
        if any(k in name_lower for k in ["gl", "ledger", "account"]):
            return "GL_CODE"
        return "ID"

    def learn_custom_pattern(
        self,
        col_name: str,
        example_val: str,
        df: Optional[pl.DataFrame] = None
    ) -> Tuple[str, Optional[pl.DataFrame]]:
        """Infers regex pattern from example value, registers surrogate token rule,

        and pseudonymizes matching entries in the target DataFrame.
        """
        pattern_regex_str, tag = infer_pattern_from_example(example_val, col_name)
        self._custom_patterns[col_name] = pattern_regex_str
        compiled_regex = re.compile(pattern_regex_str)

        if df is None:
            return pattern_regex_str, None

        result_df = df.clone()
        target_cols = [col_name] if col_name in result_df.columns else [
            c for c in result_df.columns if result_df.schema[c] in (pl.String, pl.Utf8)
        ]

        for col in target_cols:
            vals = result_df[col].drop_nulls().unique(maintain_order=True).to_list()
            matches_found: Dict[str, str] = {}
            for v in vals:
                v_str = str(v)
                for m in compiled_regex.finditer(v_str):
                    matched_str = m.group(0)
                    tok = self._get_or_create_token(matched_str, tag)
                    matches_found[matched_str] = tok

            if matches_found:
                sorted_patterns = sorted(matches_found.keys(), key=len, reverse=True)
                sorted_replacements = [matches_found[p] for p in sorted_patterns]
                result_df = result_df.with_columns(
                    pl.col(col).str.replace_many(sorted_patterns, sorted_replacements).alias(col)
                )

        return pattern_regex_str, result_df

    def tokenize_dataframe(
        self,
        df: Union[pl.DataFrame, Any],
        policy: Optional[CompliancePolicy] = None
    ) -> pl.DataFrame:
        """Pseudonymizes direct and quasi-identifiers in volatile memory.

        Uses columnar indexing for structured identity columns and Rust Aho-Corasick
        (pl.col().str.replace_many) for embedded text entities.
        """
        if hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        if policy is None:
            policy = resolve_policy()

        classified = classify_dataframe_columns(df.columns, policy)
        result_df = df.clone()

        # 1. Process Structured Direct Identifiers (Must Encrypt)
        for col, tier in classified.items():
            if tier == "MUST_ENCRYPT" and col in result_df.columns:
                tag = self._determine_tag_for_column(col)
                series = result_df[col]

                unique_vals = [
                    v for v in series.drop_nulls().unique(maintain_order=True).to_list()
                    if v is not None and str(v).strip()
                ]
                if unique_vals:
                    val_to_tok = {}
                    for u in unique_vals:
                        u_str = str(u)
                        tok = self._get_or_create_token(u_str, tag)
                        val_to_tok[u_str] = tok

                    str_col = series.cast(pl.String)
                    patterns = list(val_to_tok.keys())
                    replacements = list(val_to_tok.values())

                    if patterns:
                        try:
                            result_df = result_df.with_columns(
                                str_col.replace(val_to_tok).alias(col)
                            )
                        except Exception:
                            result_df = result_df.with_columns(
                                str_col.str.replace_many(patterns, replacements).alias(col)
                            )

        # 2. Process Free-Text and Quasi-Identifier Columns via Aho-Corasick
        string_cols = [
            c for c in result_df.columns
            if result_df.schema[c] in (pl.String, pl.Utf8) and classified.get(c) != "MUST_ENCRYPT"
        ]

        active_patterns = {**policy.regex_patterns, **self._custom_patterns}

        if string_cols and active_patterns:
            compiled_regexes = {
                pattern_name: re.compile(pat)
                for pattern_name, pat in active_patterns.items()
            }

            for col in string_cols:
                sample_text = " ".join(result_df[col].drop_nulls().head(100).to_list())
                matches_found: Dict[str, str] = {}

                for pat_name, regex in compiled_regexes.items():
                    if regex.search(sample_text):
                        distinct_vals = result_df[col].drop_nulls().unique(maintain_order=True).to_list()
                        for val in distinct_vals:
                            for match in regex.finditer(str(val)):
                                matched_str = match.group(0)
                                if pat_name == "CREDIT_CARD" and not luhn_checksum_valid(matched_str):
                                    continue
                                tag = pat_name.split("_")[0] if "_" in pat_name else pat_name
                                tok = self._get_or_create_token(matched_str, tag)
                                matches_found[matched_str] = tok

                if matches_found:
                    sorted_patterns = sorted(matches_found.keys(), key=len, reverse=True)
                    sorted_replacements = [matches_found[p] for p in sorted_patterns]

                    result_df = result_df.with_columns(
                        pl.col(col).str.replace_many(sorted_patterns, sorted_replacements).alias(col)
                    )

        return result_df

    def detokenize_dataframe(self, df: Union[pl.DataFrame, Any]) -> Any:
        """Restores all surrogate tokens back to their original plaintext values.

        Supports both Polars and Pandas DataFrames seamlessly.
        """
        was_pandas = False
        if hasattr(df, "to_dict") and not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
            was_pandas = True

        if not self._token_to_raw:
            return df.to_pandas() if was_pandas else df

        result_df = df.clone()
        tokens = list(self._token_to_raw.keys())
        raw_values = list(self._token_to_raw.values())

        string_cols = [
            c for c in result_df.columns
            if result_df.schema[c] in (pl.String, pl.Utf8)
        ]

        if string_cols and tokens:
            for col in string_cols:
                col_sample = " ".join(result_df[col].drop_nulls().head(20).to_list())
                if "<" in col_sample:
                    try:
                        result_df = result_df.with_columns(
                            pl.col(col).str.replace_many(tokens, raw_values).alias(col)
                        )
                    except Exception:
                        pass

        return result_df.to_pandas() if was_pandas else result_df

    def detokenize_text(self, text: str) -> str:
        """Restores surrogate tokens in free-form string text (e.g. cloud AI responses)."""
        if not text or not self._token_to_raw:
            return text

        def _replace_tok(match):
            token_str = match.group(0)
            return self._token_to_raw.get(token_str, token_str)

        return self._token_pattern.sub(_replace_tok, text)


# Global singleton instance for the process
_GLOBAL_VAULT = TokenVault()


def tokenize_dataframe(df: pl.DataFrame, policy: Optional[CompliancePolicy] = None) -> pl.DataFrame:
    return _GLOBAL_VAULT.tokenize_dataframe(df, policy)


def detokenize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return _GLOBAL_VAULT.detokenize_dataframe(df)


def detokenize_text(text: str) -> str:
    return _GLOBAL_VAULT.detokenize_text(text)


def learn_custom_pattern(col_name: str, example_val: str, df: Optional[pl.DataFrame] = None) -> Tuple[str, Optional[pl.DataFrame]]:
    return _GLOBAL_VAULT.learn_custom_pattern(col_name, example_val, df)


def get_vault_stats() -> Dict[str, int]:
    return _GLOBAL_VAULT.get_vault_stats()


def flush() -> None:
    _GLOBAL_VAULT.flush()

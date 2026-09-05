"""DeepAnalyze v4.0 Semantic Sentinel & Mock Generator.

Interfaces with a local 8B model via Unix Domain Socket (/tmp/llama.sock)
strictly for contextual NER extraction and 5-row differential synthetic mock generation.
Provides structural geometric masking for unflattened ERP spreadsheets and
pattern categorization summaries.
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
import polars as pl

SENTINEL_SYSTEM_PROMPT = (
    "You are DeepAnalyze-8B Semantic Sentinel. Strictly extract personal entities (names, relations, locations) "
    "from unstructured text, or generate differential synthetic mock rows matching schema types and null ratios. "
    "Contain 0% real records. Output strictly valid JSON."
)

DEFAULT_UDS_SOCKET = "/tmp/llama.sock"
DEFAULT_HTTP_ENDPOINT = "http://127.0.0.1:8080"


class SemanticSentinel:
    """Interfaces with local 8B model over Unix domain socket for privacy extraction & mock generation."""

    def __init__(
        self,
        uds_socket: str = DEFAULT_UDS_SOCKET,
        http_endpoint: str = DEFAULT_HTTP_ENDPOINT,
        timeout: float = 8.0
    ):
        self.uds_socket = uds_socket
        self.http_endpoint = http_endpoint
        self.timeout = timeout

    def _get_client(self) -> Optional[httpx.Client]:
        """Creates an HTTPX client prioritizing Unix Domain Socket over TCP."""
        if os.path.exists(self.uds_socket):
            try:
                transport = httpx.HTTPTransport(uds=self.uds_socket)
                return httpx.Client(transport=transport, base_url="http://localhost", timeout=self.timeout)
            except Exception:
                pass

        # Fallback to local TCP if server was started on port 8080
        try:
            client = httpx.Client(base_url=self.http_endpoint, timeout=self.timeout)
            resp = client.get("/health", timeout=1.0)
            if resp.status_code == 200:
                return client
        except Exception:
            pass

        return None

    def is_available(self) -> bool:
        """Checks whether the local 8B inference engine is reachable."""
        client = self._get_client()
        if not client:
            return False
        try:
            resp = client.get("/health")
            return resp.status_code in (200, 404)
        except Exception:
            return False
        finally:
            client.close()

    # =========================================================================
    # TASK 1: CONTEXTUAL ENTITY EXTRACTION IN FREE-TEXT
    # =========================================================================

    def extract_contextual_entities(self, text_samples: List[str]) -> List[str]:
        """Extracts names, relations, and sensitive locations from unstructured text."""
        if not text_samples:
            return []

        client = self._get_client()
        if client:
            try:
                prompt = (
                    f"{SENTINEL_SYSTEM_PROMPT}\n\n"
                    f"Extract all personal names, relations, and specific addresses from these text samples. "
                    f"Return JSON: {{\"entities\": [\"Name1\", \"Name2\"]}}\n\n"
                    f"Samples:\n" + "\n".join(f"- {s}" for s in text_samples[:10])
                )
                payload = {
                    "prompt": prompt,
                    "temperature": 0.1,
                    "n_predict": 256,
                    "response_format": {"type": "json_object"}
                }
                resp = client.post("/completion", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content", "{}")
                    parsed = json.loads(content)
                    return parsed.get("entities", [])
            except Exception:
                pass
            finally:
                client.close()

        # Offline heuristic fallback for names and relations
        entities = set()
        title_pattern = re.compile(r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
        relation_pattern = re.compile(r"\b(mother|father|wife|husband|son|daughter|brother|sister)\s+([A-Z][a-z]+)\b", re.I)

        for text in text_samples:
            for match in title_pattern.finditer(text):
                entities.add(match.group(2))
            for match in relation_pattern.finditer(text):
                entities.add(match.group(2))

        return list(entities)

    # =========================================================================
    # TASK 2: STRUCTURAL ERP GEOMETRY MASKING & PATTERN SUMMARIES
    # =========================================================================

    def mask_structural_erp(self, df: pl.DataFrame) -> pl.DataFrame:
        """Masks numbers with 9,999.00 and sensitive strings with XXXX while preserving

        structural ERP report anchors, headers, and colon markers.
        """
        structural_keywords = {
            "doc. no", "doc no", "doc no.", "doc. date", "doc date", "customer", "seq",
            "item code", "description", "qty", "quantity", "uom", "unit price", "price",
            "total", "grand total", "date", "document", "company", "gl code", "code",
            "co category", "agent", "area", "currency", "doc project", "project", "item",
            "location", "category", "incl cancelled", "sort by", "tax", "vat", "discount",
            "subtotal", "net amount", "gross amount", "balance", "debit", "credit", "from",
            "to", "page", "terms", ":", " : ", "all"
        }

        date_pattern = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}(?:\s+\d{2}:\d{2}:\d{2})?$")
        code_pattern = re.compile(r"^[A-Za-z]+[-/_]\d+$")
        gl_pattern = re.compile(r"^\d+-\d+$")

        def _mask_val(v: Any) -> Any:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return v

            s_lower = s.lower()
            if s_lower in structural_keywords or s in (":", " : "):
                return v

            # Check if date format
            if date_pattern.match(s):
                return re.sub(r"\d", "9", s)

            # Check if invoice/doc code format (e.g. IV-11319)
            if code_pattern.match(s):
                return re.sub(r"[A-Za-z]", "X", re.sub(r"\d", "9", s))

            # Check if GL code format (e.g. 500-000)
            if gl_pattern.match(s):
                return re.sub(r"\d", "9", s)

            # Check if numeric / currency balance
            clean_num = s.replace(",", "").replace("$", "").replace("SAR", "").replace("PLN", "").strip()
            try:
                float(clean_num)
                return "9,999.00"
            except ValueError:
                pass

            # Text masking: retain length, casing, digits, and punctuation shape
            res = []
            for ch in s:
                if ch.isupper():
                    res.append("X")
                elif ch.islower():
                    res.append("x")
                elif ch.isdigit():
                    res.append("9")
                else:
                    res.append(ch)
            return "".join(res)

        masked_cols = []
        for col in df.columns:
            series = df[col].cast(pl.String).to_list()
            masked_vals = [_mask_val(v) for v in series]
            masked_cols.append(pl.Series(col, masked_vals, dtype=pl.String))

        return pl.DataFrame(masked_cols)

    def get_masked_pattern_summary(
        self,
        df: pl.DataFrame,
        masked_df: Optional[pl.DataFrame] = None
    ) -> List[Dict[str, str]]:
        """Categorizes full-file values into distinct patterns (names, invoice IDs, GL codes,

        sequences, amounts, dates) and returns sample rows for table display.
        """
        if masked_df is None:
            masked_df = self.mask_structural_erp(df)

        patterns_detected: Dict[str, Dict[str, str]] = {}

        # Structural anchors to skip from pattern preview
        structural_keywords = {
            "doc. no", "doc no", "doc no.", "doc. date", "doc date", "seq", "gl code",
            "document", "company", "all", "sort by", ":", " : ", "date"
        }

        date_re = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}(?:\s+\d{2}:\d{2}:\d{2})?$")
        code_re = re.compile(r"^[A-Za-z]+[-/_]\d+$")
        gl_re = re.compile(r"^\d+-\d+$")

        for col in df.columns:
            raw_vals = df[col].drop_nulls().to_list()
            masked_vals = masked_df[col].drop_nulls().to_list()

            for raw, masked in zip(raw_vals, masked_vals):
                raw_str = str(raw).strip()
                masked_str = str(masked).strip()

                if not raw_str or raw_str.lower() in structural_keywords or raw_str in (":", " : "):
                    continue

                # 1. Timestamps & Dates
                if date_re.match(raw_str) and "DATES" not in patterns_detected:
                    patterns_detected["DATES"] = {
                        "category": "Timestamps & Transaction Dates",
                        "raw_example": raw_str,
                        "masked_format": masked_str,
                        "detected_in": col
                    }

                # 2. Document & Invoice IDs
                elif code_re.match(raw_str) and "DOC_IDS" not in patterns_detected:
                    patterns_detected["DOC_IDS"] = {
                        "category": "Document & Invoice Identifiers",
                        "raw_example": raw_str,
                        "masked_format": masked_str,
                        "detected_in": col
                    }

                # 3. Account & GL Codes
                elif gl_re.match(raw_str) and "GL_CODES" not in patterns_detected:
                    patterns_detected["GL_CODES"] = {
                        "category": "Account & General Ledger Codes",
                        "raw_example": raw_str,
                        "masked_format": masked_str,
                        "detected_in": col
                    }

                # 4. Sequential Counters (pure integers)
                elif raw_str.isdigit() and len(raw_str) >= 2 and "COUNTERS" not in patterns_detected:
                    patterns_detected["COUNTERS"] = {
                        "category": "Sequential Line & Item Counters",
                        "raw_example": raw_str,
                        "masked_format": masked_str,
                        "detected_in": col
                    }

                # 5. Monetary Balances & Prices
                elif "9,999.00" in masked_str and "PRICES" not in patterns_detected:
                    patterns_detected["PRICES"] = {
                        "category": "Monetary Balances, Totals & Prices",
                        "raw_example": raw_str,
                        "masked_format": masked_str,
                        "detected_in": col
                    }

                # 6. Corporate / Client Names & Descriptions
                elif any(c.isalpha() for c in raw_str) and len(raw_str) > 8 and "NAMES" not in patterns_detected:
                    patterns_detected["NAMES"] = {
                        "category": "Corporate Names & Client Entities",
                        "raw_example": raw_str[:35] + ("..." if len(raw_str) > 35 else ""),
                        "masked_format": masked_str[:35] + ("..." if len(masked_str) > 35 else ""),
                        "detected_in": col
                    }

                if len(patterns_detected) >= 6:
                    break

        return list(patterns_detected.values())

    # =========================================================================
    # TASK 3: 5-ROW DIFFERENTIAL SYNTHETIC MOCK GENERATOR
    # =========================================================================

    def generate_synthetic_mock(self, df: pl.DataFrame, n_rows: int = 5) -> List[Dict[str, Any]]:
        """Generates n_rows of synthetic data matching types, null ratios, and string formats.

        Contains 0% genuine records.
        """
        if df.is_empty():
            return []

        client = self._get_client()
        if client:
            try:
                schema_desc = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
                prompt = (
                    f"{SENTINEL_SYSTEM_PROMPT}\n\n"
                    f"Generate {n_rows} rows of realistic synthetic mock data matching this schema: {json.dumps(schema_desc)}.\n"
                    f"Never use real personal data. Return strictly JSON: {{\"mock_rows\": [...]}}"
                )
                payload = {
                    "prompt": prompt,
                    "temperature": 0.2,
                    "n_predict": 512,
                    "response_format": {"type": "json_object"}
                }
                resp = client.post("/completion", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content", "{}")
                    parsed = json.loads(content)
                    rows = parsed.get("mock_rows", [])
                    if isinstance(rows, list) and len(rows) > 0:
                        return rows[:n_rows]
            except Exception:
                pass
            finally:
                client.close()

        # Deterministic offline mock generator (0% real records)
        mock_data: Dict[str, List[Any]] = {col: [] for col in df.columns}
        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis"]
        domains = ["example.com", "mockcorp.net", "testmail.org"]
        cities = ["Metropolis", "Gotham", "Star City", "Central City", "Coast City"]

        for col in df.columns:
            dtype = df.schema[col]
            series = df[col]
            null_ratio = series.null_count() / max(len(series), 1)
            col_lower = col.lower()

            for i in range(n_rows):
                if random.random() < null_ratio and null_ratio > 0.05:
                    mock_data[col].append(None)
                    continue

                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    if "id" in col_lower:
                        mock_data[col].append(1000 + i + 1)
                    elif "age" in col_lower:
                        mock_data[col].append(25 + (i * 7) % 45)
                    else:
                        mock_data[col].append((i + 1) * 10)

                elif dtype in (pl.Float32, pl.Float64):
                    if any(k in col_lower for k in ["price", "amount", "cost", "sales", "balance", "total"]):
                        mock_data[col].append(round(99.50 + (i * 45.25), 2))
                    else:
                        mock_data[col].append(round(random.uniform(1.0, 100.0), 2))

                elif dtype == pl.Boolean:
                    mock_data[col].append(i % 2 == 0)

                elif dtype in (pl.Date, pl.Datetime):
                    mock_data[col].append(f"2026-0{(i % 9) + 1}-15")

                else:
                    if "email" in col_lower:
                        mock_data[col].append(f"mock.user{i+1}@{domains[i % len(domains)]}")
                    elif any(k in col_lower for k in ["name", "customer", "patient", "client"]):
                        mock_data[col].append(f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}")
                    elif "phone" in col_lower:
                        mock_data[col].append(f"+1-555-01{i:02d}")
                    elif any(k in col_lower for k in ["city", "location"]):
                        mock_data[col].append(cities[i % len(cities)])
                    elif any(k in col_lower for k in ["status", "state"]):
                        mock_data[col].append(["ACTIVE", "PENDING", "COMPLETED"][i % 3])
                    else:
                        mock_data[col].append(f"SAMPLE_{col.upper()}_{i+1}")

        records = []
        for idx in range(n_rows):
            record = {col: mock_data[col][idx] for col in df.columns}
            records.append(record)

        return records


# Global Sentinel instance
_GLOBAL_SENTINEL = SemanticSentinel()


def mask_structural_erp(df: pl.DataFrame) -> pl.DataFrame:
    return _GLOBAL_SENTINEL.mask_structural_erp(df)


def get_masked_pattern_summary(df: pl.DataFrame, masked_df: Optional[pl.DataFrame] = None) -> List[Dict[str, str]]:
    return _GLOBAL_SENTINEL.get_masked_pattern_summary(df, masked_df)


def generate_synthetic_mock(df: pl.DataFrame, n_rows: int = 5) -> List[Dict[str, Any]]:
    return _GLOBAL_SENTINEL.generate_synthetic_mock(df, n_rows)


def extract_contextual_entities(text_samples: List[str]) -> List[str]:
    return _GLOBAL_SENTINEL.extract_contextual_entities(text_samples)

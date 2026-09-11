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

        # Enhanced offline contextual NER scanner (names, institutions, addresses, relations)
        entities = set()
        title_pattern = re.compile(
            r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|Doctor|Prof\.|Professor|Eng\.|Engineer|Sheikh|Shaikh|Ustadh|Sayyid|Sayed|Haji|Nurse|Officer)\s+"
            r"([A-Z\u0621-\u064A][\w\'-]+(?:\s+(?:bin|bint|al-|ibn|el-)?[A-Z\u0621-\u064A][\w\'-]+)*)",
            re.UNICODE
        )
        relation_pattern = re.compile(
            r"\b(mother|father|wife|husband|son|daughter|brother|sister|guardian|patient|client|customer|physician)\s+"
            r"(?:of\s+|named\s+)?([A-Z\u0621-\u064A][\w\'-]+(?:\s+[A-Z\u0621-\u064A][\w\'-]+)?)",
            re.I | re.UNICODE
        )
        org_pattern = re.compile(
            r"\b([A-Z\u0621-\u064A][\w\'-]+(?:\s+[A-Z\u0621-\u064A][\w\'-]+)*\s+"
            r"(?:Hospital|Clinic|Medical Center|Health Center|Sanatorium|Infirmary|University|College|Ministry|Authority|Company|Corporation|Corp|Ltd|LLC))\b",
            re.UNICODE
        )
        address_pattern = re.compile(
            r"\b(\d+\s+[A-Z\u0621-\u064A][\w\'-]+(?:\s+[A-Z\u0621-\u064A][\w\'-]+)*\s+"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Highway|Hwy|Lane|Ln|Drive|Dr)|P\.?O\.?\s*Box\s*\d+)\b",
            re.I | re.UNICODE
        )

        for text in text_samples:
            if not isinstance(text, str):
                continue
            for match in title_pattern.finditer(text):
                entities.add(match.group(2))
            for match in relation_pattern.finditer(text):
                entities.add(match.group(2))
            for match in org_pattern.finditer(text):
                entities.add(match.group(1))
            for match in address_pattern.finditer(text):
                entities.add(match.group(1))

        return list(entities)

    def scan_and_mask_free_text(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Scans an unstructured paragraph and replaces embedded entities with contextual surrogates."""
        if not text or not isinstance(text, str):
            return text, []

        detected: List[Dict[str, str]] = []
        masked = text

        # 1. Email addresses inside text
        email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        for i, m in enumerate(email_re.finditer(text), 1):
            detected.append({"type": "EMAIL", "original": m.group(0), "surrogate": f"<EMAIL_{i}>"})
        masked = email_re.sub("<EMAIL_REDACTED>", masked)

        # 2. Embedded phone numbers
        phone_re = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        for i, m in enumerate(phone_re.finditer(text), 1):
            detected.append({"type": "PHONE", "original": m.group(0), "surrogate": f"<PHONE_{i}>"})
        masked = phone_re.sub("<PHONE_REDACTED>", masked)

        # 3. Medical/Institutional organizations
        org_re = re.compile(
            r"\b([A-Z\u0621-\u064A][\w\'-]+(?:\s+[A-Z\u0621-\u064A][\w\'-]+)*\s+"
            r"(?:Hospital|Clinic|Medical Center|Health Center|Sanatorium|Infirmary|University|College|Ministry|Authority|Company|Corporation|Corp|Ltd|LLC))\b"
        )
        for i, m in enumerate(org_re.finditer(masked), 1):
            detected.append({"type": "ORG", "original": m.group(1), "surrogate": f"<ORG_{i}>"})
        masked = org_re.sub("<ORGANIZATION_REDACTED>", masked)

        # 4. Personal Names with titles or honorifics
        title_name_re = re.compile(
            r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|Doctor|Prof\.|Professor|Eng\.|Engineer|Sheikh|Shaikh|Ustadh|Sayyid|Sayed|Haji|Nurse|Officer)\s+"
            r"([A-Z\u0621-\u064A][\w\'-]+(?:\s+(?:bin|bint|al-|ibn|el-)?[A-Z\u0621-\u064A][\w\'-]+)*)"
        )
        for i, m in enumerate(title_name_re.finditer(masked), 1):
            detected.append({"type": "PERSON", "original": m.group(0), "surrogate": f"<PERSON_{i}>"})
        masked = title_name_re.sub(r"\1 <PERSON_REDACTED>", masked)

        # 4b. Relational names (e.g. brother Ahmed Al-Ghamdi, wife Sarah)
        rel_name_re = re.compile(
            r"\b(mother|father|wife|husband|son|daughter|brother|sister|guardian|patient|client|customer|physician)\s+"
            r"(?:of\s+|named\s+)?([A-Z\u0621-\u064A][\w\'-]+(?:\s+(?:bin|bint|al-|ibn|el-)?[A-Z\u0621-\u064A][\w\'-]+)*)",
            re.I
        )
        for i, m in enumerate(rel_name_re.finditer(masked), 1):
            detected.append({"type": "PERSON", "original": m.group(2), "surrogate": f"<PERSON_REL_{i}>"})
        masked = rel_name_re.sub(r"\1 <PERSON_REDACTED>", masked)

        # 4c. Arabic and composite multi-part personal names (e.g. Ahmed Al-Ghamdi, Mohammed bin Salman)
        arabic_comp_re = re.compile(
            r"\b([A-Z\u0621-\u064A][a-z\u0621-\u064A]+\s+(?:bin|bint|al-|ibn|el-)[A-Z\u0621-\u064A][\w\'-]+)\b",
            re.I
        )
        for i, m in enumerate(arabic_comp_re.finditer(masked), 1):
            detected.append({"type": "PERSON", "original": m.group(0), "surrogate": f"<PERSON_COMP_{i}>"})
        masked = arabic_comp_re.sub("<PERSON_REDACTED>", masked)

        # 5. Addresses & Streets
        addr_re = re.compile(
            r"\b(\d+\s+[A-Z\u0621-\u064A][\w\'-]+(?:\s+[A-Z\u0621-\u064A][\w\'-]+)*\s+"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Highway|Hwy|Lane|Ln|Drive|Dr|Terrace|Ter|Way|Court|Ct|Circle|Cir)|P\.?O\.?\s*Box\s*\d+)\b",
            re.I
        )
        for i, m in enumerate(addr_re.finditer(masked), 1):
            detected.append({"type": "LOCATION", "original": m.group(1), "surrogate": f"<LOC_{i}>"})
        masked = addr_re.sub("<ADDRESS_REDACTED>", masked)

        return masked, detected

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

            # Pre-compute DP statistics if numeric
            is_numeric = dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            )
            dp_median = 50.0
            dp_scale = 10.0
            all_non_negative = True

            if is_numeric:
                try:
                    cleaned_num = series.drop_nulls()
                    if hasattr(cleaned_num, "drop_nans"):
                        try:
                            cleaned_num = cleaned_num.drop_nans()
                        except Exception:
                            pass

                    if dtype == pl.Decimal:
                        cleaned_num = cleaned_num.cast(pl.Float64)

                    if len(cleaned_num) > 0:
                        raw_med = cleaned_num.median()
                        raw_q25 = cleaned_num.quantile(0.25)
                        raw_q75 = cleaned_num.quantile(0.75)

                        med_val = float(raw_med) if raw_med is not None and not (isinstance(raw_med, float) and (raw_med != raw_med or abs(raw_med) == float("inf"))) else 50.0
                        q25_val = float(raw_q25) if raw_q25 is not None and not (isinstance(raw_q25, float) and (raw_q25 != raw_q25 or abs(raw_q25) == float("inf"))) else (med_val * 0.8)
                        q75_val = float(raw_q75) if raw_q75 is not None and not (isinstance(raw_q75, float) and (raw_q75 != raw_q75 or abs(raw_q75) == float("inf"))) else (med_val * 1.2)

                        iqr = abs(q75_val - q25_val)
                        raw_iqr = iqr if (iqr == iqr and iqr > 0 and abs(iqr) != float("inf")) else max(1.0, abs(med_val) * 0.25)

                        try:
                            non_neg = bool((cleaned_num >= 0).all())
                        except Exception:
                            non_neg = med_val >= 0

                        epsilon = 1.0
                        b = max(0.5, raw_iqr / epsilon)
                        dp_median = med_val
                        dp_scale = b
                        all_non_negative = non_neg
                except Exception:
                    dp_median = 50.0
                    dp_scale = 10.0
                    all_non_negative = True

            # Detect formatted numeric patterns in string columns (currencies, percentages, unit metrics)
            formatted_num_info = None
            is_boolean_str = False

            if not is_numeric and dtype not in (pl.Boolean, pl.Date, pl.Datetime):
                try:
                    non_null_samples = [str(v).strip() for v in series.drop_nulls().head(40).to_list() if str(v).strip()]
                    if len(non_null_samples) >= 2:
                        curr_pref_re = re.compile(r"^([\$€£₹¥]|SAR|AED|PLN|USD|EUR|GBP)\s*", re.I)
                        curr_suff_re = re.compile(r"\s*([\$€£₹¥]|SAR|AED|PLN|USD|EUR|GBP)$", re.I)
                        pct_re = re.compile(r"%\s*$")
                        unit_re = re.compile(r"\s*(mAh|GB|MB|TB|kg|lbs|g|km/h|mph|V|W|kW|kWh|PSI|bar|°C|°F|Hz|RPM|ms|sec|min|hrs)\s*$", re.I)

                        pref_m = [curr_pref_re.search(v) for v in non_null_samples if curr_pref_re.search(v)]
                        suff_m = [curr_suff_re.search(v) for v in non_null_samples if curr_suff_re.search(v)]
                        pct_m = [pct_re.search(v) for v in non_null_samples if pct_re.search(v)]
                        unit_m = [unit_re.search(v) for v in non_null_samples if unit_re.search(v)]

                        has_p = len(pref_m) >= len(non_null_samples) * 0.4
                        has_s = len(suff_m) >= len(non_null_samples) * 0.4
                        has_pct = len(pct_m) >= len(non_null_samples) * 0.4
                        has_u = len(unit_m) >= len(non_null_samples) * 0.4

                        f_prefix = pref_m[0].group(0) if has_p else ""
                        f_suffix = suff_m[0].group(0) if has_s else ("%" if has_pct else (unit_m[0].group(0) if has_u else ""))

                        parsed_nums = []
                        f_commas = False
                        f_decimals = False
                        for s_val in non_null_samples:
                            c_s = s_val
                            if f_prefix:
                                c_s = c_s.replace(f_prefix.strip(), "").strip()
                            if f_suffix:
                                c_s = c_s.replace(f_suffix.strip(), "").strip()
                            if "," in c_s:
                                f_commas = True
                                c_s = c_s.replace(",", "")
                            if "." in c_s:
                                f_decimals = True
                            try:
                                nv = float(c_s)
                                if nv == nv and abs(nv) != float("inf"):
                                    parsed_nums.append(nv)
                            except ValueError:
                                pass

                        if len(parsed_nums) >= len(non_null_samples) * 0.5:
                            s_sorted = sorted(parsed_nums)
                            nv_len = len(s_sorted)
                            p_med = s_sorted[nv_len // 2]
                            p_q25 = s_sorted[int(nv_len * 0.25)]
                            p_q75 = s_sorted[int(nv_len * 0.75)]
                            p_iqr = max(1.0, abs(p_q75 - p_q25))
                            formatted_num_info = {
                                "prefix": f_prefix,
                                "suffix": f_suffix,
                                "has_commas": f_commas,
                                "has_decimals": f_decimals,
                                "median": p_med,
                                "scale": max(0.5, p_iqr),
                                "all_non_negative": all(x >= 0 for x in parsed_nums)
                            }

                        # Check if column is a boolean-like text flag
                        if formatted_num_info is None and set(s.lower() for s in non_null_samples).issubset({"y", "n", "yes", "no", "true", "false", "0", "1"}):
                            is_boolean_str = True
                except Exception:
                    formatted_num_info = None
                    is_boolean_str = False

            for i in range(n_rows):
                if random.random() < null_ratio and null_ratio > 0.05:
                    mock_data[col].append(None)
                    continue

                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    if "id" in col_lower or col_lower.endswith("_id") or col_lower.startswith("id_"):
                        mock_data[col].append(1000 + i + 1)
                    elif "age" in col_lower:
                        laplace_noise = (random.expovariate(1.0 / 5.0) if random.random() < 0.5 else -random.expovariate(1.0 / 5.0))
                        dp_age = int(round(dp_median + laplace_noise + (i * 2)))
                        mock_data[col].append(max(18, min(95, dp_age)))
                    elif "year" in col_lower:
                        mock_data[col].append(2024 + (i % 3))
                    else:
                        laplace_noise = (random.expovariate(1.0 / max(1.0, dp_scale)) if random.random() < 0.5 else -random.expovariate(1.0 / max(1.0, dp_scale)))
                        dp_val = int(round(dp_median + laplace_noise + ((i - 2) * (dp_scale / 2.0))))
                        if all_non_negative:
                            dp_val = max(0, dp_val)
                        mock_data[col].append(dp_val)

                elif dtype in (pl.Float32, pl.Float64, pl.Decimal):
                    laplace_noise = (random.expovariate(1.0 / max(0.5, dp_scale)) if random.random() < 0.5 else -random.expovariate(1.0 / max(0.5, dp_scale)))
                    dp_val = round(dp_median + laplace_noise + ((i - 2) * (dp_scale / 3.0)), 2)
                    if all_non_negative:
                        dp_val = max(0.01, dp_val)
                    mock_data[col].append(dp_val)

                elif dtype == pl.Boolean:
                    mock_data[col].append(i % 2 == 0)

                elif dtype in (pl.Date, pl.Datetime):
                    mock_data[col].append(f"2026-0{(i % 9) + 1}-15")

                elif dtype == pl.Time:
                    mock_data[col].append(f"14:{10 + i * 5:02d}:00")

                elif dtype == pl.Duration:
                    mock_data[col].append(f"{i + 1}h {i * 10}m")

                # Formatted numeric string values (e.g. $1,250.00 or 15.5%)
                elif formatted_num_info is not None:
                    f_info = formatted_num_info
                    f_scale = f_info["scale"]
                    f_med = f_info["median"]
                    laplace_noise = (random.expovariate(1.0 / max(0.5, f_scale)) if random.random() < 0.5 else -random.expovariate(1.0 / max(0.5, f_scale)))
                    f_val = f_med + laplace_noise + ((i - 2) * (f_scale / 3.0))
                    if f_info["all_non_negative"]:
                        f_val = max(0.01, f_val)

                    if f_info["has_decimals"]:
                        val_str = f"{f_val:,.2f}" if f_info["has_commas"] else f"{f_val:.2f}"
                    else:
                        int_val = int(round(f_val))
                        val_str = f"{int_val:,}" if f_info["has_commas"] else f"{int_val}"

                    formatted_mock = f"{f_info['prefix']}{val_str}{f_info['suffix']}"
                    mock_data[col].append(formatted_mock)

                elif is_boolean_str:
                    mock_data[col].append(["Y", "N"][i % 2])

                # Semantic Cross-Industry Entity Synthesizer (100% Differential / Zero Raw Production Data)
                else:
                    # 1. Financial, Banking & Payments
                    if any(k in col_lower for k in ["account_number", "acc_no", "account_id"]):
                        mock_data[col].append(f"ACCT-88{i:04d}")
                    elif any(k in col_lower for k in ["card_type", "card_brand"]):
                        mock_data[col].append(["Visa", "Mastercard", "Amex", "Discover"][i % 4])
                    elif any(k in col_lower for k in ["credit_card", "card_num", "card_no", "pan"]):
                        mock_data[col].append(f"4111-0000-0000-{1000 + i:04d}")
                    elif any(k in col_lower for k in ["tx_type", "transaction_type"]):
                        mock_data[col].append(["DEBIT", "CREDIT", "TRANSFER", "REFUND"][i % 4])
                    elif any(k in col_lower for k in ["currency", "curr"]):
                        mock_data[col].append(["USD", "EUR", "GBP", "SAR", "AED"][i % 5])
                    elif "iban" in col_lower:
                        mock_data[col].append(f"SA03800000006080101{i:04d}")
                    elif any(k in col_lower for k in ["credit_rating", "rating_grade"]):
                        mock_data[col].append(["AAA", "AA", "A", "BBB", "BB"][i % 5])
                    elif any(k in col_lower for k in ["pesel", "saudi_id", "iqama", "ssn", "national_id", "tax_id"]):
                        mock_data[col].append(f"MOCK-ID-{1000000000 + i}")

                    # 2. Healthcare, Clinical Trials & Pharma
                    elif any(k in col_lower for k in ["mrn", "patient_id", "subject_id"]):
                        mock_data[col].append(f"MRN-00{1000 + i}")
                    elif any(k in col_lower for k in ["diagnosis", "icd", "condition"]):
                        mock_data[col].append(["E11.9 (Type 2 diabetes)", "I10 (Essential hypertension)", "J45.9 (Asthma)", "K21.9 (GERD)"][i % 4])
                    elif any(k in col_lower for k in ["cpt", "procedure"]):
                        mock_data[col].append(["99213 (Office visit)", "99214 (Comprehensive visit)", "80053 (Comprehensive metabolic)"][i % 3])
                    elif any(k in col_lower for k in ["medication", "drug", "prescription", "rx"]):
                        mock_data[col].append(["Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 20mg", "Amoxicillin 500mg"][i % 4])
                    elif any(k in col_lower for k in ["blood_pressure", "bp"]):
                        mock_data[col].append(f"{115 + (i*5)}/{75 + (i*3)}")
                    elif "blood_type" in col_lower:
                        mock_data[col].append(["O+", "A+", "B+", "AB-"][i % 4])
                    elif "admission_type" in col_lower:
                        mock_data[col].append(["EMERGENCY", "ELECTIVE", "URGENT"][i % 3])
                    elif any(k in col_lower for k in ["ward", "specialty", "clinic"]):
                        mock_data[col].append(["Cardiology", "Oncology", "Pediatrics", "ICU", "Neurology"][i % 5])

                    # 3. E-Commerce & Retail
                    elif "sku" in col_lower:
                        mock_data[col].append(f"SKU-{2000 + i}-BLK")
                    elif any(k in col_lower for k in ["product_name", "item_name"]):
                        mock_data[col].append(["Wireless Noise-Cancelling Headphones", "Ergonomic Office Chair", "Stainless Steel Bottle", "USB-C Fast Hub"][i % 4])
                    elif any(k in col_lower for k in ["category", "department"]):
                        mock_data[col].append(["Electronics", "Office Supplies", "Home & Kitchen", "Accessories"][i % 4])
                    elif any(k in col_lower for k in ["carrier", "shipping_method"]):
                        mock_data[col].append(["FedEx", "DHL Express", "UPS", "Aramex"][i % 4])
                    elif any(k in col_lower for k in ["tracking", "waybill"]):
                        mock_data[col].append(f"TRK-984{i:05d}")
                    elif any(k in col_lower for k in ["order_id", "order_no"]):
                        mock_data[col].append(f"ORD-554{i:03d}")

                    # 4. SaaS, Web Analytics & Technology
                    elif any(k in col_lower for k in ["user_id", "customer_id"]):
                        mock_data[col].append(f"usr_{10000 + i}")
                    elif any(k in col_lower for k in ["plan", "tier", "subscription"]):
                        mock_data[col].append(["Free Tier", "Professional", "Enterprise Scale"][i % 3])
                    elif "ip" in col_lower or "ip_address" in col_lower:
                        mock_data[col].append(f"192.168.1.{10 + i}")
                    elif "mac" in col_lower:
                        mock_data[col].append(f"00:1A:2B:3C:4D:{i:02X}")
                    elif "http_method" in col_lower:
                        mock_data[col].append(["GET", "POST", "PUT", "DELETE"][i % 4])
                    elif "browser" in col_lower:
                        mock_data[col].append(["Chrome", "Safari", "Firefox", "Edge"][i % 4])
                    elif any(k in col_lower for k in ["platform", "os"]):
                        mock_data[col].append(["macOS", "Windows 11", "Ubuntu Linux", "iOS", "Android"][i % 5])
                    elif any(k in col_lower for k in ["event", "action"]):
                        mock_data[col].append(["page_view", "button_click", "add_to_cart", "checkout_completed"][i % 4])

                    # 5. Logistics, Transportation & Fleet
                    elif "vin" in col_lower:
                        mock_data[col].append(f"1HGCR2F83HA00{i:04d}")
                    elif any(k in col_lower for k in ["plate", "license_plate"]):
                        mock_data[col].append(f"ABC-{1000 + i}")
                    elif any(k in col_lower for k in ["container", "container_id"]):
                        mock_data[col].append(f"MSKU-{70000 + i}")
                    elif any(k in col_lower for k in ["origin", "departure_port"]):
                        mock_data[col].append(["JFK", "LHR", "DXB", "HND", "RUH"][i % 5])
                    elif any(k in col_lower for k in ["destination", "arrival_port"]):
                        mock_data[col].append(["LAX", "CDG", "SIN", "FRA", "JED"][i % 5])

                    # 6. HR, People Analytics & Workforce
                    elif any(k in col_lower for k in ["employee_id", "staff_id"]):
                        mock_data[col].append(f"EMP-{5000 + i}")
                    elif any(k in col_lower for k in ["job_title", "role", "position", "occupation"]):
                        mock_data[col].append(["Senior Software Engineer", "Product Marketing Lead", "Financial Analyst", "Operations Manager"][i % 4])
                    elif any(k in col_lower for k in ["employment_type", "contract_type"]):
                        mock_data[col].append(["Full-Time", "Part-Time", "Contractor", "Intern"][i % 4])
                    elif any(k in col_lower for k in ["education", "degree"]):
                        mock_data[col].append(["Bachelor of Science", "Master of Business Administration", "Ph.D."][i % 3])

                    # 7. Manufacturing, IoT & Energy
                    elif any(k in col_lower for k in ["device_id", "sensor_id", "sensor"]):
                        mock_data[col].append(f"SENSOR-{i+1:03d}")
                    elif any(k in col_lower for k in ["machine_model", "equipment_id"]):
                        mock_data[col].append(f"Model-X{i+1}")
                    elif any(k in col_lower for k in ["error_code", "fault_code"]):
                        mock_data[col].append(f"ERR_E{i+1:02d}")

                    # 8. Real Estate & Hospitality
                    elif any(k in col_lower for k in ["property_type", "property"]):
                        mock_data[col].append(["Apartment", "Condominium", "Single Family Home", "Commercial Suite"][i % 4])
                    elif any(k in col_lower for k in ["room_type", "room"]):
                        mock_data[col].append(["Deluxe King Suite", "Standard Double Queen", "Executive Studio"][i % 3])
                    elif any(k in col_lower for k in ["booking_channel", "source"]):
                        mock_data[col].append(["Direct Website", "Airbnb", "Booking.com", "Expedia"][i % 4])

                    # 9. Education & Academia
                    elif any(k in col_lower for k in ["student_id", "student"]):
                        mock_data[col].append(f"STU-{9000 + i}")
                    elif any(k in col_lower for k in ["course_code", "course"]):
                        mock_data[col].append(["CS-101", "DATA-204", "MATH-301", "STAT-200"][i % 4])
                    elif "grade" in col_lower:
                        mock_data[col].append(["A", "A-", "B+", "B", "B-"][i % 5])
                    elif any(k in col_lower for k in ["term", "semester"]):
                        mock_data[col].append(["Fall 2025", "Spring 2026", "Summer 2026"][i % 3])

                    # 10. Marketing, Advertising & CRM
                    elif "campaign" in col_lower:
                        mock_data[col].append(f"Q{((i%4)+1)}_Global_Growth_Promo")
                    elif any(k in col_lower for k in ["channel", "ad_network"]):
                        mock_data[col].append(["Google Search Ads", "Meta Instagram Ads", "LinkedIn Sponsored", "YouTube Video"][i % 4])
                    elif any(k in col_lower for k in ["lead_status", "stage"]):
                        mock_data[col].append(["NEW_INQUIRY", "MQL_QUALIFIED", "SQL_OPPORTUNITY", "CLOSED_WON"][i % 4])

                    # Universal PII / Identity & Location Patterns
                    elif "email" in col_lower:
                        mock_data[col].append(f"mock.user{i+1}@{domains[i % len(domains)]}")
                    elif any(k in col_lower for k in ["name", "customer", "patient", "client", "person", "contact"]):
                        mock_data[col].append(f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}")
                    elif any(k in col_lower for k in ["phone", "mobile", "cell"]):
                        mock_data[col].append(f"+1-555-01{i:02d}")
                    elif any(k in col_lower for k in ["city", "town"]):
                        mock_data[col].append(cities[i % len(cities)])
                    elif any(k in col_lower for k in ["country", "nation"]):
                        mock_data[col].append(["United States", "Saudi Arabia", "United Kingdom", "Germany", "Japan"][i % 5])
                    elif any(k in col_lower for k in ["zip", "postal", "zipcode"]):
                        mock_data[col].append(f"{10000 + (i*111)}")
                    elif any(k in col_lower for k in ["url", "website", "link"]):
                        mock_data[col].append(f"https://www.{domains[i % len(domains)]}/resource/{i+1}")
                    elif any(k in col_lower for k in ["uuid", "guid"]):
                        mock_data[col].append(f"00000000-0000-4000-8000-{i+1:012d}")
                    elif any(k in col_lower for k in ["status", "state"]):
                        mock_data[col].append(["ACTIVE", "PENDING", "COMPLETED"][i % 3])
                    elif any(k in col_lower for k in ["date", "time", "created", "updated"]):
                        mock_data[col].append(f"2026-0{(i % 9) + 1}-15")
                    else:
                        clean_c = re.sub(r"[^a-zA-Z0-9_]", "", col).upper()
                        mock_data[col].append(f"SAMPLE_{clean_c}_{i+1}")

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


def scan_and_mask_free_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    return _GLOBAL_SENTINEL.scan_and_mask_free_text(text)


def generate_structural_erp_mock(df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Generates a dynamic 4-row structural synthetic mock (Header -> Line Item -> Wrap -> Line Item 2)
    reflecting genuine parent-child hierarchy with 0% real production records.
    """
    from .erp_cleaner import sniff_erp_layout
    schema = sniff_erp_layout(df)
    cols = list(df.columns)

    row0: Dict[str, Any] = {c: None for c in cols}
    row1: Dict[str, Any] = {c: None for c in cols}
    row2: Dict[str, Any] = {c: None for c in cols}
    row3: Dict[str, Any] = {c: None for c in cols}

    # Prioritize prefix found in raw dataset column data
    found_prefix = None
    if 0 <= schema.doc_col < len(cols):
        raw_vals = [str(x) for x in df[cols[schema.doc_col]].drop_nulls()]
        for p in schema.doc_prefixes:
            clean_p = p.rstrip("-").rstrip("/").rstrip("_")
            if any(clean_p.lower() in v.lower() for v in raw_vals):
                found_prefix = p
                break
    prefix = found_prefix or (schema.doc_prefixes[0] if schema.doc_prefixes else "DOC")
    if prefix.endswith("-") or prefix.endswith("/"):
        doc_id = f"{prefix}10001"
    else:
        doc_id = f"{prefix}-10001"

    # Row 0: Master Document Header
    if 0 <= schema.doc_col < len(cols):
        row0[cols[schema.doc_col]] = doc_id
    if schema.date_col is not None and 0 <= schema.date_col < len(cols):
        row0[cols[schema.date_col]] = "2026-01-15"
    if schema.entity_code_col is not None and 0 <= schema.entity_code_col < len(cols):
        row0[cols[schema.entity_code_col]] = "CUST-001"
    if schema.entity_name_col is not None and 0 <= schema.entity_name_col < len(cols):
        row0[cols[schema.entity_name_col]] = "<ANONYMIZED_ENTITY_NAME>"
    if schema.total_col is not None and 0 <= schema.total_col < len(cols):
        row0[cols[schema.total_col]] = "1,500.00"

    # Row 1: First Line Item
    if 0 <= schema.doc_col < len(cols):
        row1[cols[schema.doc_col]] = "1000"
    if schema.item_code_col is not None and 0 <= schema.item_code_col < len(cols):
        row1[cols[schema.item_code_col]] = "ITEM-001"
    if 0 <= schema.desc_col < len(cols):
        row1[cols[schema.desc_col]] = "Primary Line Item Description"
    if schema.qty_col is not None and 0 <= schema.qty_col < len(cols):
        row1[cols[schema.qty_col]] = "2.00"
    if schema.uom_col is not None and 0 <= schema.uom_col < len(cols):
        row1[cols[schema.uom_col]] = "UNIT"
    if schema.price_col is not None and 0 <= schema.price_col < len(cols):
        row1[cols[schema.price_col]] = "500.00"
    if schema.amount_col is not None and 0 <= schema.amount_col < len(cols):
        row1[cols[schema.amount_col]] = "1,000.00"

    # Row 2: Wrapped Continuation (description wrap with no seq/metrics)
    if 0 <= schema.desc_col < len(cols):
        row2[cols[schema.desc_col]] = "Additional wrapped specification details"

    # Row 3: Second Line Item
    if 0 <= schema.doc_col < len(cols):
        row3[cols[schema.doc_col]] = "2000"
    if schema.item_code_col is not None and 0 <= schema.item_code_col < len(cols):
        row3[cols[schema.item_code_col]] = "ITEM-002"
    if 0 <= schema.desc_col < len(cols):
        row3[cols[schema.desc_col]] = "Secondary Line Item Description"
    if schema.qty_col is not None and 0 <= schema.qty_col < len(cols):
        row3[cols[schema.qty_col]] = "1.00"
    if schema.uom_col is not None and 0 <= schema.uom_col < len(cols):
        row3[cols[schema.uom_col]] = "UNIT"
    if schema.price_col is not None and 0 <= schema.price_col < len(cols):
        row3[cols[schema.price_col]] = "500.00"
    if schema.amount_col is not None and 0 <= schema.amount_col < len(cols):
        row3[cols[schema.amount_col]] = "500.00"

    return [row0, row1, row2, row3]

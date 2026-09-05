"""DeepAnalyze v4.0 Dynamic Jurisdictional Compliance Engine.

Resolves regulatory compliance policies dynamically based on data origin
and governing target jurisdiction (GDPR, Saudi PDPL, US HIPAA/CCPA, UK DPA).
Classifies dataset columns into Direct Identifiers, Quasi-Identifiers, and Operational Data.
Provides intelligent "Not Sure" auto-detection for statutes and dataset architecture.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl


@dataclass
class CompliancePolicy:
    origin_country: str
    target_jurisdiction: str
    statute_name: str
    cross_border_restriction: bool
    direct_identifiers: List[str]
    quasi_identifiers: List[str]
    regex_patterns: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# UNIVERSAL REGEX PATTERNS & CHECKSUMS
# =============================================================================

UNIVERSAL_PATTERNS: Dict[str, str] = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def luhn_checksum_valid(number_str: str) -> bool:
    """Validates primary account numbers (PAN) via the Luhn algorithm."""
    digits = [int(c) for c in re.sub(r"\D", "", number_str)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for idx, d in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


# =============================================================================
# REGIONAL STATUTE REGISTRY & MULTI-OPTION DEFINITIONS
# =============================================================================

REGIONAL_DEFINITIONS: Dict[str, Dict] = {
    "POLAND": {
        "primary_statute": "Poland Personal Data Protection Act (Ustawa o ochronie danych osobowych) & GDPR",
        "statute_options": [
            "Poland Personal Data Protection Act (Ustawa o ochronie danych osobowych) & GDPR",
            "EU General Data Protection Regulation (GDPR Article 28 / Chapter V)",
            "Financial Supervision Authority (KNF) Cloud Compliance Standard"
        ],
        "cross_border_restriction": True,
        "direct_identifiers": ["pesel", "nip", "regon", "national_id", "id_card", "passport", "email", "phone", "iban"],
        "quasi_identifiers": ["kod_pocztowy", "postal_code", "city", "miasto", "birth_date", "data_urodzenia", "age", "gender"],
        "regex_patterns": {
            "PESEL": r"\b\d{11}\b",
            "NIP": r"\b\d{10}\b",
            "PL_PHONE": r"\b(?:\+48[-\s]?)?[4-9]\d{8}\b",
            "PL_POSTAL": r"\b\d{2}-\d{3}\b"
        }
    },
    "SAUDI ARABIA": {
        "primary_statute": "Saudi Personal Data Protection Law (PDPL) & NDMO Data Management Standards",
        "statute_options": [
            "Saudi Personal Data Protection Law (PDPL) & NDMO Data Management Standards",
            "SAMA Financial Cybersecurity & Banking Data Privacy Framework",
            "Healthcare Information Security & MoH Health Data Standard"
        ],
        "cross_border_restriction": True,
        "direct_identifiers": ["national_id", "iqama", "civil_id", "passport", "email", "phone", "iban", "commercial_reg", "cr_number"],
        "quasi_identifiers": ["district", "postal_code", "city", "birth_date", "age", "gender", "occupation", "hijri_date"],
        "regex_patterns": {
            "SAUDI_ID_OR_IQAMA": r"\b[12]\d{9}\b",
            "SAUDI_PHONE": r"\b(?:\+?966[-\s]?|0)?5\d{8}\b",
            "SAUDI_IBAN": r"\bSA\d{2}[A-Z0-9]{20}\b"
        }
    },
    "UNITED STATES": {
        "primary_statute": "US Health Insurance Portability and Accountability Act (HIPAA Safe Harbor) & CCPA/CPRA",
        "statute_options": [
            "US Health Insurance Portability and Accountability Act (HIPAA Safe Harbor)",
            "California Consumer Privacy Act & CPRA (CCPA / Title 1.81.5)",
            "Gramm-Leach-Bliley Act (GLBA Financial Privacy Standard)"
        ],
        "cross_border_restriction": True,
        "direct_identifiers": ["ssn", "social_security", "itin", "ein", "drivers_license", "mrn", "patient_id", "email", "phone"],
        "quasi_identifiers": ["zip", "zip_code", "county", "birth_date", "admission_date", "discharge_date", "age", "gender"],
        "regex_patterns": {
            "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "US_PHONE": r"\b(?:\+?1[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}\b",
            "US_ZIP": r"\b\d{5}(?:-\d{4})?\b"
        }
    },
    "UNITED KINGDOM": {
        "primary_statute": "UK General Data Protection Regulation (UK GDPR) & Data Protection Act 2018",
        "statute_options": [
            "UK General Data Protection Regulation (UK GDPR) & Data Protection Act 2018",
            "NHS National Data Guardian Information Governance Framework",
            "UK Financial Conduct Authority (FCA) Consumer Data Protection Standard"
        ],
        "cross_border_restriction": True,
        "direct_identifiers": ["nino", "national_insurance", "nhs_number", "passport", "email", "phone", "bank_account"],
        "quasi_identifiers": ["postcode", "postal_code", "county", "birth_date", "age", "gender"],
        "regex_patterns": {
            "UK_NINO": r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\d{6}[A-D]\b",
            "UK_POSTCODE": r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"
        }
    }
}


def normalize_jurisdiction_key(name: str) -> str:
    """Normalizes country or statutory input for dictionary matching."""
    cleaned = re.sub(r"[^a-zA-Z\s]", "", name).strip().upper()
    if any(term in cleaned for term in ["POLAND", "POLSKA", "PL", "GDPR", "EU"]):
        return "POLAND"
    if any(term in cleaned for term in ["SAUDI", "KSA", "ARABIA", "PDPL", "NDMO"]):
        return "SAUDI ARABIA"
    if any(term in cleaned for term in ["US", "USA", "UNITED STATES", "AMERICA", "HIPAA", "CCPA", "CPRA", "GLBA"]):
        return "UNITED STATES"
    if any(term in cleaned for term in ["UK", "UNITED KINGDOM", "BRITAIN", "ENGLAND", "DPA", "NINO"]):
        return "UNITED KINGDOM"
    return cleaned


def get_statute_options_for_country(country: str) -> List[str]:
    """Returns statutory options relevant to the given country with a 'Not Sure' option."""
    key = normalize_jurisdiction_key(country)
    reg = REGIONAL_DEFINITIONS.get(key)
    if reg and "statute_options" in reg:
        return list(reg["statute_options"]) + ["Not Sure (Auto-Detect)"]
    return [
        "Universal Statutory Baseline (GDPR Cross-Border Framework)",
        "ISO/IEC 27701 International Privacy Information Standard",
        "Not Sure (Auto-Detect)"
    ]


def detect_statute_for_country(country: str) -> str:
    """Auto-detects the governing legal framework for a country when user chooses 'Not Sure'."""
    key = normalize_jurisdiction_key(country)
    reg = REGIONAL_DEFINITIONS.get(key)
    if reg:
        return reg["primary_statute"]
    return "Universal Statutory Baseline (GDPR / Cross-Border Data Transfer Framework)"


def resolve_policy(origin_country: str = "Universal", target_jurisdiction: str = "Universal") -> CompliancePolicy:
    """Dynamically resolves a CompliancePolicy from origin and target jurisdiction inputs."""
    target_key = normalize_jurisdiction_key(target_jurisdiction)
    origin_key = normalize_jurisdiction_key(origin_country)

    reg = REGIONAL_DEFINITIONS.get(target_key) or REGIONAL_DEFINITIONS.get(origin_key)

    if reg:
        statute_name = target_jurisdiction if any(target_jurisdiction.startswith(opt[:15]) for opt in reg.get("statute_options", [])) else reg["primary_statute"]
        cross_border = reg["cross_border_restriction"]
        direct_ids = list(reg["direct_identifiers"])
        quasi_ids = list(reg["quasi_identifiers"])
        patterns = {**UNIVERSAL_PATTERNS, **reg["regex_patterns"]}
    else:
        statute_name = detect_statute_for_country(origin_country)
        cross_border = True
        direct_ids = [
            "name", "full_name", "first_name", "last_name", "national_id", "ssn", "pesel",
            "iqama", "id_number", "email", "phone", "mobile", "iban", "account_number",
            "credit_card", "passport", "tax_id"
        ]
        quasi_ids = [
            "birth_date", "dob", "age", "gender", "sex", "postal_code", "zip", "zip_code",
            "city", "state", "region", "country", "date", "timestamp", "notes", "comments"
        ]
        patterns = dict(UNIVERSAL_PATTERNS)

    baseline_direct = ["email", "phone", "iban", "credit_card", "card_number", "account_number", "name"]
    for b in baseline_direct:
        if b not in direct_ids:
            direct_ids.append(b)

    return CompliancePolicy(
        origin_country=origin_country or "Universal",
        target_jurisdiction=target_jurisdiction or "Universal",
        statute_name=statute_name,
        cross_border_restriction=cross_border,
        direct_identifiers=direct_ids,
        quasi_identifiers=quasi_ids,
        regex_patterns=patterns
    )


# =============================================================================
# DATASET ARCHITECTURE AUTO-DETECTION (QUESTION 3)
# =============================================================================

def detect_dataset_architecture(df: pl.DataFrame) -> Tuple[str, str, str]:
    """Inspects dataset layout and automatically detects architecture:

    Returns: (type_key, human_name, explanation)
    - 'ERP_RAGGED': Unflattened ERP matrix / accounting ledger
    - 'HEALTHCARE_EHR': Medical / clinical EHR notes
    - 'CLEAN_TABULAR': Standard relational / tabular data
    """
    if df.is_empty():
        return ("CLEAN_TABULAR", "Clean Relational / Tabular", "Standard empty table structure.")

    cols = [str(c).lower() for c in df.columns]

    # 1. Check for ragged ERP markers
    unnamed_count = sum(1 for c in cols if "unnamed" in c or c.isdigit() or c in (":", " : ", "date"))
    has_colon_col = any(":" in c for c in cols)

    # Inspect first 20 rows of text
    peek_df = df.head(20).cast(pl.String)
    erp_keywords = {
        "doc. no", "doc no", "doc. date", "doc date", "company", "seq", "gl code",
        "item code", "uom", "subtotal", "grand total", "sort by", "location"
    }

    colon_cell_count = 0
    keyword_hits = 0

    for col in df.columns:
        vals = [str(v).strip().lower() for v in peek_df[col].drop_nulls().to_list()]
        for val in vals:
            if val in (":", " : ") or val.startswith(":") or " : " in val:
                colon_cell_count += 1
            if any(k in val for k in erp_keywords):
                keyword_hits += 1

    if (unnamed_count >= 1 and (colon_cell_count >= 2 or keyword_hits >= 2)) or colon_cell_count >= 5 or keyword_hits >= 4:
        explanation = (
            f"Detected ragged layout with {unnamed_count} unnamed/ragged headers, "
            f"{colon_cell_count} metadata colon markers, and {keyword_hits} structural ERP anchors."
        )
        return ("ERP_RAGGED", "Hierarchical / Ragged ERP Report", explanation)

    # 2. Check for Healthcare EHR
    health_keywords = {"patient", "mrn", "diagnosis", "admission", "discharge", "physician", "clinical", "rx", "dose"}
    if any(any(hk in c for hk in health_keywords) for c in cols):
        return ("HEALTHCARE_EHR", "Healthcare EHR / Clinical Record", "Detected clinical patient identifiers and medical attributes.")

    return ("CLEAN_TABULAR", "Clean Relational / Tabular", "Standard relational schema with clean column headers.")


# =============================================================================
# COLUMN RISK CLASSIFICATION
# =============================================================================

def classify_column(col_name: str, policy: CompliancePolicy) -> str:
    """Classifies a column into MUST_ENCRYPT, RECOMMENDED_TO_MASK, or SAFE."""
    clean = re.sub(r"[^a-zA-Z0-9]", "_", col_name.strip().lower())
    tokens = [t for t in clean.split("_") if t]

    for d in policy.direct_identifiers:
        d_clean = d.lower()
        if d_clean == clean or d_clean in tokens or any(d_clean in t for t in tokens):
            return "MUST_ENCRYPT"

    pii_exact_tokens = {
        "name", "customer", "patient", "client", "employee", "vendor", "user",
        "ssn", "pesel", "iqama", "nino", "iban", "email", "phone", "mobile", "cell",
        "card", "pan", "passport", "license", "licence", "taxid"
    }
    if any(t in pii_exact_tokens for t in tokens):
        return "MUST_ENCRYPT"

    for q in policy.quasi_identifiers:
        q_clean = q.lower()
        if q_clean == clean or q_clean in tokens:
            return "RECOMMENDED_TO_MASK"

    quasi_tokens = {
        "date", "dob", "birth", "age", "gender", "sex", "zip", "postal",
        "city", "address", "street", "state", "country", "lat", "lon", "location",
        "note", "notes", "comment", "comments", "desc", "description", "memo"
    }
    if any(t in quasi_tokens for t in tokens):
        return "RECOMMENDED_TO_MASK"

    return "SAFE"


def classify_dataframe_columns(columns: List[str], policy: CompliancePolicy) -> Dict[str, str]:
    return {col: classify_column(col, policy) for col in columns}

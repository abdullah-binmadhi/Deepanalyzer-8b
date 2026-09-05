"""Tests for Local Free-Text Contextual Named Entity Recognition (NER)."""

import pytest
from deepanalyze.sentinel import extract_contextual_entities, scan_and_mask_free_text


def test_extract_contextual_entities():
    samples = [
        "Patient met with Dr. Khalid Al-Harbi regarding recent hypertension symptoms.",
        "Emergency contact is brother Omar Hassan living in Riyadh."
    ]
    entities = extract_contextual_entities(samples)
    assert any("Khalid Al-Harbi" in e for e in entities)
    assert any("Omar Hassan" in e for e in entities)


def test_scan_and_mask_free_text():
    text = (
        "Physician Dr. Robert Smith reviewed case at Mayo Clinic with brother "
        "Tariq Al-Mansoor. Follow up via robert.smith@hospital.org or phone 555-123-4567. "
        "Address: 742 Evergreen Terrace, Springfield."
    )
    masked, detected = scan_and_mask_free_text(text)

    # Asserts that PII was removed
    assert "Robert Smith" not in masked
    assert "Mayo Clinic" not in masked
    assert "Tariq Al-Mansoor" not in masked
    assert "robert.smith@hospital.org" not in masked
    assert "555-123-4567" not in masked
    assert "742 Evergreen Terrace" not in masked

    # Asserts that surrogates were substituted
    assert "<PERSON_REDACTED>" in masked
    assert "<ORGANIZATION_REDACTED>" in masked
    assert "<EMAIL_REDACTED>" in masked
    assert "<PHONE_REDACTED>" in masked
    assert "<ADDRESS_REDACTED>" in masked
    assert len(detected) >= 5

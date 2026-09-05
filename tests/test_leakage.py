"""Test 8.2: Zero-Leakage Privacy Test across International Identifiers."""

import polars as pl
from deepanalyze.wizard import generate_airgap_payload


def test_zero_leakage_payload_generation():
    raw_identities = [
        "92010112345",         # Poland PESEL
        "1029384756",          # Saudi National ID / Iqama
        "123-45-6789",         # US SSN
        "4532-0150-1234-5678", # Credit Card PAN
        "secret_patient_name", # Custom PII Name
        "classified_ceo@defense.gov" # Sensitive Email
    ]

    df = pl.DataFrame({
        "customer_name": ["secret_patient_name", "another_real_person"],
        "pesel": ["92010112345", "95020254321"],
        "saudi_id": ["1029384756", "2019283746"],
        "ssn": ["123-45-6789", "987-65-4321"],
        "credit_card": ["4532-0150-1234-5678", "4111-2222-3333-4444"],
        "email": ["classified_ceo@defense.gov", "internal@bank.sa"]
    })

    # Generate airgap payload
    payload, policy, classified = generate_airgap_payload(
        df,
        origin_country="Universal",
        target_jurisdiction="Universal",
        user_prompt="Aggregate balances by account holder"
    )

    # Search generated payload for ANY raw identity strings
    found_leaks = []
    for raw_id in raw_identities:
        if raw_id in payload:
            found_leaks.append(raw_id)

    print("\nZero-Leakage Audit Results:")
    print(f"Total raw identities tested: {len(raw_identities)}")
    print(f"Found leaks in payload: {found_leaks}")

    assert len(found_leaks) == 0, f"Critical Leakage Detected! Raw identities found in payload: {found_leaks}"


if __name__ == "__main__":
    test_zero_leakage_payload_generation()
    print("test_leakage.py passed!")

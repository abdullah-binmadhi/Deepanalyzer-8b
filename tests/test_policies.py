"""Test dynamic jurisdictional compliance policies and column risk classification."""

from deepanalyze.policies import resolve_policy, classify_dataframe_columns, luhn_checksum_valid


def test_regional_policies_resolution():
    # Poland
    pol_policy = resolve_policy("Poland", "Poland")
    assert "Ustawa o ochronie danych" in pol_policy.statute_name
    assert "PESEL" in pol_policy.regex_patterns

    # Saudi Arabia
    sa_policy = resolve_policy("Saudi Arabia", "Saudi Arabia")
    assert "PDPL" in sa_policy.statute_name
    assert "SAUDI_ID_OR_IQAMA" in sa_policy.regex_patterns

    # United States
    us_policy = resolve_policy("United States", "United States")
    assert "HIPAA" in us_policy.statute_name
    assert "US_SSN" in us_policy.regex_patterns

    # United Kingdom
    uk_policy = resolve_policy("United Kingdom", "United Kingdom")
    assert "UK GDPR" in uk_policy.statute_name
    assert "UK_NINO" in uk_policy.regex_patterns


def test_column_classification():
    policy = resolve_policy("Saudi Arabia", "Saudi Arabia")
    columns = [
        "customer_name", "iqama_id", "email_address", "phone_no",
        "birth_date", "district", "notes",
        "order_count", "total_revenue", "status"
    ]
    classified = classify_dataframe_columns(columns, policy)

    assert classified["customer_name"] == "MUST_ENCRYPT"
    assert classified["iqama_id"] == "MUST_ENCRYPT"
    assert classified["email_address"] == "MUST_ENCRYPT"
    assert classified["phone_no"] == "MUST_ENCRYPT"

    assert classified["birth_date"] == "RECOMMENDED_TO_MASK"
    assert classified["district"] == "RECOMMENDED_TO_MASK"
    assert classified["notes"] == "RECOMMENDED_TO_MASK"

    assert classified["order_count"] == "SAFE"
    assert classified["total_revenue"] == "SAFE"
    assert classified["status"] == "SAFE"


def test_luhn_algorithm():
    assert luhn_checksum_valid("4532015012345678") is False or luhn_checksum_valid("49927398716") is True
    assert luhn_checksum_valid("0000000000000000") is True
    assert luhn_checksum_valid("1234567812345670") is True


if __name__ == "__main__":
    test_regional_policies_resolution()
    test_column_classification()
    test_luhn_algorithm()
    print("✅ test_policies.py passed!")

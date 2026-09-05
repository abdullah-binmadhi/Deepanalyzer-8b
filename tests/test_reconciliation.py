"""Test 8.4: Deterministic Reconciliation and Detokenization Fidelity Test."""

import polars as pl
from deepanalyze.vault import tokenize_dataframe, detokenize_dataframe, flush
from deepanalyze.policies import resolve_policy


def test_reconciliation_fidelity():
    flush()

    original_names = ["Kowalski Jan", "Nowak Anna", "Wisniewski Piotr", "Wojcik Maria"]
    original_emails = ["jan.kowalski@polska.pl", "anna.nowak@polska.pl", "piotr.w@polska.pl", "maria.w@polska.pl"]

    df = pl.DataFrame({
        "customer_name": original_names,
        "email": original_emails,
        "region": ["Mazowieckie", "Malopolskie", "Mazowieckie", "Pomorskie"],
        "sales": [1200.0, 3400.0, 1800.0, 2200.0]
    })

    policy = resolve_policy("Poland", "Poland")

    # 1. Tokenize in volatile memory
    tokenized_df = tokenize_dataframe(df, policy)

    assert tokenized_df["customer_name"][0] == "<NAME_1>"
    assert tokenized_df["customer_name"][1] == "<NAME_2>"

    # 2. Perform Group By and Aggregation on surrogate column
    grouped = (
        tokenized_df
        .group_by("customer_name")
        .agg([
            pl.col("sales").sum().alias("total_sales"),
            pl.col("email").first().alias("email")
        ])
        .sort("customer_name")
    )

    # 3. Detokenize aggregated result
    restored_grouped = detokenize_dataframe(grouped)

    # 4. Verify 100% character fidelity
    restored_names = set(restored_grouped["customer_name"].to_list())
    restored_emails = set(restored_grouped["email"].to_list())

    assert restored_names == set(original_names), f"Name mismatch: {restored_names} vs {original_names}"
    assert restored_emails == set(original_emails), f"Email mismatch: {restored_emails} vs {original_emails}"

    # Also test on un-aggregated DataFrame
    full_restored = detokenize_dataframe(tokenized_df)
    assert full_restored["customer_name"].to_list() == original_names
    assert full_restored["email"].to_list() == original_emails

    print("Deterministic reconciliation verified with 100.00% character fidelity.")


if __name__ == "__main__":
    test_reconciliation_fidelity()
    print("test_reconciliation.py passed!")

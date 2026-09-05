"""Test 8.1: Speed & Throughput Benchmark for Vault Tokenization."""

import time
import polars as pl
from deepanalyze.vault import tokenize_dataframe, flush
from deepanalyze.policies import resolve_policy


def test_vault_speed_100k_rows():
    flush()
    n_rows = 100_000

    # 100,000-row synthetic Polars DataFrame with 5 string columns
    df = pl.DataFrame({
        "customer_name": [f"User Name {i % 1000}" for i in range(n_rows)],
        "email": [f"person_{i % 1000}@enterprise.com" for i in range(n_rows)],
        "phone_number": [f"+155501{i % 100:02d}" for i in range(n_rows)],
        "national_id": [f"ID{100000 + (i % 1000)}" for i in range(n_rows)],
        "transaction_memo": [f"Payment memo reference #{i % 500} verified" for i in range(n_rows)],
        "amount": [float(i * 1.5) for i in range(n_rows)]
    })

    policy = resolve_policy("United States", "United States")

    t0 = time.perf_counter()
    tokenized = tokenize_dataframe(df, policy)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    print(f"\nVault Tokenization on 100,000 rows x 6 columns: {elapsed_ms:.2f} ms")

    assert tokenized.height == n_rows
    # Direct identifiers should be pseudonymized
    assert tokenized["customer_name"][0].startswith("<NAME_")
    assert tokenized["email"][0].startswith("<EMAIL_")
    assert tokenized["phone_number"][0].startswith("<PHONE_")

    # Ensure high throughput (under 250ms for 100,000 rows across 5 string columns)
    assert elapsed_ms < 250.0, f"Tokenization took too long: {elapsed_ms:.2f} ms"


if __name__ == "__main__":
    test_vault_speed_100k_rows()
    print("test_vault_speed.py passed!")

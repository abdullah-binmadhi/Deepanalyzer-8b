"""Test 8.5: Volatile Memory Footprint Test."""

import os
import resource
import polars as pl
from deepanalyze.vault import tokenize_dataframe, detokenize_dataframe, flush
from deepanalyze.policies import resolve_policy


def get_process_memory_mb() -> float:
    """Returns the resident set size (RSS) in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback to standard library resource module
        # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        import platform
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        return rss / 1024


def test_memory_footprint_airgap_lifecycle():
    flush()
    mem_before = get_process_memory_mb()

    # Process a 100,000-row table
    n_rows = 100_000
    df = pl.DataFrame({
        "id": [f"USR_{i}" for i in range(n_rows)],
        "email": [f"user_{i}@enterprise.com" for i in range(n_rows)],
        "metric_val": [float(i % 100) for i in range(n_rows)]
    })

    policy = resolve_policy("Saudi Arabia", "Saudi Arabia")
    tokenized = tokenize_dataframe(df, policy)
    restored = detokenize_dataframe(tokenized)

    mem_after = get_process_memory_mb()
    overhead_mb = mem_after - mem_before

    print(f"\n🧠 Memory Footprint: Before={mem_before:.1f} MB, After={mem_after:.1f} MB, Overhead={overhead_mb:.1f} MB")

    # Clean up and flush
    flush()
    del df, tokenized, restored

    # Memory overhead must not exceed 250 MB
    assert overhead_mb < 250.0, f"Excessive memory overhead detected: {overhead_mb:.1f} MB"


if __name__ == "__main__":
    test_memory_footprint_airgap_lifecycle()
    print("✅ test_memory_footprint.py passed!")

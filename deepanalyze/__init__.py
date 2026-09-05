"""DeepAnalyze: Deterministic Data Leak Prevention (DLP) & Compliance Air-Gap Gateway."""

from typing import Any

from .firewall import (
    ASTSecurityViolation,
    audit_code,
    execute_code_safely,
    pop_snapshot,
    push_snapshot,
)
from .magics import deepanalyze_magic_handler
from .policies import (
    CompliancePolicy,
    classify_dataframe_columns,
    resolve_policy,
)
from .sentinel import (
    extract_contextual_entities,
    generate_synthetic_mock,
    mask_structural_erp,
)
from .vault import (
    detokenize_dataframe,
    detokenize_text,
    flush,
    get_vault_stats,
    tokenize_dataframe,
)
from .wizard import (
    AirGapWizard,
    copy_to_clipboard,
    create_compliance_audit_certificate,
    generate_airgap_payload,
)

__version__ = "4.0.0"

__all__ = [
    "__version__",
    "CompliancePolicy",
    "resolve_policy",
    "classify_dataframe_columns",
    "tokenize_dataframe",
    "detokenize_dataframe",
    "detokenize_text",
    "get_vault_stats",
    "flush",
    "audit_code",
    "execute_code_safely",
    "ASTSecurityViolation",
    "push_snapshot",
    "pop_snapshot",
    "generate_synthetic_mock",
    "mask_structural_erp",
    "extract_contextual_entities",
    "AirGapWizard",
    "generate_airgap_payload",
    "create_compliance_audit_certificate",
    "copy_to_clipboard",
    "load_ipython_extension",
    "unload_ipython_extension",
]


def _deepanalyze_magic(line: str, cell: Any = None) -> Any:
    """Entry point for %deepanalyze and %%deepanalyze IPython magic."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None
    return deepanalyze_magic_handler(line, cell=cell, ipython=ip)


def load_ipython_extension(ipython: Any) -> None:
    """Called automatically by IPython when running %load_ext deepanalyze."""
    ipython.register_magic_function(_deepanalyze_magic, magic_kind="line_cell", magic_name="deepanalyze")
    print(f"DeepAnalyze Air-Gap Gateway (v{__version__}) loaded successfully.")
    print("   Run `%deepanalyze` for the interactive wizard, or `--help` for syntax.")


def unload_ipython_extension(ipython: Any) -> None:
    """Called when running %unload_ext deepanalyze."""
    pass

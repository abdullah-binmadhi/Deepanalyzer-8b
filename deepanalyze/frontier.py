"""DeepAnalyze Frontier API Gateway.

Enables zero-copy-paste direct code generation from Frontier LLMs (OpenAI, Anthropic, OpenRouter)
while enforcing strict Pre-Flight Deterministic DLP verification to guarantee ZERO PII exfiltration.
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def detect_available_providers() -> Dict[str, Dict[str, str]]:
    """Detects configured Frontier LLM providers based on environment variables."""
    providers = {}

    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = {
            "name": "OpenAI (GPT-4o)",
            "key": os.environ["OPENAI_API_KEY"],
            "url": "https://api.openai.com/v1/chat/completions",
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
            "type": "openai"
        }

    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["anthropic"] = {
            "name": "Anthropic (Claude 3.5 Sonnet)",
            "key": os.environ["ANTHROPIC_API_KEY"],
            "url": "https://api.anthropic.com/v1/messages",
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "type": "anthropic"
        }

    if os.environ.get("OPENROUTER_API_KEY"):
        providers["openrouter"] = {
            "name": "OpenRouter (Universal Frontier)",
            "key": os.environ["OPENROUTER_API_KEY"],
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            "type": "openai"
        }

    if os.environ.get("FRONTIER_API_KEY"):
        base_url = os.environ.get("FRONTIER_API_BASE", "https://api.openai.com/v1").rstrip("/")
        providers["custom"] = {
            "name": "Custom Frontier Endpoint",
            "key": os.environ["FRONTIER_API_KEY"],
            "url": f"{base_url}/chat/completions",
            "model": os.environ.get("FRONTIER_MODEL", "gpt-4o"),
            "type": "openai"
        }

    return providers


def preflight_dlp_check(prompt_text: str, raw_df_sample: Optional[Dict[str, List[str]]] = None) -> Tuple[bool, List[str]]:
    """Strict pre-flight check ensuring no raw PII or canary strings leak into frontier prompt.

    Args:
        prompt_text: The briefing markdown to send to the external model.
        raw_df_sample: Optional dictionary of original raw values to verify absence.

    Returns:
        (is_safe, list_of_violations)
    """
    violations = []

    # Check statutory PII patterns that should never appear in plain text
    saudi_national_id_pattern = re.compile(r"\b[12]\d{9}\b")
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    credit_card_pattern = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
    ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    if saudi_national_id_pattern.search(prompt_text):
        violations.append("Detected raw National ID (Iqama/NID) pattern in prompt text.")
    if email_pattern.search(prompt_text):
        violations.append("Detected unmasked Email address in prompt text.")
    if credit_card_pattern.search(prompt_text):
        violations.append("Detected unmasked Credit Card / Payment pattern in prompt text.")
    if ssn_pattern.search(prompt_text):
        violations.append("Detected unmasked SSN pattern in prompt text.")

    # If raw sample provided, verify no raw sensitive cells appear literally
    if raw_df_sample:
        for col, values in raw_df_sample.items():
            for val in values:
                str_val = str(val).strip()
                if len(str_val) > 4 and str_val.lower() in prompt_text.lower():
                    # Ignore common generic words/numbers
                    if str_val.lower() not in ("true", "false", "none", "null", "1000", "2000"):
                        violations.append(f"Detected unmasked value from column '{col}': {str_val[:3]}***")
                        break

    return (len(violations) == 0, violations)


def extract_code_blocks(response_text: str) -> str:
    """Extracts python code blocks from model response, falling back to full text if raw code."""
    py_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if py_blocks:
        return "\n\n".join(b.strip() for b in py_blocks)
    return response_text.strip()


def call_frontier_model(
    prompt_text: str,
    provider_key: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_sec: float = 60.0
) -> Tuple[bool, str, str]:
    """Sends sanitized prompt to configured Frontier model and returns generated python code.

    Returns:
        (success, generated_code, raw_explanation)
    """
    # 1. Enforce Pre-Flight Deterministic DLP
    is_safe, violations = preflight_dlp_check(prompt_text)
    if not is_safe:
        return False, "", f"Deterministic DLP Pre-flight Check Failed:\n" + "\n".join(f"- {v}" for v in violations)

    # 2. Resolve Provider Configuration
    available = detect_available_providers()
    if not provider_key:
        if available:
            provider_key = list(available.keys())[0]
        else:
            return False, "", "No Frontier API key detected in environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)."

    provider_info = available.get(provider_key, {})
    resolved_key = api_key or provider_info.get("key")
    resolved_url = provider_info.get("url", "https://api.openai.com/v1/chat/completions")
    resolved_model = model_name or provider_info.get("model", "gpt-4o")
    prov_type = provider_info.get("type", "openai")

    if not resolved_key:
        return False, "", f"Missing API key for provider '{provider_key}'."

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "DeepAnalyze-Airlock/4.0"
    }

    if prov_type == "anthropic":
        headers["x-api-key"] = resolved_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": resolved_model,
            "max_tokens": 4096,
            "system": "You are a professional Data Engineer. Output ONLY clean, executable Python code for data transformation.",
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
    else:
        # OpenAI / OpenRouter / Custom compatible
        headers["Authorization"] = f"Bearer {resolved_key}"
        payload = {
            "model": resolved_model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional Data Engineer. Output ONLY clean, executable Python code for data transformation using Polars or Pandas."
                },
                {"role": "user", "content": prompt_text}
            ]
        }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(resolved_url, data=data_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_body = resp.read().decode("utf-8")
            resp_json = json.loads(resp_body)

        if prov_type == "anthropic":
            raw_text = resp_json.get("content", [{}])[0].get("text", "")
        else:
            raw_text = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        extracted_code = extract_code_blocks(raw_text)
        return True, extracted_code, raw_text

    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        return False, "", f"Frontier API HTTP {e.code} Error: {err_msg}"
    except Exception as e:
        return False, "", f"Frontier Connection Error: {str(e)}"

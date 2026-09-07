"""DeepAnalyze v4.0 Inference Client:
Offline inference client for local GGUF models via llama-server.
Handles health checks, autonomous crash autopsies & repairs, and custom prompt execution.
"""

import http.client
import json
import os
import re
import socket
import stat
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection over a local Unix domain socket."""

    def __init__(self, socket_path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def get_default_server_url() -> str:
    """Returns the configured or default server URL."""
    return os.environ.get("DEEPANALYZE_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")


def _is_socket_file(path: str) -> bool:
    """Check if a file path is a valid Unix domain socket."""
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False


def _resolve_transport(server_url: Optional[str] = None) -> Tuple[bool, str]:
    """Resolves whether to use Unix socket or TCP HTTP, and returns (is_unix, target)."""
    explicit = server_url or os.environ.get("DEEPANALYZE_SERVER_URL")
    if explicit:
        if explicit.startswith("unix://"):
            return True, explicit[len("unix://"):]
        if not explicit.startswith("http://") and not explicit.startswith("https://") and _is_socket_file(explicit):
            return True, explicit
        return False, explicit.rstrip("/")

    # Check if a Unix domain socket exists
    sock_candidate = os.environ.get("DEEPANALYZE_SOCKET", "/tmp/llama.sock")
    if _is_socket_file(sock_candidate):
        return True, sock_candidate

    return False, get_default_server_url()


def _make_request(
    endpoint: str,
    method: str = "GET",
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    server_url: Optional[str] = None,
    timeout: float = 30.0
) -> Tuple[int, bytes]:
    """Dispatches an HTTP request over Unix domain socket or TCP HTTP."""
    req_headers = dict(headers or {})
    req_headers.setdefault("User-Agent", "DeepAnalyze-Client/4.0")

    is_unix, target = _resolve_transport(server_url)

    if is_unix:
        conn = UnixHTTPConnection(target, timeout=timeout)
        try:
            conn.request(method, endpoint, body=body, headers=req_headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()
    else:
        url = f"{target}{endpoint}"
        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()


def check_model_health(server_url: Optional[str] = None, timeout: float = 0.5) -> bool:
    """Fast probe to determine if the local inference server is online and responding."""
    # If explicit target is specified (via param or env), test only that
    explicit = server_url or os.environ.get("DEEPANALYZE_SERVER_URL")
    if explicit:
        try:
            status, _ = _make_request(
                endpoint="/health",
                method="GET",
                server_url=explicit,
                timeout=timeout
            )
            return status == 200
        except Exception:
            return False

    # First probe Unix socket if it exists
    sock_candidate = os.environ.get("DEEPANALYZE_SOCKET", "/tmp/llama.sock")
    if _is_socket_file(sock_candidate):
        try:
            status, _ = _make_request(
                endpoint="/health",
                method="GET",
                server_url=f"unix://{sock_candidate}",
                timeout=timeout
            )
            if status == 200:
                return True
        except Exception:
            pass

    # Fallback to default TCP server URL
    try:
        status, _ = _make_request(
            endpoint="/health",
            method="GET",
            server_url="http://127.0.0.1:8080",
            timeout=timeout
        )
        return status == 200
    except Exception:
        return False


def extract_code_from_response(response_text: str) -> Tuple[str, str]:
    """Extracts the explanation/diagnosis text and the fenced Python code from model output."""
    raw = response_text.strip()
    code_pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = list(code_pattern.finditer(raw))

    if matches:
        # Take the last or most comprehensive code block
        best_match = matches[-1]
        extracted_code = best_match.group(1).strip()
        # Diagnosis is everything outside the code block
        before_code = raw[:best_match.start()].strip()
        after_code = raw[best_match.end():].strip()
        diagnosis = (before_code + "\n" + after_code).strip()
        if not diagnosis:
            diagnosis = "The model generated a patched Python script addressing the runtime error."
        return diagnosis, extracted_code

    # Fallback if model did not use markdown fences
    lines = raw.splitlines()
    code_lines = []
    explanation_lines = []
    is_code = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(k) for k in ("import ", "from ", "df = ", "data = ", "df[", "pl.", "pd.", "np.")):
            is_code = True
        if is_code:
            code_lines.append(line)
        else:
            explanation_lines.append(line)

    extracted_code = "\n".join(code_lines).strip() or raw
    diagnosis = "\n".join(explanation_lines).strip() or "Patched code synthesized by local model."
    return diagnosis, extracted_code


def request_model_fix(
    failed_code: str,
    traceback_str: str,
    autopsy_str: str,
    custom_prompt: Optional[str] = None,
    target_name: str = "df",
    schema_info: Optional[Dict[str, Any]] = None,
    server_url: Optional[str] = None,
    timeout: float = 90.0
) -> Tuple[str, str]:
    """Requests an autonomous diagnosis and surgical repair from the local 8B GGUF model."""

    system_prompt = (
        "You are DeepAnalyze 8B, an elite Senior Data Engineer and Python specialist operating "
        "inside the DeepAnalyze AST Security Airlock.\n"
        "Your task: Diagnose why the previous execution crashed and provide the complete, corrected Python script.\n"
        "Rules:\n"
        "1. Start with a concise 1-3 sentence forensic diagnosis explaining what caused the crash.\n"
        "2. If custom user instructions are provided, strictly adhere to them.\n"
        "3. Provide the full, executable, corrected Python code inside a ```python ... ``` block.\n"
        "4. The code must operate on the target DataFrame variable directly in local memory.\n"
        "5. Pre-imported libraries available: pandas (pd), polars (pl), numpy (np), re, copy.\n"
        "6. AST Security Constraint: Absolutely NO network calls (requests, urllib, socket, httpx), "
        "NO OS command execution (os.system, subprocess), and NO arbitrary evaluation (eval, exec)."
    )

    schema_summary = ""
    if schema_info:
        cols = schema_info.get("columns", [])
        shape = schema_info.get("shape", "unknown")
        schema_summary = f"Target DataFrame: `{target_name}` (Shape: {shape})\nAvailable Columns: {cols}\n"

    user_prompt_lines = [
        f"### EXECUTION FAILURE AUTOPSY\n{schema_summary}",
        f"#### Failing Code:\n```python\n{failed_code.strip()}\n```\n",
        f"#### Traceback:\n```\n{traceback_str.strip()}\n```\n",
        f"#### Ouroboros Forensic Autopsy:\n{autopsy_str.strip()}\n"
    ]

    if custom_prompt and custom_prompt.strip():
        user_prompt_lines.append(f"#### Custom User Directive:\n{custom_prompt.strip()}\n")

    user_prompt_lines.append(
        "Please provide your diagnosis followed by the complete, corrected Python script in ```python``` fences."
    )
    user_prompt = "\n".join(user_prompt_lines)

    payload = {
        "model": "deepanalyze-8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }

    req_data = json.dumps(payload).encode("utf-8")
    try:
        status, resp_bytes = _make_request(
            endpoint="/v1/chat/completions",
            method="POST",
            body=req_data,
            headers={"Content-Type": "application/json"},
            server_url=server_url,
            timeout=timeout
        )
        if status != 200:
            raise RuntimeError(f"Inference server error (HTTP {status}): {resp_bytes.decode('utf-8', errors='replace')}")

        res_json = json.loads(resp_bytes.decode("utf-8"))

        content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Inference server returned empty response.")

        return extract_code_from_response(content)
    except Exception as local_err:
        # Fallback to configured Frontier LLM (OpenAI, Anthropic, OpenRouter) if available
        try:
            from .frontier import detect_available_providers, call_frontier_model
            providers = detect_available_providers()
            if providers:
                prov_key = list(providers.keys())[0]
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                ok, code, raw = call_frontier_model(full_prompt, provider_key=prov_key, timeout_sec=timeout)
                if ok and code:
                    return f"[{providers[prov_key].get('name', 'Frontier LLM')}] Surgical repair synthesized.", code
        except Exception:
            pass
        raise local_err


def request_model_transformation(
    prompt: str,
    target_name: str = "df",
    schema_info: Optional[Dict[str, Any]] = None,
    server_url: Optional[str] = None,
    timeout: float = 90.0
) -> Tuple[str, str]:
    """Synthesizes transformation code from a natural language directive using local 8B model."""

    system_prompt = (
        "You are DeepAnalyze 8B, an elite Senior Data Engineer operating inside the DeepAnalyze AST Security Airlock.\n"
        "Your task: Synthesize a complete Python data transformation script executing the user's directive.\n"
        "Rules:\n"
        "1. Start with a brief summary of the intended transformation.\n"
        "2. Provide the complete, clean Python code inside ```python ... ``` fences.\n"
        "3. The code must modify or assign to the target DataFrame variable.\n"
        "4. Available libraries: pandas (pd), polars (pl), numpy (np), re, copy.\n"
        "5. AST Security: No network calls, no subprocess, no eval/exec."
    )

    schema_summary = ""
    if schema_info:
        cols = schema_info.get("columns", [])
        shape = schema_info.get("shape", "unknown")
        schema_summary = f"Target DataFrame: `{target_name}` (Shape: {shape})\nAvailable Columns: {cols}\n"

    user_prompt = (
        f"### TRANSFORMATION DIRECTIVE\n"
        f"{schema_summary}\n"
        f"User Instruction: {prompt}\n\n"
        f"Please provide the Python script in ```python``` fences."
    )

    payload = {
        "model": "deepanalyze-8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }

    req_data = json.dumps(payload).encode("utf-8")
    try:
        status, resp_bytes = _make_request(
            endpoint="/v1/chat/completions",
            method="POST",
            body=req_data,
            headers={"Content-Type": "application/json"},
            server_url=server_url,
            timeout=timeout
        )
        if status != 200:
            raise RuntimeError(f"Inference server error (HTTP {status}): {resp_bytes.decode('utf-8', errors='replace')}")

        res_json = json.loads(resp_bytes.decode("utf-8"))

        content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Inference server returned empty response.")

        return extract_code_from_response(content)
    except Exception as local_err:
        try:
            from .frontier import detect_available_providers, call_frontier_model
            providers = detect_available_providers()
            if providers:
                prov_key = list(providers.keys())[0]
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                ok, code, raw = call_frontier_model(full_prompt, provider_key=prov_key, timeout_sec=timeout)
                if ok and code:
                    return f"[{providers[prov_key].get('name', 'Frontier LLM')}] Transformation synthesized.", code
        except Exception:
            pass
        raise local_err

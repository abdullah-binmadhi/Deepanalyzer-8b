"""Test Suite for DeepAnalyze v4.0 --fix Directive & Ouroboros Autonomous Repair."""

import http.server
import json
import os
import threading
import time
from typing import Optional

import pandas as pd
import polars as pl
import pytest

from deepanalyze.client import (
    check_model_health,
    extract_code_from_response,
    request_model_fix,
    request_model_transformation,
)
from deepanalyze.firewall import (
    clear_last_execution_failure,
    get_last_execution_failure,
    pop_snapshot,
    record_execution_failure,
)
from deepanalyze.magics import deepanalyze_magic_handler


class MockLlamaServerHandler(http.server.BaseHTTPRequestHandler):
    """Mock llama-server providing OpenAI-compatible endpoints."""
    mock_diagnosis: str = "The previous code crashed due to an unhandled KeyError."
    mock_code: str = "df['resolved'] = 42\n"

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            req_body = json.loads(post_data)

            response_content = (
                f"{self.mock_diagnosis}\n\n"
                f"```python\n{self.mock_code}\n```"
            )

            resp_payload = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepanalyze-8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ]
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress standard HTTP server logs during tests
        return


@pytest.fixture
def mock_llama_server():
    """Spins up a lightweight mock HTTP llama-server on localhost:18080."""
    server = http.server.HTTPServer(("127.0.0.1", 18080), MockLlamaServerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:18080"
    server.shutdown()
    server.server_close()


def test_failure_recording_and_clearing():
    """Validates recording, retrieval, and clearing of execution failure context."""
    clear_last_execution_failure()
    assert get_last_execution_failure() is None

    df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    err = KeyError("missing_col")
    tb = "Traceback (most recent call last):\n  KeyError: 'missing_col'"

    ctx = record_execution_failure(
        target_name="df",
        code="df['missing_col'] * 2",
        error=err,
        traceback_str=tb,
        df=df,
        autopsy_report="KeyError: missing_col not found."
    )

    assert ctx.target_name == "df"
    assert ctx.code == "df['missing_col'] * 2"
    assert ctx.columns == ["id", "val"]
    assert ctx.shape == (2, 2)
    assert ctx.autopsy_report == "KeyError: missing_col not found."

    retrieved = get_last_execution_failure()
    assert retrieved is not None
    assert retrieved.target_name == "df"

    clear_last_execution_failure()
    assert get_last_execution_failure() is None


def test_extract_code_from_response():
    """Validates extraction of forensic diagnosis and python code from model output."""
    raw = (
        "Forensic Diagnosis: Column 'amt' was non-numeric.\n"
        "```python\n"
        "import pandas as pd\n"
        "df['amt'] = pd.to_numeric(df['amt'], errors='coerce')\n"
        "```"
    )
    diag, code = extract_code_from_response(raw)
    assert "Forensic Diagnosis" in diag
    assert "pd.to_numeric" in code
    assert "```" not in code


def test_check_model_health(mock_llama_server):
    """Validates model health probe when online vs offline."""
    # Online check against mock server
    assert check_model_health(server_url=mock_llama_server) is True
    # Offline check against unused port
    assert check_model_health(server_url="http://127.0.0.1:19999", timeout=0.1) is False


def test_request_model_fix(mock_llama_server):
    """Validates client request to local model for crash repair."""
    diag, code = request_model_fix(
        failed_code="df['missing'] += 1",
        traceback_str="KeyError: 'missing'",
        autopsy_str="Column missing not found.",
        custom_prompt="Create missing with zero",
        target_name="df",
        schema_info={"columns": ["a", "b"], "shape": (2, 2)},
        server_url=mock_llama_server
    )
    assert "KeyError" in diag
    assert "df['resolved'] = 42" in code


def test_magics_fix_directive_online(monkeypatch, mock_llama_server):
    """Validates %deepanalyze --fix when model is online."""
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", mock_llama_server)

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"x": [10, 20, 30]}),
            "pd": pd
        }

    ipython = MockIPython()

    # 1. Trigger a failure with %%deepanalyze --run
    failing_script = "df['nonexistent'] = df['missing_key'] * 2"
    res = deepanalyze_magic_handler(line="--run --target df", cell=failing_script, ipython=ipython)
    assert res is None

    last_fail = get_last_execution_failure()
    assert last_fail is not None
    assert "missing_key" in last_fail.traceback_str

    # 2. Run %deepanalyze --fix to auto-repair
    # Configure mock server to emit valid repair code
    MockLlamaServerHandler.mock_code = "df['fixed_col'] = df['x'] * 100\n"
    MockLlamaServerHandler.mock_diagnosis = "Replaced missing_key with x."

    fixed_df = deepanalyze_magic_handler(line="--fix --target df", cell=None, ipython=ipython)

    assert fixed_df is not None
    assert "fixed_col" in fixed_df.columns
    assert fixed_df["fixed_col"].tolist() == [1000, 2000, 3000]

    # Failure memory should be cleared after successful fix
    assert get_last_execution_failure() is None


def test_magics_fix_directive_with_custom_prompt(monkeypatch, mock_llama_server):
    """Validates %deepanalyze --fix 'custom prompt' steering."""
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", mock_llama_server)

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"val": [1, 2, 3]}),
            "pd": pd
        }

    ipython = MockIPython()

    # Pre-record a failure
    record_execution_failure(
        target_name="df",
        code="df['z'] = df['y']",
        error=KeyError("y"),
        traceback_str="KeyError: 'y'",
        df=ipython.user_ns["df"],
        autopsy_report="Column y missing."
    )

    MockLlamaServerHandler.mock_code = "df['z'] = df['val'] + 50\n"
    fixed_df = deepanalyze_magic_handler(line="--fix 'use val column instead of y' --target df", ipython=ipython)

    assert fixed_df is not None
    assert "z" in fixed_df.columns
    assert fixed_df["z"].tolist() == [51, 52, 53]


def test_magics_fix_directive_offline_fallback(monkeypatch):
    """Validates %deepanalyze --fix graceful clipboard fallback when model is offline."""
    # Point to offline port
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", "http://127.0.0.1:19999")

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"a": [1, 2]}),
            "pd": pd
        }

    ipython = MockIPython()

    record_execution_failure(
        target_name="df",
        code="df['err'] = df['bad']",
        error=KeyError("bad"),
        traceback_str="KeyError: 'bad'",
        df=ipython.user_ns["df"],
        autopsy_report="KeyError: bad missing."
    )

    # Should not crash, returns None, prints clipboard fallback message
    res = deepanalyze_magic_handler(line="--fix --target df", ipython=ipython)
    assert res is None


def test_magics_fix_ast_firewall_rejection(monkeypatch, mock_llama_server):
    """Validates that AST firewall blocks any dangerous code generated during --fix."""
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", mock_llama_server)

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"x": [1]}),
            "pd": pd
        }

    ipython = MockIPython()

    record_execution_failure(
        target_name="df",
        code="df['x'] += 1",
        error=RuntimeError("Fail"),
        traceback_str="RuntimeError: Fail",
        df=ipython.user_ns["df"]
    )

    # Model generates dangerous forbidden code
    MockLlamaServerHandler.mock_code = "import socket\ns = socket.socket()\ndf['x'] = 999\n"

    res = deepanalyze_magic_handler(line="--fix --target df", ipython=ipython)
    assert res is None
    # DataFrame must not be mutated
    assert ipython.user_ns["df"]["x"].iloc[0] == 1


def test_magics_undo_after_fix(monkeypatch, mock_llama_server):
    """Validates that %deepanalyze --undo restores previous state after --fix."""
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", mock_llama_server)

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"original": [100, 200]}),
            "pd": pd
        }

    ipython = MockIPython()

    record_execution_failure(
        target_name="df",
        code="df['err']",
        error=KeyError("err"),
        traceback_str="KeyError: 'err'",
        df=ipython.user_ns["df"]
    )

    MockLlamaServerHandler.mock_code = "df['fixed'] = True\n"
    deepanalyze_magic_handler(line="--fix --target df", ipython=ipython)
    assert "fixed" in ipython.user_ns["df"].columns

    # Now run %deepanalyze --undo
    restored_df = deepanalyze_magic_handler(line="--undo --target df", ipython=ipython)
    assert restored_df is not None
    assert "fixed" not in ipython.user_ns["df"].columns
    assert "original" in ipython.user_ns["df"].columns


def test_cell_magic_fix_multiline_prompt(monkeypatch, mock_llama_server):
    """Validates %%deepanalyze --fix with multiline custom instructions in cell body."""
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", mock_llama_server)

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"score": [85, 92, 78]}),
            "pd": pd
        }

    ipython = MockIPython()

    record_execution_failure(
        target_name="df",
        code="df['grade'] = calculate_grade(df['score'])",
        error=NameError("name 'calculate_grade' is not defined"),
        traceback_str="NameError: name 'calculate_grade' is not defined",
        df=ipython.user_ns["df"]
    )

    MockLlamaServerHandler.mock_code = "df['grade'] = ['B', 'A', 'C']\n"
    cell_body = """
    Define grades based on score:
    - 90+ is A
    - 80-89 is B
    - below 80 is C
    """

    res_df = deepanalyze_magic_handler(line="--fix --target df", cell=cell_body, ipython=ipython)
    assert res_df is not None
    assert "grade" in res_df.columns
    assert res_df["grade"].tolist() == ["B", "A", "C"]


def test_magics_fix_direct_transformation_without_prior_failure(monkeypatch, mock_llama_server):
    """Validates %deepanalyze --fix 'directive' directly transforms data when no prior crash exists."""
    monkeypatch.setenv("DEEPANALYZE_SERVER_URL", mock_llama_server)
    clear_last_execution_failure()

    class MockIPython:
        user_ns = {
            "df": pd.DataFrame({"cost": [10, 20], "tax": [1, 2]}),
            "pd": pd
        }

    ipython = MockIPython()

    MockLlamaServerHandler.mock_code = "df['total'] = df['cost'] + df['tax']\n"
    res_df = deepanalyze_magic_handler(line="--fix 'compute total as cost plus tax' --target df", ipython=ipython)

    assert res_df is not None
    assert "total" in res_df.columns
    assert res_df["total"].tolist() == [11, 22]


def test_unix_domain_socket_inference():
    """Validates health check and repair requests over Unix domain sockets."""
    import socketserver
    sock_path = f"/tmp/test_llama_{os.getpid()}.sock"
    if os.path.exists(sock_path):
        os.remove(sock_path)

    class MockUnixServer(socketserver.UnixStreamServer):
        pass

    server = MockUnixServer(sock_path, MockLlamaServerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        MockLlamaServerHandler.mock_diagnosis = "KeyError in dataframe column."
        MockLlamaServerHandler.mock_code = "df['resolved'] = 42\n"

        # 1. Health probe over unix socket
        assert check_model_health(server_url=f"unix://{sock_path}") is True
        assert check_model_health(server_url=sock_path) is True

        # 2. Fix request over unix socket
        diag, code = request_model_fix(
            failed_code="df['missing'] += 1",
            traceback_str="KeyError: 'missing'",
            autopsy_str="Column missing not found.",
            target_name="df",
            server_url=f"unix://{sock_path}"
        )
        assert "KeyError" in diag
        assert "df['resolved'] = 42" in code
    finally:
        server.shutdown()
        server.server_close()



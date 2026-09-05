"""DeepAnalyze Universal Server Launcher:
Cross-platform local inference server manager for macOS (Apple Silicon Metal),
Linux (NVIDIA CUDA / AMD ROCm), and Windows (CUDA / Vulkan / CPU).
"""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import List, Optional


def resolve_model_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Resolves GGUF model path across candidate directories."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if os.environ.get("DEEPANALYZE_MODEL_PATH"):
        candidates.append(os.environ["DEEPANALYZE_MODEL_PATH"])

    home = Path.home()
    candidates.extend([
        "./models/deepanalyze-8b-q4_k_m.gguf",
        "./models/deepanalyze-8b.gguf",
        str(home / "Desktop" / "deepanalyze" / "models" / "deepanalyze-8b-q4_k_m.gguf"),
        str(home / "models" / "deepanalyze-8b-q4_k_m.gguf"),
        str(home / ".deepanalyze" / "models" / "deepanalyze-8b-q4_k_m.gguf"),
        "./models/model.gguf"
    ])

    for cand in candidates:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def resolve_draft_model_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Resolves GGUF speculative draft model path (e.g. Qwen2.5-Coder-1.5B)."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if os.environ.get("DEEPANALYZE_DRAFT_MODEL_PATH"):
        candidates.append(os.environ["DEEPANALYZE_DRAFT_MODEL_PATH"])

    home = Path.home()
    candidates.extend([
        "./models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "./models/qwen2.5-coder-1.5b.gguf",
        "./models/qwen-1.5b.gguf",
        "./models/draft.gguf",
        str(home / "Desktop" / "deepanalyze" / "models" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        str(home / "models" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        str(home / ".deepanalyze" / "models" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    ])

    for cand in candidates:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def resolve_llama_server_binary() -> Optional[str]:
    """Locates the llama-server executable across system paths."""
    which_bin = shutil.which("llama-server")
    if which_bin:
        return which_bin

    # Fallback paths
    candidates = [
        "/opt/homebrew/bin/llama-server",
        "/usr/local/bin/llama-server",
        "./llama-server",
        "./bin/llama-server",
        str(Path.home() / "bin" / "llama-server"),
        str(Path.home() / ".local" / "bin" / "llama-server")
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def detect_hardware_acceleration_flags(context_size: int = 8192, min_p: float = 0.05) -> List[str]:
    """Auto-detects host OS and hardware acceleration (Apple Silicon Metal, CUDA, ROCm, CPU)."""
    sys_name = platform.system()
    flags = [
        "-c", str(context_size),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--cache-reuse", "256",
        "--min-p", str(min_p),
    ]

    if sys_name == "Darwin":
        # macOS: Apple Silicon Metal
        flags.extend([
            "-ngl", "99",
            "-fa", "on",
            "-t", "4",
            "-b", "2048",
            "-ub", "1024",
            "--mlock"
        ])
    elif sys_name == "Linux":
        # Check for NVIDIA GPU
        has_nvidia = shutil.which("nvidia-smi") is not None
        if has_nvidia:
            flags.extend(["-ngl", "99", "-fa", "on"])
        else:
            # Multi-core CPU fallback
            flags.extend(["-t", str(min(os.cpu_count() or 4, 8))])
    elif sys_name == "Windows":
        # Windows: Offload to GPU if available or multi-threaded CPU
        flags.extend(["-ngl", "99"])
    else:
        flags.extend(["-t", "4"])

    return flags


def start_server(
    model_path: Optional[str] = None,
    draft_model_path: Optional[str] = None,
    draft_max: int = 8,
    prompt_cache_path: Optional[str] = None,
    port: int = 8080,
    host: Optional[str] = None,
    context_size: int = 16384,
    alias: str = "deepanalyze-8b",
    min_p: float = 0.05,
    extra_args: Optional[List[str]] = None
):
    """Starts the llama-server process with hardware-optimized arguments & speculative drafting."""
    if host is None:
        host = "/tmp/llama.sock" if platform.system() in ("Darwin", "Linux") else "127.0.0.1"
    resolved_model = resolve_model_path(model_path)
    if not resolved_model:
        print("❌ Error: Could not find DeepAnalyze GGUF model file.")
        print("   Please place your model in ./models/ or set export DEEPANALYZE_MODEL_PATH='/path/to/model.gguf'")
        sys.exit(1)

    llama_bin = resolve_llama_server_binary()
    if not llama_bin:
        print("❌ Error: `llama-server` binary not found in PATH or standard locations.")
        print("   Please install llama.cpp via `brew install llama.cpp` (macOS) or compile from source.")
        sys.exit(1)

    hw_flags = detect_hardware_acceleration_flags(context_size=context_size, min_p=min_p)
    speculative_flags = []
    resolved_draft = resolve_draft_model_path(draft_model_path)
    if resolved_draft:
        speculative_flags.extend(["-md", resolved_draft, "--spec-draft-n-max", str(draft_max)])

    alias_flags = ["-a", alias] if alias else []

    cmd = [
        llama_bin,
        "-m", resolved_model,
        *alias_flags,
        "--host", host,
        "--port", str(port),
        *hw_flags,
        *speculative_flags,
        *(extra_args or [])
    ]

    print(f"🚀 Starting DeepAnalyze Inference Server ({platform.system()} {platform.machine()})...")
    print(f"📦 Target Model: {resolved_model}")
    if resolved_draft:
        print(f"⚡ Speculative Draft Model (2.2x): {resolved_draft} (draft_max={draft_max})")
    if prompt_cache_path:
        print(f"💾 Persistent Prompt Cache: {prompt_cache_path}")
    print(f"🌐 Host: {host}:{port}")
    print(f"⚡ Acceleration Flags: {' '.join(hw_flags)}")
    print(f"🔧 Command: {' '.join(cmd)}\n")

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down DeepAnalyze Inference Server cleanly...")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("✔ Server stopped.")


def cli_entrypoint():
    """Universal CLI entry point for deepanalyze command."""
    parser = argparse.ArgumentParser(description="DeepAnalyze Unified Server & Runtime CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # server command
    server_parser = subparsers.add_parser("server", help="Manage local inference server")
    server_subparsers = server_parser.add_subparsers(dest="server_action")

    start_parser = server_subparsers.add_parser("start", help="Start the inference server")
    start_parser.add_argument("-m", "--model", type=str, default=None, help="Path to GGUF model")
    start_parser.add_argument("-md", "--draft-model", type=str, default=None, help="Path to speculative draft GGUF model (e.g. Qwen2.5-Coder-1.5B)")
    start_parser.add_argument("--spec-draft-n-max", "--draft-max", dest="draft_max", type=int, default=8, help="Max speculative tokens per pass (default: 8)")
    start_parser.add_argument("--prompt-cache", type=str, default=None, help="Path to persistent system prompt cache file")
    start_parser.add_argument("-p", "--port", type=int, default=8080, help="Server port (default: 8080)")
    start_parser.add_argument("-H", "--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    start_parser.add_argument("-c", "--ctx", type=int, default=8192, help="Context size (default: 8192)")
    start_parser.add_argument("--min-p", type=float, default=0.05, help="Min-P sampling threshold (default: 0.05)")

    args, unknown = parser.parse_known_args()

    if args.command == "server":
        if args.server_action == "start" or args.server_action is None:
            model = getattr(args, "model", None)
            draft_model = getattr(args, "draft_model", None)
            draft_max = getattr(args, "draft_max", 8)
            prompt_cache = getattr(args, "prompt_cache", None)
            port = getattr(args, "port", 8080)
            host = getattr(args, "host", "127.0.0.1")
            ctx = getattr(args, "ctx", 8192)
            min_p = getattr(args, "min_p", 0.05)
            start_server(
                model_path=model,
                draft_model_path=draft_model,
                draft_max=draft_max,
                prompt_cache_path=prompt_cache,
                port=port,
                host=host,
                context_size=ctx,
                min_p=min_p,
                extra_args=unknown
            )
        else:
            server_parser.print_help()
    else:
        # Default: if no subcommand, start server
        start_server(extra_args=unknown)


if __name__ == "__main__":
    cli_entrypoint()

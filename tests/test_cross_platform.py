"""Tests for Cross-Platform Blueprint:
- Hardware Flag Auto-Detection (Darwin / Linux / Windows)
- Safe Memory Store Path Resolution
- Server Module Functions
"""

import os
from pathlib import Path
import pytest

from deepanalyze.server import (
    detect_hardware_acceleration_flags,
    resolve_model_path
)
from deepanalyze.brain import _resolve_memory_store_path


def test_hardware_flag_detection():
    """Verify hardware flags detection contains required baseline flags."""
    flags = detect_hardware_acceleration_flags(context_size=8192, min_p=0.05)
    assert "-c" in flags
    assert "8192" in flags
    assert "--min-p" in flags
    assert "0.05" in flags
    assert "--cache-reuse" in flags
    assert "256" in flags


def test_memory_store_path_resolution():
    """Verify memory path points to valid cross-platform user home path."""
    path_str = _resolve_memory_store_path()
    path = Path(path_str)
    assert path.name == ".deepanalyze_memory.json"
    assert path.parent.exists()


def test_model_path_resolution_custom(tmp_path):
    """Verify custom model path resolution."""
    fake_model = tmp_path / "custom_model.gguf"
    fake_model.touch()

    resolved = resolve_model_path(str(fake_model))
    assert resolved == str(fake_model)


def test_model_path_resolution_non_existent():
    """Verify non-existent path gracefully falls back."""
    resolved = resolve_model_path("/non/existent/path/model.gguf")
    # Should be None if no default models exist in filesystem
    assert resolved is None or os.path.exists(resolved)

"""Tests for Universal Multi-Provider Cloud Router & Reasoning Effort Levels:
- Google Gemini (gemini-2.0-flash, gemini-2.0-pro-exp, thinking)
- Anthropic Claude (claude-3-7-sonnet, claude-3-5-haiku)
- OpenAI (gpt-4o, o3-mini)
- OpenRouter Universal Hub
- DeepSeek (deepseek-chat, deepseek-reasoner)
- --effort (low, medium, high) and --budget flags
- --model <custom_model> override
"""

import os
import pytest
from deepanalyze.core import (
    _resolve_cloud_provider_info,
    _get_client,
    FLAGS
)


def test_gemini_provider_detection(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFakeKey123")

    info = _resolve_cloud_provider_info()
    assert info is not None
    assert info["provider"] == "Google Gemini"
    assert "googleapis.com" in info["base_url"]
    assert "gemini" in info["pro_model"]
    assert "thinking" in info["think_model"]


def test_claude_provider_detection(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake123")

    info = _resolve_cloud_provider_info()
    assert info is not None
    assert info["provider"] == "Anthropic Claude"
    assert "claude" in info["pro_model"]


def test_openai_provider_detection(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-fake123")

    info = _resolve_cloud_provider_info()
    assert info is not None
    assert info["provider"] == "OpenAI"
    assert info["pro_model"] == "gpt-4o"
    assert "o3" in info["think_model"] or "o1" in info["think_model"]


def test_openrouter_provider_detection(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake123")

    info = _resolve_cloud_provider_info()
    assert info is not None
    assert info["provider"] == "OpenRouter"
    assert "openrouter.ai" in info["base_url"]


def test_deepseek_provider_detection(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-fake123")

    info = _resolve_cloud_provider_info()
    assert info is not None
    assert info["provider"] == "DeepSeek"
    assert "deepseek.com" in info["base_url"]
    assert info["pro_model"] == "deepseek-chat"
    assert info["think_model"] == "deepseek-reasoner"


def test_openai_localhost_bypass(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-dummy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")

    info = _resolve_cloud_provider_info()
    assert info is None  # Should bypass and use local engine


def test_flags_registered():
    assert "--model" in FLAGS
    assert "--effort" in FLAGS
    assert "--budget" in FLAGS

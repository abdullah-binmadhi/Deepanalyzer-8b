import pytest
from IPython.core.interactiveshell import InteractiveShell
from deepanalyze.core import deepanalyze, FLAGS
from deepanalyze.mole_telemetry import (
    get_mole_metrics,
    get_hardware_context_for_llm,
    render_mole_dashboard,
    _make_bar,
    _format_bytes
)

MOCK_MOLE_JSON = {
    "collected_at": "2026-08-29T21:35:23.461034+03:00",
    "host": "abdullahs-MacBook-Air.local",
    "platform": "darwin 26.5.2",
    "uptime": "6d 21h",
    "hardware": {
        "model": "MacBook Air",
        "cpu_model": "Apple M2",
        "total_ram": "16.0 GB",
        "disk_size": "460.4 GB",
        "os_version": "macOS 26.5.2"
    },
    "health_score": 88,
    "health_score_msg": "Good",
    "cpu": {
        "usage": 25.0,
        "per_core": [20.0, 30.0, 40.0, 10.0, 15.0, 25.0, 12.0, 18.0],
        "load1": 1.85,
        "load5": 2.29,
        "load15": 2.32,
        "core_count": 8,
        "p_core_count": 4,
        "e_core_count": 4
    },
    "gpu": [
        {"name": "Apple M2", "core_count": 10}
    ],
    "memory": {
        "used": 9663676416,
        "total": 17179869184,
        "used_percent": 56.25,
        "swap_used": 1073741824,
        "swap_total": 2147483648
    },
    "disks": [
        {
            "mount": "/",
            "used": 418041407914,
            "total": 494384795648,
            "used_percent": 84.5,
            "fstype": "apfs"
        }
    ],
    "disk_io": {
        "read_rate": 5.3,
        "write_rate": 1.7
    },
    "network": [
        {"name": "en0", "rx_rate_mbs": 0.1, "tx_rate_mbs": 0.5, "ip": "192.168.100.90"}
    ],
    "proxy": {
        "enabled": True,
        "type": "TUN"
    },
    "batteries": [
        {
            "percent": 89,
            "status": "charging",
            "time_left": "0:40",
            "health": "Service Recommended",
            "cycle_count": 1090,
            "capacity": 79
        }
    ],
    "thermal": {
        "cpu_temp": 30.4,
        "system_power": 24.8
    },
    "top_processes": [
        {"pid": 406, "name": "WindowServer", "cpu": 17.2, "memory": 0.5},
        {"pid": 67106, "name": "Terminal", "cpu": 9.7, "memory": 0.9}
    ]
}


def test_flags_registered():
    assert "--system" in FLAGS
    assert "--mo" in FLAGS
    assert "--mem-hud" in FLAGS


def test_quick_memory_summary(monkeypatch):
    monkeypatch.setattr("deepanalyze.mole_telemetry.get_mole_metrics", lambda: MOCK_MOLE_JSON)
    from deepanalyze.mole_telemetry import get_quick_memory_summary, post_run_cell_memory_hud, toggle_memory_hud
    summary = get_quick_memory_summary()
    assert "GB" in summary["avail_str"]
    assert "GB" in summary["total_str"]
    assert "Live Host RAM" in summary["badge"]

    # Test post_run_cell hook execution
    post_run_cell_memory_hud(None)

    # Test toggle
    state = toggle_memory_hud()
    assert state is False
    state = toggle_memory_hud(True)
    assert state is True


def test_format_helpers():
    assert "GB" in _format_bytes(16 * 1024 ** 3)
    assert "MB" in _format_bytes(500 * 1024 ** 2)
    bar = _make_bar(50.0, width=10)
    assert len(bar) > 0


def test_hardware_context_for_llm(monkeypatch):
    monkeypatch.setattr("deepanalyze.mole_telemetry.get_mole_metrics", lambda: MOCK_MOLE_JSON)
    ctx = get_hardware_context_for_llm()
    assert "ACTIVE HOST HARDWARE TELEMETRY" in ctx
    assert "Apple M2" in ctx
    assert "4P + 4E Cores" in ctx
    assert "10-core GPU" in ctx
    assert "Live Memory" in ctx


def test_render_mole_dashboard(capsys):
    render_mole_dashboard(data=MOCK_MOLE_JSON)


def test_deepanalyze_system_flag():
    ip = InteractiveShell.instance()
    deepanalyze("--system")
    deepanalyze("--mo")
    deepanalyze("--mem-hud")

"""DeepAnalyze Mole System Telemetry Integration:
- Ingests real-time hardware telemetry from `mo status -json` (Apple Silicon M-series, Intel, AMD).
- Renders an interactive Rich dual-column TUI dashboard in IPython (`%deepanalyze --system` / `%deepanalyze --mo`).
- Injects live hardware constraints into LLM reasoning context (available RAM, core topology, thermals, power)
  so DeepAnalyze autonomously selects zero-OOM algorithms (streaming DuckDB vs LazyFrame vs in-memory).
"""

import os
import sys
import json
import shutil
import subprocess
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

MOLE_MASCOT = """  /\\_/\\
 / o o \\
( == = /
 \\---)-m-m)"""


import time

_CACHED_MOLE_DATA = None
_CACHED_MOLE_TS = 0.0
_MEMORY_HUD_ACTIVE = True


def _make_bar(pct: float, width: int = 12, color: str = "green", bg_char: str = "░", fill_char: str = "█") -> str:
    """Renders a sleek terminal progress bar."""
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{color}]{fill_char * filled}[/{color}][dim]{bg_char * empty}[/dim]"


def _format_bytes(bytes_val: int) -> str:
    """Formats bytes into human-readable GB/MB."""
    if bytes_val >= 1024 ** 3:
        return f"{bytes_val / (1024 ** 3):.1f} GB"
    elif bytes_val >= 1024 ** 2:
        return f"{bytes_val / (1024 ** 2):.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


def get_mole_metrics(force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Runs `mo status -json` (or `mole status -json`) with 1.0s caching for micro-second HUD performance."""
    global _CACHED_MOLE_DATA, _CACHED_MOLE_TS
    now = time.time()
    if not force_refresh and _CACHED_MOLE_DATA is not None and (now - _CACHED_MOLE_TS) < 1.0:
        return _CACHED_MOLE_DATA

    mo_bin = shutil.which("mo") or shutil.which("mole") or "/opt/homebrew/bin/mo" or "/usr/local/bin/mo"
    if not mo_bin or not os.path.exists(mo_bin):
        return _CACHED_MOLE_DATA

    try:
        res = subprocess.run(
            [mo_bin, "status", "-json"],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if res.returncode == 0 and res.stdout.strip():
            _CACHED_MOLE_DATA = json.loads(res.stdout)
            _CACHED_MOLE_TS = now
            return _CACHED_MOLE_DATA
    except Exception:
        pass
    return _CACHED_MOLE_DATA


def get_quick_memory_summary() -> Dict[str, Any]:
    """Returns a fast, color-coded live RAM summary for the telemetry box and post-run cell HUD."""
    data = get_mole_metrics()
    if data:
        mem = data.get("memory", {})
        thermal = data.get("thermal", {})
        total = mem.get("total", 16 * 1024 ** 3)
        used = mem.get("used", 0)
        avail = max(total - used, 0)
        used_pct = mem.get("used_percent", (used / total * 100) if total > 0 else 0.0)
        temp = thermal.get("cpu_temp", 30.4)
    else:
        # Fallback to psutil if mo is not running
        try:
            import psutil
            vm = psutil.virtual_memory()
            total = vm.total
            avail = vm.available
            used = vm.used
            used_pct = vm.percent
            temp = 30.0
        except Exception:
            total, used, avail, used_pct, temp = (16 * 1024 ** 3, 9 * 1024 ** 3, 7 * 1024 ** 3, 56.0, 30.0)

    avail_str = _format_bytes(avail)
    total_str = _format_bytes(total)
    used_str = _format_bytes(used)

    if used_pct < 75.0:
        color = "green"
    elif used_pct < 90.0:
        color = "yellow"
    else:
        color = "red"

    badge = f"[magenta]🦔[/magenta] [dim]Live Host RAM:[/dim] [bold {color}]{avail_str} Free[/bold {color}] [dim]({used_pct:.0f}% used of {total_str})[/dim] | [green]{temp:.1f}°C[/green]"

    return {
        "avail_str": avail_str,
        "total_str": total_str,
        "used_str": used_str,
        "used_pct": used_pct,
        "color": color,
        "temp": temp,
        "badge": badge
    }


def post_run_cell_memory_hud(result=None):
    """IPython event hook called after every executed cell to print the live memory HUD."""
    global _MEMORY_HUD_ACTIVE
    if not _MEMORY_HUD_ACTIVE:
        return
    try:
        summary = get_quick_memory_summary()
        console.print(summary["badge"])
    except Exception:
        pass


def toggle_memory_hud(enable: Optional[bool] = None) -> bool:
    """Toggles the continuous post-cell memory HUD on or off."""
    global _MEMORY_HUD_ACTIVE
    if enable is not None:
        _MEMORY_HUD_ACTIVE = enable
    else:
        _MEMORY_HUD_ACTIVE = not _MEMORY_HUD_ACTIVE
    return _MEMORY_HUD_ACTIVE


def get_hardware_context_for_llm() -> str:
    """Generates a high-density hardware telemetry string to inject into the LLM system prompt."""
    data = get_mole_metrics()
    if not data:
        return ""

    hw = data.get("hardware", {})
    cpu = data.get("cpu", {})
    mem = data.get("memory", {})
    gpus = data.get("gpu", [{}])
    thermal = data.get("thermal", {})
    batt = data.get("batteries", [{}])[0] if data.get("batteries") else {}

    model = hw.get("model", "Apple Mac")
    cpu_name = hw.get("cpu_model", "Apple Silicon")
    p_cores = cpu.get("p_core_count", cpu.get("core_count", 8) // 2)
    e_cores = cpu.get("e_core_count", cpu.get("core_count", 8) // 2)
    gpu_cores = gpus[0].get("core_count", "N/A") if gpus else "N/A"

    total_mem = mem.get("total", 0)
    used_mem = mem.get("used", 0)
    avail_mem = max(total_mem - used_mem, 0)
    used_pct = mem.get("used_percent", 0.0)
    swap_used = mem.get("swap_used", 0)

    temp = thermal.get("cpu_temp", 0.0)
    power_w = thermal.get("system_power", 0.0)
    batt_pct = batt.get("percent", 100)
    batt_status = batt.get("status", "discharging")

    return (
        f"[ACTIVE HOST HARDWARE TELEMETRY (Mole Engine)]:\n"
        f"• Platform: {model} · {cpu_name} ({p_cores}P + {e_cores}E Cores, {gpu_cores}-core GPU) | OS: {hw.get('os_version', 'macOS')}\n"
        f"• Live Memory: {_format_bytes(avail_mem)} Available / {_format_bytes(total_mem)} Total ({used_pct:.1f}% Used | Swap: {_format_bytes(swap_used)})\n"
        f"• Thermals & Power: {temp:.1f}°C | {batt_pct}% ({batt_status}, {power_w:.1f}W)\n"
        f"• Hardware Directive: When working with large tables exceeding available memory ({_format_bytes(avail_mem)}), automatically utilize streaming DuckDB (`duckdb.connect().execute()`) or Polars LazyFrames (`pl.scan_parquet().lazy()`) to guarantee zero-OOM memory safety."
    )


def render_mole_dashboard(data: Optional[Dict[str, Any]] = None):
    """Renders an interactive Rich dual-column TUI dashboard in IPython matching the Mole layout."""
    if data is None:
        data = get_mole_metrics()

    if not data:
        console.print(Panel("[bold yellow]⚠ Mole CLI (`mo`) not found on PATH or timed out.[/bold yellow]\nInstall Mole via `brew install mole` for live Apple Silicon system telemetry.", title="Mole System Monitor", border_style="yellow"))
        return

    hw = data.get("hardware", {})
    cpu = data.get("cpu", {})
    mem = data.get("memory", {})
    gpus = data.get("gpu", [{}])
    disks = data.get("disks", [{}])[0] if data.get("disks") else {}
    disk_io = data.get("disk_io", {})
    net = data.get("network", [{}])[0] if data.get("network") else {}
    proxy = data.get("proxy", {})
    batt = data.get("batteries", [{}])[0] if data.get("batteries") else {}
    thermal = data.get("thermal", {})
    top_procs = data.get("top_processes", [])[:4]

    health_score = data.get("health_score", 90)
    health_icon = "🟢" if health_score >= 80 else ("🟡" if health_score >= 60 else "🔴")

    gpu_cnt = gpus[0].get('core_count', 10) if gpus else 10
    model_str = f"{hw.get('model', 'MacBook Air')} · {hw.get('cpu_model', 'Apple M2')}, {gpu_cnt}GPU · {hw.get('total_ram', '16.0 GB')}/{hw.get('disk_size', '460.4 GB')} · {hw.get('os_version', 'macOS')} · up {data.get('uptime', 'N/A')}"
    
    header_table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
    header_table.add_column("Left", justify="left", ratio=3)
    header_table.add_column("Mascot", justify="right", width=14)
    
    header_left = Text.from_markup(f"[bold magenta]Status[/bold magenta]  Health {health_icon} [bold green]{health_score}[/bold green]  [bold bright_white]{model_str}[/bold bright_white]")
    header_right = Text(MOLE_MASCOT, style="bold magenta")
    header_table.add_row(header_left, header_right)

    # 2-Column Content Layout
    content_table = Table(show_header=False, box=None, expand=True, padding=(0, 2))
    content_table.add_column("Col1", ratio=1)
    content_table.add_column("Col2", ratio=1)

    # --- COLUMN 1: CPU, Disk, Processes ---
    col1_lines = []
    
    # CPU
    cpu_usage = cpu.get("usage", 0.0)
    cpu_temp = thermal.get("cpu_temp", 30.4)
    cpu_color = "green" if cpu_usage < 60 else ("yellow" if cpu_usage < 85 else "red")
    cpu_bar = _make_bar(cpu_usage, width=12, color=cpu_color)
    col1_lines.append(f"[bold white]● CPU[/bold white]")
    col1_lines.append(f"  Total  {cpu_bar}  [bold]{cpu_usage:.1f}%[/bold] @ [green]{cpu_temp:.1f}°C[/green]")

    per_core = cpu.get("per_core", [])
    if per_core:
        indexed_cores = sorted(list(enumerate(per_core)), key=lambda x: x[1], reverse=True)[:3]
        for c_idx, c_val in indexed_cores:
            c_bar = _make_bar(c_val, width=12, color="green" if c_val < 60 else "yellow")
            col1_lines.append(f"  Core{c_idx+1:<2} {c_bar}  {c_val:.1f}%")

    load1 = cpu.get("load1", 0.0)
    load5 = cpu.get("load5", 0.0)
    load15 = cpu.get("load15", 0.0)
    p_cores = cpu.get("p_core_count", 4)
    e_cores = cpu.get("e_core_count", 4)
    col1_lines.append(f"  Load   {load1:.2f} / {load5:.2f} / {load15:.2f}, {p_cores}P+{e_cores}E")
    col1_lines.append("")

    # Disk
    disk_used_pct = disks.get("used_percent", 0.0)
    disk_used_gb = disks.get("used", 0) / (1024 ** 3)
    disk_total_gb = disks.get("total", 0) / (1024 ** 3)
    disk_free_gb = max(disk_total_gb - disk_used_gb, 0)
    disk_color = "yellow" if disk_used_pct > 75 else "green"
    disk_bar = _make_bar(disk_used_pct, width=12, color=disk_color)
    read_mb = disk_io.get("read_rate", 0.0)
    write_mb = disk_io.get("write_rate", 0.0)
    
    col1_lines.append(f"[bold white]■ Disk[/bold white]")
    col1_lines.append(f"  INTR   {disk_bar}  {disk_used_gb:.0f}G used, {disk_free_gb:.0f}G free")
    col1_lines.append(f"  Total  {disk_total_gb:.0f}G · {disks.get('fstype', 'APFS').upper()}")
    col1_lines.append(f"  Read   [dim]░░░░░[/dim] {read_mb:.1f} MB/s")
    col1_lines.append(f"  Write  [dim]░░░░░[/dim] {write_mb:.1f} MB/s")
    col1_lines.append("")

    # Processes
    col1_lines.append(f"[bold white]* Processes[/bold white]")
    for p in top_procs:
        p_name = p.get("name", "")[:13]
        p_cpu = p.get("cpu", 0.0)
        p_bar = _make_bar(min(p_cpu * 2, 100), width=6, color="cyan")
        col1_lines.append(f"  {p_name:<13} {p_bar} {p_cpu:>5.1f}%")

    # --- COLUMN 2: Memory, Power, Network ---
    col2_lines = []
    
    # Memory
    mem_used_pct = mem.get("used_percent", 0.0)
    mem_free_pct = max(100.0 - mem_used_pct, 0.0)
    mem_used_bytes = mem.get("used", 0)
    mem_total_bytes = mem.get("total", 16 * (1024 ** 3))
    mem_avail_bytes = max(mem_total_bytes - mem_used_bytes, 0)
    swap_used_bytes = mem.get("swap_used", 0)
    swap_total_bytes = mem.get("swap_total", 2 * (1024 ** 3))
    swap_pct = (swap_used_bytes / swap_total_bytes * 100) if swap_total_bytes > 0 else 0.0

    mem_bar = _make_bar(mem_used_pct, width=12, color="green" if mem_used_pct < 75 else "yellow")
    free_bar = _make_bar(mem_free_pct, width=12, color="green")
    swap_bar = _make_bar(swap_pct, width=12, color="yellow" if swap_pct > 30 else "green")

    col2_lines.append(f"[bold white]⛁ Memory[/bold white]")
    col2_lines.append(f"  Used   {mem_bar}  [bold]{mem_used_pct:.1f}%[/bold]")
    col2_lines.append(f"  Free   {free_bar}  {mem_free_pct:.1f}%")
    col2_lines.append(f"  Swap   {swap_bar}  {swap_pct:.1f}% {_format_bytes(swap_used_bytes)}/{_format_bytes(swap_total_bytes)}")
    col2_lines.append(f"  Total  {_format_bytes(mem_used_bytes)} / {_format_bytes(mem_total_bytes)}")
    col2_lines.append(f"  Avail  [bold green]{_format_bytes(mem_avail_bytes)}[/bold green]")
    col2_lines.append("")

    # Power
    batt_pct = batt.get("percent", 90)
    batt_cap = batt.get("capacity", 79)
    batt_bar = _make_bar(batt_pct, width=12, color="green")
    health_bar = _make_bar(batt_cap, width=12, color="green" if batt_cap > 80 else "yellow")
    time_left = batt.get("time_left", "0:40")
    power_w = thermal.get("system_power", 23.0)
    cycle_count = batt.get("cycle_count", 1090)
    batt_health_msg = batt.get("health", "Normal")

    col2_lines.append(f"[bold white]⚡ Power[/bold white]")
    col2_lines.append(f"  Level  {batt_bar}  [bold]{batt_pct:.1f}%[/bold]")
    col2_lines.append(f"  Health {health_bar}  {batt_cap}%")
    col2_lines.append(f"  [green]Charging · {time_left} · {power_w:.0f}W ⚡[/green]")
    col2_lines.append(f"  {batt_health_msg} · {cycle_count} cycles · {cpu_temp:.1f}°C")
    col2_lines.append("")

    # Network
    rx_rate = net.get("rx_rate_mbs", 0.0)
    tx_rate = net.get("tx_rate_mbs", 0.0)
    net_ip = net.get("ip", "192.168.100.90")
    proxy_type = proxy.get("type", "TUN") if proxy.get("enabled") else "Direct"

    col2_lines.append(f"[bold white]⇅ Network[/bold white]")
    col2_lines.append(f"  Down   [dim]━━━━━━━━━━━━[/dim] {rx_rate:.1f} MB/s")
    col2_lines.append(f"  Up     [dim]━━━━━[/dim][green]━[/green][dim]━━━━━━[/dim] {tx_rate:.1f} MB/s")
    col2_lines.append(f"  Proxy {proxy_type} · [bold cyan]{net_ip}[/bold cyan]")

    content_table.add_row("\n".join(col1_lines), "\n".join(col2_lines))

    # Main Container Panel
    full_table = Table(show_header=False, box=None, expand=True, padding=(0, 0))
    full_table.add_row(header_table)
    full_table.add_row(Text("─" * 78, style="dim"))
    full_table.add_row(content_table)

    console.print(Panel(full_table, border_style="magenta", expand=False))

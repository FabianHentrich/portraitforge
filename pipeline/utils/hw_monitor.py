"""
Hardware monitoring: CPU, RAM, GPU utilization, VRAM.

GPU metrics are queried via Windows Performance Counters
(works for AMD, NVIDIA, and Intel — no CUDA/ROCm required).
Results are cached for 3 seconds, since the PowerShell process
requires ~300-500 ms startup time.
"""
import logging
import subprocess
import time

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    logger.warning("psutil not installed — CPU/RAM display disabled. pip install psutil")

# Batch query: VRAM usage (Dedicated Usage) + GPU utilization (3D + Compute).
# One PowerShell process for all GPU metrics -> one startup overhead instead of three.
# engtype_3D     -> Rendering (Gradio, Browser)
# engtype_Compute -> DirectML / AI inference
_PS_BATCH = r"""
$ea = 'SilentlyContinue'
$vram = try {
    (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -EA Stop).CounterSamples |
    Measure-Object CookedValue -Sum | ForEach-Object { [long]$_.Sum }
} catch { -1 }
$g3d = try {
    (Get-Counter '\GPU Engine(*engtype_3D*)\Utilization Percentage' -EA Stop).CounterSamples |
    Measure-Object CookedValue -Sum | ForEach-Object { $_.Sum }
} catch { 0 }
$gcmp = try {
    (Get-Counter '\GPU Engine(*engtype_Compute_0*)\Utilization Percentage' -EA Stop).CounterSamples |
    Measure-Object CookedValue -Sum | ForEach-Object { $_.Sum }
} catch { 0 }
$gutil = [Math]::Round([Math]::Min(100, $g3d + $gcmp), 1)
Write-Output "$vram $gutil"
"""

_CACHE_TTL = 3.0   # Seconds between PowerShell calls
_cache_ts: float = 0.0
_cache_val: tuple[float, float] = (-1.0, -1.0)   # (vram_used_gb, gpu_util_pct)


def _query_gpu_windows() -> tuple[float, float]:
    """
    Returns (vram_used_gb, gpu_util_pct).
    Returns (-1, -1) if the query fails.
    Result is cached for _CACHE_TTL seconds.
    """
    global _cache_ts, _cache_val
    now = time.monotonic()
    if now - _cache_ts < _CACHE_TTL:
        return _cache_val

    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_BATCH],
            capture_output=True,
            text=True,
            timeout=8,
        )
        parts = proc.stdout.strip().split()
        if len(parts) >= 2 and parts[0] != "-1":
            vram_gb = int(parts[0]) / 1024 ** 3
            gpu_pct = float(parts[1])
            _cache_val = (vram_gb, gpu_pct)
        else:
            _cache_val = (-1.0, -1.0)
    except Exception as e:
        logger.debug(f"GPU metrics not available: {e}")
        _cache_val = (-1.0, -1.0)

    _cache_ts = now
    return _cache_val


def _color(pct: float) -> str:
    if pct >= 85:
        return "#e74c3c"   # red
    if pct >= 60:
        return "#f39c12"   # orange
    return "#2ecc71"       # green


def _mini_bar(pct: float, color: str, width_px: int = 56) -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div style="display:inline-block;background:#333;border-radius:3px;'
        f'height:6px;width:{width_px}px;vertical-align:middle;margin:0 5px 0 4px">'
        f'<div style="background:{color};width:{pct:.0f}%;height:6px;border-radius:3px"></div>'
        f'</div>'
    )


def status_html(vram_total_gb: float = 16.0) -> str:
    """
    Returns an HTML status line with:
      CPU %  |  RAM x.x / xx GB  |  GPU %  |  VRAM x.x / 16 GB

    Each metric has a color-coded mini bar.
    Unavailable values are omitted.
    """
    segments: list[str] = []

    # -- CPU & RAM (psutil) --------------------------------------------------------
    if _PSUTIL:
        cpu_pct = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        ram_used_gb = vm.used / 1024 ** 3
        ram_total_gb = vm.total / 1024 ** 3
        ram_pct = vm.percent

        c = _color(cpu_pct)
        segments.append(
            f'<span style="white-space:nowrap"><b>CPU</b>'
            f'{_mini_bar(cpu_pct, c)}'
            f'<span style="color:{c}">{cpu_pct:.0f}&thinsp;%</span></span>'
        )

        c = _color(ram_pct)
        segments.append(
            f'<span style="white-space:nowrap"><b>RAM</b>'
            f'{_mini_bar(ram_pct, c)}'
            f'<span style="color:{c}">{ram_used_gb:.1f}&thinsp;/&thinsp;{ram_total_gb:.0f}&thinsp;GB</span></span>'
        )

    # -- GPU & VRAM (Windows Performance Counters) ---------------------------------
    vram_used_gb, gpu_pct = _query_gpu_windows()

    if gpu_pct >= 0:
        c = _color(gpu_pct)
        segments.append(
            f'<span style="white-space:nowrap"><b>GPU</b>'
            f'{_mini_bar(gpu_pct, c)}'
            f'<span style="color:{c}">{gpu_pct:.0f}&thinsp;%</span></span>'
        )

    if vram_used_gb >= 0:
        vram_pct = (vram_used_gb / vram_total_gb) * 100
        c = _color(vram_pct)
        segments.append(
            f'<span style="white-space:nowrap"><b>VRAM</b>'
            f'{_mini_bar(vram_pct, c)}'
            f'<span style="color:{c}">{vram_used_gb:.1f}&thinsp;/&thinsp;{vram_total_gb:.0f}&thinsp;GB</span></span>'
        )

    if not segments:
        return (
            '<div style="font-family:monospace;font-size:0.82em;padding:4px 0;color:#888">'
            'Hardware metrics not available (install psutil)</div>'
        )

    divider = '<span style="color:#555">&ensp;|&ensp;</span>'
    return (
        f'<div style="font-family:monospace;font-size:0.82em;padding:6px 0;'
        f'display:flex;flex-wrap:wrap;align-items:center;gap:2px">'
        f'{divider.join(segments)}</div>'
    )

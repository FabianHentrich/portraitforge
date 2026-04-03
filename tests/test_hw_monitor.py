"""Tests für pipeline/utils/hw_monitor.py — pure Hilfsfunktionen und Caching."""
import pytest
import time
from unittest.mock import patch

import pipeline.utils.hw_monitor as hw


# ── _color() ─────────────────────────────────────────────────────────────────

def test_color_high_returns_red():
    assert hw._color(90) == "#e74c3c"


def test_color_medium_returns_orange():
    assert hw._color(70) == "#f39c12"


def test_color_low_returns_green():
    assert hw._color(30) == "#2ecc71"


def test_color_boundary_85_is_red():
    assert hw._color(85) == "#e74c3c"


def test_color_boundary_60_is_orange():
    assert hw._color(60) == "#f39c12"


def test_color_boundary_59_is_green():
    assert hw._color(59) == "#2ecc71"


# ── _mini_bar() ───────────────────────────────────────────────────────────────

def test_mini_bar_returns_string():
    result = hw._mini_bar(50, "#2ecc71")
    assert isinstance(result, str)


def test_mini_bar_contains_color():
    result = hw._mini_bar(50, "#e74c3c")
    assert "#e74c3c" in result


def test_mini_bar_clamps_above_100():
    result = hw._mini_bar(150, "#2ecc71")
    assert "100%" in result


def test_mini_bar_clamps_below_0():
    result = hw._mini_bar(-10, "#2ecc71")
    assert "0%" in result


def test_mini_bar_custom_width():
    result = hw._mini_bar(50, "#fff", width_px=80)
    assert "80px" in result


# ── _query_gpu_windows() — Caching ────────────────────────────────────────────

def test_query_gpu_caches_result(monkeypatch):
    """Zweiter Aufruf innerhalb TTL liefert gecachten Wert ohne subprocess."""
    hw._cache_ts = 0.0
    hw._cache_val = (-1.0, -1.0)

    call_count = [0]

    def fake_run(*args, **kwargs):
        call_count[0] += 1
        from unittest.mock import MagicMock
        proc = MagicMock()
        proc.stdout = "8589934592 45.0\n"  # 8 GB, 45%
        return proc

    monkeypatch.setattr("pipeline.utils.hw_monitor.subprocess.run", fake_run)

    result1 = hw._query_gpu_windows()
    result2 = hw._query_gpu_windows()  # Should use cache

    assert call_count[0] == 1  # subprocess called only once
    assert result1 == result2


def test_query_gpu_returns_minus_one_on_failure(monkeypatch):
    hw._cache_ts = 0.0
    hw._cache_val = (-1.0, -1.0)

    monkeypatch.setattr(
        "pipeline.utils.hw_monitor.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(Exception("fail")),
    )

    result = hw._query_gpu_windows()
    assert result == (-1.0, -1.0)


# ── status_html() ─────────────────────────────────────────────────────────────

def test_status_html_returns_string():
    result = hw.status_html()
    assert isinstance(result, str)


def test_status_html_contains_html_tags():
    result = hw.status_html()
    assert "<div" in result


def test_status_html_with_psutil_available(monkeypatch):
    """Wenn psutil verfügbar: CPU und RAM erscheinen im Output."""
    monkeypatch.setattr(hw, "_PSUTIL", True)

    class FakeVM:
        used = 8 * 1024 ** 3
        total = 32 * 1024 ** 3
        percent = 25.0

    import types
    fake_psutil = types.SimpleNamespace(
        cpu_percent=lambda interval: 42.0,
        virtual_memory=lambda: FakeVM(),
    )
    monkeypatch.setattr(hw, "psutil", fake_psutil)
    monkeypatch.setattr(hw, "_cache_ts", time.monotonic())  # skip subprocess
    monkeypatch.setattr(hw, "_cache_val", (-1.0, -1.0))

    result = hw.status_html()
    assert "CPU" in result
    assert "RAM" in result


def test_status_html_without_psutil(monkeypatch):
    monkeypatch.setattr(hw, "_PSUTIL", False)
    monkeypatch.setattr(hw, "_cache_ts", time.monotonic())
    monkeypatch.setattr(hw, "_cache_val", (-1.0, -1.0))

    result = hw.status_html()
    # Either fallback message or GPU-only; must still return a string
    assert isinstance(result, str)

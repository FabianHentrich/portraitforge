"""Tests für VRAMTracker: register_load/unload, used_gb, free_gb, summary."""
import pytest
from pipeline.utils.vram_manager import VRAMTracker


class _P:
    """Minimaler Dummy-Provider für Tests."""
    def __init__(self, name: str, vram_gb: float):
        self.name = name
        self.vram_gb = vram_gb


@pytest.fixture
def tracker():
    return VRAMTracker()


# ── Initialzustand ────────────────────────────────────────────────────────────

def test_initial_used_is_zero(tracker):
    assert tracker.used_gb == 0.0


def test_initial_free_is_total(tracker):
    assert tracker.free_gb == 16.0


def test_initial_providers_empty(tracker):
    assert tracker.loaded_providers == []


# ── register_load ─────────────────────────────────────────────────────────────

def test_register_load_increases_used(tracker):
    tracker.register_load(_P("SDXL", 7.0))
    assert tracker.used_gb == pytest.approx(7.0)


def test_register_load_adds_to_provider_list(tracker):
    tracker.register_load(_P("SDXL", 7.0))
    assert "SDXL" in tracker.loaded_providers


def test_register_load_multiple_providers(tracker):
    tracker.register_load(_P("SDXL", 7.0))
    tracker.register_load(_P("CodeFormer", 1.5))
    assert tracker.used_gb == pytest.approx(8.5)
    assert len(tracker.loaded_providers) == 2


def test_register_load_zero_vram_provider(tracker):
    tracker.register_load(_P("U2Net", 0.0))
    assert tracker.used_gb == 0.0
    assert "U2Net" in tracker.loaded_providers


# ── register_unload ───────────────────────────────────────────────────────────

def test_register_unload_decreases_used(tracker):
    p = _P("SDXL", 7.0)
    tracker.register_load(p)
    tracker.register_unload(p)
    assert tracker.used_gb == pytest.approx(0.0)


def test_register_unload_removes_from_list(tracker):
    p = _P("SDXL", 7.0)
    tracker.register_load(p)
    tracker.register_unload(p)
    assert "SDXL" not in tracker.loaded_providers


def test_register_unload_unknown_provider_does_not_raise(tracker):
    tracker.register_unload(_P("Ghost", 5.0))


def test_register_unload_partial(tracker):
    tracker.register_load(_P("A", 3.0))
    tracker.register_load(_P("B", 2.0))
    tracker.register_unload(_P("A", 3.0))
    assert tracker.used_gb == pytest.approx(2.0)
    assert "A" not in tracker.loaded_providers
    assert "B" in tracker.loaded_providers


# ── free_gb ───────────────────────────────────────────────────────────────────

def test_free_gb_decreases_on_load(tracker):
    tracker.register_load(_P("SDXL", 7.0))
    assert tracker.free_gb == pytest.approx(9.0)


def test_free_gb_never_negative(tracker):
    tracker.register_load(_P("Huge", 100.0))
    assert tracker.free_gb == 0.0


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_empty(tracker):
    s = tracker.summary()
    assert "nothing loaded" in s
    assert "16" in s


def test_summary_contains_provider_name(tracker):
    tracker.register_load(_P("SDXL", 7.0))
    assert "SDXL" in tracker.summary()


def test_summary_contains_used_gb(tracker):
    tracker.register_load(_P("SDXL", 7.0))
    assert "7.0" in tracker.summary()


# ── loaded_providers returns copy ─────────────────────────────────────────────

def test_loaded_providers_returns_copy(tracker):
    tracker.register_load(_P("A", 1.0))
    lst = tracker.loaded_providers
    lst.append("injected")
    assert "injected" not in tracker.loaded_providers

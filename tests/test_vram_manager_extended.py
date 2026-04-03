"""Tests für VRAMTracker: CPU-Offload, RAM-Check, _effective_vram."""
import pytest
from pipeline.utils.vram_manager import VRAMTracker, _effective_vram


class _P:
    """Minimaler Provider-Stub."""
    def __init__(self, name, vram_gb, cpu_offload=False, vram_gb_offloaded=0.0, ram_gb=0.0):
        self.name = name
        self.vram_gb = vram_gb
        self.cpu_offload = cpu_offload
        self.vram_gb_offloaded = vram_gb_offloaded
        self.ram_gb = ram_gb


# ── _effective_vram() ────────────────────────────────────────────────────────


def test_effective_vram_no_offload():
    """Ohne CPU-Offload → vram_gb direkt."""
    p = _P("SDXL", vram_gb=7.0)
    assert _effective_vram(p) == 7.0


def test_effective_vram_with_offload():
    """Mit CPU-Offload → vram_gb_offloaded wenn gesetzt."""
    p = _P("FLUX", vram_gb=12.0, cpu_offload=True, vram_gb_offloaded=7.0)
    assert _effective_vram(p) == 7.0


def test_effective_vram_offload_without_offloaded_value():
    """cpu_offload=True aber vram_gb_offloaded=0 → Fallback auf vram_gb."""
    p = _P("Old", vram_gb=8.0, cpu_offload=True, vram_gb_offloaded=0.0)
    assert _effective_vram(p) == 8.0


def test_effective_vram_offload_false_ignores_offloaded():
    """cpu_offload=False → vram_gb_offloaded wird ignoriert."""
    p = _P("Test", vram_gb=10.0, cpu_offload=False, vram_gb_offloaded=5.0)
    assert _effective_vram(p) == 10.0


# ── check_available() VRAM ───────────────────────────────────────────────────


def test_check_available_sufficient_vram():
    t = VRAMTracker()
    p = _P("Small", vram_gb=4.0)
    assert t.check_available(p) is True


def test_check_available_insufficient_vram():
    t = VRAMTracker()
    # Belege 14 GB, dann prüfe 4 GB (nur 2 frei)
    t.register_load(_P("Big", vram_gb=14.0))
    p = _P("Another", vram_gb=4.0)
    assert t.check_available(p) is False


def test_check_available_zero_vram_always_ok():
    t = VRAMTracker()
    t.register_load(_P("Full", vram_gb=16.0))
    p = _P("U2Net", vram_gb=0.0)
    assert t.check_available(p) is True


# ── check_available() RAM ────────────────────────────────────────────────────


def test_check_available_sufficient_ram():
    t = VRAMTracker()
    p = _P("FLUX", vram_gb=7.0, cpu_offload=True, vram_gb_offloaded=7.0, ram_gb=12.0)
    # 32 GB - 6 GB reserved = 26 GB frei, 12 GB needed → OK
    assert t.check_available(p) is True


def test_check_available_insufficient_ram():
    t = VRAMTracker()
    # Lade einen Provider der viel RAM braucht
    t.register_load(_P("Heavy", vram_gb=0.0, ram_gb=20.0))
    # Jetzt nur noch 6 GB RAM frei, aber 12 GB benötigt.
    # vram_gb muss >0 sein, sonst short-circuited check_available bei needed<=0.
    p = _P("FLUX", vram_gb=7.0, cpu_offload=True, vram_gb_offloaded=7.0, ram_gb=12.0)
    assert t.check_available(p) is False


def test_check_available_ram_zero_not_checked():
    """Provider ohne ram_gb → kein RAM-Check."""
    t = VRAMTracker()
    p = _P("NoRAM", vram_gb=2.0, ram_gb=0.0)
    assert t.check_available(p) is True


# ── RAM-Tracking bei register_load/unload ────────────────────────────────────


def test_register_load_tracks_ram():
    t = VRAMTracker()
    t.register_load(_P("FLUX", vram_gb=7.0, ram_gb=12.0))
    assert t.used_ram_gb == pytest.approx(12.0)


def test_register_unload_frees_ram():
    t = VRAMTracker()
    p = _P("FLUX", vram_gb=7.0, ram_gb=12.0)
    t.register_load(p)
    t.register_unload(p)
    assert t.used_ram_gb == pytest.approx(0.0)


def test_free_ram_gb_accounts_for_reserved():
    t = VRAMTracker()
    # 32 GB total - 6 GB reserved = 26 GB frei
    assert t.free_ram_gb == pytest.approx(26.0)


def test_free_ram_decreases_on_load():
    t = VRAMTracker()
    t.register_load(_P("A", vram_gb=0.0, ram_gb=10.0))
    assert t.free_ram_gb == pytest.approx(16.0)  # 26 - 10


# ── Offload-aware register_load ──────────────────────────────────────────────


def test_register_load_uses_effective_vram():
    """CPU-Offload-Provider wird mit reduziertem VRAM getrackt."""
    t = VRAMTracker()
    t.register_load(_P("FLUX", vram_gb=12.0, cpu_offload=True, vram_gb_offloaded=7.0, ram_gb=12.0))
    assert t.used_vram_gb == pytest.approx(7.0)
    assert t.used_ram_gb == pytest.approx(12.0)

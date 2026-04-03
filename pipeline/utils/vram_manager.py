import gc
import logging

logger = logging.getLogger(__name__)

_TOTAL_VRAM_GB = 16.0  # RX 9070 XT
_TOTAL_RAM_GB = 32.0   # System RAM
_RAM_RESERVED_GB = 6.0  # OS + Python + Gradio overhead


def _effective_vram(provider) -> float:
    """
    Returns the actual VRAM requirement of a provider.
    Takes CPU offload into account: if active and vram_gb_offloaded is set,
    the reduced value is used.
    """
    if getattr(provider, "cpu_offload", False):
        offloaded = getattr(provider, "vram_gb_offloaded", 0.0)
        if offloaded > 0:
            return offloaded
    return provider.vram_gb


class VRAMTracker:
    """
    Tracks which providers are currently loaded and how much VRAM/RAM (estimated) is in use.
    Based on provider.vram_gb / vram_gb_offloaded / ram_gb — no DirectML API call,
    since AMD/Windows does not provide programmatic access to used VRAM via PyTorch.

    Offload-aware: Providers with cpu_offload=True report the reduced VRAM requirement
    (vram_gb_offloaded) and the additional RAM requirement (ram_gb).
    """

    def __init__(self):
        self._loaded_vram: dict[str, float] = {}  # name -> effective vram_gb
        self._loaded_ram: dict[str, float] = {}    # name -> ram_gb

    def check_available(self, provider) -> bool:
        """Checks whether enough VRAM is estimated to be available for the provider. Returns True if OK."""
        needed = _effective_vram(provider)
        if needed <= 0:
            return True
        ok = True
        if needed > self.free_vram_gb:
            logger.warning(
                f"[VRAM] ⚠ {provider.name} requires ~{needed:.1f} GB, "
                f"but only ~{self.free_vram_gb:.1f} GB free ({self.summary()}). "
                f"OOM risk!"
            )
            ok = False
        # RAM check (CPU offload moves weights into RAM)
        ram_needed = getattr(provider, "ram_gb", 0.0)
        if ram_needed > 0 and ram_needed > self.free_ram_gb:
            logger.warning(
                f"[RAM] ⚠ {provider.name} requires ~{ram_needed:.1f} GB RAM, "
                f"but only ~{self.free_ram_gb:.1f} GB free. Swapping possible!"
            )
            ok = False
        return ok

    def register_load(self, provider) -> None:
        vram = _effective_vram(provider)
        ram = getattr(provider, "ram_gb", 0.0)
        self._loaded_vram[provider.name] = vram
        self._loaded_ram[provider.name] = ram
        logger.info(
            f"[VRAM] +{vram:.1f} GB — {provider.name} loaded"
            + (f" (+{ram:.1f} GB RAM)" if ram > 0 else "")
            + f" | {self.summary()}"
        )

    def register_unload(self, provider) -> None:
        vram = self._loaded_vram.pop(provider.name, 0.0)
        ram = self._loaded_ram.pop(provider.name, 0.0)
        gc.collect()
        logger.info(
            f"[VRAM] -{vram:.1f} GB — {provider.name} unloaded"
            + (f" (-{ram:.1f} GB RAM)" if ram > 0 else "")
            + f" | {self.summary()}"
        )

    @property
    def used_vram_gb(self) -> float:
        return sum(self._loaded_vram.values())

    @property
    def used_ram_gb(self) -> float:
        return sum(self._loaded_ram.values())

    @property
    def free_vram_gb(self) -> float:
        return max(0.0, _TOTAL_VRAM_GB - self.used_vram_gb)

    @property
    def free_ram_gb(self) -> float:
        return max(0.0, _TOTAL_RAM_GB - _RAM_RESERVED_GB - self.used_ram_gb)

    # Backward compatibility: existing callers use .free_gb / .used_gb
    @property
    def free_gb(self) -> float:
        return self.free_vram_gb

    @property
    def used_gb(self) -> float:
        return self.used_vram_gb

    @property
    def loaded_providers(self) -> list[str]:
        return list(self._loaded_vram.keys())

    def summary(self) -> str:
        if not self._loaded_vram:
            return f"VRAM 0.0/{_TOTAL_VRAM_GB:.0f} GB · RAM 0.0/{_TOTAL_RAM_GB - _RAM_RESERVED_GB:.0f} GB — nothing loaded"
        names = ", ".join(self._loaded_vram)
        return (
            f"VRAM {self.used_vram_gb:.1f}/{_TOTAL_VRAM_GB:.0f} GB · "
            f"RAM {self.used_ram_gb:.1f}/{_TOTAL_RAM_GB - _RAM_RESERVED_GB:.0f} GB — {names}"
        )


# Singleton — used by Orchestrator and app.py
vram_tracker = VRAMTracker()

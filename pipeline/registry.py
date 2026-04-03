from pipeline.base.base_generator import BaseGeneratorProvider
from pipeline.base.base_enhancer import BaseFaceRestoreProvider, BaseUpscaleProvider
from pipeline.base.base_background import BaseBackgroundProvider


class ProviderRegistry:
    """
    Zentrale Registrierung aller Provider.
    Adding a new provider: create file in providers/,
    register in providers/__init__.py. Nothing else to touch.

    Instances are cached (singleton per key). get_*() always returns
    the same instance — prevents orphaned loaded providers and VRAM
    leaks when orchestrator and app.py use the same provider.
    """

    def __init__(self):
        self._generators:   dict[str, type[BaseGeneratorProvider]]   = {}
        self._face_restore: dict[str, type[BaseFaceRestoreProvider]] = {}
        self._upscale:      dict[str, type[BaseUpscaleProvider]]     = {}
        self._background:   dict[str, type[BaseBackgroundProvider]]  = {}
        # Instanz-Caches (Singleton pro Key)
        self._gen_instances:  dict[str, BaseGeneratorProvider]   = {}
        self._fr_instances:   dict[str, BaseFaceRestoreProvider] = {}
        self._up_instances:   dict[str, BaseUpscaleProvider]     = {}
        self._bg_instances:   dict[str, BaseBackgroundProvider]  = {}

    def register_generator(self, key: str, cls: type[BaseGeneratorProvider]) -> None:
        self._generators[key] = cls

    def register_face_restore(self, key: str, cls: type[BaseFaceRestoreProvider]) -> None:
        self._face_restore[key] = cls

    def register_upscale(self, key: str, cls: type[BaseUpscaleProvider]) -> None:
        self._upscale[key] = cls

    def register_background(self, key: str, cls: type[BaseBackgroundProvider]) -> None:
        self._background[key] = cls

    def get_generator(self, key: str) -> BaseGeneratorProvider:
        if key not in self._generators:
            raise KeyError(f"Generator '{key}' not registered. Available: {list(self._generators)}")
        if key not in self._gen_instances:
            self._gen_instances[key] = self._generators[key]()
        return self._gen_instances[key]

    def get_face_restore(self, key: str) -> BaseFaceRestoreProvider:
        if key not in self._face_restore:
            raise KeyError(f"Face-Restore provider '{key}' not registered. Available: {list(self._face_restore)}")
        if key not in self._fr_instances:
            self._fr_instances[key] = self._face_restore[key]()
        return self._fr_instances[key]

    def get_upscale(self, key: str) -> BaseUpscaleProvider:
        if key not in self._upscale:
            raise KeyError(f"Upscale provider '{key}' not registered. Available: {list(self._upscale)}")
        if key not in self._up_instances:
            self._up_instances[key] = self._upscale[key]()
        return self._up_instances[key]

    def get_background(self, key: str) -> BaseBackgroundProvider:
        if key not in self._background:
            raise KeyError(f"Background provider '{key}' not registered. Available: {list(self._background)}")
        if key not in self._bg_instances:
            self._bg_instances[key] = self._background[key]()
        return self._bg_instances[key]

    def list_generators(self) -> dict[str, type]:
        return dict(self._generators)

    def list_face_restore(self) -> dict[str, type]:
        return dict(self._face_restore)

    def list_upscale(self) -> dict[str, type]:
        return dict(self._upscale)

    def list_background(self) -> dict[str, type]:
        return dict(self._background)


registry = ProviderRegistry()   # Singleton

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class EnhancerConfig:
    fidelity: float = 0.7
    extra: dict = field(default_factory=dict)


class BaseFaceRestoreProvider(ABC):
    name: str
    model_id: str
    vram_gb: float

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def restore(self, image: Image.Image, config: EnhancerConfig) -> Image.Image: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...


class BaseUpscaleProvider(ABC):
    name: str
    model_id: str
    vram_gb: float
    supported_scales: list[int]

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def upscale(self, image: Image.Image, scale: int = 4) -> Image.Image: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

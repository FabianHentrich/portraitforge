from abc import ABC, abstractmethod
from PIL import Image


class BaseBackgroundProvider(ABC):
    name: str
    model_id: str
    vram_gb: float

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def remove(self, image: Image.Image) -> Image.Image:
        """Returns RGBA image with transparent background."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

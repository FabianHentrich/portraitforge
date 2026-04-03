from pipeline.base.base_background import BaseBackgroundProvider
from PIL import Image


class BiRefNetProvider(BaseBackgroundProvider):
    """
    BiRefNet: High-quality background removal via ONNX DirectML.
    Status: Stub — not yet implemented.
    """

    name = "BiRefNet"
    model_id = "models/background/birefnet.onnx"
    vram_gb = 1.0

    def load(self) -> None:
        raise NotImplementedError("BiRefNetProvider is not yet implemented.")

    def unload(self) -> None:
        raise NotImplementedError("BiRefNetProvider is not yet implemented.")

    def remove(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError("BiRefNetProvider is not yet implemented.")

    @property
    def is_loaded(self) -> bool:
        return False

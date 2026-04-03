from pipeline.base.base_enhancer import BaseFaceRestoreProvider, EnhancerConfig
from PIL import Image


class GFPGANProvider(BaseFaceRestoreProvider):
    """
    GFPGAN: Face restoration via ONNX DirectML.
    Status: Stub — not yet implemented.
    """

    name = "GFPGAN"
    model_id = "models/enhancer/gfpgan.onnx"
    vram_gb = 1.0

    def load(self) -> None:
        raise NotImplementedError("GFPGANProvider is not yet implemented.")

    def unload(self) -> None:
        raise NotImplementedError("GFPGANProvider is not yet implemented.")

    def restore(self, image: Image.Image, config: EnhancerConfig) -> Image.Image:
        raise NotImplementedError("GFPGANProvider is not yet implemented.")

    @property
    def is_loaded(self) -> bool:
        return False

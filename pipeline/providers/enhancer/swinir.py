from pipeline.base.base_enhancer import BaseUpscaleProvider
from PIL import Image


class SwinIRProvider(BaseUpscaleProvider):
    """
    SwinIR: Transformer-based upscaling via ONNX DirectML.
    Status: Stub — not yet implemented.
    """

    name = "SwinIR"
    model_id = "models/enhancer/swinir.onnx"
    vram_gb = 1.0
    supported_scales = [4]

    def load(self) -> None:
        raise NotImplementedError("SwinIRProvider is not yet implemented.")

    def unload(self) -> None:
        raise NotImplementedError("SwinIRProvider is not yet implemented.")

    def upscale(self, image: Image.Image, scale: int = 4) -> Image.Image:
        raise NotImplementedError("SwinIRProvider is not yet implemented.")

    @property
    def is_loaded(self) -> bool:
        return False

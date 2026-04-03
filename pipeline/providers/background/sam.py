from pipeline.base.base_background import BaseBackgroundProvider
from PIL import Image


class SAMProvider(BaseBackgroundProvider):
    """
    Segment Anything Model (SAM): Interactive segmentation via torch-directml.
    Status: Stub — not yet implemented.
    """

    name = "SAM"
    model_id = "models/background/sam.pth"
    vram_gb = 2.5

    def load(self) -> None:
        raise NotImplementedError("SAMProvider is not yet implemented.")

    def unload(self) -> None:
        raise NotImplementedError("SAMProvider is not yet implemented.")

    def remove(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError("SAMProvider is not yet implemented.")

    @property
    def is_loaded(self) -> bool:
        return False

import gc
import logging
import time

from pipeline.base.base_background import BaseBackgroundProvider
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from rembg import new_session, remove
except ImportError:
    new_session = None  # type: ignore[assignment]
    remove = None       # type: ignore[assignment]


class U2NetProvider(BaseBackgroundProvider):
    """
    U2Net: Background removal via rembg.
    Source: automatically downloaded via rembg on first call -> ~/.u2net/
    No manual download required. One-time network access.
    VRAM: 0 GB — runs exclusively on CPU.
    """

    name = "U2Net"
    model_id = "u2net"
    vram_gb = 0.0

    def __init__(self):
        self._session = None

    def load(self) -> None:
        logger.info(f"Loading {self.name} (CPU)")
        try:
            self._session = new_session(self.model_id)
        except Exception as e:
            logger.error(f"{self.name}: Loading failed — {e}")
            raise
        logger.info(f"{self.name} loaded")

    def unload(self) -> None:
        self._session = None
        gc.collect()
        logger.info(f"{self.name} unloaded")

    def remove(self, image: Image.Image) -> Image.Image:
        """Returns an RGBA image with transparent background."""
        if self._session is None:
            raise RuntimeError(f"{self.name} is not loaded.")

        w, h = image.size
        logger.info(f"{self.name} — Remove background: {w}x{h}")
        t0 = time.monotonic()
        result = remove(image, session=self._session)
        logger.info(f"{self.name} — Done in {time.monotonic() - t0:.1f}s")
        return result

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

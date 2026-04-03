import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> Image.Image:
    """Load image, correct EXIF rotation, and convert to RGB."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        logger.error(f"Could not load image: {path} — {e}")
        raise
    return to_rgb(img)


def to_rgb(image: Image.Image) -> Image.Image:
    """Safely convert image to RGB."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def resize_to_max(image: Image.Image, max_size: int) -> Image.Image:
    """Proportionally resize image so the longest side equals max_size."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h), Image.LANCZOS)


# Maximum input size for various pipeline steps.
# Larger images are proportionally downscaled to save RAM.
_MAX_REFERENCE_PX = 1536     # Reference images (PhotoMaker): max 1536px longest side
_MAX_ENHANCER_INPUT_PX = 2048  # Enhancer input: max 2048px (may be upscaled afterwards)


def prepare_reference_images(
    images: list[Image.Image] | None,
    max_size: int = _MAX_REFERENCE_PX,
) -> list[Image.Image] | None:
    """
    Prepares reference images for the generator pipeline:
    - Converts to RGB
    - Downscales to max_size (saves RAM: 6000x4000 = 72 MB -> 1536x1024 = ~5 MB)
    - Returns None if no images are present
    """
    if not images:
        return None
    result = []
    for img in images:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            logger.info(f"Reference image {w}x{h} -> downscaled to max {max_size}px")
            img = resize_to_max(img, max_size)
        result.append(img)
    return result or None


def prepare_input_image(
    image: Image.Image,
    max_size: int = _MAX_ENHANCER_INPUT_PX,
) -> Image.Image:
    """
    Prepares an input image for Enhancer/Background:
    - Converts to RGB
    - Downscales to max_size if necessary
    """
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        logger.info(f"Input image {w}x{h} -> downscaled to max {max_size}px")
        image = resize_to_max(image, max_size)
    return image


def encode_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _cover_fit(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Scales the image using 'cover' mode (fill + center crop, no distortion)."""
    tw, th = target_size
    iw, ih = image.size
    scale = max(tw / iw, th / ih)
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # Center crop
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return image.crop((left, top, left + tw, top + th))


def composite_background(
    foreground: Image.Image,           # must be RGBA
    background: Image.Image | str,     # PIL.Image or path
) -> Image.Image:                      # returns RGB
    """
    Composite RGBA foreground onto background.
    Background is scaled using cover-fit (no distortion).
    """
    if foreground.mode != "RGBA":
        foreground = foreground.convert("RGBA")

    if isinstance(background, (str, Path)):
        try:
            bg = Image.open(background).convert("RGB")
        except Exception as e:
            logger.error(f"Could not load background image: {background} — {e}")
            raise
    else:
        bg = background.convert("RGB")

    bg = _cover_fit(bg, foreground.size)
    bg = bg.convert("RGBA")
    result = Image.alpha_composite(bg, foreground)
    return result.convert("RGB")

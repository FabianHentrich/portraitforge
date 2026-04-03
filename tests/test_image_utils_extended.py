"""Tests für prepare_reference_images, prepare_input_image, _cover_fit."""
import pytest
from PIL import Image

from pipeline.utils.image_utils import (
    prepare_reference_images,
    prepare_input_image,
    composite_background,
)


# ── prepare_reference_images() ───────────────────────────────────────────────


def test_prepare_reference_images_none():
    assert prepare_reference_images(None) is None


def test_prepare_reference_images_empty_list():
    assert prepare_reference_images([]) is None


def test_prepare_reference_images_converts_rgba_to_rgb():
    img = Image.new("RGBA", (100, 100))
    result = prepare_reference_images([img])
    assert result[0].mode == "RGB"


def test_prepare_reference_images_converts_grayscale_to_rgb():
    img = Image.new("L", (100, 100))
    result = prepare_reference_images([img])
    assert result[0].mode == "RGB"


def test_prepare_reference_images_no_resize_when_small():
    img = Image.new("RGB", (500, 500))
    result = prepare_reference_images([img], max_size=1536)
    assert result[0].size == (500, 500)


def test_prepare_reference_images_resize_when_large():
    img = Image.new("RGB", (4000, 3000))
    result = prepare_reference_images([img], max_size=1536)
    w, h = result[0].size
    assert max(w, h) == 1536


def test_prepare_reference_images_multiple():
    imgs = [Image.new("RGB", (100, 100)) for _ in range(4)]
    result = prepare_reference_images(imgs)
    assert len(result) == 4


def test_prepare_reference_images_preserves_aspect_ratio():
    img = Image.new("RGB", (4000, 2000))  # 2:1
    result = prepare_reference_images([img], max_size=1000)
    w, h = result[0].size
    assert w == 1000
    assert h == 500


# ── prepare_input_image() ────────────────────────────────────────────────────


def test_prepare_input_image_converts_to_rgb():
    img = Image.new("RGBA", (100, 100))
    result = prepare_input_image(img)
    assert result.mode == "RGB"


def test_prepare_input_image_no_resize_when_small():
    img = Image.new("RGB", (1000, 1000))
    result = prepare_input_image(img, max_size=2048)
    assert result.size == (1000, 1000)


def test_prepare_input_image_resize_when_large():
    img = Image.new("RGB", (4000, 3000))
    result = prepare_input_image(img, max_size=2048)
    w, h = result[0].size if isinstance(result, tuple) else result.size
    assert max(w, h) == 2048


# ── _cover_fit (via composite_background) ────────────────────────────────────


def test_composite_landscape_bg_portrait_fg():
    """Landscape-BG wird auf Portrait-FG zugeschnitten (Cover-Fit)."""
    fg = Image.new("RGBA", (600, 800), (0, 0, 0, 128))
    bg = Image.new("RGB", (1920, 1080))
    result = composite_background(fg, bg)
    assert result.size == (600, 800)
    assert result.mode == "RGB"


def test_composite_portrait_bg_landscape_fg():
    """Portrait-BG wird auf Landscape-FG zugeschnitten."""
    fg = Image.new("RGBA", (800, 600), (0, 0, 0, 128))
    bg = Image.new("RGB", (1080, 1920))
    result = composite_background(fg, bg)
    assert result.size == (800, 600)


def test_composite_same_aspect_no_crop():
    """Gleiche Seitenverhältnisse → kein Crop nötig."""
    fg = Image.new("RGBA", (512, 512), (0, 0, 0, 128))
    bg = Image.new("RGB", (1024, 1024))
    result = composite_background(fg, bg)
    assert result.size == (512, 512)


def test_composite_tiny_bg_upscaled():
    """Sehr kleines BG wird auf FG-Größe hochskaliert."""
    fg = Image.new("RGBA", (1024, 1024), (0, 0, 0, 128))
    bg = Image.new("RGB", (50, 50))
    result = composite_background(fg, bg)
    assert result.size == (1024, 1024)

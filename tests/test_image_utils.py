"""Tests für image_utils: composite_background(), resize, RGB-Konvertierung."""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

from pipeline.utils.image_utils import (
    load_image,
    to_rgb,
    resize_to_max,
    encode_base64,
    composite_background,
)


def make_rgb_image(w=100, h=80) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


def make_rgba_image(w=100, h=80) -> Image.Image:
    return Image.new("RGBA", (w, h), color=(128, 64, 32, 200))


def make_greyscale_image(w=50, h=50) -> Image.Image:
    return Image.new("L", (w, h), color=128)


def test_to_rgb_from_rgb():
    img = make_rgb_image()
    result = to_rgb(img)
    assert result.mode == "RGB"
    assert result is img  # Same object, no conversion


def test_to_rgb_from_rgba():
    img = make_rgba_image()
    result = to_rgb(img)
    assert result.mode == "RGB"


def test_to_rgb_from_greyscale():
    img = make_greyscale_image()
    result = to_rgb(img)
    assert result.mode == "RGB"


def test_resize_to_max_no_change_when_smaller():
    img = make_rgb_image(100, 80)
    result = resize_to_max(img, 200)
    assert result.size == (100, 80)
    assert result is img


def test_resize_to_max_landscape():
    img = make_rgb_image(400, 200)
    result = resize_to_max(img, 200)
    assert result.size[0] == 200
    assert result.size[1] == 100


def test_resize_to_max_portrait():
    img = make_rgb_image(200, 400)
    result = resize_to_max(img, 200)
    assert result.size[0] == 100
    assert result.size[1] == 200


def test_resize_to_max_square():
    img = make_rgb_image(300, 300)
    result = resize_to_max(img, 150)
    assert result.size == (150, 150)


def test_encode_base64_returns_string():
    img = make_rgb_image()
    result = encode_base64(img)
    assert isinstance(result, str)
    assert len(result) > 0


def test_encode_base64_is_decodable():
    import base64
    img = make_rgb_image()
    b64 = encode_base64(img)
    data = base64.b64decode(b64)
    decoded_img = Image.open(BytesIO(data))
    assert decoded_img is not None


def test_composite_background_returns_rgb():
    fg = make_rgba_image(100, 100)
    bg = make_rgb_image(200, 150)
    result = composite_background(fg, bg)
    assert result.mode == "RGB"


def test_composite_background_size_matches_foreground():
    fg = make_rgba_image(100, 80)
    bg = make_rgb_image(200, 150)
    result = composite_background(fg, bg)
    assert result.size == (100, 80)


def test_composite_background_with_path(tmp_path):
    bg_path = tmp_path / "bg.png"
    bg = make_rgb_image(200, 150)
    bg.save(str(bg_path))

    fg = make_rgba_image(100, 80)
    result = composite_background(fg, str(bg_path))
    assert result.mode == "RGB"
    assert result.size == (100, 80)


def test_composite_background_rgb_foreground_converted():
    """RGB Vordergrund wird zu RGBA konvertiert."""
    fg = make_rgb_image(100, 100)
    bg = make_rgb_image(100, 100)
    result = composite_background(fg, bg)
    assert result.mode == "RGB"

"""Tests für app.py-Hilfsfunktionen (keine Gradio-Abhängigkeit)."""
import pytest
from unittest.mock import MagicMock
from PIL import Image
import numpy as np

# Importiere direkt aus app (Gradio-Launch wird dabei nicht ausgelöst)
from app import (
    _estimate_tokens,
    _parse_resolution,
    _token_warning_html,
    _parse_reference_images,
    _provider_guidelines_html,
    RESOLUTION_PRESETS,
    SCHEDULER_PRESETS,
)


# ── _estimate_tokens() ────────────────────────────────────────────────────────

def test_estimate_tokens_empty():
    assert _estimate_tokens("") == 0


def test_estimate_tokens_whitespace_only():
    assert _estimate_tokens("   ") == 0


def test_estimate_tokens_single_word():
    assert _estimate_tokens("portrait") == 1


def test_estimate_tokens_comma_separated():
    # "portrait, photo, studio" → 3 Wörter + 2 Kommas = 5
    count = _estimate_tokens("portrait, photo, studio")
    assert count == 5


def test_estimate_tokens_returns_int():
    assert isinstance(_estimate_tokens("test"), int)


def test_estimate_tokens_longer_prompt():
    prompt = "portrait of a person, professional photo, sharp focus, studio lighting"
    count = _estimate_tokens(prompt)
    assert count > 0


# ── _parse_resolution() ───────────────────────────────────────────────────────

def test_parse_resolution_default():
    h, w = _parse_resolution("1024×1024 — Square")
    assert h == 1024
    assert w == 1024


def test_parse_resolution_portrait():
    h, w = _parse_resolution("832×1216  — Portrait 2:3")
    assert h == 832
    assert w == 1216


def test_parse_resolution_landscape():
    h, w = _parse_resolution("1216×832  — Landscape 3:2")
    assert h == 1216
    assert w == 832


def test_parse_resolution_unknown_key_returns_default():
    h, w = _parse_resolution("nonexistent preset")
    assert h == 1024
    assert w == 1024


def test_parse_resolution_all_presets_valid():
    for key in RESOLUTION_PRESETS:
        h, w = _parse_resolution(key)
        assert h > 0
        assert w > 0
        assert h % 64 == 0, f"{key}: height {h} not divisible by 64"
        assert w % 64 == 0, f"{key}: width {w} not divisible by 64"


# ── _token_warning_html() ─────────────────────────────────────────────────────

def test_token_warning_empty_prompt():
    assert _token_warning_html("", 77) == ""


def test_token_warning_whitespace_prompt():
    assert _token_warning_html("   ", 77) == ""


def test_token_warning_ok_returns_green():
    result = _token_warning_html("short prompt", 77)
    assert "#2ecc71" in result


def test_token_warning_near_limit_returns_orange():
    # Erstelle Prompt mit ~85 Tokens (über 77, unter 97)
    prompt = " ".join(["word"] * 82)
    result = _token_warning_html(prompt, 77)
    assert "#f39c12" in result


def test_token_warning_over_limit_returns_red():
    # Erstelle Prompt mit >97 Tokens
    prompt = " ".join(["word"] * 100)
    result = _token_warning_html(prompt, 77)
    assert "#e74c3c" in result


def test_token_warning_returns_html_string():
    result = _token_warning_html("test prompt", 77)
    assert "<div" in result


def test_token_warning_shows_count():
    result = _token_warning_html("hello world", 77)
    assert "2" in result  # 2 Tokens


def test_token_warning_flux_limit_512():
    # Langer Prompt ist OK für FLUX (512 Token-Limit)
    prompt = " ".join(["word"] * 100)
    result = _token_warning_html(prompt, 512)
    assert "#2ecc71" in result


# ── _parse_reference_images() ─────────────────────────────────────────────────

def test_parse_reference_images_none():
    assert _parse_reference_images(None) is None


def test_parse_reference_images_empty_list():
    assert _parse_reference_images([]) is None


def test_parse_reference_images_pil_image():
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    result = _parse_reference_images([img])
    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], Image.Image)
    assert result[0].mode == "RGB"


def test_parse_reference_images_numpy_array():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    result = _parse_reference_images([arr])
    assert result is not None
    assert len(result) == 1
    assert result[0].mode == "RGB"


def test_parse_reference_images_tuple_format():
    """Gradio Gallery gibt (image, caption)-Tupel zurück."""
    img = Image.new("RGB", (64, 64))
    result = _parse_reference_images([(img, "caption")])
    assert result is not None
    assert len(result) == 1


def test_parse_reference_images_multiple():
    imgs = [Image.new("RGB", (64, 64)) for _ in range(3)]
    result = _parse_reference_images(imgs)
    assert len(result) == 3


def test_parse_reference_images_converts_to_rgb():
    img = Image.new("RGBA", (64, 64), (100, 150, 200, 128))
    result = _parse_reference_images([img])
    assert result[0].mode == "RGB"


# ── _provider_guidelines_html() ──────────────────────────────────────────────

def _make_dummy_provider(prompt_template="", negative_prompt_hint="", max_tokens=77):
    p = MagicMock()
    p.prompt_template = prompt_template
    p.negative_prompt_hint = negative_prompt_hint
    p.max_prompt_tokens = max_tokens
    return p


def test_provider_guidelines_returns_string():
    result = _provider_guidelines_html(_make_dummy_provider())
    assert isinstance(result, str)


def test_provider_guidelines_contains_token_limit():
    result = _provider_guidelines_html(_make_dummy_provider(max_tokens=77))
    assert "77" in result


def test_provider_guidelines_shows_prompt_template():
    result = _provider_guidelines_html(_make_dummy_provider(prompt_template="portrait of img"))
    assert "portrait of img" in result


def test_provider_guidelines_shows_negative_prompt_hint():
    result = _provider_guidelines_html(_make_dummy_provider(negative_prompt_hint="blurry, ugly"))
    assert "blurry, ugly" in result


def test_provider_guidelines_orange_for_clip_limit():
    result = _provider_guidelines_html(_make_dummy_provider(max_tokens=77))
    assert "#f39c12" in result  # CLIP-Limit warnt in Orange


def test_provider_guidelines_green_for_large_limit():
    result = _provider_guidelines_html(_make_dummy_provider(max_tokens=512))
    assert "#2ecc71" in result  # FLUX-Limit ist OK in Grün


# ── SCHEDULER_PRESETS Vollständigkeit ─────────────────────────────────────────

def test_scheduler_presets_euler_present():
    assert "euler" in SCHEDULER_PRESETS.values()


def test_scheduler_presets_all_values_nonempty():
    for label, key in SCHEDULER_PRESETS.items():
        assert key, f"Leerer Key für '{label}'"

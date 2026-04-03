"""Tests für RealESRGANProvider: upscale() verdoppelt/vervierfacht Auflösung."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from PIL import Image

import pipeline.providers.enhancer.realesrgan as esrgan_module
from pipeline.providers.enhancer.realesrgan import RealESRGANProvider


@pytest.fixture
def provider():
    return RealESRGANProvider()


def _make_mock_ort(input_w=64, input_h=64):
    """Mock-ort-Modul für Real-ESRGAN x4."""
    mock_input = MagicMock()
    mock_input.name = "input"

    # Output: [1, 3, H*4, W*4]
    out_h = input_h * 4
    out_w = input_w * 4
    fake_output = np.random.rand(1, 3, out_h, out_w).astype(np.float32)

    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = [mock_input]
    mock_sess.run.return_value = [fake_output]

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_sess
    return mock_ort, mock_sess


def test_initial_state(provider):
    assert not provider.is_loaded
    assert provider._sess is None


def test_metadata(provider):
    assert provider.name == "Real-ESRGAN"
    assert "realesrgan.onnx" in provider.model_id
    assert provider.vram_gb == 1.5
    assert 4 in provider.supported_scales
    assert 2 in provider.supported_scales


def test_load_creates_session(provider, mocker):
    mock_ort, mock_sess = _make_mock_ort()
    mocker.patch.object(esrgan_module, "ort", mock_ort)
    provider.load()
    assert provider.is_loaded


def test_unload_clears_session(provider, mocker):
    mock_ort, _ = _make_mock_ort()
    mocker.patch.object(esrgan_module, "ort", mock_ort)
    provider.load()
    provider.unload()
    assert not provider.is_loaded


def test_upscale_x4_quadruples_resolution(provider, mocker):
    mock_ort, _ = _make_mock_ort(64, 64)
    mocker.patch.object(esrgan_module, "ort", mock_ort)
    provider.load()

    input_img = Image.new("RGB", (64, 64), (100, 150, 200))
    result = provider.upscale(input_img, scale=4)

    assert result.size == (256, 256)
    assert result.mode == "RGB"


def test_upscale_x2_doubles_resolution(provider, mocker):
    """scale=2: Ergebnis wird von 256 auf 128 zurück resized."""
    mock_ort, _ = _make_mock_ort(64, 64)
    mocker.patch.object(esrgan_module, "ort", mock_ort)
    provider.load()

    input_img = Image.new("RGB", (64, 64), (100, 150, 200))
    result = provider.upscale(input_img, scale=2)

    assert result.size == (128, 128)


def test_upscale_raises_on_unsupported_scale(provider, mocker):
    mock_ort, _ = _make_mock_ort()
    mocker.patch.object(esrgan_module, "ort", mock_ort)
    provider.load()

    with pytest.raises(ValueError):
        provider.upscale(Image.new("RGB", (64, 64)), scale=3)


def test_upscale_raises_when_not_loaded(provider):
    with pytest.raises(RuntimeError):
        provider.upscale(Image.new("RGB", (64, 64)))


def test_upscale_returns_pil_image(provider, mocker):
    mock_ort, _ = _make_mock_ort(32, 32)
    mocker.patch.object(esrgan_module, "ort", mock_ort)
    provider.load()

    result = provider.upscale(Image.new("RGB", (32, 32)), scale=4)
    assert isinstance(result, Image.Image)

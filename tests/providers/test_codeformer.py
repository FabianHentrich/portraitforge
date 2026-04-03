"""Tests für CodeFormerProvider: load/unload, restore() gibt PIL.Image zurück."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from PIL import Image

import pipeline.providers.enhancer.codeformer as cf_module
from pipeline.providers.enhancer.codeformer import CodeFormerProvider
from pipeline.base.base_enhancer import EnhancerConfig


@pytest.fixture
def provider():
    return CodeFormerProvider()


def _make_mock_ort(output_shape=(1, 3, 512, 512)):
    """Erstellt ein Mock-ort-Modul mit einer realistischen InferenceSession."""
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input2 = MagicMock()
    mock_input2.name = "w"

    fake_output = np.zeros(output_shape, dtype=np.float32)

    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = [mock_input, mock_input2]
    mock_sess.run.return_value = [fake_output]

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_sess
    return mock_ort, mock_sess


def test_initial_state(provider):
    assert not provider.is_loaded
    assert provider._sess is None


def test_metadata(provider):
    assert provider.name == "CodeFormer"
    assert "codeformer.onnx" in provider.model_id
    assert provider.vram_gb == 1.5


def test_load_creates_session(provider, mocker):
    mock_ort, mock_sess = _make_mock_ort()
    mocker.patch.object(cf_module, "ort", mock_ort)
    provider.load()
    assert provider.is_loaded
    assert provider._sess is mock_sess


def test_unload_clears_session(provider, mocker):
    mock_ort, mock_sess = _make_mock_ort()
    mocker.patch.object(cf_module, "ort", mock_ort)
    provider.load()
    provider.unload()
    assert not provider.is_loaded
    assert provider._sess is None


def test_restore_returns_pil_image(provider, mocker):
    mock_ort, _ = _make_mock_ort()
    mocker.patch.object(cf_module, "ort", mock_ort)
    provider.load()

    input_img = Image.new("RGB", (256, 256), (100, 150, 200))
    config = EnhancerConfig(fidelity=0.7)
    result = provider.restore(input_img, config)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


def test_restore_output_resized_to_input(provider, mocker):
    """Ergebnis wird auf Eingabegröße zurück resized."""
    mock_ort, _ = _make_mock_ort()
    mocker.patch.object(cf_module, "ort", mock_ort)
    provider.load()

    input_img = Image.new("RGB", (256, 192), (100, 150, 200))
    config = EnhancerConfig(fidelity=0.5)
    result = provider.restore(input_img, config)

    assert result.size == (256, 192)


def test_restore_raises_when_not_loaded(provider):
    with pytest.raises(RuntimeError):
        provider.restore(Image.new("RGB", (64, 64)), EnhancerConfig())


def test_restore_uses_directml_providers(provider, mocker):
    """InferenceSession wird mit DmlExecutionProvider aufgerufen."""
    mock_ort, _ = _make_mock_ort()
    mocker.patch.object(cf_module, "ort", mock_ort)
    provider.load()

    call_args = mock_ort.InferenceSession.call_args
    providers = call_args[1].get("providers") or call_args[0][1]
    assert "DmlExecutionProvider" in providers

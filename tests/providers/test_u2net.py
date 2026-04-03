"""Tests für U2NetProvider: load/unload/is_loaded, remove() gibt RGBA zurück."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from pipeline.providers.background.u2net import U2NetProvider


@pytest.fixture
def provider():
    return U2NetProvider()


def test_initial_state(provider):
    assert not provider.is_loaded
    assert provider._session is None


def test_metadata(provider):
    assert provider.name == "U2Net"
    assert provider.model_id == "u2net"
    assert provider.vram_gb == 0.0


def test_load_sets_session(provider, mocker):
    mock_session = MagicMock()
    mocker.patch("pipeline.providers.background.u2net.new_session", return_value=mock_session)
    provider.load()
    assert provider.is_loaded
    assert provider._session is mock_session


def test_unload_clears_session(provider, mocker):
    mock_session = MagicMock()
    mocker.patch("pipeline.providers.background.u2net.new_session", return_value=mock_session)
    provider.load()
    provider.unload()
    assert not provider.is_loaded
    assert provider._session is None


def test_remove_returns_rgba(provider, mocker):
    mock_session = MagicMock()
    mocker.patch("pipeline.providers.background.u2net.new_session", return_value=mock_session)

    rgba_result = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
    mocker.patch("pipeline.providers.background.u2net.remove", return_value=rgba_result)

    provider.load()
    input_img = Image.new("RGB", (64, 64), (200, 100, 50))
    result = provider.remove(input_img)

    assert result.mode == "RGBA"
    assert result.size == (64, 64)


def test_remove_raises_when_not_loaded(provider):
    with pytest.raises(RuntimeError):
        provider.remove(Image.new("RGB", (64, 64)))


def test_remove_passes_session_to_rembg(provider, mocker):
    mock_session = MagicMock()
    mocker.patch("pipeline.providers.background.u2net.new_session", return_value=mock_session)

    mock_remove = mocker.patch(
        "pipeline.providers.background.u2net.remove",
        return_value=Image.new("RGBA", (64, 64)),
    )

    provider.load()
    input_img = Image.new("RGB", (64, 64))
    provider.remove(input_img)

    mock_remove.assert_called_once_with(input_img, session=mock_session)

"""Tests für neue app.py-Hilfsfunktionen: _to_pil, _resolve_background."""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from app import _to_pil, _resolve_background


# ── _to_pil() ────────────────────────────────────────────────────────────────


def test_to_pil_from_pil():
    img = Image.new("RGB", (64, 64))
    result = _to_pil(img)
    assert result is img  # Gleiches Objekt, keine Konvertierung


def test_to_pil_from_ndarray():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    result = _to_pil(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)


def test_to_pil_from_rgba_ndarray():
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    result = _to_pil(arr)
    assert isinstance(result, Image.Image)


# ── _resolve_background() ───────────────────────────────────────────────────


def test_resolve_background_none():
    assert _resolve_background(None) is None


def test_resolve_background_none_string():
    assert _resolve_background("None") is None


def test_resolve_background_empty_string():
    assert _resolve_background("") is None


def test_resolve_background_existing_file(tmp_path, monkeypatch):
    """Existierende Datei → gibt Pfad als String zurück."""
    bg = tmp_path / "test_bg.png"
    bg.write_bytes(b"fake png")
    monkeypatch.chdir(tmp_path.parent)

    # Muss das assets/backgrounds-Verzeichnis mocken da _resolve_background
    # hardcoded "assets/backgrounds" nutzt
    bg_dir = tmp_path / "assets" / "backgrounds"
    bg_dir.mkdir(parents=True)
    bg_file = bg_dir / "office.png"
    bg_file.write_bytes(b"fake png")
    monkeypatch.chdir(tmp_path)

    result = _resolve_background("office.png")
    assert result is not None
    assert "office.png" in result


def test_resolve_background_missing_file_raises():
    """Nicht existierende Datei -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve_background("nonexistent_bg_12345.png")

"""Tests für PhotoMaker ref_scale_cap und Trigger-Token-Logik."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from pipeline.providers.generator.photomaker import PhotoMakerProvider
from pipeline.base.base_generator import GeneratorConfig


def _make_loaded_photomaker(mocker):
    """Erzeugt einen PhotoMaker-Provider mit gemockter Pipeline."""
    import pipeline.providers.generator.photomaker as pm_module

    mock_torch = MagicMock()
    mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
    mock_dml = MagicMock()
    mock_dml.device.return_value = "dml"
    mocker.patch.object(pm_module, "torch", mock_torch)
    mocker.patch.object(pm_module, "torch_directml", mock_dml)

    # Scheduler-Factory mocken → verhindert echten HF-Hub-Aufruf
    mocker.patch.object(pm_module, "SCHEDULERS", {"euler": lambda cfg: MagicMock()})

    p = PhotoMakerProvider()

    # Mock-Pipeline mit generate-Output
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (896, 896))]
    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_output

    p._pipe = mock_pipe
    p._device = "dml"
    return p, mock_pipe


# ── ref_scale_cap ────────────────────────────────────────────────────────────


def test_gen_scale_capped_with_3_refs(mocker):
    """3 Referenzbilder → gen_scale auf 0.875 gekappt (896×896 statt 1024×1024)."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    refs = [Image.new("RGB", (64, 64)) for _ in range(3)]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait of a person img", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    # 1024 × 0.875 = 896, gerundet auf 64er-Raster = 896
    assert call_kwargs["width"] == 896
    assert call_kwargs["height"] == 896


def test_gen_scale_capped_with_4_refs(mocker):
    """4 Referenzbilder → gen_scale ebenfalls auf 0.875 gekappt."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    refs = [Image.new("RGB", (64, 64)) for _ in range(4)]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait img", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["width"] == 896
    assert call_kwargs["height"] == 896


def test_gen_scale_not_capped_with_2_refs(mocker):
    """2 Referenzbilder → kein Cap, volle 1024×1024."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    refs = [Image.new("RGB", (64, 64)) for _ in range(2)]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait img", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["width"] == 1024
    assert call_kwargs["height"] == 1024


def test_gen_scale_not_capped_with_1_ref(mocker):
    """1 Referenzbild → kein Cap."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    refs = [Image.new("RGB", (64, 64))]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait img", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["width"] == 1024
    assert call_kwargs["height"] == 1024


def test_gen_scale_respects_max_gen_scale_over_ref_cap(mocker):
    """max_gen_scale (1.0) hat Vorrang wenn kleiner als ref_scale_cap (sollte nicht vorkommen)."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    # Manuell max_gen_scale reduzieren
    p.max_gen_scale = 0.75
    refs = [Image.new("RGB", (64, 64)) for _ in range(2)]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait img", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    # 0.75 ist kleiner als 1.0 (ref_cap bei 2 Refs) → 0.75 gewinnt
    assert call_kwargs["width"] == 768
    assert call_kwargs["height"] == 768


# ── Trigger-Token ────────────────────────────────────────────────────────────


def test_trigger_token_auto_appended(mocker):
    """Wenn 'img' fehlt, wird es automatisch ans Prompt-Ende angehängt."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    refs = [Image.new("RGB", (64, 64))]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait of a man in a suit", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["prompt"].endswith(", img")


def test_trigger_token_not_duplicated(mocker):
    """Wenn 'img' bereits im Prompt ist, wird es nicht doppelt angehängt."""
    p, mock_pipe = _make_loaded_photomaker(mocker)
    refs = [Image.new("RGB", (64, 64))]
    config = GeneratorConfig(width=1024, height=1024, seed=42)

    p.generate("portrait of a person img in a suit", "", config, refs)

    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["prompt"].count("img") == 1

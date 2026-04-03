"""Tests für alle Generator-Provider: Metadaten, load/unload/is_loaded."""
import pytest
from unittest.mock import MagicMock
from PIL import Image

import pipeline.providers.generator.photomaker as pm_module
import pipeline.providers.generator.sdxl_base as sdxl_module
import pipeline.providers.generator.flux as flux_module
import pipeline.providers.generator.sd35_medium as sd35_module

from pipeline.providers.generator.photomaker import PhotoMakerProvider
from pipeline.providers.generator.sdxl_base import SDXLBaseProvider
from pipeline.providers.generator.flux import FluxProvider
from pipeline.providers.generator.sd35_medium import SD35MediumProvider
from pipeline.base.base_generator import BaseGeneratorProvider, GeneratorConfig


ALL_PROVIDERS = [
    PhotoMakerProvider,
    SDXLBaseProvider,
    FluxProvider,
    SD35MediumProvider,
]


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_has_name(ProviderCls):
    p = ProviderCls()
    assert isinstance(p.name, str) and len(p.name) > 0


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_has_model_id(ProviderCls):
    p = ProviderCls()
    assert isinstance(p.model_id, str) and len(p.model_id) > 0


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_has_vram_gb(ProviderCls):
    p = ProviderCls()
    assert isinstance(p.vram_gb, float)
    assert p.vram_gb > 0


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_has_requires_reference(ProviderCls):
    p = ProviderCls()
    assert isinstance(p.requires_reference, bool)


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_has_prompt_hint(ProviderCls):
    p = ProviderCls()
    assert isinstance(p.prompt_hint, str) and len(p.prompt_hint) > 0


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_initial_not_loaded(ProviderCls):
    p = ProviderCls()
    assert not p.is_loaded


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_is_instance_of_base(ProviderCls):
    p = ProviderCls()
    assert isinstance(p, BaseGeneratorProvider)


def test_photomaker_requires_reference():
    assert PhotoMakerProvider().requires_reference is True


def test_sdxl_no_reference():
    assert SDXLBaseProvider().requires_reference is False


def test_flux_no_reference():
    assert FluxProvider().requires_reference is False


def test_sd35_no_reference():
    assert SD35MediumProvider().requires_reference is False


@pytest.mark.parametrize("ProviderCls", ALL_PROVIDERS)
def test_has_max_gen_scale(ProviderCls):
    p = ProviderCls()
    assert isinstance(p.max_gen_scale, float)
    assert 0.0 < p.max_gen_scale <= 1.0


def test_sd35_max_gen_scale_is_reduced():
    """SD3.5 muss gen_scale auf 0.75 begrenzen — Attention-OOM-Schutz auf DirectML."""
    assert SD35MediumProvider().max_gen_scale == 0.75


def test_sdxl_max_gen_scale_is_full():
    """SDXL-UNet skaliert linear — 1.0 ist sicher."""
    assert SDXLBaseProvider().max_gen_scale == 1.0


def test_photomaker_max_gen_scale_is_full():
    assert PhotoMakerProvider().max_gen_scale == 1.0



def _make_torch_dml_mocks():
    """Erstellt Mock-Objekte für torch und torch_directml."""
    mock_torch = MagicMock()
    mock_dml = MagicMock()
    mock_dml.device.return_value = "dml"
    return mock_torch, mock_dml


def _make_mock_pipe_cls():
    """Mock-Pipeline-Klasse die .images[0] zurückgibt."""
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (512, 512))]
    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_output
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value.to.return_value = mock_pipe
    return mock_cls, mock_pipe


def test_photomaker_load_unload(mocker):
    mock_torch, mock_dml = _make_torch_dml_mocks()
    mock_cls, mock_pipe = _make_mock_pipe_cls()
    mock_pipe.load_photomaker_adapter = MagicMock()
    # PhotoMaker lädt die Pipeline über einen lokalen community-Import, nicht via DiffusionPipeline.
    # Wir mocken den Import-Pfad der Pipeline-Klasse im community-Modul.
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch.object(pm_module, "torch", mock_torch)
    mocker.patch.object(pm_module, "torch_directml", mock_dml)
    # Mock des dynamischen Imports aus pipeline.community.photomaker_src.pipeline
    mock_pm_pipeline_module = MagicMock()
    mock_pm_pipeline_module.PhotoMakerStableDiffusionXLPipeline = mock_cls
    mocker.patch.dict(
        "sys.modules",
        {"pipeline.community.photomaker_src.pipeline": mock_pm_pipeline_module},
    )

    p = PhotoMakerProvider()
    p.load()
    assert p.is_loaded

    p.unload()
    assert not p.is_loaded


def test_sdxl_load_unload(mocker):
    mock_torch, mock_dml = _make_torch_dml_mocks()
    mock_cls, _ = _make_mock_pipe_cls()
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch.object(sdxl_module, "torch", mock_torch)
    mocker.patch.object(sdxl_module, "torch_directml", mock_dml)
    mocker.patch.object(sdxl_module, "StableDiffusionXLPipeline", mock_cls)

    p = SDXLBaseProvider()
    p.load()
    assert p.is_loaded

    p.unload()
    assert not p.is_loaded


def test_flux_load_unload(mocker):
    mock_torch, mock_dml = _make_torch_dml_mocks()
    mock_cls, _ = _make_mock_pipe_cls()
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch.object(flux_module, "torch", mock_torch)
    mocker.patch.object(flux_module, "torch_directml", mock_dml)
    mocker.patch.object(flux_module, "FluxPipeline", mock_cls)

    p = FluxProvider()
    p.load()
    assert p.is_loaded

    p.unload()
    assert not p.is_loaded



def test_sd35_load_unload(mocker):
    mock_torch, mock_dml = _make_torch_dml_mocks()
    mock_cls, _ = _make_mock_pipe_cls()
    mocker.patch("os.path.isdir", return_value=True)
    mocker.patch.object(sd35_module, "torch", mock_torch)
    mocker.patch.object(sd35_module, "torch_directml", mock_dml)
    mocker.patch.object(sd35_module, "StableDiffusion3Pipeline", mock_cls)

    p = SD35MediumProvider()
    p.load()
    assert p.is_loaded

    p.unload()
    assert not p.is_loaded


def test_generate_raises_when_not_loaded():
    p = PhotoMakerProvider()
    with pytest.raises(RuntimeError):
        p.generate("prompt", "neg", GeneratorConfig())

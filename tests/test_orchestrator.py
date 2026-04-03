"""Tests für PipelineOrchestrator.run_full_pipeline() mit gemockten Providern."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.base.base_generator import GeneratorConfig
from pipeline.base.base_enhancer import EnhancerConfig


def make_mock_generator(output_img=None):
    if output_img is None:
        # Muss GeneratorConfig-Defaults matchen (1024×1024), sonst greift Auto-Quality-Upscale
        output_img = Image.new("RGB", (1024, 1024))
    mock = MagicMock()
    mock.is_loaded = False   # Startet ungeladen; _ensure_loaded() ruft load() auf
    mock.name = "MockGenerator"
    mock.vram_gb = 7.0
    mock.cpu_offload = False
    mock.vram_gb_offloaded = 0.0
    mock.ram_gb = 0.0
    mock.generate.return_value = output_img
    # Nach load() soll is_loaded True sein (für _finish/unload)
    def _set_loaded():
        mock.is_loaded = True
    mock.load.side_effect = _set_loaded
    def _set_unloaded():
        mock.is_loaded = False
    mock.unload.side_effect = _set_unloaded
    return mock


def make_mock_face_restore(output_img=None):
    if output_img is None:
        output_img = Image.new("RGB", (1024, 1024))
    mock = MagicMock()
    mock.is_loaded = False
    mock.name = "MockFaceRestore"
    mock.vram_gb = 1.5
    mock.cpu_offload = False
    mock.vram_gb_offloaded = 0.0
    mock.ram_gb = 0.0
    mock.restore.return_value = output_img
    def _set_loaded():
        mock.is_loaded = True
    mock.load.side_effect = _set_loaded
    def _set_unloaded():
        mock.is_loaded = False
    mock.unload.side_effect = _set_unloaded
    return mock


def make_mock_upscale(output_img=None):
    if output_img is None:
        output_img = Image.new("RGB", (4096, 4096))
    mock = MagicMock()
    mock.is_loaded = False
    mock.name = "MockUpscale"
    mock.vram_gb = 1.5
    mock.cpu_offload = False
    mock.vram_gb_offloaded = 0.0
    mock.ram_gb = 0.0
    mock.upscale.return_value = output_img
    def _set_loaded():
        mock.is_loaded = True
    mock.load.side_effect = _set_loaded
    def _set_unloaded():
        mock.is_loaded = False
    mock.unload.side_effect = _set_unloaded
    return mock


def make_mock_background(output_img=None):
    if output_img is None:
        output_img = Image.new("RGBA", (1024, 1024))
    mock = MagicMock()
    mock.is_loaded = False
    mock.name = "MockBackground"
    mock.vram_gb = 0.0
    mock.cpu_offload = False
    mock.vram_gb_offloaded = 0.0
    mock.ram_gb = 0.0
    mock.remove.return_value = output_img
    def _set_loaded():
        mock.is_loaded = True
    mock.load.side_effect = _set_loaded
    def _set_unloaded():
        mock.is_loaded = False
    mock.unload.side_effect = _set_unloaded
    return mock


@pytest.fixture
def orchestrator():
    return PipelineOrchestrator()


def test_run_full_pipeline_generator_only(orchestrator, mocker):
    gen = make_mock_generator()
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)

    config = GeneratorConfig()
    result = orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="test",
        negative_prompt="",
        generator_key="dummy",
        generator_config=config,
    )

    assert "generated" in result
    assert "output" in result
    gen.load.assert_called_once()
    gen.generate.assert_called_once()
    gen.unload.assert_called_once()


def test_run_full_pipeline_with_face_restore(orchestrator, mocker):
    gen = make_mock_generator()
    fr = make_mock_face_restore()
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)
    mocker.patch("pipeline.orchestrator.registry.get_face_restore", return_value=fr)

    result = orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="test",
        negative_prompt="",
        generator_key="dummy",
        face_restore_key="codeformer",
        generator_config=GeneratorConfig(),
        enhancer_config=EnhancerConfig(),
    )

    assert "restored" in result
    fr.load.assert_called_once()
    fr.restore.assert_called_once()
    # Leichter Provider wird bei _finish() entladen (nicht sofort nach jedem Schritt)
    fr.unload.assert_called()


def test_run_full_pipeline_with_upscale(orchestrator, mocker):
    gen = make_mock_generator()
    up = make_mock_upscale()
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)
    mocker.patch("pipeline.orchestrator.registry.get_upscale", return_value=up)

    result = orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="test",
        negative_prompt="",
        generator_key="dummy",
        upscale_key="realesrgan",
        generator_config=GeneratorConfig(),
        enhancer_config=EnhancerConfig(),
    )

    assert "upscaled" in result
    up.load.assert_called_once()
    up.upscale.assert_called_once()
    # Leichter Provider wird bei _finish() entladen
    up.unload.assert_called()


def test_run_full_pipeline_with_background(orchestrator, mocker):
    gen = make_mock_generator()
    bg = make_mock_background()
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)
    mocker.patch("pipeline.orchestrator.registry.get_background", return_value=bg)

    result = orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="test",
        negative_prompt="",
        generator_key="dummy",
        background_key="u2net",
        generator_config=GeneratorConfig(),
        enhancer_config=EnhancerConfig(),
    )

    assert "final" in result
    bg.load.assert_called_once()
    bg.remove.assert_called_once()
    bg.unload.assert_called_once()


def test_run_full_pipeline_full_chain(orchestrator, mocker):
    gen_img = Image.new("RGB", (1024, 1024), "red")
    fr_img = Image.new("RGB", (1024, 1024), "green")
    up_img = Image.new("RGB", (4096, 4096), "blue")
    bg_img = Image.new("RGBA", (4096, 4096), (0, 0, 0, 0))

    gen = make_mock_generator(gen_img)
    fr = make_mock_face_restore(fr_img)
    up = make_mock_upscale(up_img)
    bg = make_mock_background(bg_img)

    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)
    mocker.patch("pipeline.orchestrator.registry.get_face_restore", return_value=fr)
    mocker.patch("pipeline.orchestrator.registry.get_upscale", return_value=up)
    mocker.patch("pipeline.orchestrator.registry.get_background", return_value=bg)

    result = orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="full pipeline test",
        negative_prompt="",
        generator_key="gen",
        face_restore_key="fr",
        upscale_key="up",
        background_key="bg",
        generator_config=GeneratorConfig(),
        enhancer_config=EnhancerConfig(),
    )

    # "generated" bleibt erhalten (kein Quality-Upscale nötig bei 1024×1024)
    assert result["generated"] is gen_img
    # "restored" wird vor Upscale aus RAM freigegeben (result.pop)
    assert "restored" not in result
    assert result["upscaled"] is up_img
    assert "final" in result
    assert "output" in result


def test_vram_switching_unloads_previous(orchestrator, mocker):
    """Bei Provider-Wechsel wird der vorherige schwere Provider entladen."""
    gen1 = make_mock_generator()
    gen1.is_loaded = True  # Simuliere geladenen Provider

    orchestrator._active_heavy = gen1

    gen2 = make_mock_generator()
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen2)

    orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="test",
        negative_prompt="",
        generator_key="gen2",
        generator_config=GeneratorConfig(),
    )

    # gen1 muss entladen worden sein (durch _ensure_loaded → _unload_heavy)
    gen1.unload.assert_called_once()


def test_optional_steps_not_called_when_none(orchestrator, mocker):
    """Optionale Schritte werden nicht aufgerufen wenn key=None."""
    gen = make_mock_generator()
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)

    mock_get_fr = mocker.patch("pipeline.orchestrator.registry.get_face_restore")
    mock_get_up = mocker.patch("pipeline.orchestrator.registry.get_upscale")
    mock_get_bg = mocker.patch("pipeline.orchestrator.registry.get_background")

    orchestrator.run_full_pipeline(
        reference_images=None,
        prompt="test",
        negative_prompt="",
        generator_key="gen",
        face_restore_key=None,
        upscale_key=None,
        background_key=None,
        generator_config=GeneratorConfig(),
    )

    mock_get_fr.assert_not_called()
    mock_get_up.assert_not_called()
    mock_get_bg.assert_not_called()

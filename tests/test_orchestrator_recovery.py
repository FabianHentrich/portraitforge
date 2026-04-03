"""Tests für OOM-Recovery, Quality-Upscale-Fallback und State-Konsistenz im Orchestrator."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.base.base_generator import GeneratorConfig


def _mock_provider(name="Mock", vram_gb=7.0, is_heavy=True):
    """Erstellt einen Mock-Provider mit load/unload State-Tracking."""
    mock = MagicMock()
    mock.name = name
    mock.vram_gb = vram_gb
    mock.cpu_offload = False
    mock.vram_gb_offloaded = 0.0
    mock.ram_gb = 0.0
    mock.is_loaded = False

    def _load():
        mock.is_loaded = True
    mock.load.side_effect = _load

    def _unload():
        mock.is_loaded = False
    mock.unload.side_effect = _unload
    return mock


# ── OOM-Recovery in generate() ───────────────────────────────────────────────


def test_generate_runtime_error_unloads_heavy(mocker):
    """RuntimeError (OOM) in generate() → Provider wird entladen, State sauber."""
    orch = PipelineOrchestrator()
    gen = _mock_provider("OOM-Gen")
    gen.generate.side_effect = RuntimeError("Could not allocate tensor")
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)

    with pytest.raises(RuntimeError, match="Could not allocate"):
        orch.generate("prompt", "", "dummy", GeneratorConfig())

    # Provider muss entladen sein
    gen.unload.assert_called_once()
    # _active_heavy muss zurückgesetzt sein
    assert orch._active_heavy is None


def test_generate_oom_allows_retry(mocker):
    """Nach OOM-Recovery kann ein neuer generate()-Aufruf sauber starten."""
    orch = PipelineOrchestrator()
    gen = _mock_provider("RetryGen")
    call_count = 0

    def _generate_with_oom(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("OOM")
        return Image.new("RGB", (1024, 1024))

    gen.generate.side_effect = _generate_with_oom
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)

    # Erster Aufruf: OOM
    with pytest.raises(RuntimeError):
        orch.generate("prompt", "", "dummy", GeneratorConfig())

    # Zweiter Aufruf: muss sauber laden und generieren
    result = orch.generate("prompt", "", "dummy", GeneratorConfig())
    assert result.size == (1024, 1024)
    # load() wurde 2× aufgerufen (einmal pro Versuch)
    assert gen.load.call_count == 2


def test_generate_non_runtime_error_not_caught(mocker):
    """Nicht-RuntimeError (z.B. ValueError) wird direkt durchgereicht."""
    orch = PipelineOrchestrator()
    gen = _mock_provider("ErrGen")
    gen.generate.side_effect = ValueError("bad config")
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)

    with pytest.raises(ValueError, match="bad config"):
        orch.generate("prompt", "", "dummy", GeneratorConfig())


# ── Quality-Upscale Fallback ─────────────────────────────────────────────────


def test_quality_upscale_noop_when_large_enough():
    """Kein Upscale wenn Bild bereits >= Zielauflösung."""
    orch = PipelineOrchestrator()
    img = Image.new("RGB", (1024, 1024))
    result = orch.quality_upscale(img, 1024, 1024)
    assert result is img  # Exakt gleiches Objekt, kein Processing


def test_quality_upscale_lanczos_fallback_when_no_realesrgan(mocker):
    """Fallback auf LANCZOS wenn Real-ESRGAN nicht registriert ist."""
    orch = PipelineOrchestrator()
    mocker.patch("pipeline.orchestrator.registry.list_upscale", return_value={})

    small = Image.new("RGB", (512, 512))
    result = orch.quality_upscale(small, 1024, 1024)

    assert result.size == (1024, 1024)


def test_quality_upscale_lanczos_fallback_on_exception(mocker):
    """Fallback auf LANCZOS wenn Real-ESRGAN bei Inferenz crasht."""
    orch = PipelineOrchestrator()
    up = _mock_provider("ESRGAN", vram_gb=1.5, is_heavy=False)
    up.upscale.side_effect = RuntimeError("OOM in tile")
    mocker.patch("pipeline.orchestrator.registry.list_upscale", return_value={"realesrgan": up})
    mocker.patch("pipeline.orchestrator.registry.get_upscale", return_value=up)

    small = Image.new("RGB", (512, 512))
    result = orch.quality_upscale(small, 1024, 1024)

    assert result.size == (1024, 1024)  # LANCZOS-Fallback liefert korrekte Größe


def test_quality_upscale_selects_scale_2_for_small_factor(mocker):
    """Faktor <= 2.0 → Real-ESRGAN ×2."""
    orch = PipelineOrchestrator()
    up = _mock_provider("ESRGAN", vram_gb=1.5)
    up.upscale.return_value = Image.new("RGB", (1024, 1024))
    mocker.patch("pipeline.orchestrator.registry.list_upscale", return_value={"realesrgan": up})
    mocker.patch("pipeline.orchestrator.registry.get_upscale", return_value=up)

    small = Image.new("RGB", (768, 768))
    orch.quality_upscale(small, 1024, 1024)

    up.upscale.assert_called_once_with(small, scale=2)


def test_quality_upscale_selects_scale_4_for_large_factor(mocker):
    """Faktor > 2.0 → Real-ESRGAN ×4."""
    orch = PipelineOrchestrator()
    up = _mock_provider("ESRGAN", vram_gb=1.5)
    up.upscale.return_value = Image.new("RGB", (1024, 1024))
    mocker.patch("pipeline.orchestrator.registry.list_upscale", return_value={"realesrgan": up})
    mocker.patch("pipeline.orchestrator.registry.get_upscale", return_value=up)

    small = Image.new("RGB", (256, 256))
    orch.quality_upscale(small, 1024, 1024)

    up.upscale.assert_called_once_with(small, scale=4)


# ── State-Konsistenz bei _unload Fehlern ─────────────────────────────────────


def test_unload_heavy_exception_clears_state():
    """Wenn unload() fehlschlägt, wird _active_heavy trotzdem zurückgesetzt."""
    orch = PipelineOrchestrator()
    broken = _mock_provider("BrokenProvider")
    broken.is_loaded = True
    broken.unload.side_effect = RuntimeError("unload crashed")
    orch._active_heavy = broken

    # Darf nicht crashen, muss State trotzdem aufräumen
    orch._unload_heavy()

    assert orch._active_heavy is None


def test_unload_all_light_exception_clears_list():
    """Wenn ein leichter Provider beim Entladen crasht, wird die Liste trotzdem geleert."""
    orch = PipelineOrchestrator()
    p1 = _mock_provider("Good", vram_gb=1.0)
    p1.is_loaded = True
    p2 = _mock_provider("Broken", vram_gb=1.0)
    p2.is_loaded = True
    p2.unload.side_effect = RuntimeError("crash")
    orch._loaded_light = [p1, p2]

    # Darf nicht crashen
    orch._unload_all_light()

    assert orch._loaded_light == []
    p1.unload.assert_called_once()  # Erster Provider wurde trotzdem entladen


def test_finish_clears_both_heavy_and_light():
    """finish() räumt _active_heavy UND _loaded_light auf."""
    orch = PipelineOrchestrator()
    heavy = _mock_provider("Heavy", vram_gb=8.0)
    heavy.is_loaded = True
    light = _mock_provider("Light", vram_gb=1.0)
    light.is_loaded = True
    orch._active_heavy = heavy
    orch._loaded_light = [light]

    orch.finish()

    assert orch._active_heavy is None
    assert orch._loaded_light == []
    heavy.unload.assert_called_once()
    light.unload.assert_called_once()


# ── run_full_pipeline Exception → finish() ───────────────────────────────────


def test_run_full_pipeline_exception_calls_finish(mocker):
    """Exception in der Pipeline → finish() wird aufgerufen → VRAM frei."""
    orch = PipelineOrchestrator()
    gen = _mock_provider("FailGen")
    gen.generate.side_effect = RuntimeError("OOM")
    mocker.patch("pipeline.orchestrator.registry.get_generator", return_value=gen)

    with pytest.raises(RuntimeError):
        orch.run_full_pipeline(
            reference_images=None,
            prompt="test",
            negative_prompt="",
            generator_key="dummy",
            generator_config=GeneratorConfig(),
        )

    # Provider muss entladen sein
    assert not gen.is_loaded
    assert orch._active_heavy is None

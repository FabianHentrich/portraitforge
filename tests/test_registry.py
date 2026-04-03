"""Tests für ProviderRegistry: Register, abrufen, list_*()"""
import pytest
from unittest.mock import MagicMock

from pipeline.registry import ProviderRegistry
from pipeline.base.base_generator import BaseGeneratorProvider, GeneratorConfig
from pipeline.base.base_enhancer import BaseFaceRestoreProvider, BaseUpscaleProvider, EnhancerConfig
from pipeline.base.base_background import BaseBackgroundProvider
from PIL import Image


class DummyGenerator(BaseGeneratorProvider):
    name = "Dummy Generator"
    model_id = "dummy/generator"
    vram_gb = 1.0
    requires_reference = False
    prompt_hint = "Test"

    def load(self): pass
    def unload(self): pass
    def generate(self, prompt, negative_prompt, config, reference_images=None):
        return Image.new("RGB", (64, 64))
    @property
    def is_loaded(self): return False


class DummyFaceRestore(BaseFaceRestoreProvider):
    name = "Dummy FaceRestore"
    model_id = "dummy/facerestore"
    vram_gb = 0.5

    def load(self): pass
    def unload(self): pass
    def restore(self, image, config): return image
    @property
    def is_loaded(self): return False


class DummyUpscale(BaseUpscaleProvider):
    name = "Dummy Upscale"
    model_id = "dummy/upscale"
    vram_gb = 0.5
    supported_scales = [2, 4]

    def load(self): pass
    def unload(self): pass
    def upscale(self, image, scale=4): return image
    @property
    def is_loaded(self): return False


class DummyBackground(BaseBackgroundProvider):
    name = "Dummy Background"
    model_id = "dummy/background"
    vram_gb = 0.0

    def load(self): pass
    def unload(self): pass
    def remove(self, image): return image.convert("RGBA")
    @property
    def is_loaded(self): return False


@pytest.fixture
def reg():
    return ProviderRegistry()


def test_register_and_list_generator(reg):
    reg.register_generator("dummy_gen", DummyGenerator)
    assert "dummy_gen" in reg.list_generators()


def test_register_and_list_face_restore(reg):
    reg.register_face_restore("dummy_fr", DummyFaceRestore)
    assert "dummy_fr" in reg.list_face_restore()


def test_register_and_list_upscale(reg):
    reg.register_upscale("dummy_up", DummyUpscale)
    assert "dummy_up" in reg.list_upscale()


def test_register_and_list_background(reg):
    reg.register_background("dummy_bg", DummyBackground)
    assert "dummy_bg" in reg.list_background()


def test_get_generator_returns_instance(reg):
    reg.register_generator("dummy_gen", DummyGenerator)
    instance = reg.get_generator("dummy_gen")
    assert isinstance(instance, DummyGenerator)


def test_get_generator_returns_cached_instance(reg):
    """Registry gibt immer dieselbe Instanz zurück (Singleton pro Key)."""
    reg.register_generator("dummy_gen", DummyGenerator)
    a = reg.get_generator("dummy_gen")
    b = reg.get_generator("dummy_gen")
    assert a is b


def test_get_face_restore_returns_instance(reg):
    reg.register_face_restore("dummy_fr", DummyFaceRestore)
    instance = reg.get_face_restore("dummy_fr")
    assert isinstance(instance, DummyFaceRestore)


def test_get_upscale_returns_instance(reg):
    reg.register_upscale("dummy_up", DummyUpscale)
    instance = reg.get_upscale("dummy_up")
    assert isinstance(instance, DummyUpscale)


def test_get_background_returns_instance(reg):
    reg.register_background("dummy_bg", DummyBackground)
    instance = reg.get_background("dummy_bg")
    assert isinstance(instance, DummyBackground)


def test_list_generators_empty(reg):
    assert reg.list_generators() == {}


def test_list_face_restore_empty(reg):
    assert reg.list_face_restore() == {}


def test_list_upscale_empty(reg):
    assert reg.list_upscale() == {}


def test_list_background_empty(reg):
    assert reg.list_background() == {}


def test_key_not_found_raises(reg):
    with pytest.raises(KeyError):
        reg.get_generator("nonexistent")


def test_multiple_generators(reg):
    reg.register_generator("gen1", DummyGenerator)
    reg.register_generator("gen2", DummyGenerator)
    assert len(reg.list_generators()) == 2


def test_list_returns_copy(reg):
    reg.register_generator("dummy_gen", DummyGenerator)
    listing = reg.list_generators()
    listing["injected"] = None
    assert "injected" not in reg.list_generators()

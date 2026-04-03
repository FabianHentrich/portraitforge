import gc
import logging
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from pipeline.registry import registry
from pipeline.utils.image_utils import (
    composite_background,
    prepare_reference_images,
    prepare_input_image,
)
from pipeline.utils.vram_manager import vram_tracker

logger = logging.getLogger(__name__)

# Threshold: providers with vram_gb above this value are considered "heavy".
# Heavy providers are never loaded simultaneously. Light providers (enhancers)
# may co-exist if the estimated total VRAM usage fits.
_HEAVY_THRESHOLD_GB = 3.0

_INTERMEDIATE_DIR = Path("outputs/.intermediates")


class PipelineOrchestrator:
    """
    Coordinates VRAM between modules and provides individual pipeline steps
    as standalone methods.

    Each public method (generate, face_restore, upscale, remove_background)
    handles VRAM scheduling itself via _ensure_loaded(). Callers (app.py)
    do not need to perform manual load/unload.

    Scheduling rules:
      - Heavy providers (vram_gb > 3 GB, e.g. generators): never loaded simultaneously.
      - Light providers (vram_gb <= 3 GB, e.g. CodeFormer + Real-ESRGAN): may be
        loaded simultaneously if the estimated total VRAM fits.
    """

    def __init__(self):
        self._active_heavy = None       # Currently loaded heavy provider
        self._loaded_light: list = []   # Currently loaded light providers

    # ──────────────────────────────────────────────────────────────────────────
    # VRAM Scheduling (internal)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_heavy(self, provider) -> bool:
        return provider.vram_gb > _HEAVY_THRESHOLD_GB

    def _unload_heavy(self) -> None:
        """Unloads the active heavy provider and frees VRAM."""
        if self._active_heavy is not None:
            try:
                if self._active_heavy.is_loaded:
                    self._active_heavy.unload()
            except Exception as e:
                logger.warning(f"Error unloading {self._active_heavy.name}: {e}")
            vram_tracker.register_unload(self._active_heavy)
            gc.collect()
            gc.collect()
            time.sleep(0.5)
        self._active_heavy = None

    def _unload_all_light(self) -> None:
        """Unloads all light providers."""
        for p in self._loaded_light:
            try:
                if p.is_loaded:
                    p.unload()
            except Exception as e:
                logger.warning(f"Error unloading {p.name}: {e}")
            vram_tracker.register_unload(p)
        self._loaded_light.clear()

    def _ensure_loaded(self, provider) -> None:
        """
        Ensures that a provider is loaded.
        Heavy providers: unloads the previous heavy provider + all light providers.
        Light providers: unloads the heavy provider if necessary, leaves other light ones loaded.

        State mutation (_active_heavy, _loaded_light) only happens AFTER successful load() —
        on exception the orchestrator state remains consistent.
        """
        if provider.is_loaded:
            return

        is_heavy = self._is_heavy(provider)
        if is_heavy:
            self._unload_all_light()
            self._unload_heavy()
        else:
            if self._active_heavy is not None:
                self._unload_heavy()
            if not vram_tracker.check_available(provider):
                self._unload_all_light()

        if not vram_tracker.check_available(provider):
            logger.warning(
                f"VRAM warning: {provider.name} requires ~{provider.vram_gb:.1f} GB, "
                f"only ~{vram_tracker.free_gb:.1f} GB free. Loading anyway (OOM possible)."
            )
        provider.load()
        # State mutation only after successful load()
        vram_tracker.register_load(provider)
        if is_heavy:
            self._active_heavy = provider
        else:
            self._loaded_light.append(provider)

    def finish(self) -> None:
        """Unloads all active providers. Called after pipeline completion."""
        self._unload_all_light()
        self._unload_heavy()

    # ──────────────────────────────────────────────────────────────────────────
    # Intermediate Caching
    # ──────────────────────────────────────────────────────────────────────────

    def _save_intermediate(self, image: Image.Image, label: str) -> Path:
        """Saves an intermediate result to disk (crash protection)."""
        _INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _INTERMEDIATE_DIR / f"{label}_{ts}.png"
        image.save(path, format="PNG")
        logger.debug(f"Intermediate saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Public Pipeline Steps
    # ──────────────────────────────────────────────────────────────────────────

    def compress_prompt(self, prompt: str, generator_key: str) -> tuple[str, bool]:
        """
        Compresses the prompt to the generator's token limit.
        Returns (prompt, was_compressed). Frees flan-t5 after compression.
        """
        gen = registry.get_generator(generator_key)
        try:
            from pipeline.utils.prompt_compressor import compressor
            token_limit = getattr(gen, "max_prompt_tokens", 77)
            prompt, was_compressed = compressor.compress(prompt, token_limit=token_limit)
            if was_compressed:
                logger.info(
                    f"Prompt compressed to <={token_limit} tokens "
                    f"({compressor.count_tokens(prompt)} tokens): '{prompt[:100]}...'"
                )
            compressor.unload()
            return prompt, was_compressed
        except Exception as e:
            logger.warning(f"Prompt compression failed, using original: {e}")
            return prompt, False

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        generator_key: str,
        generator_config,
        reference_images=None,
    ) -> Image.Image:
        """
        Generates an image. Loads/unloads the generator automatically via VRAM scheduling.
        Returns the native generation resolution (no upscale).
        Reference images are automatically downscaled to max 1536px (RAM protection).
        """
        gen = registry.get_generator(generator_key)
        # Preprocess reference images: resize saves RAM (6000x4000 DSLR -> 1536px)
        reference_images = prepare_reference_images(reference_images)
        self._ensure_loaded(gen)
        t0 = time.monotonic()
        try:
            result = gen.generate(prompt, negative_prompt, generator_config, reference_images)
        except RuntimeError as e:
            # OOM or DirectML error: unload provider immediately to free VRAM.
            # Without this cleanup the provider remains registered as _active_heavy
            # with a broken pipeline -> next attempt skips load() and fails again.
            logger.error(f"Generator '{generator_key}' RuntimeError: {e}")
            self._unload_heavy()
            gc.collect()
            raise
        logger.info(
            f"Generator '{generator_key}' finished in {time.monotonic() - t0:.1f}s, "
            f"output {result.size[0]}x{result.size[1]}"
        )
        self._save_intermediate(result, "01_generated")
        return result

    def quality_upscale(
        self,
        image: Image.Image,
        target_w: int,
        target_h: int,
    ) -> Image.Image:
        """
        Upscales a too-small image to the target resolution using Real-ESRGAN.
        Automatically selects x2 or x4. Fallback: LANCZOS.
        Automatically unloads the generator (heavy provider) before loading Real-ESRGAN.
        No-op if the image is already large enough.
        """
        w, h = image.size
        if w >= target_w and h >= target_h:
            return image

        scale_needed = max(target_w / w, target_h / h)
        logger.info(
            f"Quality upscale: {w}x{h} -> {target_w}x{target_h} "
            f"(factor {scale_needed:.2f})"
        )

        try:
            upscale_providers = registry.list_upscale()
            if "realesrgan" not in upscale_providers:
                raise KeyError("realesrgan not registered")

            up = registry.get_upscale("realesrgan")
            self._ensure_loaded(up)
            scale = 4 if scale_needed > 2.0 else 2
            result = up.upscale(image, scale=scale)

            if result.size != (target_w, target_h):
                result = result.resize((target_w, target_h), Image.LANCZOS)

            self._save_intermediate(result, "01b_quality_upscaled")
            logger.info(f"Quality upscale finished: {result.size[0]}x{result.size[1]}")
            return result

        except Exception as e:
            logger.warning(f"Real-ESRGAN failed, using LANCZOS: {e}")
            return image.resize((target_w, target_h), Image.LANCZOS)

    def face_restore(
        self,
        image: Image.Image,
        face_restore_key: str,
        enhancer_config=None,
    ) -> Image.Image:
        """
        Face restoration. Loads the provider automatically via VRAM scheduling.
        Input image is downscaled to max 2048px if necessary.
        """
        image = prepare_input_image(image)
        fr = registry.get_face_restore(face_restore_key)
        self._ensure_loaded(fr)
        t0 = time.monotonic()
        result = fr.restore(image, enhancer_config)
        logger.info(f"Face Restore '{face_restore_key}' finished in {time.monotonic() - t0:.1f}s")
        self._save_intermediate(result, "02_restored")
        return result

    def upscale(
        self,
        image: Image.Image,
        upscale_key: str,
        scale: int = 4,
    ) -> Image.Image:
        """
        Upscaling. Loads the provider automatically via VRAM scheduling.
        Real-ESRGAN may still be loaded from the quality_upscale step -> no reload needed.
        """
        up = registry.get_upscale(upscale_key)
        self._ensure_loaded(up)
        t0 = time.monotonic()
        result = up.upscale(image, scale=scale)
        logger.info(f"Upscale '{upscale_key}' x{scale} finished in {time.monotonic() - t0:.1f}s")
        self._save_intermediate(result, "03_upscaled")
        return result

    def remove_background(
        self,
        image: Image.Image,
        background_key: str,
        background_image=None,
    ) -> Image.Image:
        """
        Remove background and optionally replace it.
        Loads the provider via _ensure_loaded (consistent also for GPU providers like BiRefNet).
        Input image is downscaled to max 2048px if necessary.
        """
        image = prepare_input_image(image)
        bg = registry.get_background(background_key)
        self._ensure_loaded(bg)
        t0 = time.monotonic()
        rgba = bg.remove(image)
        logger.info(f"Background '{background_key}' finished in {time.monotonic() - t0:.1f}s")

        if background_image:
            result = composite_background(rgba, background_image)
        else:
            result = rgba

        self._save_intermediate(result, "04_background")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Full Pipeline (combines all steps)
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        reference_images,
        prompt: str,
        negative_prompt: str,
        generator_key: str,
        face_restore_key: str | None = None,
        upscale_key: str | None = None,
        background_key: str | None = None,
        background_image=None,
        generator_config=None,
        enhancer_config=None,
    ) -> dict:
        """
        Runs the complete pipeline: Generate -> Quality-Upscale -> Face Restore ->
        Upscale -> Background. Each step is optional (except Generate).

        Delegates to the individual public methods — no duplicated VRAM management.

        RAM optimization: intermediate results are removed from the result dict after
        saving to disk. Only `current` (= input for next step) and `output` (= final result)
        remain as PIL objects in RAM. For the gallery, images are reloaded from the saved
        intermediate files.
        """
        result = {}
        intermediate_paths: dict[str, Path] = {}
        t_pipeline = time.monotonic()

        target_w = generator_config.width if generator_config else 1024
        target_h = generator_config.height if generator_config else 1024

        steps_active = (
            ["Generator"]
            + (["Face Restore"] if face_restore_key else [])
            + (["Upscale"] if upscale_key else [])
            + (["Background"] if background_key else [])
        )
        total = len(steps_active)
        step = 1
        logger.info(f"Pipeline starting — steps: {' -> '.join(steps_active)}")

        try:
            # 1. Prompt compression
            prompt, was_compressed = self.compress_prompt(prompt, generator_key)
            if was_compressed:
                result["compressed_prompt"] = prompt

            # 2. Generate
            logger.info(f"[{step}/{total}] Generator — starting '{generator_key}' ...")
            current = self.generate(prompt, negative_prompt, generator_key, generator_config, reference_images)
            result["generated"] = current
            step += 1

            # 3. Quality upscale (automatic when generator produced at reduced resolution)
            gen_w, gen_h = current.size
            if gen_w < target_w or gen_h < target_h:
                logger.info(
                    f"Generator output {gen_w}x{gen_h} < target {target_w}x{target_h} "
                    f"— starting automatic quality upscale"
                )
                # generated is no longer needed — only current matters
                del result["generated"]
                current = self.quality_upscale(current, target_w, target_h)

            # 4. Face Restore (optional)
            if face_restore_key:
                logger.info(f"[{step}/{total}] Face Restore — starting '{face_restore_key}' ...")
                current = self.face_restore(current, face_restore_key, enhancer_config)
                result["restored"] = current
                step += 1

            # 5. Upscale (optional — user-requested, in addition to quality upscale)
            if upscale_key:
                logger.info(f"[{step}/{total}] Upscale — starting '{upscale_key}' ...")
                # Release previous intermediate image (upscale produces a much larger image)
                result.pop("restored", None)
                current = self.upscale(current, upscale_key)
                result["upscaled"] = current
                step += 1

            # 6. Unload all GPU providers before Background (CPU)
            self.finish()

            # 7. Background (optional)
            if background_key:
                logger.info(f"[{step}/{total}] Background — starting '{background_key}' ...")
                current = self.remove_background(current, background_key, background_image)
                result["final"] = current

        except Exception:
            # On error, unload all providers — don't leave VRAM blocked
            self.finish()
            raise

        # Cleanup
        self.finish()

        logger.info(f"Pipeline total: {time.monotonic() - t_pipeline:.1f}s")
        result["output"] = current
        return result

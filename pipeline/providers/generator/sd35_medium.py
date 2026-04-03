import gc
import logging
import os
import time

from pipeline.base.base_generator import BaseGeneratorProvider, GeneratorConfig
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import torch_directml
    from diffusers import StableDiffusion3Pipeline
    if hasattr(torch, "privateuseone") and not hasattr(torch.privateuseone, "empty_cache"):
        torch.privateuseone.empty_cache = lambda: None
except (ImportError, RuntimeError):
    torch = None                        # type: ignore[assignment]
    torch_directml = None               # type: ignore[assignment]
    StableDiffusion3Pipeline = None     # type: ignore[assignment]


class SD35MediumProvider(BaseGeneratorProvider):
    """
    Stable Diffusion 3.5 Medium (2.5B parameters).
    Source: huggingface.co/stabilityai/stable-diffusion-3.5-medium
    VRAM: ~6 GB (without T5-XXL encoder; T5 disabled to save VRAM)
    Backend: torch-directml (float16, no bfloat16 on DirectML)

    T5 encoder note:
        SD3.5 has three text encoders (CLIP-L, CLIP-G, T5-XXL).
        T5-XXL alone requires ~9 GB VRAM → passed as None.
        CLIP-L + CLIP-G are sufficient for good prompt following.
        For T5: place local model with T5 weights and enable cpu_offload.

    Scheduler:
        SD3.5 uses FlowMatchEulerDiscreteScheduler (flow matching, not a diffusion scheduler).
        Switching schedulers (euler, dpm++ etc.) is not meaningful for SD3 models
        and is ignored.

    Download:
        python scripts/download_models.py --models sd35_medium
    """

    name = "SD3.5 Medium"
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    _local_dir = "models/generator/sd35_medium"
    vram_gb = 6.0
    vram_gb_offloaded = 2.0   # Sequential offload: ~2 GB VRAM peak
    ram_gb = 6.0              # Full weights in RAM for sequential offload shuttle
    requires_reference = False
    cpu_offload = True
    max_prompt_tokens = 77   # CLIP limit; T5 would be 512 but is disabled
    # SD3.5 transformer scales quadratically with resolution. DirectML materializes the full
    # attention matrix (no Flash Attention). At 1024x1024, a single attention allocation
    # is ~1.6 GB → OOM on 16 GB. At 768x768 (0.75), matrices are ~44% smaller.
    max_gen_scale = 0.75
    prompt_hint = "No trigger token · CLIP limit 77 tokens · guidance 3.5–4.5 recommended · generates at 75% resolution"
    prompt_template = (
        "portrait of [subject], [style], [lighting], "
        "professional photo, sharp focus, highly detailed"
    )
    negative_prompt_hint = (
        "blurry, deformed, extra limbs, bad anatomy, watermark, "
        "text, low quality, cartoon, painting"
    )

    def __init__(self):
        self._pipe = None
        self._device = None

    def load(self) -> None:
        self._device = torch_directml.device()
        if not os.path.isdir(self._local_dir):
            raise FileNotFoundError(
                f"{self.name}: Model not found at '{self._local_dir}'. "
                "Please download first: python scripts/download_models.py --models sd35_medium"
            )
        logger.info(
            f"Loading {self.name} from '{self._local_dir}' on {self._device}"
            + (" — CPU offload active" if self.cpu_offload else "")
        )
        t0 = time.monotonic()
        try:
            self._pipe = StableDiffusion3Pipeline.from_pretrained(
                self._local_dir,
                torch_dtype=torch.float16,   # bfloat16 not supported by DirectML
                use_safetensors=True,
                # Disable T5-XXL: ~9 GB VRAM saved, CLIP-L+G handle encoding
                text_encoder_3=None,
                tokenizer_3=None,
            )
            if self.cpu_offload:
                # Sequential instead of model CPU offload: moves individual transformer blocks
                # to GPU instead of the entire model. Peak VRAM drops from ~6 GB to ~2 GB.
                # Slower (constant CPU↔GPU transfers), but VRAM-safe for 16 GB cards.
                self._pipe.enable_sequential_cpu_offload(device=self._device)
            else:
                self._pipe = self._pipe.to(self._device)

            # VAE tiling: SD3.5 VAE has 16 channels (vs. SDXL 4 channels) — larger intermediate tensors.
            # We disable tiling at 768x768 entirely (by setting min_size high),
            # since 16GB VRAM is sufficient for 768x768 VAE decode, avoiding all artificial tile seams.
            self._pipe.vae.enable_tiling()
            self._pipe.vae.tile_sample_min_size = 1024
            self._pipe.vae.tile_latent_min_size = 128
            self._pipe.vae.tile_overlap_factor = 0.5
            self._pipe.vae.force_upcast = True
            self._pipe.vae.enable_slicing()

        except Exception as e:
            logger.error(f"{self.name}: Loading failed — {e}")
            raise
        logger.info(f"{self.name} loaded in {time.monotonic() - t0:.1f}s")

    def unload(self) -> None:
        if self._pipe is not None:
            # CPU offload hooks (accelerate) hold strong references to transformer/VAE/CLIP.
            # Remove hooks first, otherwise GC cannot free the tensors.
            if self.cpu_offload:
                try:
                    from accelerate.hooks import remove_hook_from_module
                    remove_hook_from_module(self._pipe, recurse=True)
                except Exception as e:
                    logger.debug(f"{self.name}: Could not remove hooks — {e}")
            for attr in ("transformer", "vae", "text_encoder", "text_encoder_2",
                         "tokenizer", "tokenizer_2", "scheduler"):
                if hasattr(self._pipe, attr):
                    setattr(self._pipe, attr, None)
            del self._pipe
            self._pipe = None
            gc.collect()
            gc.collect()
            logger.info(f"{self.name} unloaded")

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        config: GeneratorConfig,
        reference_images: list[Image.Image] | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError(f"{self.name} is not loaded. Please call load() first.")

        # SD3.5 uses FlowMatchEulerDiscreteScheduler — other schedulers are not compatible
        if config.scheduler != "euler":
            logger.warning(
                f"{self.name}: Scheduler '{config.scheduler}' is ignored — "
                "SD3.5 uses FlowMatchEulerDiscreteScheduler (flow matching, not interchangeable)"
            )

        logger.info(
            f"{self.name} — Generate: {config.width}×{config.height}, "
            f"steps={config.num_steps}, guidance={config.guidance_scale}, seed={config.seed}"
        )
        logger.debug(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

        # Generation resolution: gen_scale × target resolution (rounded to 64px grid).
        # gen_scale is capped at max_gen_scale (SD3.5: 0.75 due to attention OOM on DirectML).
        gen_scale = config.extra.get("gen_scale", config.gen_scale)
        gen_scale = min(gen_scale, self.max_gen_scale)
        gen_w = round(config.width  * gen_scale / 64) * 64
        gen_h = round(config.height * gen_scale / 64) * 64
        if gen_w != config.width or gen_h != config.height:
            logger.info(
                f"{self.name}: Reduced resolution — generating {gen_w}×{gen_h} "
                f"(target {config.width}×{config.height}, upscale by pipeline)"
            )

        t0 = time.monotonic()
        generator = torch.Generator("cpu").manual_seed(config.seed) if config.seed is not None else None
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_steps,
            guidance_scale=config.guidance_scale,
            height=gen_h,
            width=gen_w,
            generator=generator,
            # T5 is disabled — but diffusers internally allocates a zero-padding tensor
            # for T5. Default max_sequence_length=256 → joint attention 4506 tokens.
            # With 77 → 4327 tokens, ~17% smaller attention matrix, significantly less peak VRAM.
            max_sequence_length=77,
        ).images[0]

        # No LANCZOS upscale here — the orchestrator uses Real-ESRGAN for quality upscale.
        # Returned at native generation resolution (gen_w × gen_h).
        logger.info(
            f"{self.name} — Done in {time.monotonic() - t0:.1f}s, "
            f"output {result.size[0]}×{result.size[1]}"
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

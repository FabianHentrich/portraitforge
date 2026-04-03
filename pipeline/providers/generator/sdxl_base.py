import gc
import logging
import os
import time

from pipeline.base.base_generator import BaseGeneratorProvider, GeneratorConfig
from PIL import Image

logger = logging.getLogger(__name__)

from pipeline.utils.schedulers import SCHEDULERS

try:
    import torch
    import torch_directml
    from diffusers import StableDiffusionXLPipeline
    # diffusers 0.36.0 calls torch.privateuseone.empty_cache() in enable_model_cpu_offload(),
    # which does not exist in the DirectML backend. Register a no-op before loading the pipeline.
    if hasattr(torch, "privateuseone") and not hasattr(torch.privateuseone, "empty_cache"):
        torch.privateuseone.empty_cache = lambda: None
except (ImportError, RuntimeError):
    torch = None                        # type: ignore[assignment]
    torch_directml = None               # type: ignore[assignment]
    StableDiffusionXLPipeline = None    # type: ignore[assignment]


class SDXLBaseProvider(BaseGeneratorProvider):
    """
    Stable Diffusion XL Base 1.0: Standard text-to-image.
    Source: huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    VRAM: ~7 GB
    Backend: torch-directml
    """

    name = "SDXL Base"
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    _local_dir = "models/generator/sdxl_base"
    vram_gb = 7.0
    vram_gb_offloaded = 5.2   # Model CPU offload: text encoder (~1.8 GB) in RAM
    ram_gb = 7.0              # Full weights in RAM for CPU offload shuttle
    requires_reference = False
    cpu_offload = True   # Text encoder (~1.8 GB) moved to RAM after prompt encoding → ~1.8 GB freed for UNet activations
    max_prompt_tokens = 77
    prompt_hint = "Comma-separated descriptors · max 77 tokens · no trigger token"
    prompt_template = (
        "portrait of [subject], [style], [lighting], [quality], "
        "professional photo, highly detailed"
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
                "Please download first: python scripts/download_models.py --models sdxl"
            )
        logger.info(
            f"Loading {self.name} from '{self._local_dir}' on {self._device}"
            + (" — CPU offload active" if self.cpu_offload else "")
        )
        t0 = time.monotonic()
        try:
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self._local_dir,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            if self.cpu_offload:
                self._pipe.enable_model_cpu_offload(device=self._device)
            else:
                self._pipe = self._pipe.to(self._device)
            # VAE tiling: SDXL VAE has 4 channels (sample_size=1024, block_out_channels=[128,256,512,512]).
            # At 1024x1024, the latent is 128x128. Tiling condition: latent_dim > tile_latent_min_size.
            # Tiling at 1024x1024 causes visible horizontal streaks at tile boundaries.
            # Fix: tile_latent_min_size=128 → 128 > 128 = False → no tiling at generation resolution.
            # SDXL VAE with 4 channels needs only ~500 MB for full decode at 1024x1024 — fits in 16 GB.
            # enable_tiling() remains as safety net for larger resolutions (e.g., after upscale).
            self._pipe.vae.enable_tiling()
            self._pipe.vae.tile_sample_min_size = 1024
            self._pipe.vae.tile_latent_min_size = 128
            self._pipe.vae.tile_overlap_factor = 0.5
            # force_upcast: VAE decoder runs in fp32, even if weights are fp16.
            # SDXL VAE systematically produces banding artifacts in fp16 (known issue).
            self._pipe.vae.force_upcast = True
            self._pipe.vae.enable_slicing()
            # "auto": chunk-based slicing — less aggressive than (1), but VRAM-safe
            self._pipe.enable_attention_slicing("auto")
            # SDPA: PyTorch 2.4 optimized attention — with try/except as DirectML support varies
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self._pipe.unet.set_attn_processor(AttnProcessor2_0())
                logger.info(f"{self.name}: SDPA attention enabled")
            except Exception as e:
                logger.debug(f"{self.name}: SDPA not available — {e}")
        except Exception as e:
            logger.error(f"{self.name}: Loading failed — {e}")
            raise
        logger.info(f"{self.name} loaded in {time.monotonic() - t0:.1f}s")

    def unload(self) -> None:
        if self._pipe is not None:
            # CPU offload hooks (accelerate) hold strong references to unet/VAE/CLIP.
            # Remove hooks first, otherwise GC cannot free the tensors.
            if self.cpu_offload:
                try:
                    from accelerate.hooks import remove_hook_from_module
                    remove_hook_from_module(self._pipe, recurse=True)
                except Exception as e:
                    logger.debug(f"{self.name}: Could not remove hooks — {e}")
            for attr in ("unet", "vae", "text_encoder", "text_encoder_2",
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

        if config.scheduler in SCHEDULERS:
            self._pipe.scheduler = SCHEDULERS[config.scheduler](self._pipe.scheduler.config)
            logger.info(f"{self.name}: Scheduler → {config.scheduler}")

        logger.info(
            f"{self.name} — Generate: {config.width}×{config.height}, "
            f"steps={config.num_steps}, scheduler={config.scheduler}, "
            f"guidance={config.guidance_scale}, seed={config.seed}"
        )
        logger.debug(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

        # Generation resolution: gen_scale × target resolution (rounded to 64px grid)
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
        ).images[0]

        logger.info(
            f"{self.name} — Done in {time.monotonic() - t0:.1f}s, "
            f"output {result.size[0]}×{result.size[1]}"
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

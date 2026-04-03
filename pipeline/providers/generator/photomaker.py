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
    if hasattr(torch, "privateuseone") and not hasattr(torch.privateuseone, "empty_cache"):
        torch.privateuseone.empty_cache = lambda: None
except (ImportError, RuntimeError):
    torch = None            # type: ignore[assignment]
    torch_directml = None   # type: ignore[assignment]


class PhotoMakerProvider(BaseGeneratorProvider):
    """
    PhotoMaker: Identity-preserving portrait generation via Community-Pipeline.
    Base model: SG161222/RealVisXL_V4.0 (SDXL)
    Adapter: TencentARC/PhotoMaker — photomaker-v1.bin → models/generator/photomaker-v1.bin
    VRAM: ~8 GB
    Backend: torch-directml

    Adapter download:
        from huggingface_hub import hf_hub_download
        hf_hub_download("TencentARC/PhotoMaker", "photomaker-v1.bin",
                        local_dir="models/generator")
    """

    name = "PhotoMaker (SDXL)"
    model_id = "SG161222/RealVisXL_V5.0"
    vram_gb = 8.0
    vram_gb_offloaded = 8.0   # No offload active → same as vram_gb
    ram_gb = 0.5              # Only PIL images and small caches
    requires_reference = True
    cpu_offload = False  # DirectML + cpu_offload: id_encoder device mismatch → load directly on device
    max_prompt_tokens = 77
    prompt_hint = "Trigger token 'img' required · max 77 tokens · upload reference photos"
    prompt_template = (
        "portrait of a person img, [clothing/setting], professional photo, "
        "sharp focus, studio lighting, highly detailed"
    )
    negative_prompt_hint = (
        "blurry, deformed, extra limbs, bad anatomy, watermark, "
        "text, low quality, cartoon, painting, ugly"
    )

    _local_dir = "models/generator/realvisxl_v5"
    _adapter_path = "models/generator/photomaker-v1.bin"
    # Locally downloaded Python source files from TencentARC/PhotoMaker.
    # Download: python scripts/download_models.py --models photomaker_pipeline
    _src_dir = "pipeline/community/photomaker_src"

    def __init__(self):
        self._pipe = None
        self._device = None

    def load(self) -> None:
        self._device = torch_directml.device()
        if not os.path.isdir(self._local_dir):
            raise FileNotFoundError(
                f"{self.name}: Model not found at '{self._local_dir}'. "
                "Please download first: python scripts/download_models.py --models photomaker photomaker_adapter"
            )
        if not os.path.isfile(os.path.join(self._src_dir, "pipeline.py")):
            raise FileNotFoundError(
                f"{self.name}: Pipeline code not found at '{self._src_dir}'. "
                "Please download first: python scripts/download_models.py --models photomaker_pipeline"
            )
        logger.info(
            f"Loading {self.name} from '{self._local_dir}' on {self._device}"
            + (" — CPU offload active" if self.cpu_offload else "")
        )
        t0 = time.monotonic()
        try:
            # Import directly from the locally downloaded package.
            # No custom_pipeline mechanism from diffusers — avoids HF requests at runtime
            # and solves the relative import problem (from . import PhotoMakerIDEncoder).
            from pipeline.community.photomaker_src.pipeline import (
                PhotoMakerStableDiffusionXLPipeline,
            )
            self._pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                self._local_dir,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            # Load adapter BEFORE CPU offload: load_lora_weights() internally calls
            # enable_model_cpu_offload() and needs the "bare" pipeline state for that.
            logger.info(f"{self.name}: Loading PhotoMaker adapter from '{self._adapter_path}'")
            self._pipe.load_photomaker_adapter(
                os.path.dirname(self._adapter_path),
                weight_name=os.path.basename(self._adapter_path),
                trigger_word="img",
                pm_version="v1",
            )
            if self.cpu_offload:
                self._pipe.enable_model_cpu_offload(device=self._device)
            else:
                self._pipe = self._pipe.to(self._device)
            # VAE tiling: Same SDXL VAE as SDXL Base (4 channels, sample_size=1024).
            # At 1024x1024 the latent is 128x128. Tiling causes streaks at tile boundaries.
            # tile_latent_min_size=128 → 128 > 128 = False → no tiling at generation resolution.
            # PhotoMaker without CPU offload: ~8 GB model + ~500 MB VAE decode = ~8.5 GB → fits in 16 GB.
            self._pipe.vae.enable_tiling()
            self._pipe.vae.tile_sample_min_size = 1024
            self._pipe.vae.tile_latent_min_size = 128
            self._pipe.vae.tile_overlap_factor = 0.5
            # force_upcast: VAE decoder in fp32 — SDXL VAE produces banding artifacts in fp16.
            self._pipe.vae.force_upcast = True
            self._pipe.vae.enable_slicing()
            self._pipe.enable_attention_slicing(1)      # 1 head per slice: maximum VRAM savings
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self._pipe.unet.set_attn_processor(AttnProcessor2_0())
                logger.info(f"{self.name}: SDPA attention enabled")
            except Exception as e:
                logger.debug(f"{self.name}: SDPA not available — {e}")
            logger.info(f"{self.name}: Adapter loaded (trigger_word='img')")
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
                         "tokenizer", "tokenizer_2", "scheduler",
                         "id_encoder", "fuse_module"):  # PhotoMaker-specific
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

        n_refs = len(reference_images) if reference_images else 0
        if config.scheduler in SCHEDULERS:
            self._pipe.scheduler = SCHEDULERS[config.scheduler](self._pipe.scheduler.config)
            logger.info(f"{self.name}: Scheduler → {config.scheduler}")

        logger.info(
            f"{self.name} — Generate: {config.width}x{config.height}, "
            f"steps={config.num_steps}, scheduler={config.scheduler}, "
            f"guidance={config.guidance_scale}, seed={config.seed}, "
            f"style_strength={config.extra.get('style_strength', 0.65)}, refs={n_refs}"
        )
        logger.debug(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

        if "img" not in prompt.split():
            prompt = prompt.strip().rstrip(",") + ", img"
            logger.info(f"{self.name}: Trigger token 'img' automatically appended")

        # Ensure encoders are on GPU (they may be on CPU from a previous generate() call)
        for attr in ("text_encoder", "text_encoder_2", "id_encoder"):
            mod = getattr(self._pipe, attr, None)
            if mod is not None:
                try:
                    if next(mod.parameters()).device.type == "cpu":
                        mod.to(self._device)
                        logger.debug(f"{self.name}: {attr} moved back to {self._device}")
                except StopIteration:
                    pass

        # Generation resolution: gen_scale x target resolution (rounded to 64px grid)
        gen_scale = config.extra.get("gen_scale", config.gen_scale)
        # With >=3 reference images, identity embeddings are 3-4x larger → cross-attention
        # in UNet uses more VRAM. During VAE decode, remaining space is insufficient for
        # the (1,256,1024,1024) fp32 tensor (~1 GB) in the last upsampler. At 896x896
        # this shrinks to ~768 MB. The orchestrator upscales to target resolution with Real-ESRGAN.
        ref_scale_cap = 0.875 if n_refs >= 3 else 1.0
        gen_scale = min(gen_scale, self.max_gen_scale, ref_scale_cap)
        if ref_scale_cap < 1.0 and gen_scale == ref_scale_cap:
            logger.info(
                f"{self.name}: {n_refs} reference images → gen_scale reduced to {ref_scale_cap} "
                f"(VRAM protection for VAE decode)"
            )
        gen_w = round(config.width  * gen_scale / 64) * 64
        gen_h = round(config.height * gen_scale / 64) * 64
        if gen_w != config.width or gen_h != config.height:
            logger.info(
                f"{self.name}: Reduced resolution — generating {gen_w}x{gen_h} "
                f"(target {config.width}x{config.height}, upscale by pipeline)"
            )

        # Callback: Move text encoder + ID encoder to CPU once embeddings are computed.
        # From step 0 only UNet + VAE need VRAM → ~2.5 GB freed.
        _self = self
        def _offload_encoders(pipe, step_idx: int, timestep, callback_kwargs: dict) -> dict:
            if step_idx == 0:
                for attr in ("text_encoder", "text_encoder_2", "id_encoder"):
                    mod = getattr(pipe, attr, None)
                    if mod is not None:
                        mod.to("cpu")
                gc.collect()
                logger.info(f"{_self.name}: Encoders moved to CPU after embedding computation (~2.5 GB freed)")
            return callback_kwargs

        t0 = time.monotonic()
        generator = torch.Generator("cpu").manual_seed(config.seed) if config.seed is not None else None
        result = self._pipe(
            prompt=prompt,
            input_id_images=reference_images,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_steps,
            guidance_scale=config.guidance_scale,
            height=gen_h,
            width=gen_w,
            style_strength_ratio=config.extra.get("style_strength", 0.65),
            generator=generator,
            callback_on_step_end=_offload_encoders,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        logger.info(
            f"{self.name} — Done in {time.monotonic() - t0:.1f}s, "
            f"output {result.size[0]}x{result.size[1]}"
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

import gc
import logging
import os
import time
from contextlib import contextmanager

from pipeline.base.base_generator import BaseGeneratorProvider, GeneratorConfig
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import torch_directml
    from diffusers import FluxPipeline
    if hasattr(torch, "privateuseone") and not hasattr(torch.privateuseone, "empty_cache"):
        torch.privateuseone.empty_cache = lambda: None
except (ImportError, RuntimeError):
    torch = None          # type: ignore[assignment]
    torch_directml = None # type: ignore[assignment]
    FluxPipeline = None   # type: ignore[assignment]


@contextmanager
def _no_mmap_safetensors():
    """
    Disables safetensors mmap when loading models.

    Workaround for Windows: safetensors crashes with Access Violation (0xC0000005)
    when mmapping large files (>2 GB). The FLUX transformer shards are ~10 GB
    per shard and reliably trigger this crash. With this patch, each shard file
    is read entirely into RAM instead of being mmapped.

    Peak RAM during loading: ~25 GB (model + largest shard as bytes + parsed tensors).
    Sufficient with 32 GB system RAM.

    The patch only affects the loading process within the ``with`` block and is
    cleanly reverted afterwards.
    """
    import diffusers.models.model_loading_utils as _mlu
    import diffusers.models.modeling_utils as _mu

    _orig = _mlu.load_state_dict

    def _patched(*args, **kwargs):
        kwargs["disable_mmap"] = True
        return _orig(*args, **kwargs)

    # Patch in both modules — _load_shard_file (modeling_utils) resolves the name
    # via the module global dict, so it must be replaced there as well.
    _mlu.load_state_dict = _patched
    _mu.load_state_dict = _patched
    try:
        yield
    finally:
        _mlu.load_state_dict = _orig
        _mu.load_state_dict = _orig


class FluxProvider(BaseGeneratorProvider):
    """
    FLUX.1-schnell: Highest image quality with few steps.
    Source: huggingface.co/black-forest-labs/FLUX.1-schnell
    VRAM: ~12 GB (without CPU offload), ~7 GB (with CPU offload, T5 in RAM)
    Backend: torch-directml
    Download: ~23 GB

    DirectML fix:
        FLUX weights are bfloat16. DirectML does not support bfloat16.
        The pipeline is loaded directly as fp16 (torch_dtype=float16), diffusers
        automatically casts the bf16 weights during loading. Additionally,
        safetensors mmap is disabled because Windows crashes with Access Violation
        on large files (>2 GB) — the same issue as with T5.
    """

    name = "FLUX.1-schnell"
    model_id = "black-forest-labs/FLUX.1-schnell"
    _local_dir = "models/generator/flux_schnell"
    vram_gb = 12.0
    vram_gb_offloaded = 7.0   # With CPU offload: ~7 GB VRAM, rest in RAM
    ram_gb = 12.0             # Full weights in RAM for CPU offload shuttle
    requires_reference = False
    cpu_offload = True    # T5 encoder (~4-5 GB) in RAM → ~7 GB VRAM instead of 12 GB
    # FLUX DiT at 1024x1024 needs ~12 GB; with CPU offload ~7 GB.
    # At 0.85 (896x896 or 832x1152) attention activations fit safely in 16 GB.
    max_gen_scale = 0.85
    max_prompt_tokens = 77
    prompt_hint = "max 77 tokens (CLIP) · 4-8 steps · negative prompt is ignored"
    prompt_template = (
        "A professional portrait of [description of person]. "
        "[Clothing]. [Setting/background]. [Lighting]. "
        "Photorealistic, sharp focus, high resolution."
    )
    negative_prompt_hint = "Warning: FLUX ignores the negative prompt — field is not evaluated."

    def __init__(self):
        self._pipe = None
        self._device = None
        self._t5_zero_embeds = None  # Cached zero tensor for T5 placeholder

    def load(self) -> None:
        self._device = torch_directml.device()
        if not os.path.isdir(self._local_dir):
            raise FileNotFoundError(
                f"{self.name}: Model not found at '{self._local_dir}'. "
                "Please download first: python scripts/download_models.py --models flux"
            )
        t0 = time.monotonic()
        logger.info(
            f"Loading {self.name} from '{self._local_dir}' as fp16 (mmap disabled)"
            + (" — CPU offload active" if self.cpu_offload else "")
        )
        try:
            # Step 1: Load pipeline as fp16, without T5-XXL.
            #
            # torch_dtype=float16: diffusers automatically casts bf16 weights
            # to fp16 during loading — no manual conversion needed.
            #
            # _no_mmap_safetensors(): safetensors mmap crashes on Windows (0xC0000005)
            # with large files. Affects the transformer shards (~10 GB per shard)
            # and T5 shards (~3.7 GB). By patching load_state_dict, each shard
            # is read entirely into RAM instead of being mmapped.
            #
            # text_encoder_2=None: T5-XXL disabled. FLUX works with CLIP
            # alone (77 tokens instead of 512), sufficient for portrait prompts.
            with _no_mmap_safetensors():
                self._pipe = FluxPipeline.from_pretrained(
                    self._local_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    text_encoder_2=None,
                    tokenizer_2=None,
                )

            # Step 2: Move to device
            if self.cpu_offload:
                self._pipe.enable_model_cpu_offload(device=self._device)
                logger.info(
                    f"{self.name}: CPU offload active — "
                    f"estimated VRAM usage ~7 GB (T5 encoder in RAM)"
                )
            else:
                self._pipe = self._pipe.to(self._device)

            # VAE tiling: FLUX VAE has 16 channels (like SD3.5) → large intermediate tensors.
            # At max_gen_scale=0.85, FLUX generates at ~870x870. 16 GB VRAM is sufficient for
            # VAE decode at this resolution without tiling. Tiling with 16-channel VAEs
            # causes visible streaks at tile boundaries (same issue as SD3.5).
            # Fix: tile_sample_min_size=1024 → tiling only activates at 1024x1024, so never
            # at our generation resolution. enable_tiling() remains as a safety net
            # in case someone manually forces higher resolutions.
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
            if self.cpu_offload:
                try:
                    from accelerate.hooks import remove_hook_from_module
                    remove_hook_from_module(self._pipe, recurse=True)
                except Exception as e:
                    logger.debug(f"{self.name}: Could not remove hooks — {e}")
            for attr in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
                if hasattr(self._pipe, attr):
                    setattr(self._pipe, attr, None)
            del self._pipe
            self._pipe = None
            self._t5_zero_embeds = None
            gc.collect()
            gc.collect()
            logger.info(f"{self.name} unloaded")

    def _encode_prompt_clip_only(self, prompt: str):
        """Encode prompt with CLIP only, returning zero T5 embeddings.

        diffusers 0.36.0 FluxPipeline.encode_prompt() calls _get_t5_prompt_embeds
        even when tokenizer_2=None, causing an AttributeError.  This method
        bypasses that path entirely.
        """
        pipe = self._pipe
        # CLIP tokenize + encode
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # With cpu_offload, modules move between CPU and GPU — generate embeddings
        # consistently on CPU, the pipeline moves them during forward pass.
        embed_device = "cpu" if self.cpu_offload else pipe.text_encoder.device

        text_input_ids = text_inputs.input_ids.to(pipe.text_encoder.device)
        clip_out = pipe.text_encoder(text_input_ids, output_hidden_states=False)
        pooled_prompt_embeds = clip_out.pooler_output.to(
            dtype=pipe.text_encoder.dtype, device=embed_device,
        )

        # T5 embeddings: cached zero tensor with correct shape
        # FLUX expects (batch, seq_len, 4096) from T5-XXL
        if self._t5_zero_embeds is None or self._t5_zero_embeds.device != embed_device:
            self._t5_zero_embeds = torch.zeros(
                1, 512, 4096,
                dtype=pipe.transformer.dtype,
                device=embed_device,
            )
        prompt_embeds = self._t5_zero_embeds

        return prompt_embeds, pooled_prompt_embeds

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        config: GeneratorConfig,
        reference_images: list[Image.Image] | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError(f"{self.name} is not loaded. Please call load() first.")

        # Generation resolution: gen_scale x target resolution (rounded to 64px grid)
        gen_scale = config.extra.get("gen_scale", config.gen_scale)
        gen_scale = min(gen_scale, self.max_gen_scale)
        gen_w = round(config.width  * gen_scale / 64) * 64
        gen_h = round(config.height * gen_scale / 64) * 64

        logger.info(
            f"{self.name} — Generate: {gen_w}x{gen_h}, "
            f"steps={config.num_steps}, seed={config.seed} "
            f"(guidance and negative_prompt are ignored)"
        )
        logger.debug(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

        t0 = time.monotonic()
        generator = torch.Generator("cpu").manual_seed(config.seed) if config.seed is not None else None

        # T5 encoder is disabled (tokenizer_2=None), but diffusers 0.36.0
        # calls _get_t5_prompt_embeds anyway → AttributeError.
        # Fix: Generate CLIP embeddings manually, T5 embeddings as zero tensor,
        # and pass prompt_embeds + pooled_prompt_embeds directly.
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt_clip_only(prompt)

        # FLUX.1-schnell: no CFG — guidance_scale and negative_prompt are ignored
        result = self._pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=config.num_steps,
            height=gen_h,
            width=gen_w,
            generator=generator,
            guidance_scale=0.0,
        ).images[0]

        logger.info(
            f"{self.name} — Done in {time.monotonic() - t0:.1f}s, "
            f"output {result.size[0]}x{result.size[1]}"
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

import logging
import random
import re
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

# Pipeline imports
import pipeline.providers  # Registers all providers
from pipeline.registry import registry
from pipeline.base.base_generator import GeneratorConfig
from pipeline.base.base_enhancer import EnhancerConfig
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.utils.vram_manager import vram_tracker
from pipeline.utils import hw_monitor

logging.basicConfig(
    level=logging.WARNING,  # Silence third-party loggers (httpx, gradio, huggingface_hub)
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Set own loggers to INFO
logging.getLogger("pipeline").setLevel(logging.INFO)
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Suppress known harmless warnings
# torch.amp warns about CUDA even though we use DirectML (no CUDA on AMD)
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'", category=UserWarning)
# transformers warns when prompt tokens exceed model maximum — handled by PromptCompressor
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than", category=UserWarning)
# huggingface_hub warns about symlinks on Windows — no action needed
warnings.filterwarnings("ignore", message=".*HF_HUB_DISABLE_SYMLINKS_WARNING.*", category=UserWarning)
warnings.filterwarnings("ignore", message="cache-system uses symlinks", category=UserWarning)

orchestrator = PipelineOrchestrator()

_OUTPUT_DIR = Path("outputs")
_OUTPUT_DIR.mkdir(exist_ok=True)


def _auto_save(image: Image.Image, prefix: str = "result") -> Path:
    """Automatically save result to outputs/. Returns the file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _OUTPUT_DIR / f"{prefix}_{ts}.png"
    image.save(path, format="PNG")
    logger.info(f"Result saved: {path}")
    return path

# Startup inventory
logger.info("=" * 55)
logger.info("PortraitForge — Provider Inventory")
logger.info("=" * 55)
for _cat, _items in [
    ("Generator", registry.list_generators()),
    ("Face-Restore", registry.list_face_restore()),
    ("Upscale", registry.list_upscale()),
    ("Background", registry.list_background()),
]:
    for _key, _cls in _items.items():
        _offload = " · CPU-Offload" if getattr(_cls, "cpu_offload", False) else ""
        _vram = getattr(_cls, "vram_gb", 0)
        _name = getattr(_cls, "name", _key)
        logger.info(f"  [{_cat:<12}] {_key:<15} — {_name}  ({_vram:.1f} GB{_offload})")
logger.info("=" * 55)

# Active generator key (for manual load/unload in the Generator tab).
# The instance itself lives in the registry cache — no separate object,
# so that Orchestrator and app.py always see the same instance.
_active_gen_key: str | None = None
_active_max_tokens: int = 77   # updated on provider change


def _estimate_tokens(text: str) -> int:
    """
    Estimates the token count for CLIP-based models.
    CLIP tokenizes at sub-word level: words + punctuation separately.
    """
    return len(re.findall(r"\w+|[^\w\s]", text.strip()))


def _provider_guidelines_html(provider) -> str:
    """Returns formatted prompt guidelines as HTML. Expects a provider instance."""
    lines = []

    if provider.prompt_template:
        lines.append(f"<b>Structure:</b> <code>{provider.prompt_template}</code>")

    if provider.negative_prompt_hint:
        lines.append(f"<b>Negative Prompt:</b> <code>{provider.negative_prompt_hint}</code>")

    token_color = "#f39c12" if provider.max_prompt_tokens <= 77 else "#2ecc71"
    lines.append(
        f"<b>Token Limit:</b> "
        f"<span style='color:{token_color};font-weight:bold'>{provider.max_prompt_tokens} Tokens</span>"
        + (" — Prompt will be silently truncated!" if provider.max_prompt_tokens <= 77 else "")
    )

    body = "<br>".join(lines)
    return (
        f'<div style="font-family:monospace;font-size:0.82em;'
        f'background:#1e1e1e;border-radius:6px;padding:10px 12px;'
        f'border-left:3px solid {token_color};margin-top:4px">'
        f"{body}</div>"
    )


# SDXL-native resolutions (multiples of 64, ~1 MP — trained on these values)
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "1024×1024 — Square":        (1024, 1024),
    "832×1216  — Portrait 2:3":  (832,  1216),
    "1216×832  — Landscape 3:2": (1216, 832),
    "1152×896  — 4:3":           (1152, 896),
    "896×1152  — 3:4":           (896,  1152),
}
_DEFAULT_RES = "1024×1024 — Square"

# Scheduler options (only SDXL-based providers; FLUX ignores these)
SCHEDULER_PRESETS: dict[str, str] = {
    "Euler (Standard)":           "euler",
    "DPM++ 2M  — 15–20 Steps":   "dpm++2m",
    "DPM++ 2M Karras — 15 Steps": "dpm++2m_karras",
    "Euler Ancestral":            "euler_a",
    "DDIM":                       "ddim",
}
_DEFAULT_SCHED = "Euler (Standard)"


def _parse_resolution(preset: str) -> tuple[int, int]:
    """Returns (width, height) for a preset key."""
    return RESOLUTION_PRESETS.get(preset, (1024, 1024))


def _token_warning_html(prompt: str, max_tokens: int) -> str:
    """Returns a token counter banner (empty if everything is OK)."""
    if not prompt or not prompt.strip():
        return ""
    count = _estimate_tokens(prompt)
    if count <= max_tokens:
        color, icon, msg = "#2ecc71", "✓", f"{count} / {max_tokens} Tokens"
    elif count <= max_tokens + 20:
        color, icon, msg = "#f39c12", "⚠", f"{count} / {max_tokens} Tokens — close to limit, consider shortening"
    else:
        color, icon, msg = "#e74c3c", "✗", (
            f"{count} / {max_tokens} Tokens — "
            f"approx. {count - max_tokens} tokens will be truncated!"
        )
    return (
        f'<div style="font-family:monospace;font-size:0.82em;padding:4px 10px;'
        f'border-radius:4px;background:#1e1e1e;border-left:3px solid {color};margin-top:2px">'
        f'<span style="color:{color}">{icon}</span> {msg}</div>'
    )


def _to_pil(image) -> Image.Image:
    """Normalize Gradio input (ndarray or PIL) to PIL.Image."""
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _parse_reference_images(reference_images) -> list[Image.Image] | None:
    """Normalize Gradio gallery output to a list of PIL RGB images."""
    if not reference_images:
        return None
    result = []
    for item in reference_images:
        img = item[0] if isinstance(item, tuple) else item
        if isinstance(img, str):
            img = Image.open(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        result.append(ImageOps.exif_transpose(img).convert("RGB"))
    return result or None


def _resolve_background(background_choice: str | None) -> str | None:
    """Resolve background choice to file path. Returns None if no background selected."""
    if not background_choice or background_choice == "None":
        return None
    bg_path = Path("assets/backgrounds") / background_choice
    if not bg_path.exists():
        raise FileNotFoundError(f"Background image not found: {bg_path}")
    return str(bg_path)


def get_background_choices() -> list[str]:
    """Load background images from assets/backgrounds/."""
    bg_dir = Path("assets/backgrounds")
    if not bg_dir.exists():
        return ["None"]
    choices = ["None"] + [f.name for f in bg_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    return choices if choices else ["None"]


# ─── Generator Tab ────────────────────────────────────────────────────────────

def on_generator_change(provider_key: str):
    """On provider change: update metadata and UI."""
    global _active_gen_key, _active_max_tokens
    providers = registry.list_generators()
    if provider_key not in providers:
        return gr.update(visible=False), "", "<p>No provider selected</p>", ""

    prov = registry.get_generator(provider_key)
    _active_max_tokens = prov.max_prompt_tokens
    ref_visible = prov.requires_reference
    loaded = "loaded ✓" if prov.is_loaded else "not loaded"
    status = f"{prov.name}: {prov.vram_gb} GB — {loaded}"
    return (
        gr.update(visible=ref_visible),
        _provider_guidelines_html(prov),
        status,
        "",   # Clear token warning on provider change
    )


def on_load_generator(provider_key: str):
    """Load generator model via Orchestrator VRAM scheduling."""
    global _active_gen_key
    providers = registry.list_generators()
    if provider_key not in providers:
        gr.Warning("No provider selected.")
        return gr.update(value="No provider selected"), hw_monitor.status_html()

    try:
        # Orchestrator handles: unloads previous heavy provider, loads new one
        gen = registry.get_generator(provider_key)
        orchestrator._ensure_loaded(gen)
        _active_gen_key = provider_key
        return gr.update(value=f"{gen.name}: {gen.vram_gb} GB — loaded ✓"), hw_monitor.status_html()
    except Exception as e:
        _active_gen_key = None
        orchestrator.finish()
        logger.exception("Error in on_load_generator")
        gr.Error(f"Error loading model: {e}")
        return gr.update(value=f"Error: {e}"), hw_monitor.status_html()


def on_unload_generator():
    """Unload generator model via Orchestrator."""
    global _active_gen_key
    if _active_gen_key is not None:
        gen = registry.get_generator(_active_gen_key)
        if gen.is_loaded:
            name = gen.name
            orchestrator.finish()
            return gr.update(value=f"{name}: unloaded"), hw_monitor.status_html()
    return gr.update(value="No model loaded"), hw_monitor.status_html()


def on_random_seed():
    return random.randint(0, 2**32 - 1)


def on_generate(
    provider_key: str,
    reference_images,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    style_strength: float,
    seed: int | None,
    resolution: str = _DEFAULT_RES,
    scheduler: str = _DEFAULT_SCHED,
):
    """
    Generation via Orchestrator. No manual VRAM management —
    orchestrator.generate() and quality_upscale() handle everything.
    """
    global _active_gen_key, _active_max_tokens
    if _active_gen_key is None:
        gr.Warning("Please load the model first.")
        return None, prompt

    try:
        width, height = _parse_resolution(resolution)

        # Compress prompt
        prompt, was_compressed = orchestrator.compress_prompt(prompt, _active_gen_key)
        if was_compressed:
            gr.Info(f"Prompt compressed to <={_active_max_tokens} tokens")

        config = GeneratorConfig(
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed) if seed is not None else None,
            height=height,
            width=width,
            scheduler=SCHEDULER_PRESETS.get(scheduler, "euler"),
            extra={"style_strength": style_strength},
        )

        ref_imgs = _parse_reference_images(reference_images)

        # Orchestrator handles VRAM: lazy reload if needed (_ensure_loaded)
        result = orchestrator.generate(prompt, negative_prompt, _active_gen_key, config, ref_imgs)

        # Quality upscale: Orchestrator automatically unloads generator, loads Real-ESRGAN
        result = orchestrator.quality_upscale(result, width, height)

        # Unload lightweight providers (Real-ESRGAN) after quality upscale.
        # Generator is NOT reloaded immediately (lazy reload on next generate).
        orchestrator.finish()

        _auto_save(result, prefix="gen")
        return result, prompt
    except Exception as e:
        # OOM recovery: unload provider to free VRAM so the next
        # attempt can start cleanly (otherwise is_loaded stays True with broken pipe).
        orchestrator.finish()
        logger.exception("Error in on_generate")
        gr.Error(str(e))
        return None, prompt


# ─── Enhancer Tab ─────────────────────────────────────────────────────────────

def on_enhance(
    input_image,
    face_restore_key: str,
    upscale_key: str,
    fidelity: float,
    operations: list[str],
):
    """
    Enhancer via Orchestrator. _ensure_loaded automatically unloads
    the active generator and coordinates CodeFormer + Real-ESRGAN co-loading.
    """
    if input_image is None:
        gr.Warning("Please upload an image.")
        return None

    try:
        img = _to_pil(input_image)
        current = img

        enhancer_config = EnhancerConfig(fidelity=fidelity)

        if "Face Restore" in operations and face_restore_key:
            current = orchestrator.face_restore(current, face_restore_key, enhancer_config)

        if "Upscaling" in operations and upscale_key:
            current = orchestrator.upscale(current, upscale_key)

        orchestrator.finish()
        _auto_save(current, prefix="enh")
        return [img, current]
    except Exception as e:
        orchestrator.finish()
        logger.exception("Error in on_enhance")
        gr.Error(str(e))
        return []


# ─── Background Tab ──────────────────────────────────────────────────────────

def on_remove_background(
    input_image,
    provider_key: str,
    background_choice: str,
    custom_bg,
):
    """
    Background removal via Orchestrator. _ensure_loaded automatically
    unloads the active generator if needed.
    """
    if input_image is None:
        gr.Warning("Please upload an image.")
        return None

    try:
        img = _to_pil(input_image)

        # Determine background image
        background_image = _to_pil(custom_bg) if custom_bg is not None else _resolve_background(background_choice)

        result = orchestrator.remove_background(img, provider_key, background_image)
        orchestrator.finish()

        _auto_save(result, prefix="bg")
        return result
    except Exception as e:
        orchestrator.finish()
        logger.exception("Error in on_remove_background")
        gr.Error(str(e))
        return None


# ─── Full Pipeline Tab ──────────────────────────────────────────────────────

def on_run_pipeline(
    reference_images,
    prompt: str,
    negative_prompt: str,
    generator_key: str,
    use_face_restore: bool,
    face_restore_key: str,
    use_upscale: bool,
    upscale_key: str,
    use_background: bool,
    background_key: str,
    background_choice: str,
    num_steps: int,
    guidance_scale: float,
    style_strength: float,
    fidelity: float,
    seed: int | None,
    resolution: str = _DEFAULT_RES,
    scheduler: str = _DEFAULT_SCHED,
    progress=gr.Progress(),
):
    """
    Full pipeline via Orchestrator. No manual VRAM management —
    orchestrator.run_full_pipeline() coordinates everything.
    """
    try:
        ref_imgs = _parse_reference_images(reference_images)
        width, height = _parse_resolution(resolution)

        config = GeneratorConfig(
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed) if seed is not None else None,
            height=height,
            width=width,
            scheduler=SCHEDULER_PRESETS.get(scheduler, "euler"),
            extra={"style_strength": style_strength},
        )
        enhancer_cfg = EnhancerConfig(fidelity=fidelity)

        background_image = _resolve_background(background_choice) if use_background else None

        progress(0, desc="Starting pipeline...")
        result = orchestrator.run_full_pipeline(
            reference_images=ref_imgs,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator_key=generator_key,
            face_restore_key=face_restore_key if use_face_restore else None,
            upscale_key=upscale_key if use_upscale else None,
            background_key=background_key if use_background else None,
            background_image=background_image,
            generator_config=config,
            enhancer_config=enhancer_cfg,
        )
        progress(1.0, desc="Done!")
        if "output" in result and result["output"] is not None:
            _auto_save(result["output"], prefix="pipe")

        gallery_images = []
        for key in ["generated", "restored", "upscaled", "final", "output"]:
            if key in result and result[key] is not None:
                gallery_images.append((result[key], key))

        # Deduplicate: output is often the same object as the last step
        seen_ids: set[int] = set()
        unique = []
        for img, label in gallery_images:
            if id(img) not in seen_ids:
                seen_ids.add(id(img))
                unique.append((img, label))

        return unique, hw_monitor.status_html()
    except Exception as e:
        orchestrator.finish()
        logger.exception("Error in on_run_pipeline")
        gr.Error(str(e))
        return [], hw_monitor.status_html()


# ─── Gradio App ───────────────────────────────────────────────────────────────

def build_app():
    gen_keys = list(registry.list_generators().keys())
    fr_keys = list(registry.list_face_restore().keys())
    up_keys = list(registry.list_upscale().keys())
    bg_keys = list(registry.list_background().keys())
    bg_choices = get_background_choices()

    with gr.Blocks(title="PortraitForge — Local AI Image Pipeline") as app:
        gr.Markdown("# PortraitForge\nLocal AI image pipeline for professional portraits · AMD DirectML · Offline")
        hw_banner = gr.HTML(hw_monitor.status_html())
        hw_timer = gr.Timer(3)   # refresh every 3 seconds

        with gr.Tabs():
            # ── Tab 1: Generator ──────────────────────────────────────────────
            with gr.Tab("Generator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_provider = gr.Dropdown(
                            choices=gen_keys,
                            value=gen_keys[0] if gen_keys else None,
                            label="Provider",
                        )
                        ref_gallery = gr.Gallery(
                            label="Reference Photos (max. 4)",
                            columns=2,
                            height=200,
                            visible=True,
                            type="pil",
                        )
                        gen_prompt = gr.Textbox(label="Prompt", lines=3)
                        gen_token_warning = gr.HTML("")
                        gen_hint = gr.HTML(
                            _provider_guidelines_html(
                                registry.get_generator(gen_keys[0])
                            ) if gen_keys else ""
                        )
                        gen_neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                        with gr.Row():
                            gen_resolution = gr.Dropdown(
                                choices=list(RESOLUTION_PRESETS.keys()),
                                value=_DEFAULT_RES,
                                label="Resolution",
                            )
                            gen_scheduler = gr.Dropdown(
                                choices=list(SCHEDULER_PRESETS.keys()),
                                value=_DEFAULT_SCHED,
                                label="Scheduler",
                            )
                        with gr.Row():
                            gen_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                            gen_guidance = gr.Slider(3.0, 10.0, value=5.0, step=0.5, label="Guidance Scale")
                        gen_style = gr.Slider(0.3, 0.9, value=0.65, step=0.05, label="Style Strength (PhotoMaker)")
                        with gr.Row():
                            gen_seed = gr.Number(label="Seed", value=None, precision=0)
                            gen_rand_btn = gr.Button("Random", size="sm")
                        with gr.Row():
                            gen_load_btn = gr.Button("Load Model", variant="primary")
                            gen_unload_btn = gr.Button("Unload")
                        gen_vram_status = gr.HTML("<p>No model loaded</p>")
                        gen_run_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        gen_output = gr.Image(label="Result", type="pil")

                # Events
                gen_provider.change(
                    on_generator_change,
                    inputs=[gen_provider],
                    outputs=[ref_gallery, gen_hint, gen_vram_status, gen_token_warning],
                )
                gen_prompt.change(
                    lambda p: _token_warning_html(p, _active_max_tokens),
                    inputs=[gen_prompt],
                    outputs=[gen_token_warning],
                )
                gen_load_btn.click(on_load_generator, inputs=[gen_provider], outputs=[gen_vram_status, hw_banner])
                gen_unload_btn.click(on_unload_generator, outputs=[gen_vram_status, hw_banner])
                gen_rand_btn.click(on_random_seed, outputs=[gen_seed])
                gen_run_btn.click(
                    on_generate,
                    inputs=[gen_provider, ref_gallery, gen_prompt, gen_neg_prompt,
                            gen_steps, gen_guidance, gen_style, gen_seed, gen_resolution, gen_scheduler],
                    outputs=[gen_output, gen_prompt],
                )

            # ── Tab 2: Enhancer ───────────────────────────────────────────────
            with gr.Tab("Enhancer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        enh_input = gr.Image(label="Input Image", type="numpy")
                        enh_fr_provider = gr.Dropdown(
                            choices=fr_keys,
                            value=fr_keys[0] if fr_keys else None,
                            label="Face Restore Provider",
                        )
                        enh_up_provider = gr.Dropdown(
                            choices=up_keys,
                            value=up_keys[0] if up_keys else None,
                            label="Upscale Provider",
                        )
                        enh_fidelity = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Fidelity (CodeFormer)")
                        enh_ops = gr.CheckboxGroup(
                            choices=["Face Restore", "Upscaling"],
                            value=["Face Restore"],
                            label="Operations",
                        )
                        enh_run_btn = gr.Button("Enhance", variant="primary")

                    with gr.Column(scale=1):
                        enh_output = gr.Gallery(label="Before / After", columns=2)

                enh_run_btn.click(
                    on_enhance,
                    inputs=[enh_input, enh_fr_provider, enh_up_provider, enh_fidelity, enh_ops],
                    outputs=[enh_output],
                )

            # ── Tab 3: Background ────────────────────────────────────────────
            with gr.Tab("Background"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bg_input = gr.Image(label="Input Image", type="numpy")
                        bg_provider = gr.Dropdown(
                            choices=bg_keys,
                            value=bg_keys[0] if bg_keys else None,
                            label="Background Provider",
                        )
                        bg_choice = gr.Dropdown(
                            choices=bg_choices,
                            value=bg_choices[0] if bg_choices else "None",
                            label="Background",
                        )
                        bg_custom = gr.Image(label="Custom Background (optional)", type="numpy")
                        bg_run_btn = gr.Button("Remove/Replace Background", variant="primary")

                    with gr.Column(scale=1):
                        bg_output = gr.Image(label="Result", type="pil")

                bg_run_btn.click(
                    on_remove_background,
                    inputs=[bg_input, bg_provider, bg_choice, bg_custom],
                    outputs=[bg_output],
                )

            # ── Tab 4: Full Pipeline ─────────────────────────────────────────
            with gr.Tab("Full Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pipe_refs = gr.Gallery(
                            label="Reference Photos (optional)", columns=2, height=200, type="pil"
                        )
                        pipe_prompt = gr.Textbox(label="Prompt", lines=3)
                        pipe_neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                        pipe_gen = gr.Dropdown(
                            choices=gen_keys,
                            value=gen_keys[0] if gen_keys else None,
                            label="Generator",
                        )
                        with gr.Row():
                            pipe_use_fr = gr.Checkbox(label="Enable Face Restore", value=True)
                            pipe_fr = gr.Dropdown(choices=fr_keys, value=fr_keys[0] if fr_keys else None, label="Face Restore")
                        with gr.Row():
                            pipe_use_up = gr.Checkbox(label="Enable Upscaling", value=True)
                            pipe_up = gr.Dropdown(choices=up_keys, value=up_keys[0] if up_keys else None, label="Upscale")
                        with gr.Row():
                            pipe_use_bg = gr.Checkbox(label="Enable Background", value=False)
                            pipe_bg = gr.Dropdown(choices=bg_keys, value=bg_keys[0] if bg_keys else None, label="Background")
                        pipe_bg_choice = gr.Dropdown(
                            choices=bg_choices,
                            value=bg_choices[0] if bg_choices else "None",
                            label="Background Image",
                        )
                        with gr.Row():
                            pipe_resolution = gr.Dropdown(
                                choices=list(RESOLUTION_PRESETS.keys()),
                                value=_DEFAULT_RES,
                                label="Resolution",
                            )
                            pipe_scheduler = gr.Dropdown(
                                choices=list(SCHEDULER_PRESETS.keys()),
                                value=_DEFAULT_SCHED,
                                label="Scheduler",
                            )
                        with gr.Row():
                            pipe_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                            pipe_guidance = gr.Slider(3.0, 10.0, value=5.0, step=0.5, label="Guidance")
                        with gr.Row():
                            pipe_style = gr.Slider(0.3, 0.9, value=0.65, step=0.05, label="Style Strength (PhotoMaker)")
                            pipe_fidelity = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Fidelity (CodeFormer)")
                        pipe_seed = gr.Number(label="Seed", value=None, precision=0)
                        pipe_run_btn = gr.Button("Start Pipeline", variant="primary")

                    with gr.Column(scale=1):
                        pipe_output = gr.Gallery(label="Results & Intermediate Steps", columns=2)

                pipe_run_btn.click(
                    on_run_pipeline,
                    inputs=[
                        pipe_refs, pipe_prompt, pipe_neg_prompt, pipe_gen,
                        pipe_use_fr, pipe_fr, pipe_use_up, pipe_up,
                        pipe_use_bg, pipe_bg, pipe_bg_choice,
                        pipe_steps, pipe_guidance, pipe_style, pipe_fidelity,
                        pipe_seed, pipe_resolution, pipe_scheduler,
                    ],
                    outputs=[pipe_output, hw_banner],
                )

        hw_timer.tick(hw_monitor.status_html, outputs=[hw_banner])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

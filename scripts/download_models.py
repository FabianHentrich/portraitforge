"""
Download models and store them locally.

Usage:
    python scripts/download_models.py                          # all models
    python scripts/download_models.py --models sdxl            # specific ones only
    python scripts/download_models.py --models flux            # FLUX (token required)

Token:
    Read from the HF_READ_TOKEN environment variable.
    Alternatively: --token hf_xxxx

Prerequisites for FLUX.1-schnell:
    1. Accept the license: huggingface.co/black-forest-labs/FLUX.1-schnell
    2. Set HF_READ_TOKEN in .env
"""

import argparse
import logging
import os
import sys
import time

# Paths always relative to the project root (not the CWD)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Load .env from project root (before anything else)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # dotenv optional — environment variables from the shell work without it

# Enable hf_transfer — Rust-based downloader, 3-5x faster than default.
# Must be set as an env variable before the huggingface_hub import.
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "0":
    pass  # already set (e.g. from .env)
else:
    try:
        import hf_transfer  # noqa: F401 — only check if installed
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        pass  # hf_transfer not installed — fallback to default downloader


def _project_path(rel: str) -> str:
    """Return absolute path relative to the project root."""
    return os.path.join(PROJECT_ROOT, rel)

# ---------------------------------------------------------------------------
# Logging — Format: time + level + message
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # Root logger at WARNING — suppresses httpx/urllib3 spam
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Only our own logger outputs at INFO level
logger = logging.getLogger("download")
logger.setLevel(logging.INFO)

try:
    from huggingface_hub import snapshot_download, hf_hub_download, repo_info
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    logger.error("huggingface-hub not installed — pip install huggingface-hub")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "photomaker_pipeline": {
        # Python source files from GitHub (~50 KB) — no HF repo, code only.
        # The three files form the photomaker package with PhotoMakerIDEncoder.
        # Imported as a local package by pipeline/providers/generator/photomaker.py.
        "type": "github_files",
        "base_url": "https://raw.githubusercontent.com/TencentARC/PhotoMaker/main/photomaker/",
        # __init__.py does NOT come from GitHub (pulls in insightface/v2 deps).
        # Created as a minimal version after download (v1 classes only).
        "files": ["model.py", "resampler.py", "pipeline.py"],
        "local_dir": _project_path("pipeline/community/photomaker_src"),
        "requires_token": False,
        "description": "PhotoMaker pipeline code (Python, GitHub)",
        "size_hint": "~50 KB",
    },
    "photomaker": {
        # diffusers uses subdirectories (unet/, text_encoder/, ...) — not the root files.
        # Skip fp32 file (13.9 GB) and fp16 single file (6.94 GB) — only subdirectories needed.
        "type": "snapshot",
        "repo_id": "SG161222/RealVisXL_V5.0",
        "local_dir": _project_path("models/generator/realvisxl_v5"),
        "ignore_patterns": [
            "*.ckpt", "*.bin",               # Pickle format, unsafe
            "*.msgpack", "*.h5", "flax_model*", "*.ot",  # non-PyTorch
            "RealVisXL_V5.0_fp16.safetensors",  # Root single file (A1111 format, 6.94 GB)
            "RealVisXL_V5.0_fp32.safetensors",  # Root single file fp32 (13.9 GB)
            "*.png", "*.jpg", "*.jpeg",      # Example images
            # Within subdirectories: skip fp32 where fp16 exists
            "*/model.safetensors",                        # text_encoder fp32
            "*/diffusion_pytorch_model.safetensors",      # unet/vae fp32 (single file)
            "*/diffusion_pytorch_model-*-of-*.safetensors",  # unet fp32 (sharded, e.g. -00001-of-00002)
        ],
        "requires_token": False,
        "description": "PhotoMaker base model (RealVisXL V5)",
        "size_hint": "~4 GB (instead of 41 GB)",
    },
    "photomaker_adapter": {
        "type": "file",
        "repo_id": "TencentARC/PhotoMaker",
        "filename": "photomaker-v1.bin",
        "local_dir": _project_path("models/generator"),
        "requires_token": False,
        "description": "PhotoMaker V1 Adapter",
        "size_hint": "~934 MB",
    },
    "sdxl": {
        # Skip root checkpoints (sd_xl_base_1.0.safetensors = 6.94 GB, 0.9vae = 6.94 GB).
        # Alternative VAE directories (vae_1_0/, vae_decoder/, vae_encoder/) not needed.
        "type": "snapshot",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_dir": _project_path("models/generator/sdxl_base"),
        "ignore_patterns": [
            "*.ckpt",
            "*.msgpack", "*.h5", "flax_model*", "*.ot",
            "*.onnx", "*.onnx_data",             # ONNX exports
            "openvino_model.*",                  # OpenVINO exports (not needed for diffusers)
            "sd_xl_base_1.0.safetensors",        # Root single file (A1111 format, 6.94 GB)
            "sd_xl_base_1.0_0.9vae.safetensors", # Old VAE single file (6.94 GB)
            "*lora*",                            # Example LoRA (50 MB)
            "vae_1_0/*", "vae_decoder/*", "vae_encoder/*",  # Alternative VAE components
            "*.png", "*.jpg", "*.jpeg",
            "*/diffusion_pytorch_model.safetensors",         # fp32 in subdirectories (single file)
            "*/diffusion_pytorch_model-*-of-*.safetensors",  # fp32 in subdirectories (sharded)
            "*/model.safetensors",
        ],
        "requires_token": False,
        "description": "SDXL Base 1.0",
        "size_hint": "~5 GB (instead of 77 GB)",
    },
    "sd35_medium": {
        # SD3.5 Medium: DiT transformer (2.5B), 16-channel VAE, three text encoders.
        # The repo contains two formats in parallel:
        #   1. diffusers format: text_encoder/, text_encoder_2/, text_encoder_3/, transformer/, vae/
        #   2. ComfyUI format:   text_encoders/ (collection), sd3.5_medium.safetensors (monolith)
        # We only need the diffusers format. T5-XXL (~9+5 GB) is not loaded
        # (provider passes text_encoder_3=None) and therefore does not need to be downloaded.
        "type": "snapshot",
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "local_dir": _project_path("models/generator/sd35_medium"),
        "ignore_patterns": [
            "*.ckpt",
            "*.msgpack", "*.h5", "flax_model*", "*.ot",
            "*.onnx", "*.onnx_data",
            "*.png", "*.jpg", "*.jpeg",
            # ComfyUI monolith: transformer + all text encoders as single file (not for diffusers)
            "sd3.5_medium.safetensors",
            # ComfyUI text_encoders/ directory: clip_l, clip_g, t5xxl_fp16 (~9 GB), t5xxl_fp8 (~5 GB)
            # NOTE: "text_encoders" (plural) != "text_encoder" / "text_encoder_2" (diffusers format)
            "text_encoders/*",
            # Exclude diffusers T5-XXL encoder (~9 GB) — provider uses text_encoder_3=None
            "text_encoder_3/*",
            "tokenizer_3/*",
            # NOTE: Unlike SDXL, SD3.5 in diffusers format has NO fp16 variants
            # in the subdirectories. The fp32 files (diffusion_pytorch_model.safetensors,
            # model.safetensors) are the only weights here — do NOT exclude!
            # diffusers loads fp32 and casts via torch_dtype=float16 on-the-fly.
        ],
        "requires_token": True,
        "description": "Stable Diffusion 3.5 Medium (2.5B, without T5-XXL)",
        "size_hint": "~7 GB (instead of ~23 GB with ComfyUI files + T5)",
    },
    "flux": {
        # FLUX: barely reducible. Skip root single file — transformer/ subdirectory is used.
        "type": "snapshot",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "local_dir": _project_path("models/generator/flux_schnell"),
        "ignore_patterns": [
            "*.msgpack", "*.h5", "flax_model*",
            "flux1-schnell.safetensors",  # Root single file (ComfyUI format) — transformer/ is sufficient
            "*.png", "*.jpg", "*.jpeg",
        ],
        "requires_token": True,
        "description": "FLUX.1-schnell",
        "size_hint": "~23 GB",
    },
    # Real-ESRGAN: no pre-built ONNX available on HuggingFace.
    # -> Use scripts/export_realesrgan_onnx.py (downloads .pth + exports to ONNX).

    "flan_t5_small": {
        # Used by PromptCompressor (pipeline/utils/prompt_compressor.py).
        # Compresses long prompts to CLIP-compatible 77 tokens.
        # Auto-download on first compress() call possible (requires internet).
        "type": "snapshot",
        "repo_id": "google/flan-t5-small",
        "local_dir": _project_path("models/utils/flan-t5-small"),
        "ignore_patterns": ["*.msgpack", "*.h5", "flax_model*", "*.ot"],
        "requires_token": False,
        "description": "flan-T5-small for prompt compression (PromptCompressor)",
        "size_hint": "~300 MB",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _separator(title: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info(f"  {title}")
    logger.info("=" * width)


def _check_repo_accessible(repo_id: str, token: str | None) -> bool:
    """Check if the repository is accessible before starting the actual download."""
    try:
        repo_info(repo_id, token=token)
        return True
    except RepositoryNotFoundError:
        logger.error(f"  Repository not found: {repo_id}")
        return False
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            logger.error(f"  Access denied (403) — license not yet accepted or token missing")
            logger.error(f"  1. Log in and accept the license: https://huggingface.co/{repo_id}")
            logger.error(f"  2. Set HF_READ_TOKEN in .env (token from huggingface.co/settings/tokens)")
        else:
            logger.error(f"  Accessibility check failed: {e}")
        return False


def _already_downloaded(cfg: dict) -> bool:
    """Check if the model is already present locally."""
    if cfg["type"] == "file":
        rename_to = cfg.get("rename_to", cfg["filename"])
        target = os.path.join(cfg["local_dir"], rename_to)
        return os.path.isfile(target)
    elif cfg["type"] == "github_files":
        return all(
            os.path.isfile(os.path.join(cfg["local_dir"], f))
            for f in cfg["files"]
        )
    else:
        local_dir = cfg["local_dir"]
        if not os.path.isdir(local_dir):
            return False
        # Directory must contain more than just .gitkeep
        files = [f for f in os.listdir(local_dir) if not f.startswith(".")]
        return len(files) > 0


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------
def download_model(key: str, token: str | None) -> bool:
    cfg = MODELS[key]

    _separator(f"{key.upper()}  —  {cfg['description']}  ({cfg['size_hint']})")

    # github_files: direct URL downloads, no HF repo
    if cfg["type"] == "github_files":
        import urllib.request
        os.makedirs(cfg["local_dir"], exist_ok=True)
        t_start = time.monotonic()
        for filename in cfg["files"]:
            url = cfg["base_url"] + filename
            dest = os.path.join(cfg["local_dir"], filename)
            logger.info(f"  Downloading: {url}")
            try:
                urllib.request.urlretrieve(url, dest)
                size_kb = os.path.getsize(dest) / 1024
                logger.info(f"  Saved: {dest}  ({size_kb:.1f} KB)")
            except Exception as e:
                logger.error(f"  Download failed: {url} — {e}")
                return False
        # Write minimal __init__.py:
        # - PhotoMaker v1 only (no insightface, no model_v2)
        # - No import of pipeline.py (would cause circular import:
        #   __init__ -> pipeline -> from . import -> __init__)
        # - PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken as stub, since pipeline.py
        #   imports the name but v2 is never used
        init_path = os.path.join(cfg["local_dir"], "__init__.py")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(
                "# Automatically generated by download_models.py\n"
                "# PhotoMaker v1 only — no circular import, no insightface\n"
                "from .model import PhotoMakerIDEncoder\n\n"
                "class PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken:\n"
                '    """Stub — PhotoMaker v2 not installed."""\n'
                "    pass\n\n"
                '__all__ = ["PhotoMakerIDEncoder", "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken"]\n'
            )
        logger.info(f"  __init__.py created (PhotoMaker v1, no insightface)")
        logger.info(f"  Done in {time.monotonic() - t_start:.0f}s")
        return True

    # Token check
    if cfg["requires_token"] and not token:
        logger.error(f"  FLUX requires a HuggingFace token.")
        logger.error(f"  1. Accept the license: huggingface.co/{cfg['repo_id']}")
        logger.error(f"  2. Set HF_READ_TOKEN in .env")
        return False

    # Already present?
    if _already_downloaded(cfg):
        logger.info(f"  Already present — skipping: {cfg['local_dir']}")
        return True

    # Repository accessible?
    logger.info(f"  Checking accessibility: {cfg['repo_id']} ...")
    if not _check_repo_accessible(cfg["repo_id"], token):
        return False
    logger.info(f"  Repository accessible.")

    os.makedirs(cfg["local_dir"], exist_ok=True)
    t_start = time.monotonic()

    try:
        if cfg["type"] == "file":
            logger.info(f"  Downloading file: {cfg['filename']}")
            dest = hf_hub_download(
                repo_id=cfg["repo_id"],
                filename=cfg["filename"],
                local_dir=cfg["local_dir"],
                token=token,
            )
            if "rename_to" in cfg:
                target = os.path.join(cfg["local_dir"], cfg["rename_to"])
                os.replace(dest, target)
                logger.info(f"  Renamed: {cfg['filename']} -> {cfg['rename_to']}")
                dest = target
            size_mb = os.path.getsize(dest) / 1024 / 1024
            logger.info(f"  File saved: {dest}  ({size_mb:.0f} MB)")

        else:
            logger.info(f"  Downloading repository snapshot: {cfg['repo_id']}")
            logger.info(f"  Target directory: {cfg['local_dir']}")
            if cfg.get("allow_patterns"):
                logger.info(f"  Only: {', '.join(cfg['allow_patterns'])}")
            if cfg.get("ignore_patterns"):
                logger.info(f"  Ignoring: {', '.join(cfg['ignore_patterns'])}")
            snapshot_download(
                repo_id=cfg["repo_id"],
                local_dir=cfg["local_dir"],
                token=token,
                allow_patterns=cfg.get("allow_patterns") or None,
                ignore_patterns=cfg.get("ignore_patterns") or None,
                max_workers=8,  # parallel file downloads (default: 1)
            )
            # Determine size of the downloaded directory
            total_bytes = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(cfg["local_dir"])
                for f in files
            )
            logger.info(f"  Snapshot saved: {cfg['local_dir']}  ({total_bytes / 1024**3:.2f} GB)")

        elapsed = time.monotonic() - t_start
        logger.info(f"  Done in {elapsed:.0f}s")
        return True

    except EntryNotFoundError as e:
        logger.error(f"  File not found in repository: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"  Cancelled (Ctrl+C)")
        raise
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models for portraitforge",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=list(MODELS.keys()) + ["all"],
        metavar="MODEL",
        help=(
            "Which models to download. Available:\n"
            + "\n".join(f"  {k:<20} {v['description']} ({v['size_hint']})" for k, v in MODELS.items())
            + "\n  all                  All models (default)"
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_READ_TOKEN"),
        help="HuggingFace API token (default: env variable HF_READ_TOKEN)",
    )
    args = parser.parse_args()

    if args.token:
        logger.info("HF_READ_TOKEN found — gated repositories accessible")
    else:
        logger.warning("No HF_READ_TOKEN — FLUX download not possible")

    keys = list(MODELS.keys()) if "all" in args.models else args.models

    results: dict[str, bool] = {}
    for key in keys:
        try:
            results[key] = download_model(key, args.token)
        except KeyboardInterrupt:
            logger.warning("Cancelled.")
            break

    # Final report
    _separator("RESULT")
    for key, ok in results.items():
        icon = "OK     " if ok else "FAILED "
        cfg = MODELS[key]
        logger.info(f"  [{icon}]  {key:<20} {cfg['local_dir']}")

    skipped = [k for k in keys if k not in results]
    for key in skipped:
        logger.info(f"  [--     ]  {key:<20} (cancelled)")

    if not results or not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

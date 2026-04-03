# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local AI image pipeline for application photos. Runs fully offline on Windows with AMD RX 9070 XT (16 GB VRAM, 32 GB RAM). No cloud upload, no CUDA — exclusively AMD-compatible backends (torch-directml, ONNX Runtime DirectML).

**Three modules, fully interchangeable providers:**
- **Generator** — Portrait generation from reference photos + prompt (PhotoMaker, SDXL, SD3.5 Medium, FLUX.1-schnell)
- **Enhancer** — Face restoration + upscaling (CodeFormer/GFPGAN + ESRGAN/SwinIR)
- **Background** — Background removal / replacement (U2Net, BiRefNet, SAM)

**Priority: Image quality over speed.** Processing time is not critical.

Architecture details: `ARCHITECTURE.md`

---

## Development Commands

```bash
# Virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Start app (Gradio on localhost:7860)
python app.py

# Run all tests
pytest tests/ -v

# Single test file
pytest tests/test_registry.py -v

# Single test
pytest tests/test_registry.py::test_name -v

# Download models (examples)
python scripts/download_models.py --models photomaker photomaker_adapter photomaker_pipeline sdxl
python scripts/download_models.py --models sd35_medium   # Token + license required
python scripts/download_models.py --models flux           # Token + license required
# Real-ESRGAN ONNX (downloads .pth + exports to ONNX)
python scripts/export_realesrgan_onnx.py
```

HF_READ_TOKEN must be placed in `.env` in the project root (for SD3.5 and FLUX).

---

## Constraints — never ignore

- **No CUDA.** No `torch.cuda`, no `onnxruntime-gpu`. Always `torch-directml` or `onnxruntime-directml`.
- **No cloud API calls.** All models run locally. No external requests in the pipeline.
- **Python 3.12.** torch-directml supports 3.8–3.12; Python 3.13 is not supported.
- **torch==2.4.x + torch-directml==0.2.x.** Versions pinned in `requirements.txt` — do not upgrade.
- **Models reside in `/models/`.** Never commit weights. `/models/` is in `.gitignore`.
- **VRAM management is critical.** Only one heavy provider may reside in VRAM at a time. Each provider implements `load()` / `unload()`. The orchestrator coordinates this.
- **Provider registry is the single source of truth.** New models are registered, not hardcoded.
- **Set `max_gen_scale` per provider.** Transformer models (SD3.5, FLUX) must set a reduced value because DirectML materializes the full attention matrix (no Flash Attention). Missing `max_gen_scale` on a new transformer provider leads to OOM.
- **Generators return native resolution.** No LANCZOS upscale in the `generate()` body. The orchestrator automatically inserts Real-ESRGAN when the native resolution is smaller than the target.

---

## Core Architecture Decisions

### `max_gen_scale` — VRAM-safe Resolution Cap

Transformer models (SD3.5, FLUX) scale quadratically with resolution. DirectML materializes the full attention matrix (no Flash Attention / Memory-Efficient Attention as on CUDA). At 1024x1024, attention tensors of 1–2 GB per step are created -> OOM on 16 GB.

Each generator provider declares `max_gen_scale` (0.0–1.0). In `generate()`, the effective gen_scale is capped:

```python
gen_scale = min(config.extra.get("gen_scale", config.gen_scale), self.max_gen_scale)
```

| Provider | max_gen_scale | Reason |
|---|---|---|
| PhotoMaker | 1.0 | UNet, linear scaling |
| SDXL Base | 1.0 | UNet, linear scaling |
| SD3.5 Medium | 0.75 | DiT, OOM at 1024x1024 on DirectML |
| FLUX.1-schnell | 0.85 | DiT + 12 GB weights |

### Auto Quality-Upscale in the Orchestrator

Generators return their native resolution (no internal LANCZOS). When the output is smaller than the target resolution, the orchestrator automatically inserts a Real-ESRGAN step:

```
generate() → 768×768
  ↓ _quality_upscale(): Real-ESRGAN ×2 → resize auf 1024×1024
face_restore() → 1024×1024
upscale() → 4096×4096  (optional, user-gesteuert)
```

This is significantly better in quality than LANCZOS, as Real-ESRGAN generates real details. Falls back to LANCZOS when Real-ESRGAN is not available.

### CodeFormer — Face Detection before Restoration

CodeFormer no longer processes the full image. Instead:
1. Face detection via OpenCV Haar Cascade (`haarcascade_frontalface_alt2.xml`, ships with opencv-python, no download needed)
2. Each face is cropped with 1.7x padding
3. Crop scaled to 512x512, processed through CodeFormer
4. Scaled back to crop size, blended in with Gaussian feathering (radius 25)
5. Falls back to full image when no faces are detected

### Real-ESRGAN — Tile-based Inference

The ONNX model processes 256x256 tiles instead of the entire image. Tiles overlap by 32 px. Overlap regions are weighted with linear blend masks and accumulated. Prevents OOM on large input images, delivers seamless results.

### FLUX — DirectML bf16->fp16 Fix

FLUX weights are bfloat16. DirectML does not support bfloat16. The pipeline is loaded directly as `torch_dtype=float16` (diffusers casts bf16->fp16 automatically). Additionally, safetensors mmap is disabled via `_no_mmap_safetensors()`, since Windows crashes with Access Violation on large files (>2 GB).

### SD3.5 — Sequential CPU-Offload

SD3.5 uses `enable_sequential_cpu_offload()` instead of `enable_model_cpu_offload()`. Sequential offload moves individual transformer blocks to GPU, not the entire model. Peak VRAM drops from ~6 GB to ~2 GB. T5-XXL encoder is disabled (saves ~9 GB, CLIP-L + CLIP-G are sufficient).

---

## Quality Rules

- **No global state.** `registry` and `vram_tracker` are the only permitted singletons.
- **Lazy loading.** `__init__` initializes only Python objects. No model loading.
- **Type hints on all public methods.**
- **Stubs instead of omission.** Unimplemented providers: file + `raise NotImplementedError` + **do not** register in the registry.
- **No print debugging.** Python `logging`, level INFO.
- **Errors are user-facing.** Pipeline exceptions -> `gr.Warning()` / `gr.Error()`.
- **Respect VRAM management.** When switching between tabs (Enhancer, Background), `_active_generator` is unloaded. Do not load a provider without first ensuring no other heavy provider is loaded.
- **Auto-save.** All pipeline results are automatically saved to `outputs/`.

**Adding a new provider = two steps:**
1. Create file in `providers/<module>/`, fully implement the base class; for generator providers, set `max_gen_scale` correctly
2. Register in `providers/__init__.py`

Nothing else needs to be changed.

# PortraitForge

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white" />
  <img alt="GPU" src="https://img.shields.io/badge/GPU-AMD%20DirectML-ED1C24?logo=amd&logoColor=white" />
  <img alt="Offline" src="https://img.shields.io/badge/Offline-100%25-brightgreen" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-not%20required-lightgrey" />
  <img alt="License" src="https://img.shields.io/badge/License-PRUL%20v1.0-yellow" />
</p>

<p align="center">
  Local AI pipeline for generating professional application photos.<br/>
  Fully offline · no cloud upload · no CUDA · AMD GPU-native on Windows.
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Download Models](#download-models)
- [Launch the App](#launch-the-app)
- [Project Structure](#project-structure)
- [Tests](#tests)
- [Known Limitations](#known-limitations)

---

## Project Overview

PortraitForge combines multiple specialized AI models into a unified pipeline:

| Module | Task | Implemented Providers |
|---|---|---|
| **Generator** | Generate portrait from text/reference photo | PhotoMaker · SDXL Base · SD3.5 Medium · FLUX.1-schnell |
| **Enhancer** | Face restoration + upscaling | CodeFormer · Real-ESRGAN |
| **Background** | Remove / replace background | U2Net |

**Open Tasks & Missing Providers (in development):**
- **Enhancer**: GFPGAN, SwinIR (currently implemented as stubs).
- **Background**: BiRefNet, SAM (currently implemented as stubs).

**Technical Highlights:**

- **Provider Pattern + Registry** — Each model is a swappable provider behind an abstract base class. Adding a new model: create the file, register it in `providers/__init__.py`. No other code changes needed.
- **VRAM Scheduling** — The orchestrator ensures that no more than one heavy provider resides in GPU memory at any time. Automatic loading/unloading between pipeline steps.
- **Auto Quality Upscale** — When a generator produces output at reduced resolution (e.g., SD3.5 at 75% due to attention OOM), Real-ESRGAN is automatically used instead of LANCZOS to upscale to the target resolution.
- **Two Inference Backends** — Diffusers models (SDXL, FLUX, PhotoMaker, SD3.5) run via PyTorch + `torch-directml`, enhancer models via `onnxruntime-directml`.
- **AMD / DirectML instead of CUDA** — Full GPU acceleration on AMD under Windows without ROCm.

> Technical details on architecture decisions, provider interface, and VRAM management: [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## Architecture

```
User (Browser)
       │
       ▼
Gradio Web UI  (localhost:7860)
       │
       ▼
PipelineOrchestrator          ← VRAM scheduling: max 1 heavy provider at a time
       │
       ├──▶ Generator         ──▶  PhotoMaker · SDXL · SD3.5 · FLUX.1-schnell
       │         │
       │         └── Auto Quality Upscale (Real-ESRGAN, when output < target resolution)
       ├──▶ Enhancer          ──▶  CodeFormer (Haar Cascade + Crop) · Real-ESRGAN (Tiles)
       └──▶ Background        ──▶  U2Net
```

```
pipeline/
├── base/           # Abstract base classes — BaseGeneratorProvider, ...
├── providers/      # Concrete implementations
│   ├── generator/  # PhotoMaker, SDXL, SD3.5, FLUX
│   ├── enhancer/   # CodeFormer, Real-ESRGAN
│   └── background/ # U2Net
├── registry.py     # Provider registry (singleton)
├── orchestrator.py # Pipeline coordination + VRAM management + Auto Quality Upscale
└── utils/          # image_utils, vram_manager, hw_monitor, schedulers
```

---

## Requirements

| Requirement | Details |
|---|---|
| OS | Windows 10 / 11 |
| GPU | AMD with DirectML support (recommended: RX 9070 XT, 16 GB VRAM) |
| RAM | min. 16 GB (32 GB recommended — CPU offload moves parts into RAM) |
| Python | 3.12 — `torch-directml` supports up to 3.12, Python 3.13 is not compatible |
| VRAM | min. 8 GB (PhotoMaker/SDXL), 16 GB for FLUX.1-schnell without CPU offload |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<user>/portraitforge.git
cd portraitforge
```

### 2. Verify Python version

```bash
py -3.12 --version   # must output 3.12.x
```

### 3. Create virtual environment

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
HF_READ_TOKEN=hf_...
```

Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (type: **Read**).

> For **FLUX.1-schnell** and **SD3.5 Medium**, the license must also be accepted once:
> - [huggingface.co/black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
> - [huggingface.co/stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

---

## Download Models

### Generator Models

```bash
# PhotoMaker + SDXL (no token required):
python scripts/download_models.py --models photomaker photomaker_adapter photomaker_pipeline sdxl

# SD3.5 Medium (~5 GB, token + license required):
python scripts/download_models.py --models sd35_medium

# FLUX.1-schnell (~23 GB, token + license required):
python scripts/download_models.py --models flux
```

The script saves all files to `models/` and supports **resume** — interrupted downloads are automatically continued.

### Enhancer Models

```bash
# Real-ESRGAN ONNX (~67 MB, downloads .pth + exports to ONNX):
python scripts/export_realesrgan_onnx.py
```

### CodeFormer (one-time ONNX export)

CodeFormer is not distributed as ONNX and must be exported once from the original weights:

```bash
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer
py -3.12 -m venv .venv-cf && .venv-cf\Scripts\activate
pip install -r requirements.txt

# Download weights:
py basicsr/scripts/download_pretrained_models.py CodeFormer

# Export to ONNX:
py -c "
import torch, sys; sys.path.insert(0, '.')
from basicsr.archs.codeformer_arch import CodeFormer
net = CodeFormer(num_in_ch=3, num_out_ch=3, num_feat=512,
                 num_heads=8, num_layers=9,
                 connect_list=['32','64','128','256'])
ckpt = torch.load('weights/CodeFormer/codeformer.pth', map_location='cpu')
net.load_state_dict(ckpt['params_ema']); net.eval()
torch.onnx.export(net, (torch.zeros(1,3,512,512), torch.tensor([0.7])),
                  'codeformer.onnx', input_names=['input','w'],
                  output_names=['output'], opset_version=17)
"

copy codeformer.onnx ..\portraitforge\models\enhancer\codeformer.onnx
```

### U2Net

Downloaded **automatically** by `rembg` on first launch (~170 MB, one-time).

---

## Launch the App

```bash
python app.py
```

Gradio automatically opens [`http://localhost:7860`](http://localhost:7860).

---

## Project Structure

```
portraitforge/
├── app.py                    # Gradio entry point
├── CLAUDE.md                 # AI system instructions & architecture specs
├── ARCHITECTURE.md           # Architecture details (VRAM, providers, registry)
├── aufgabenliste.md          # Open project goals
├── requirements.txt
├── .env                      # HF_READ_TOKEN (do not commit)
│
├── pipeline/
│   ├── orchestrator.py       # VRAM scheduling + workflow coordination + Auto Quality Upscale
│   ├── registry.py           # Provider registry (singleton)
│   ├── base/                 # Abstract base classes
│   ├── providers/            # Concrete provider implementations
│   │   ├── generator/        # photomaker, sdxl_base, sd35_medium, flux
│   │   ├── enhancer/         # codeformer, realesrgan
│   │   └── background/       # u2net
│   ├── community/            # Local copies of community pipelines (PhotoMaker)
│   └── utils/                # image_utils, vram_manager, hw_monitor, schedulers
│
├── scripts/
│   ├── download_models.py          # Model download with resume support
│   └── export_realesrgan_onnx.py   # Real-ESRGAN PyTorch → ONNX export
│
├── models/                   # Weights — gitignored
│   ├── generator/
│   ├── enhancer/
│   └── background/
│
├── inputs/                   # Input images
├── outputs/                  # Result images (auto-save) — gitignored
├── assets/backgrounds/       # Background images for compositing
└── tests/                    # Pytest
```

---

## Tests

```bash
pytest tests/ -v
```

233 tests (all categories: provider, orchestrator, registry, VRAM manager, app helpers, HW monitor, image utils, schedulers).

---

## Known Limitations

| Problem | Cause | Note |
|---|---|---|
| First inference slow | DirectML shader compilation | normal from the 2nd run onward |
| SD3.5 generates at 75% resolution | Attention OOM on DirectML at 1024x1024 | Auto Quality Upscale with Real-ESRGAN compensates |
| FLUX tight on 16 GB | ~12 GB model, CPU offload → ~7 GB | CPU offload enabled by default |
| No `cuda.empty_cache()` | DirectML has no equivalent | `gc.collect()` + deleting references is correct |
| U2Net requires internet (one-time) | rembg download on first launch | fully offline afterwards |
| Haar Cascade with side profile | Frontal face detection | Fallback to full image; sufficient for application photos |

---

## License

**PortraitForge Responsible Use License (PRUL) v1.0** — see [`LICENSE`](LICENSE)

Non-commercial use permitted. Commercial use requires a separate agreement. The license includes binding usage restrictions (no deepfakes, consent required, AI labeling obligation, no CSAM, no discrimination, no disinformation). Also observe the licenses of the third-party models used (Section 6 of the LICENSE).

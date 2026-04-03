"""
Real-ESRGAN x4plus PyTorch → ONNX export.

Downloads RealESRGAN_x4plus.pth (xinntao/Real-ESRGAN, GitHub Releases)
and exports the model to models/enhancer/realesrgan.onnx.

Requires no external packages beyond the normal project environment
(torch is already available).

Usage:
    python scripts/export_realesrgan_onnx.py

Options:
    --pth PATH      Use local .pth file (skips download)
    --out PATH      Output file (default: models/enhancer/realesrgan.onnx)
    --opset N       ONNX opset version (default: 17)
"""

import argparse
import logging
import os
import sys
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths always relative to project root (not CWD)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("export_realesrgan")

# Official source: github.com/xinntao/Real-ESRGAN (Releases v0.1.0)
_PTH_URL = (
    "https://github.com/xinntao/Real-ESRGAN"
    "/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


# ---------------------------------------------------------------------------
# Minimal RRDBNet implementation (identical to basicsr RRDBNet)
# Architecture parameters for Real-ESRGAN x4plus:
#   num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
# ---------------------------------------------------------------------------

class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class _RRDBNet(nn.Module):
    """
    RRDBNet — Backbone of Real-ESRGAN x4plus.
    Identical to basicsr.archs.rrdbnet_arch.RRDBNet (original source).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # x4 upsampling via nearest interpolation + conv (2x x2)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# Download helper with progress display
# ---------------------------------------------------------------------------

def _download_pth(dest: str) -> None:
    logger.info(f"Lade RealESRGAN_x4plus.pth von GitHub Releases ...")
    logger.info(f"  URL:  {_PTH_URL}")
    logger.info(f"  Ziel: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {pct:3d}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(_PTH_URL, dest, reporthook=_progress)
    print()  # Zeilenumbruch nach Fortschrittsbalken
    size_mb = os.path.getsize(dest) / 1024 / 1024
    logger.info(f"  Gespeichert: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Export-Logik
# ---------------------------------------------------------------------------

def export(pth_path: str, out_path: str, opset: int) -> None:
    logger.info("=" * 55)
    logger.info("  Real-ESRGAN x4plus → ONNX Export")
    logger.info("=" * 55)

    # 1. Gewichte laden
    logger.info(f"Lade Gewichte: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)

    # Das .pth enthält üblicherweise {'params_ema': {...}} oder {'params': {...}}
    if "params_ema" in ckpt:
        state_dict = ckpt["params_ema"]
        logger.info("  State-Dict-Schlüssel: params_ema")
    elif "params" in ckpt:
        state_dict = ckpt["params"]
        logger.info("  State-Dict-Schlüssel: params")
    else:
        # Direktes State-Dict (kein Wrapper)
        state_dict = ckpt
        logger.info("  State-Dict: direkt (kein Wrapper)")

    # 2. Modell initialisieren und Gewichte laden
    logger.info("Initialisiere RRDBNet (num_feat=64, num_block=23) ...")
    model = _RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info("  Gewichte geladen — OK")

    # 3. Dummy-Input für den Trace.
    # Klein halten (64×64) — dynamic_axes lösen die echte Tile-Größe zur Laufzeit auf.
    # torch.onnx.export traced das Modell intern nochmals; 512×512 würde mit 23 RRDB-Blöcken
    # auf CPU ~35s pro Pass dauern. 64×64 reicht zum Aufzeichnen des Graphen.
    dummy = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    # Sanity-Check: Forward-Pass vor dem Export
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 3, 256, 256), f"Unerwartete Output-Shape: {out.shape}"
    logger.info(f"  Forward-Pass: (1,3,64,64) → {tuple(out.shape)} — OK")

    # 4. ONNX-Export
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logger.info(f"Exportiere ONNX (opset={opset}) → {out_path} ...")
    logger.info("  (Kann 1–2 Minuten dauern — torch traced den Graphen intern)")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            # Dynamische Achsen: H/W können variieren (kleine Rand-Tiles möglich)
            dynamic_axes={
                "input":  {2: "height", 3: "width"},
                "output": {2: "out_height", 3: "out_width"},
            },
            # Constant Folding deaktivieren: spart mehrere zusätzliche Forward-Passes
            # beim Export (kein Qualitätsverlust zur Laufzeit).
            do_constant_folding=False,
        )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    logger.info(f"  Gespeichert: {out_path}  ({size_mb:.1f} MB)")
    logger.info("Export erfolgreich.")


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main() -> None:
    default_out = os.path.join(PROJECT_ROOT, "models", "enhancer", "realesrgan.onnx")
    default_pth = os.path.join(PROJECT_ROOT, "models", "enhancer", "RealESRGAN_x4plus.pth")

    parser = argparse.ArgumentParser(
        description="Real-ESRGAN x4plus PyTorch → ONNX",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pth",
        default=default_pth,
        help=f"Lokale .pth-Datei (default: {default_pth})\n"
             "Fehlt die Datei, wird sie automatisch heruntergeladen.",
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help=f"Ausgabe-Pfad (default: {default_out})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX-Opset-Version (default: 17)",
    )
    args = parser.parse_args()

    # .pth herunterladen falls nicht vorhanden
    if not os.path.isfile(args.pth):
        logger.info(f".pth nicht gefunden: {args.pth}")
        _download_pth(args.pth)
    else:
        logger.info(f".pth vorhanden: {args.pth}")

    export(args.pth, args.out, args.opset)


if __name__ == "__main__":
    main()

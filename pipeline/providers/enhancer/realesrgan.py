import gc
import logging
import pathlib
import time
import numpy as np

from pipeline.base.base_enhancer import BaseUpscaleProvider
from PIL import Image

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


class RealESRGANProvider(BaseUpscaleProvider):
    """
    Real-ESRGAN: ONNX-based upscaling with tile-based inference.
    Source: huggingface.co/onnx-community/real-esrgan-x4plus
    File: realesrgan-x4plus.onnx -> models/enhancer/realesrgan.onnx
    Input: float32, shape [1,3,H,W], values in [0,1]
    Output: [1,3,H*4,W*4]
    VRAM: ~1.5 GB (DirectML)

    Tile-based inference prevents OOM on large images.
    The image is split into overlapping tiles, each tile is processed
    individually through the model, and then seamlessly reassembled
    using linear ramps.
    """

    name = "Real-ESRGAN"
    model_id = str(_PROJECT_ROOT / "models" / "enhancer" / "realesrgan.onnx")
    vram_gb = 1.5
    supported_scales = [2, 4]

    _tile_size: int = 256   # 512px -> 1 GB upsample buffer (OOM on CPU+DML); 256px -> 268 MB
    _tile_overlap: int = 32
    _native_scale: int = 4

    def __init__(self):
        self._sess = None
        self._sess_cpu = None  # CPU fallback when DML fails during inference

    def load(self) -> None:
        logger.info(f"Loading {self.name} from {self.model_id}")
        try:
            # Limit ANSI output from onnxruntime to ERROR level (suppresses
            # colored warnings that appear as cryptic character sequences in the terminal).
            ort.set_default_logger_severity(3)
            self._sess = ort.InferenceSession(
                self.model_id,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as e:
            logger.error(f"{self.name}: Loading failed — {e}")
            raise
        logger.info(f"{self.name} loaded")

    def unload(self) -> None:
        self._sess = None
        self._sess_cpu = None
        gc.collect()
        logger.info(f"{self.name} unloaded")

    def _build_ramp(self, length: int) -> np.ndarray:
        """Creates a linear ramp from 0.0 to 1.0 over `length` pixels."""
        if length <= 0:
            return np.array([], dtype=np.float32)
        return np.linspace(0.0, 1.0, length, dtype=np.float32)

    def _build_blend_mask(
        self,
        tile_h: int,
        tile_w: int,
        overlap_top: int,
        overlap_bottom: int,
        overlap_left: int,
        overlap_right: int,
    ) -> np.ndarray:
        """
        Creates a 2D weight mask (H, W) for a single tile.
        In the overlap regions, the weight falls linearly from 1 to 0,
        so that when adding neighboring tiles the sum of weights = 1.
        """
        mask = np.ones((tile_h, tile_w), dtype=np.float32)

        if overlap_top > 0:
            ramp = self._build_ramp(overlap_top)
            mask[:overlap_top, :] *= ramp[:, np.newaxis]

        if overlap_bottom > 0:
            ramp = self._build_ramp(overlap_bottom)
            mask[-overlap_bottom:, :] *= ramp[::-1, np.newaxis]

        if overlap_left > 0:
            ramp = self._build_ramp(overlap_left)
            mask[:, :overlap_left] *= ramp[np.newaxis, :]

        if overlap_right > 0:
            ramp = self._build_ramp(overlap_right)
            mask[:, -overlap_right:] *= ramp[::-1][np.newaxis, :]

        return mask

    def _infer_tile(self, tile_array: np.ndarray) -> np.ndarray:
        """
        Runs ONNX inference for a single tile.
        Input:  float32 (H, W, 3) in [0, 1]
        Output: float32 (H*4, W*4, 3) in [0, 1]

        Tries DML first. If DML fails (e.g. Windows-1252-encoded error
        message or unsupported op), automatically falls back to CPU.
        The CPU fallback applies to all subsequent tiles.
        """
        inp = tile_array.transpose(2, 0, 1)  # HWC -> CHW
        inp = np.expand_dims(inp, axis=0)     # -> [1,3,H,W]
        input_name = self._sess.get_inputs()[0].name
        try:
            outputs = self._sess.run(None, {input_name: inp})
        except Exception as e:
            # DML errors often come as Windows-1252 bytes — make them readable
            err_str = str(e) if isinstance(e, str) else repr(e)
            try:
                if isinstance(e.args[0], bytes):
                    err_str = e.args[0].decode("latin-1")
            except Exception:
                pass
            logger.warning(
                f"{self.name}: DML inference failed, switching to CPU — {err_str}"
            )
            if self._sess_cpu is None:
                logger.info(f"{self.name}: Creating CPU fallback session...")
                self._sess_cpu = ort.InferenceSession(
                    self.model_id,
                    providers=["CPUExecutionProvider"],
                )
            outputs = self._sess_cpu.run(None, {input_name: inp})
            # From now on all tiles via CPU, no longer attempting DML
            self._sess = self._sess_cpu
        out = outputs[0][0]                   # [3, H*4, W*4]
        out = out.transpose(1, 2, 0)          # -> HWC
        return np.clip(out, 0.0, 1.0)

    def _compute_tile_starts(self, total: int, tile: int, overlap: int) -> list[int]:
        """
        Computes the start positions for tiles along an axis.
        Ensures the entire image is covered and the last tile
        ends flush at the edge.
        """
        if total <= tile:
            return [0]

        stride = tile - overlap
        starts: list[int] = []
        pos = 0
        while pos + tile < total:
            starts.append(pos)
            pos += stride
        # Last tile flush at the edge
        starts.append(total - tile)
        return starts

    def upscale(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """
        Upscale image with Real-ESRGAN (tile-based).
        ONNX model is x4. For scale=2, the result is resized back to 2x.
        """
        if self._sess is None:
            raise RuntimeError(f"{self.name} is not loaded.")

        if scale not in self.supported_scales:
            raise ValueError(f"Scale {scale} not supported. Supported: {self.supported_scales}")

        img = image.convert("RGB")
        w, h = img.size
        logger.info(f"{self.name} — Upscale x{scale}: {w}x{h} -> {w * scale}x{h * scale}")
        t0 = time.monotonic()

        img_array = np.array(img).astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]

        s = self._native_scale
        tile = self._tile_size
        overlap = self._tile_overlap

        # Compute tile start positions
        y_starts = self._compute_tile_starts(h, tile, overlap)
        x_starts = self._compute_tile_starts(w, tile, overlap)

        total_tiles = len(y_starts) * len(x_starts)
        logger.info(
            f"{self.name} — {total_tiles} tiles ({len(x_starts)}x{len(y_starts)}), "
            f"tile size {tile}x{tile}, overlap {overlap}px"
        )

        # Output canvas: weighted accumulation
        out_h = h * s
        out_w = w * s
        output_accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_accum = np.zeros((out_h, out_w), dtype=np.float32)

        tile_idx = 0
        for i, y0 in enumerate(y_starts):
            for j, x0 in enumerate(x_starts):
                tile_idx += 1

                # Input tile coordinates (clamped to image boundaries)
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                tile_h = y1 - y0
                tile_w = x1 - x0

                tile_data = img_array[y0:y1, x0:x1, :]

                # ONNX inference for this tile
                tile_out = self._infer_tile(tile_data)

                # Output tile coordinates (in 4x space)
                out_y0 = y0 * s
                out_x0 = x0 * s
                out_tile_h = tile_h * s
                out_tile_w = tile_w * s

                # Compute actual overlap with neighboring tiles (in output space).
                # The last tile position may be much closer to the predecessor
                # than the normal stride due to the forced edge alignment.
                ol_top    = (y_starts[i - 1] + tile - y0) * s if i > 0 else 0
                ol_bottom = (y0 + tile - y_starts[i + 1]) * s if i < len(y_starts) - 1 else 0
                ol_left   = (x_starts[j - 1] + tile - x0) * s if j > 0 else 0
                ol_right  = (x0 + tile - x_starts[j + 1]) * s if j < len(x_starts) - 1 else 0

                mask = self._build_blend_mask(
                    out_tile_h, out_tile_w,
                    ol_top, ol_bottom, ol_left, ol_right,
                )

                # Weighted accumulation
                output_accum[out_y0:out_y0 + out_tile_h, out_x0:out_x0 + out_tile_w, :] += (
                    tile_out[:out_tile_h, :out_tile_w, :] * mask[:, :, np.newaxis]
                )
                weight_accum[out_y0:out_y0 + out_tile_h, out_x0:out_x0 + out_tile_w] += mask

                if tile_idx % 10 == 0 or tile_idx == total_tiles:
                    logger.info(f"{self.name} — Tile {tile_idx}/{total_tiles}")

        # Normalize by weight sum
        weight_accum = np.maximum(weight_accum, 1e-8)
        output_accum /= weight_accum[:, :, np.newaxis]

        out_pixels = np.clip(output_accum * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(out_pixels, "RGB")

        # For scale=2: resize result back to 2x
        if scale == 2:
            result = result.resize((w * 2, h * 2), Image.LANCZOS)

        logger.info(f"{self.name} — Done in {time.monotonic() - t0:.1f}s")
        return result

    @property
    def is_loaded(self) -> bool:
        return self._sess is not None or self._sess_cpu is not None

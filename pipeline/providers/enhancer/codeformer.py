import gc
import logging
import time
import numpy as np
import cv2

from pipeline.base.base_enhancer import BaseFaceRestoreProvider, EnhancerConfig
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

# Haar cascade path — ships with opencv-python, no extra download needed
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

# Factor by which the detected face bounding box is expanded (1.7 = 70% larger)
_BBOX_EXPAND = 1.7

# Gaussian blur radius for the feathered blending mask
_FEATHER_RADIUS = 25


class CodeFormerProvider(BaseFaceRestoreProvider):
    """
    CodeFormer: ONNX-based face restoration.
    Source: github.com/sczhou/CodeFormer
    ONNX export: python scripts/convert_to_onnx.py --model_path weights/CodeFormer/codeformer.pth
    Input: float32, shape [1,3,512,512], values in [-1, 1]
    Fidelity w: 0 = maximum restoration, 1 = maximum identity fidelity
    VRAM: ~1.5 GB (DirectML)

    Processing:
    - Face detection via OpenCV Haar Cascade
    - Each detected face is cropped with padding, restored at 512x512,
      and blended back with Gaussian feathering
    - Fallback: full image when no faces are detected
    """

    name = "CodeFormer"
    model_id = "models/enhancer/codeformer.onnx"
    vram_gb = 1.5
    _cascade = None  # Lazy-loaded, shared across instances

    def __init__(self):
        self._sess = None

    def load(self) -> None:
        logger.info(f"Loading {self.name} from {self.model_id}")
        try:
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
        gc.collect()
        logger.info(f"{self.name} unloaded")

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    @classmethod
    def _get_cascade(cls) -> cv2.CascadeClassifier:
        if cls._cascade is None:
            cls._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        return cls._cascade

    @classmethod
    def _detect_faces(cls, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV Haar Cascade and return expanded, clipped
        bounding boxes as (x1, y1, x2, y2) tuples.
        """
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        cascade = cls._get_cascade()
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64),
        )

        if detections is None or len(detections) == 0:
            return []

        boxes: list[tuple[int, int, int, int]] = []
        for (fx, fy, fw, fh) in detections:
            # Centre of the detected box
            cx = fx + fw / 2.0
            cy = fy + fh / 2.0

            # Expand by _BBOX_EXPAND factor
            new_w = fw * _BBOX_EXPAND
            new_h = fh * _BBOX_EXPAND

            x1 = int(max(0, cx - new_w / 2.0))
            y1 = int(max(0, cy - new_h / 2.0))
            x2 = int(min(w, cx + new_w / 2.0))
            y2 = int(min(h, cy + new_h / 2.0))

            boxes.append((x1, y1, x2, y2))

        return boxes

    # ------------------------------------------------------------------
    # ONNX inference on a single 512x512 crop
    # ------------------------------------------------------------------

    def _infer_512(self, crop: Image.Image, fidelity: float) -> Image.Image:
        """
        Run CodeFormer ONNX inference on a single RGB crop resized to 512x512.
        Returns the restored crop at 512x512.
        """
        img = crop.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # -> [1,3,512,512]

        w = np.array([fidelity], dtype=np.float32)

        input_names = [inp.name for inp in self._sess.get_inputs()]
        if len(input_names) >= 2:
            outputs = self._sess.run(None, {input_names[0]: img_array, input_names[1]: w})
        else:
            outputs = self._sess.run(None, {input_names[0]: img_array})

        out = outputs[0][0]  # [3,512,512]
        out = out.transpose(1, 2, 0)  # -> HWC
        out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")

    # ------------------------------------------------------------------
    # Feathered paste-back
    # ------------------------------------------------------------------

    @staticmethod
    def _create_feathered_mask(width: int, height: int) -> Image.Image:
        """
        Create a white-centred mask with Gaussian-feathered (soft) edges.
        The mask is used as an alpha channel when compositing the restored
        face crop back onto the original image so that the seam is invisible.
       """
        # Create a black (transparent) mask
        mask = Image.new("L", (width, height), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        # Draw a white rectangle in the center, leaving space for the blur radius
        pad = _FEATHER_RADIUS * 2
        draw.rectangle([pad, pad, width - pad, height - pad], fill=255)

        # Apply Gaussian blur to soften edges — larger radius = softer transition
        mask = mask.filter(ImageFilter.GaussianBlur(radius=_FEATHER_RADIUS))
        return mask

    # ------------------------------------------------------------------
    # Fallback: process entire image (original behaviour)
    # ------------------------------------------------------------------

    def _restore_full_image(self, image: Image.Image, config: EnhancerConfig) -> Image.Image:
        """Fallback when no faces are detected — process the whole image at 512x512."""
        original_size = image.size
        result = self._infer_512(image.convert("RGB"), config.fidelity)
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restore(self, image: Image.Image, config: EnhancerConfig) -> Image.Image:
        """
        Face restoration with CodeFormer.

        1. Detects faces in the image via Haar Cascade.
        2. Crops each face (with padding) and restores it at 512x512.
        3. Blends the result back with Gaussian feathering.
        4. Fallback: full image is processed when no faces are detected.
        """
        if self._sess is None:
            raise RuntimeError(f"{self.name} is not loaded.")

        original_size = image.size
        logger.info(
            f"{self.name} — Restore: fidelity={config.fidelity}, "
            f"size={original_size[0]}x{original_size[1]}"
        )
        t0 = time.monotonic()

        rgb_image = image.convert("RGB")

        # --- Face detection ---
        faces = self._detect_faces(rgb_image)

        if not faces:
            logger.warning(
                f"{self.name} — No faces detected, processing full image as fallback"
            )
            result = self._restore_full_image(rgb_image, config)
            logger.info(f"{self.name} — Done (fallback) in {time.monotonic() - t0:.1f}s")
            return result

        logger.info(f"{self.name} — {len(faces)} face(s) detected")

        # Start with a copy of the original image
        result = rgb_image.copy()

        for idx, (x1, y1, x2, y2) in enumerate(faces):
            crop_w = x2 - x1
            crop_h = y2 - y1
            logger.info(
                f"{self.name} — Face {idx + 1}/{len(faces)}: "
                f"bbox=({x1},{y1},{x2},{y2}), crop={crop_w}x{crop_h}"
            )

            # Crop the face region
            face_crop = rgb_image.crop((x1, y1, x2, y2))

            # Run ONNX inference at 512x512
            restored_crop = self._infer_512(face_crop, config.fidelity)

            # Resize restored crop back to the original crop dimensions
            restored_crop = restored_crop.resize((crop_w, crop_h), Image.LANCZOS)

            # Create feathered mask and paste back
            mask = self._create_feathered_mask(crop_w, crop_h)
            result.paste(restored_crop, (x1, y1), mask)

        logger.info(f"{self.name} — Done in {time.monotonic() - t0:.1f}s")
        return result

    @property
    def is_loaded(self) -> bool:
        return self._sess is not None

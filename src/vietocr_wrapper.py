"""
VietOCR Wrapper — Vietnamese Text Recognition with full diacritical marks.

Uses VietOCR (pbcquoc/vietocr) Transformer model trained on 10M Vietnamese images.
Designed to work with PaddleOCR detection (hybrid pipeline):
    PaddleOCR (detect text boxes) → crop → VietOCR (recognize Vietnamese text)

Features:
    - Transformer or Seq2Seq model support
    - GPU/CPU auto-detection
    - Batch prediction for speed
    - Auto-download pretrained weights
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils import get_logger

logger = get_logger(__name__)

# Model configs available in VietOCR
VIETOCR_MODELS = {
    "vgg_transformer": {
        "name": "vgg_transformer",
        "description": "VGG + Transformer — Higher accuracy, slower",
    },
    "vgg_seq2seq": {
        "name": "vgg_seq2seq",
        "description": "VGG + Seq2Seq (Attention) — Faster, slightly less accurate",
    },
}


class VietOCRWrapper:
    """Wrapper for VietOCR text recognition.

    Example:
        >>> wrapper = VietOCRWrapper(model_name="vgg_transformer", device="cuda")
        >>> text = wrapper.predict(cropped_image)
        >>> texts = wrapper.predict_batch([img1, img2, img3])
    """

    def __init__(
        self,
        model_name: str = "vgg_transformer",
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_name: 'vgg_transformer' (accurate) or 'vgg_seq2seq' (fast).
            device: 'cuda' or 'cpu'.
        """
        self.model_name = model_name
        self.device = device
        self._predictor = None
        self._config = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize VietOCR model with pretrained weights."""
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
        except ImportError:
            raise ImportError(
                "VietOCR is required for Vietnamese text recognition. "
                "Install it: pip install vietocr"
            )

        logger.info("📦 Loading VietOCR (model=%s, device=%s)...",
                     self.model_name, self.device)

        # Load pretrained config
        self._config = Cfg.load_config_from_name(self.model_name)

        # Set device
        self._config['device'] = self.device
        self._config['cnn']['pretrained'] = True
        self._config['predictor']['beamsearch'] = False  # greedy for speed

        self._predictor = Predictor(self._config)
        logger.info("✅ VietOCR ready (model=%s)", self.model_name)

    def predict(self, image: np.ndarray) -> str:
        """Predict text from a single cropped image.

        Args:
            image: Cropped text region (BGR numpy array).

        Returns:
            Recognized Vietnamese text with diacritical marks.
        """
        from PIL import Image

        # Convert BGR (OpenCV) → RGB (PIL)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1]
        else:
            rgb = image

        pil_img = Image.fromarray(rgb)
        text = self._predictor.predict(pil_img)
        return text.strip()

    def predict_batch(self, images: List[np.ndarray]) -> List[str]:
        """Predict text from multiple cropped images.

        Args:
            images: List of cropped text regions (BGR numpy arrays).

        Returns:
            List of recognized texts.
        """
        if not images:
            return []

        results = []
        for img in images:
            try:
                text = self.predict(img)
                results.append(text)
            except Exception as e:
                logger.warning("⚠️ VietOCR prediction failed for a crop: %s", e)
                results.append("")

        return results

    @staticmethod
    def crop_text_region(
        image: np.ndarray,
        bbox: List[List[float]],
        padding: int = 2,
    ) -> np.ndarray:
        """Crop a text region from image using bounding box.

        Handles rotated bounding boxes by applying perspective transform
        to get a straight horizontal text crop.

        Args:
            image: Full image (BGR).
            bbox: 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
            padding: Extra pixels around the crop.

        Returns:
            Cropped and straightened text region.
        """
        import cv2

        pts = np.array(bbox, dtype=np.float32)

        # Calculate width and height of the bounding box
        width = int(max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3]),
        ))
        height = int(max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2]),
        ))

        if width <= 0 or height <= 0:
            return np.zeros((32, 100, 3), dtype=np.uint8)

        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ], dtype=np.float32)

        # Perspective transform to straighten the text
        M = cv2.getPerspectiveTransform(pts, dst)
        cropped = cv2.warpPerspective(image, M, (width, height))

        # Add padding
        if padding > 0:
            cropped = cv2.copyMakeBorder(
                cropped, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )

        return cropped

    def cleanup(self) -> None:
        """Release model resources."""
        self._predictor = None
        self._config = None
        logger.info("🧹 VietOCR released")

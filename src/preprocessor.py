"""
Image Preprocessor — Auto-rotate, deskew, resize, denoise.

Handles common issues with phone-captured document photos:
    - EXIF rotation metadata
    - Skewed/tilted scans
    - Oversized images
    - Noise reduction (optional)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.utils import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Document image preprocessing pipeline.

    Example:
        >>> prep = Preprocessor(max_size=1280, denoise=True)
        >>> image = prep.process("scan.jpg")
        >>> # Returns preprocessed numpy array ready for OCR
    """

    def __init__(
        self,
        max_size: int = 1280,
        denoise: bool = False,
        auto_rotate: bool = True,
        deskew: bool = True,
    ) -> None:
        self.max_size = max_size
        self.denoise = denoise
        self.auto_rotate = auto_rotate
        self.deskew = deskew
        logger.info(
            "🔧 Preprocessor | max_size=%d | denoise=%s | deskew=%s",
            max_size, denoise, deskew,
        )

    def process(self, image_path: str) -> np.ndarray:
        """Full preprocessing pipeline.

        Args:
            image_path: Path to image file.

        Returns:
            Preprocessed image as numpy array (BGR).
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load image
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")

        h, w = image.shape[:2]
        logger.info("📷 Loaded: %s (%dx%d)", path.name, w, h)

        # Step 1: EXIF rotation
        if self.auto_rotate:
            image = self._fix_exif_orientation(str(path), image)

        # Step 2: Resize
        image = self._smart_resize(image)

        # Step 3: Deskew
        if self.deskew:
            image = self._deskew(image)

        # Step 4: Denoise (optional)
        if self.denoise:
            image = self._denoise(image)

        h, w = image.shape[:2]
        logger.info("✅ Preprocessed: %dx%d", w, h)
        return image

    # ========================================================================
    # EXIF ROTATION
    # ========================================================================

    @staticmethod
    def _fix_exif_orientation(image_path: str, image: np.ndarray) -> np.ndarray:
        """Fix image orientation based on EXIF metadata.

        Phone cameras store rotation in EXIF rather than rotating pixels.
        """
        try:
            from PIL import Image
            from PIL.ExifTags import Base as ExifBase

            pil_img = Image.open(image_path)
            exif = pil_img.getexif()

            if not exif:
                return image

            orientation = exif.get(ExifBase.Orientation, 1)

            rotation_map = {
                3: cv2.ROTATE_180,
                6: cv2.ROTATE_90_CLOCKWISE,
                8: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }

            if orientation in rotation_map:
                logger.info("🔄 EXIF rotation: orientation=%d", orientation)
                image = cv2.rotate(image, rotation_map[orientation])

        except Exception as e:
            logger.debug("EXIF read failed (OK): %s", e)

        return image

    # ========================================================================
    # SMART RESIZE
    # ========================================================================

    def _smart_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image if larger than max_size, preserving aspect ratio."""
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim <= self.max_size:
            return image

        scale = self.max_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info("📐 Resize: %dx%d → %dx%d (scale=%.2f)", w, h, new_w, new_h, scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ========================================================================
    # DESKEW
    # ========================================================================

    @staticmethod
    def _deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """Correct small rotation/skew in scanned documents.

        Uses Hough line detection to find dominant angle.
        Only corrects angles within ±max_angle degrees.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return image

        # Calculate dominant angle from detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only near-horizontal lines (within ±max_angle of 0° or 180°)
            if abs(angle) < max_angle:
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        # Only deskew if angle is significant (> 0.5°)
        if abs(median_angle) < 0.5:
            return image

        logger.info("📐 Deskew: %.1f°", median_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # ========================================================================
    # DENOISE
    # ========================================================================

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for noise reduction while preserving edges."""
        logger.info("🧹 Applying denoise filter")
        return cv2.bilateralFilter(image, 9, 75, 75)

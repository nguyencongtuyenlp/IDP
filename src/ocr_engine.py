"""
OCR Engine — PaddleOCR + VietOCR Hybrid Pipeline.

Hybrid architecture for Vietnamese OCR with full diacritical marks:
    - PaddleOCR → text detection (bounding boxes)
    - VietOCR  → text recognition (Vietnamese with dấu)

Fallback: PaddleOCR-only for non-Vietnamese or when VietOCR unavailable.

Features:
    - Multi-language support (vi, en, ch, ja, ko, ...)
    - VietOCR Transformer for accurate Vietnamese recognition
    - GPU/CPU auto-detection
    - Configurable quality presets
    - Structured output: List[OCRResult]
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils import get_logger, timer
from src.device_manager import DeviceManager
from src.preprocessor import Preprocessor

logger = get_logger(__name__)


class OCRResult:
    """Single OCR detection result."""

    def __init__(self, bbox: List[List[float]], text: str, confidence: float):
        self.bbox = bbox          # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        self.text = text
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }

    def __repr__(self) -> str:
        return f"OCRResult(text='{self.text[:30]}', conf={self.confidence:.2f})"


class OCREngine:
    """PaddleOCR wrapper with preprocessing + GPU/CPU fallback.

    Example:
        >>> engine = OCREngine(device="auto", mode="balanced", lang="vi")
        >>> results = engine.process("document.jpg")
        >>> for r in results:
        ...     print(f"{r.text} ({r.confidence:.0%})")
    """

    def __init__(
        self,
        device: str = "auto",
        mode: str = "balanced",
        lang: str = "vi",
        denoise: bool = False,
        use_vietocr: bool = True,
        vietocr_model: str = "vgg_transformer",
    ) -> None:
        self.device_mgr = DeviceManager(device=device, mode=mode)
        self.lang = lang
        self.use_vietocr = use_vietocr and lang == "vi"
        self.preprocessor = Preprocessor(
            max_size=self.device_mgr.preset.max_image_size,
            denoise=denoise,
            auto_rotate=True,
            deskew=True,
        )
        self._ocr = None
        self._vietocr = None
        self._init_paddle()
        if self.use_vietocr:
            self._init_vietocr(vietocr_model)

    def _init_paddle(self) -> None:
        """Initialize PaddleOCR with current device/mode settings."""
        from paddleocr import PaddleOCR

        paddle_kwargs = self.device_mgr.get_paddle_kwargs()

        # Always init PaddleOCR in full mode; detection-only is controlled
        # at call time via ocr(rec=False, cls=False)
        mode_label = "detection-only" if self.use_vietocr else "full"
        logger.info("📦 Initializing PaddleOCR [%s] (lang=%s, gpu=%s)...",
                     mode_label, self.lang, paddle_kwargs["use_gpu"])

        self._ocr = PaddleOCR(
            lang=self.lang,
            show_log=False,
            **paddle_kwargs,
        )
        logger.info("✅ PaddleOCR ready")

    def _init_vietocr(self, model_name: str) -> None:
        """Initialize VietOCR for Vietnamese text recognition."""
        try:
            from src.vietocr_wrapper import VietOCRWrapper

            vietocr_device = "cuda" if self.device_mgr.use_gpu else "cpu"
            self._vietocr = VietOCRWrapper(
                model_name=model_name,
                device=vietocr_device,
            )
            logger.info("✅ VietOCR ready (model=%s)", model_name)
        except ImportError:
            logger.warning(
                "⚠️ VietOCR not installed. Falling back to PaddleOCR recognition. "
                "Install: pip install vietocr"
            )
            self.use_vietocr = False
            # Reinit PaddleOCR in full mode
            self._init_paddle()

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    @timer
    def process(
        self,
        image_input: Union[str, Path, np.ndarray],
        confidence_threshold: float = 0.3,
    ) -> List[OCRResult]:
        """Run OCR on an image.

        Args:
            image_input: Path to image or numpy array.
            confidence_threshold: Minimum confidence to include result.

        Returns:
            List of OCRResult objects, sorted by reading order.
        """
        # Preprocess if path given
        if isinstance(image_input, (str, Path)):
            image = self.preprocessor.process(str(image_input))
        else:
            image = image_input

        if self.use_vietocr and self._vietocr is not None:
            return self._process_hybrid(image, confidence_threshold)
        else:
            return self._process_paddle_only(image, confidence_threshold)

    def _process_hybrid(
        self,
        image: np.ndarray,
        confidence_threshold: float,
    ) -> List[OCRResult]:
        """Hybrid mode: PaddleOCR detection + VietOCR recognition."""
        from src.vietocr_wrapper import VietOCRWrapper

        logger.info("🔍 Running hybrid OCR (PaddleOCR detect → VietOCR recognize)...")

        try:
            # Step 1: PaddleOCR detection only (cls=False required when rec=False)
            raw_results = self._ocr.ocr(image, rec=False, cls=False)
        except (ValueError, AttributeError, TypeError) as e:
            # PaddleOCR version incompatibility
            logger.warning(
                "⚠️ PaddleOCR detection-only failed (%s). "
                "Falling back to full mode (no VietOCR).", str(e)[:60]
            )
            return self._process_paddle_only(image, confidence_threshold)

        results = []
        if raw_results and raw_results[0]:
            bboxes = raw_results[0]  # List of bounding boxes

            # Step 2: Crop each text region
            crops = []
            valid_bboxes = []
            for bbox in bboxes:
                try:
                    bbox_float = [[float(p[0]), float(p[1])] for p in bbox]
                    crop = VietOCRWrapper.crop_text_region(image, bbox_float)
                    if crop.size > 0:
                        crops.append(crop)
                        valid_bboxes.append(bbox_float)
                except Exception:
                    continue

            # Step 3: VietOCR recognition on all crops
            if crops:
                texts = self._vietocr.predict_batch(crops)

                for bbox, text in zip(valid_bboxes, texts):
                    text = text.strip()
                    if text:
                        results.append(OCRResult(
                            bbox=bbox,
                            text=text,
                            confidence=0.95,  # VietOCR doesn't return confidence
                        ))

        logger.info("📝 Detected %d text regions (VietOCR hybrid)", len(results))
        return results

    def _process_paddle_only(
        self,
        image: np.ndarray,
        confidence_threshold: float,
    ) -> List[OCRResult]:
        """PaddleOCR-only mode: detection + recognition."""
        logger.info("🔍 Running OCR (PaddleOCR only)...")
        raw_results = self._ocr.ocr(image, cls=self.device_mgr.preset.use_angle_cls)

        # Parse results: PaddleOCR returns [ [ [bbox, (text, conf)], ... ] ]
        results = []
        if raw_results and raw_results[0]:
            for line in raw_results[0]:
                try:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = str(line[1][0]).strip()
                    confidence = float(line[1][1])

                    if confidence >= confidence_threshold and text:
                        results.append(OCRResult(
                            bbox=[[float(p[0]), float(p[1])] for p in bbox],
                            text=text,
                            confidence=confidence,
                        ))
                except (IndexError, ValueError, TypeError):
                    continue

        logger.info("📝 Detected %d text regions (threshold=%.0f%%)",
                     len(results), confidence_threshold * 100)

        return results

    @timer
    def process_to_text(
        self,
        image_input: Union[str, Path, np.ndarray],
        confidence_threshold: float = 0.3,
    ) -> str:
        """Run OCR and return concatenated text (reading order)."""
        results = self.process(image_input, confidence_threshold)
        sorted_results = self._sort_reading_order(results)
        return "\n".join(r.text for r in sorted_results)

    # ========================================================================
    # READING ORDER
    # ========================================================================

    @staticmethod
    def _sort_reading_order(results: List[OCRResult]) -> List[OCRResult]:
        """Sort results in reading order: top-to-bottom, left-to-right."""
        if not results:
            return results

        sorted_by_y = sorted(results, key=lambda r: r.bbox[0][1])

        lines = []
        current_line = [sorted_by_y[0]]
        line_y = sorted_by_y[0].bbox[0][1]

        for result in sorted_by_y[1:]:
            y = result.bbox[0][1]
            bbox_height = abs(result.bbox[2][1] - result.bbox[0][1])
            threshold = max(bbox_height * 0.5, 10)

            if abs(y - line_y) < threshold:
                current_line.append(result)
            else:
                lines.append(current_line)
                current_line = [result]
                line_y = y

        lines.append(current_line)

        ordered = []
        for line in lines:
            line.sort(key=lambda r: r.bbox[0][0])
            ordered.extend(line)

        return ordered

    # ========================================================================
    # ANNOTATION
    # ========================================================================

    @staticmethod
    def draw_boxes(
        image: np.ndarray,
        results: List[OCRResult],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        show_text: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and text on image."""
        import cv2
        annotated = image.copy()

        for r in results:
            pts = np.array(r.bbox, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, color, thickness)

            if show_text:
                label = r.text
                if show_confidence:
                    label += f" ({r.confidence:.0%})"

                x, y = int(r.bbox[0][0]), int(r.bbox[0][1]) - 5
                cv2.putText(
                    annotated, label, (x, max(y, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA,
                )

        return annotated

    def cleanup(self) -> None:
        """Release OCR engine resources."""
        self._ocr = None
        if self._vietocr is not None:
            self._vietocr.cleanup()
            self._vietocr = None
        logger.info("🧹 OCR engine released")

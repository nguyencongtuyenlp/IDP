"""
Postprocessor — Merge lines, format output, export files.

Handles:
    - Merging text regions into logical lines/paragraphs
    - Exporting to text, JSON, and annotated images
    - Summary statistics
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class Postprocessor:
    """Process and export OCR results in multiple formats.

    Example:
        >>> post = Postprocessor(output_dir="data/output")
        >>> post.export_all(results, image, "invoice.jpg")
    """

    def __init__(self, output_dir: str = "data/output") -> None:
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def merge_lines(self, results: list) -> List[str]:
        """Merge OCR results into coherent text lines.

        Groups nearby text regions into same line based on
        Y-coordinate proximity, then concatenates with spaces.
        """
        if not results:
            return []

        # Sort by Y first, then X
        sorted_results = sorted(results, key=lambda r: (r.bbox[0][1], r.bbox[0][0]))

        lines = []
        current_line_texts = [sorted_results[0].text]
        current_y = sorted_results[0].bbox[0][1]

        for r in sorted_results[1:]:
            y = r.bbox[0][1]
            bbox_height = abs(r.bbox[2][1] - r.bbox[0][1])
            threshold = max(bbox_height * 0.5, 10)

            if abs(y - current_y) < threshold:
                current_line_texts.append(r.text)
            else:
                lines.append(" ".join(current_line_texts))
                current_line_texts = [r.text]
                current_y = y

        lines.append(" ".join(current_line_texts))
        return lines

    def export_text(self, results: list, filename: str) -> Path:
        """Export OCR results as plain text file."""
        lines = self.merge_lines(results)
        text = "\n".join(lines)

        out_path = self.output_dir / f"{Path(filename).stem}_extracted.txt"
        out_path.write_text(text, encoding="utf-8")
        logger.info("💾 Text saved: %s (%d lines)", out_path.name, len(lines))
        return out_path

    def export_json(
        self,
        results: list,
        filename: str,
        processing_time: float = 0,
        device: str = "cpu",
        mode: str = "balanced",
    ) -> Path:
        """Export OCR results as structured JSON."""
        lines = self.merge_lines(results)

        output = {
            "file": filename,
            "device": device,
            "mode": mode,
            "num_regions": len(results),
            "text": "\n".join(lines),
            "regions": [r.to_dict() for r in results],
            "metadata": {
                "processing_time_seconds": round(processing_time, 3),
                "avg_confidence": round(
                    sum(r.confidence for r in results) / max(len(results), 1), 4
                ),
                "status": "success" if results else "no_text",
            },
        }

        out_path = self.output_dir / f"{Path(filename).stem}_result.json"
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("💾 JSON saved: %s", out_path.name)
        return out_path

    def export_annotated(
        self,
        image: np.ndarray,
        results: list,
        filename: str,
        draw_func=None,
    ) -> Path:
        """Export annotated image with bounding boxes.

        Args:
            image: Original image (BGR).
            results: OCR results.
            filename: Original filename.
            draw_func: Optional custom drawing function.
        """
        if draw_func:
            annotated = draw_func(image, results)
        else:
            # Default: draw green boxes
            annotated = image.copy()
            for r in results:
                pts = np.array(r.bbox, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

                # Label
                label = f"{r.text[:25]} ({r.confidence:.0%})"
                x, y = int(r.bbox[0][0]), int(r.bbox[0][1]) - 5
                cv2.putText(
                    annotated, label, (x, max(y, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
                )

        out_path = self.output_dir / f"{Path(filename).stem}_annotated.png"
        cv2.imwrite(str(out_path), annotated)
        logger.info("💾 Annotated image saved: %s", out_path.name)
        return out_path

    def export_all(
        self,
        results: list,
        image: np.ndarray,
        filename: str,
        processing_time: float = 0,
        device: str = "cpu",
        mode: str = "balanced",
        draw_func=None,
    ) -> Dict[str, Path]:
        """Export all formats: text + JSON + annotated image.

        Returns:
            Dict with keys: 'text', 'json', 'annotated'.
        """
        return {
            "text": self.export_text(results, filename),
            "json": self.export_json(results, filename, processing_time, device, mode),
            "annotated": self.export_annotated(image, results, filename, draw_func),
        }

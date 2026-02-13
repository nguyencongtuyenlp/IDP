"""
PDF Handler — PDF to images conversion + optional searchable PDF.

Uses PyMuPDF (fitz) for PDF→image conversion.
Supports multi-page PDFs with progress tracking.
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


class PDFHandler:
    """Handle PDF input/output for OCR pipeline.

    Example:
        >>> handler = PDFHandler(dpi=200)
        >>> images = handler.pdf_to_images("document.pdf")
        >>> # Process each page image through OCR
    """

    def __init__(self, dpi: int = 200, max_pages: int = 50) -> None:
        """
        Args:
            dpi: Resolution for PDF rendering (higher = better quality, slower).
            max_pages: Maximum pages to process (safety limit).
        """
        self.dpi = dpi
        self.max_pages = max_pages
        logger.info("📄 PDFHandler | dpi=%d | max_pages=%d", dpi, max_pages)

    def pdf_to_images(
        self,
        pdf_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file.
            page_range: Optional (start, end) page range (0-indexed).

        Returns:
            List of page images as numpy arrays (BGR).
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install it: pip install PyMuPDF"
            )

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(str(path))
        total_pages = len(doc)
        logger.info("📄 PDF: %s | %d pages", path.name, total_pages)

        # Determine page range
        start = 0
        end = min(total_pages, self.max_pages)
        if page_range:
            start = max(0, page_range[0])
            end = min(total_pages, page_range[1])

        if total_pages > self.max_pages:
            logger.warning(
                "⚠️  PDF has %d pages, processing only first %d",
                total_pages, self.max_pages,
            )

        images = []
        zoom = self.dpi / 72  # 72 DPI is PDF default
        matrix = fitz.Matrix(zoom, zoom)

        for page_idx in range(start, end):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)

            # Convert to numpy array (RGB → BGR for OpenCV)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_data.reshape((pix.height, pix.width, pix.n))

            if pix.n == 4:  # RGBA
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            images.append(img)
            logger.debug("  📄 Page %d/%d: %dx%d", page_idx + 1, end, pix.width, pix.height)

        doc.close()
        logger.info("✅ Converted %d pages to images", len(images))
        return images

    def get_page_count(self, pdf_path: str) -> int:
        """Get number of pages without converting."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    @staticmethod
    def is_pdf(file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == ".pdf"

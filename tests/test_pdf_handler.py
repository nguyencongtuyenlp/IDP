"""
Tests for src/pdf_handler.py — PDF conversion and helpers.
"""

import pytest

from src.pdf_handler import PDFHandler


class TestIsPdf:
    def test_pdf_file(self):
        assert PDFHandler.is_pdf("document.pdf") is True

    def test_pdf_uppercase(self):
        assert PDFHandler.is_pdf("DOCUMENT.PDF") is True

    def test_jpg_file(self):
        assert PDFHandler.is_pdf("photo.jpg") is False

    def test_png_file(self):
        assert PDFHandler.is_pdf("image.png") is False

    def test_no_extension(self):
        assert PDFHandler.is_pdf("filename") is False

    def test_pdf_in_name(self):
        """File with 'pdf' in name but different extension."""
        assert PDFHandler.is_pdf("pdf_report.txt") is False


class TestPDFHandler:
    def test_init_defaults(self):
        handler = PDFHandler()
        assert handler.dpi == 200
        assert handler.max_pages == 50

    def test_init_custom(self):
        handler = PDFHandler(dpi=300, max_pages=10)
        assert handler.dpi == 300
        assert handler.max_pages == 10

    def test_get_page_count_nonexistent(self):
        handler = PDFHandler()
        count = handler.get_page_count("nonexistent.pdf")
        assert count == 0

    def test_pdf_to_images_file_not_found(self):
        handler = PDFHandler()
        with pytest.raises(FileNotFoundError):
            handler.pdf_to_images("nonexistent.pdf")

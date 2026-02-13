"""
Tests for src/ocr_engine.py — PaddleOCR wrapper.
"""

import numpy as np
import pytest

from src.ocr_engine import OCRResult, OCREngine


# ============================================================================
# OCRResult
# ============================================================================


class TestOCRResult:
    def test_init(self):
        r = OCRResult(
            bbox=[[0, 0], [100, 0], [100, 30], [0, 30]],
            text="Hello",
            confidence=0.95,
        )
        assert r.text == "Hello"
        assert r.confidence == 0.95
        assert len(r.bbox) == 4

    def test_to_dict(self):
        r = OCRResult(
            bbox=[[10, 20], [100, 20], [100, 50], [10, 50]],
            text="Test text",
            confidence=0.8765,
        )
        d = r.to_dict()
        assert d["text"] == "Test text"
        assert d["confidence"] == 0.8765
        assert d["bbox"] == [[10, 20], [100, 20], [100, 50], [10, 50]]

    def test_repr(self):
        r = OCRResult(
            bbox=[[0, 0], [50, 0], [50, 20], [0, 20]],
            text="Short",
            confidence=0.99,
        )
        repr_str = repr(r)
        assert "Short" in repr_str
        assert "0.99" in repr_str

    def test_repr_long_text(self):
        """Long text should be truncated in repr."""
        r = OCRResult(
            bbox=[[0, 0], [50, 0], [50, 20], [0, 20]],
            text="A" * 100,
            confidence=0.5,
        )
        repr_str = repr(r)
        assert len(repr_str) < 200


# ============================================================================
# Reading Order Sort
# ============================================================================


class TestSortReadingOrder:
    def test_empty(self):
        assert OCREngine._sort_reading_order([]) == []

    def test_single_result(self):
        r = OCRResult([[0, 0], [50, 0], [50, 20], [0, 20]], "Only", 0.9)
        result = OCREngine._sort_reading_order([r])
        assert len(result) == 1

    def test_top_to_bottom(self):
        """Results should be sorted top to bottom."""
        r1 = OCRResult([[0, 100], [50, 100], [50, 120], [0, 120]], "Bottom", 0.9)
        r2 = OCRResult([[0, 10], [50, 10], [50, 30], [0, 30]], "Top", 0.9)
        result = OCREngine._sort_reading_order([r1, r2])
        assert result[0].text == "Top"
        assert result[1].text == "Bottom"

    def test_left_to_right_same_line(self):
        """Same-line results should be sorted left to right."""
        r1 = OCRResult([[200, 10], [300, 10], [300, 30], [200, 30]], "Right", 0.9)
        r2 = OCRResult([[10, 10], [100, 10], [100, 30], [10, 30]], "Left", 0.9)
        result = OCREngine._sort_reading_order([r1, r2])
        assert result[0].text == "Left"
        assert result[1].text == "Right"

    def test_multi_line(self, sample_ocr_results):
        result = OCREngine._sort_reading_order(sample_ocr_results)
        assert len(result) == 3
        # First result should be top-most
        assert result[0].text == "Hello World"


# ============================================================================
# Draw Boxes
# ============================================================================


class TestDrawBoxes:
    def test_output_shape(self, sample_image, sample_ocr_results):
        annotated = OCREngine.draw_boxes(sample_image, sample_ocr_results)
        assert annotated.shape == sample_image.shape

    def test_does_not_modify_original(self, sample_image, sample_ocr_results):
        original_copy = sample_image.copy()
        OCREngine.draw_boxes(sample_image, sample_ocr_results)
        np.testing.assert_array_equal(sample_image, original_copy)

    def test_empty_results(self, sample_image):
        annotated = OCREngine.draw_boxes(sample_image, [])
        assert annotated.shape == sample_image.shape


# ============================================================================
# Integration Test (requires PaddleOCR installed)
# ============================================================================


class TestOCREngineIntegration:
    @pytest.fixture(autouse=True)
    def _check_paddle(self):
        """Skip if PaddleOCR is not available."""
        try:
            import paddleocr
        except ImportError:
            pytest.skip("PaddleOCR not installed")

    def test_process_real_image(self, test_image_path):
        engine = OCREngine(device="cpu", mode="fast", lang="vi")
        results = engine.process(str(test_image_path))

        assert isinstance(results, list)
        assert len(results) > 0

        for r in results:
            assert isinstance(r, OCRResult)
            assert len(r.text) > 0
            assert 0 <= r.confidence <= 1
            assert len(r.bbox) == 4

        engine.cleanup()

    def test_process_to_text(self, test_image_path):
        engine = OCREngine(device="cpu", mode="fast", lang="vi")
        text = engine.process_to_text(str(test_image_path))

        assert isinstance(text, str)
        assert len(text) > 0

        engine.cleanup()

    def test_cleanup(self):
        engine = OCREngine(device="cpu", mode="fast", lang="vi")
        engine.cleanup()
        assert engine._ocr is None

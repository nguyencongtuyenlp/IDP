"""
Tests for src/preprocessor.py — Image preprocessing pipeline.
"""

import numpy as np
import pytest

from src.preprocessor import Preprocessor


class TestSmartResize:
    def test_large_image_resized(self, large_image):
        prep = Preprocessor(max_size=1280, deskew=False, denoise=False)
        result = prep._smart_resize(large_image)
        h, w = result.shape[:2]
        assert max(h, w) <= 1280

    def test_small_image_unchanged(self, sample_image):
        prep = Preprocessor(max_size=1280, deskew=False, denoise=False)
        result = prep._smart_resize(sample_image)
        assert result.shape == sample_image.shape

    def test_preserves_aspect_ratio(self, large_image):
        prep = Preprocessor(max_size=1280, deskew=False, denoise=False)
        h_orig, w_orig = large_image.shape[:2]
        result = prep._smart_resize(large_image)
        h_new, w_new = result.shape[:2]
        ratio_orig = w_orig / h_orig
        ratio_new = w_new / h_new
        assert abs(ratio_orig - ratio_new) < 0.02


class TestDeskew:
    def test_straight_image_unchanged(self, sample_image):
        """Straight image should not be significantly altered."""
        result = Preprocessor._deskew(sample_image)
        assert result.shape == sample_image.shape

    def test_output_shape_preserved(self, sample_image):
        result = Preprocessor._deskew(sample_image, max_angle=15.0)
        assert result.shape == sample_image.shape


class TestDenoise:
    def test_output_shape_preserved(self, sample_image):
        result = Preprocessor._denoise(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        result = Preprocessor._denoise(sample_image)
        assert result.dtype == np.uint8


class TestPreprocessorProcess:
    def test_process_real_image(self, test_image_path):
        """Integration test with real image file."""
        prep = Preprocessor(max_size=960, denoise=False, deskew=True)
        result = prep.process(str(test_image_path))
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # H, W, C
        assert result.shape[2] == 3   # BGR
        assert max(result.shape[:2]) <= 960

    def test_process_file_not_found(self):
        prep = Preprocessor()
        with pytest.raises(FileNotFoundError):
            prep.process("nonexistent_image.jpg")

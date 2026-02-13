"""
Tests for VietOCR wrapper.

Tests cover:
    - Wrapper initialization and model config
    - Image cropping (perspective transform)
    - Prediction (mocked)
    - Batch prediction
    - Cleanup
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestCropTextRegion:
    """Test static crop_text_region method."""

    def test_crop_basic_rectangle(self):
        """Test cropping a simple rectangular region."""
        from src.vietocr_wrapper import VietOCRWrapper

        # Create a 200x200 test image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Simple rectangle bbox
        bbox = [[10, 10], [100, 10], [100, 40], [10, 40]]
        crop = VietOCRWrapper.crop_text_region(image, bbox)

        assert crop is not None
        assert len(crop.shape) == 3
        # Should be roughly the width and height of the bbox + padding
        assert crop.shape[0] > 0  # height
        assert crop.shape[1] > 0  # width

    def test_crop_with_padding(self):
        """Test that padding is applied."""
        from src.vietocr_wrapper import VietOCRWrapper

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = [[20, 20], [80, 20], [80, 50], [20, 50]]

        crop_no_pad = VietOCRWrapper.crop_text_region(image, bbox, padding=0)
        crop_with_pad = VietOCRWrapper.crop_text_region(image, bbox, padding=5)

        # Padded version should be larger
        assert crop_with_pad.shape[0] > crop_no_pad.shape[0]
        assert crop_with_pad.shape[1] > crop_no_pad.shape[1]

    def test_crop_zero_size_returns_fallback(self):
        """Test that zero-size bbox returns fallback image."""
        from src.vietocr_wrapper import VietOCRWrapper

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = [[10, 10], [10, 10], [10, 10], [10, 10]]  # Zero area

        crop = VietOCRWrapper.crop_text_region(image, bbox)
        assert crop is not None
        assert crop.shape == (32, 100, 3)  # Fallback shape

    def test_crop_rotated_bbox(self):
        """Test cropping a rotated bounding box."""
        from src.vietocr_wrapper import VietOCRWrapper

        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        # Slightly rotated bbox
        bbox = [[50, 30], [150, 25], [152, 55], [52, 60]]
        crop = VietOCRWrapper.crop_text_region(image, bbox)

        assert crop is not None
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0


class TestVietOCRModels:
    """Test model config constants."""

    def test_model_names(self):
        from src.vietocr_wrapper import VIETOCR_MODELS

        assert "vgg_transformer" in VIETOCR_MODELS
        assert "vgg_seq2seq" in VIETOCR_MODELS

    def test_model_has_description(self):
        from src.vietocr_wrapper import VIETOCR_MODELS

        for name, cfg in VIETOCR_MODELS.items():
            assert "name" in cfg
            assert "description" in cfg


class TestVietOCRWrapperMocked:
    """Test VietOCR wrapper with mocked VietOCR library."""

    @patch("src.vietocr_wrapper.VietOCRWrapper._init_model")
    def test_init_sets_attributes(self, mock_init):
        """Test constructor sets model_name and device."""
        from src.vietocr_wrapper import VietOCRWrapper

        wrapper = VietOCRWrapper(model_name="vgg_seq2seq", device="cpu")
        assert wrapper.model_name == "vgg_seq2seq"
        assert wrapper.device == "cpu"
        mock_init.assert_called_once()

    @patch("src.vietocr_wrapper.VietOCRWrapper._init_model")
    def test_predict_converts_bgr_to_rgb(self, mock_init):
        """Test that predict converts BGR to RGB for PIL."""
        from src.vietocr_wrapper import VietOCRWrapper

        wrapper = VietOCRWrapper(model_name="vgg_transformer", device="cpu")

        # Mock the predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = "Tiếng suối trong như tiếng hát xa"
        wrapper._predictor = mock_predictor

        bgr_image = np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
        result = wrapper.predict(bgr_image)

        assert result == "Tiếng suối trong như tiếng hát xa"
        mock_predictor.predict.assert_called_once()

    @patch("src.vietocr_wrapper.VietOCRWrapper._init_model")
    def test_predict_batch(self, mock_init):
        """Test batch prediction."""
        from src.vietocr_wrapper import VietOCRWrapper

        wrapper = VietOCRWrapper(model_name="vgg_transformer", device="cpu")

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = ["Dòng 1", "Dòng 2", "Dòng 3"]
        wrapper._predictor = mock_predictor

        images = [
            np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        results = wrapper.predict_batch(images)

        assert len(results) == 3
        assert results[0] == "Dòng 1"
        assert results[1] == "Dòng 2"
        assert results[2] == "Dòng 3"

    @patch("src.vietocr_wrapper.VietOCRWrapper._init_model")
    def test_predict_batch_empty(self, mock_init):
        """Test batch prediction with empty list."""
        from src.vietocr_wrapper import VietOCRWrapper

        wrapper = VietOCRWrapper(model_name="vgg_transformer", device="cpu")
        results = wrapper.predict_batch([])
        assert results == []

    @patch("src.vietocr_wrapper.VietOCRWrapper._init_model")
    def test_predict_batch_handles_errors(self, mock_init):
        """Test batch prediction handles individual prediction errors."""
        from src.vietocr_wrapper import VietOCRWrapper

        wrapper = VietOCRWrapper(model_name="vgg_transformer", device="cpu")

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = [
            "OK text",
            ValueError("bad image"),
            "Another text",
        ]
        wrapper._predictor = mock_predictor

        images = [
            np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        results = wrapper.predict_batch(images)

        assert len(results) == 3
        assert results[0] == "OK text"
        assert results[1] == ""  # Error → empty string
        assert results[2] == "Another text"

    @patch("src.vietocr_wrapper.VietOCRWrapper._init_model")
    def test_cleanup(self, mock_init):
        """Test cleanup releases resources."""
        from src.vietocr_wrapper import VietOCRWrapper

        wrapper = VietOCRWrapper(model_name="vgg_transformer", device="cpu")
        wrapper._predictor = MagicMock()
        wrapper._config = {"key": "value"}

        wrapper.cleanup()

        assert wrapper._predictor is None
        assert wrapper._config is None

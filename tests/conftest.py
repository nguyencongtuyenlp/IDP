"""
Shared test fixtures for Document OCR Extractor.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a simple test image (BGR, 200x300)."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # Add some white text-like rectangles
    img[50:70, 30:150] = 255
    img[90:110, 30:200] = 255
    img[130:150, 30:180] = 255
    return img


@pytest.fixture
def large_image():
    """Create a large image that should be resized."""
    return np.zeros((3000, 4000, 3), dtype=np.uint8)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory, cleaned up after test."""
    tmpdir = tempfile.mkdtemp(prefix="ocr_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_ocr_results():
    """Create mock OCRResult objects for testing."""
    from src.ocr_engine import OCRResult

    return [
        OCRResult(
            bbox=[[30, 50], [150, 50], [150, 70], [30, 70]],
            text="Hello World",
            confidence=0.95,
        ),
        OCRResult(
            bbox=[[30, 90], [200, 90], [200, 110], [30, 110]],
            text="Second line of text",
            confidence=0.88,
        ),
        OCRResult(
            bbox=[[30, 130], [180, 130], [180, 150], [30, 150]],
            text="Third line here",
            confidence=0.92,
        ),
    ]


@pytest.fixture
def test_image_path():
    """Path to test image (tho.jpg)."""
    path = Path("data/input/tho.jpg")
    if not path.exists():
        pytest.skip("Test image tho.jpg not found in data/input/")
    return path

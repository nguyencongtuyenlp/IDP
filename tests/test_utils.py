"""
Tests for src/utils.py — Logging, timing, file helpers.
"""

import logging
import tempfile
import shutil
from pathlib import Path

import pytest

from src.utils import (
    format_time,
    format_size,
    get_image_files,
    get_supported_files,
    ensure_dir,
    get_logger,
    setup_logging,
    IMAGE_EXTENSIONS,
)


# ============================================================================
# format_time
# ============================================================================


class TestFormatTime:
    def test_milliseconds(self):
        assert format_time(0.5) == "500ms"

    def test_milliseconds_small(self):
        assert format_time(0.023) == "23ms"

    def test_seconds(self):
        assert format_time(3.5) == "3.5s"

    def test_seconds_boundary(self):
        assert format_time(1.0) == "1.0s"

    def test_minutes(self):
        result = format_time(125)
        assert result == "2m5s"

    def test_zero(self):
        assert format_time(0) == "0ms"


# ============================================================================
# format_size
# ============================================================================


class TestFormatSize:
    def test_bytes(self):
        assert format_size(512) == "512.0B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0KB"

    def test_megabytes(self):
        assert format_size(5 * 1024 * 1024) == "5.0MB"

    def test_gigabytes(self):
        assert format_size(2 * 1024 ** 3) == "2.0GB"


# ============================================================================
# File helpers
# ============================================================================


class TestFileHelpers:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ocr_test_"))
        # Create test files
        (self.tmpdir / "photo.jpg").touch()
        (self.tmpdir / "scan.png").touch()
        (self.tmpdir / "doc.pdf").touch()
        (self.tmpdir / "notes.txt").touch()
        (self.tmpdir / "image.bmp").touch()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_image_files(self):
        images = get_image_files(self.tmpdir)
        names = [f.name for f in images]
        assert "photo.jpg" in names
        assert "scan.png" in names
        assert "image.bmp" in names
        assert "doc.pdf" not in names
        assert "notes.txt" not in names

    def test_get_image_files_sorted(self):
        images = get_image_files(self.tmpdir)
        assert images == sorted(images)

    def test_get_supported_files(self):
        files = get_supported_files(self.tmpdir)
        names = [f.name for f in files]
        assert "photo.jpg" in names
        assert "doc.pdf" in names
        assert "notes.txt" not in names

    def test_get_supported_files_count(self):
        files = get_supported_files(self.tmpdir)
        assert len(files) == 4  # jpg, png, bmp, pdf

    def test_ensure_dir_creates(self):
        new_dir = self.tmpdir / "sub" / "dir"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_dir_existing(self):
        result = ensure_dir(self.tmpdir)
        assert self.tmpdir.exists()
        assert result == self.tmpdir


# ============================================================================
# Logging
# ============================================================================


class TestLogging:
    def test_get_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert "docx." in logger.name

    def test_setup_logging(self):
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_image_extensions(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".pdf" not in IMAGE_EXTENSIONS

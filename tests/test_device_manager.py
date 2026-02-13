"""
Tests for src/device_manager.py — Device detection + quality presets.
"""

import pytest

from src.device_manager import (
    DeviceManager,
    DeviceInfo,
    QualityPreset,
    QUALITY_PRESETS,
)


# ============================================================================
# Quality Presets
# ============================================================================


class TestQualityPresets:
    def test_all_presets_exist(self):
        assert "fast" in QUALITY_PRESETS
        assert "balanced" in QUALITY_PRESETS
        assert "accurate" in QUALITY_PRESETS

    def test_fast_preset(self):
        p = QUALITY_PRESETS["fast"]
        assert p.max_image_size == 960
        assert p.use_angle_cls is False
        assert p.rec_batch_num == 16

    def test_balanced_preset(self):
        p = QUALITY_PRESETS["balanced"]
        assert p.max_image_size == 1280
        assert p.use_angle_cls is True
        assert p.rec_batch_num == 8

    def test_accurate_preset(self):
        p = QUALITY_PRESETS["accurate"]
        assert p.max_image_size == 1920
        assert p.use_angle_cls is True
        assert p.rec_batch_num == 4

    def test_accurate_has_lowest_thresholds(self):
        """Accurate mode should have lower thresholds = more sensitive."""
        assert QUALITY_PRESETS["accurate"].det_db_thresh < QUALITY_PRESETS["fast"].det_db_thresh

    def test_fast_has_highest_batch(self):
        """Fast mode should have highest batch for speed."""
        assert QUALITY_PRESETS["fast"].rec_batch_num > QUALITY_PRESETS["accurate"].rec_batch_num


# ============================================================================
# DeviceInfo
# ============================================================================


class TestDeviceInfo:
    def test_cpu_device(self):
        info = DeviceInfo(device="cpu")
        assert info.device == "cpu"
        assert info.gpu_name is None
        assert info.gpu_memory_mb is None

    def test_cuda_device(self):
        info = DeviceInfo(device="cuda", gpu_name="GTX 1650", gpu_memory_mb=4096)
        assert info.device == "cuda"
        assert info.gpu_name == "GTX 1650"


# ============================================================================
# DeviceManager
# ============================================================================


class TestDeviceManager:
    def test_cpu_mode(self):
        dm = DeviceManager(device="cpu", mode="balanced")
        assert dm.device_info.device == "cpu"
        assert dm.use_gpu is False

    def test_default_mode(self):
        dm = DeviceManager(device="cpu", mode="balanced")
        assert dm.preset.name == "balanced"

    def test_fast_mode(self):
        dm = DeviceManager(device="cpu", mode="fast")
        assert dm.preset.name == "fast"

    def test_invalid_mode_fallback(self):
        dm = DeviceManager(device="cpu", mode="nonexistent")
        assert dm.preset.name == "balanced"  # fallback

    def test_get_paddle_kwargs(self):
        dm = DeviceManager(device="cpu", mode="balanced")
        kwargs = dm.get_paddle_kwargs()

        assert "use_gpu" in kwargs
        assert "use_angle_cls" in kwargs
        assert "det_db_thresh" in kwargs
        assert "det_db_box_thresh" in kwargs
        assert "rec_batch_num" in kwargs
        assert kwargs["use_gpu"] is False

    def test_get_paddle_kwargs_values(self):
        dm = DeviceManager(device="cpu", mode="fast")
        kwargs = dm.get_paddle_kwargs()
        assert kwargs["use_angle_cls"] is False
        assert kwargs["rec_batch_num"] == 16

    def test_ram_info(self):
        dm = DeviceManager(device="cpu")
        assert dm.device_info.ram_total_mb > 0
        assert dm.device_info.ram_available_mb > 0

    def test_log_status(self):
        dm = DeviceManager(device="cpu")
        dm.log_status("Test")  # Should not raise

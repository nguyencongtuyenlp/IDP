"""
Device Manager — GPU/CPU Detection + Fallback.

Supports 3 device modes:
    - auto:  Use GPU if available, fallback to CPU
    - cuda:  Force GPU (raises error if unavailable)
    - cpu:   Force CPU

Supports 3 quality presets:
    - fast:     Smaller models, lower resolution → max speed
    - balanced: Default models, standard resolution
    - accurate: Larger models, higher resolution → best quality
"""

import platform
import psutil
from dataclasses import dataclass
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    """Current device configuration."""
    device: str  # "cuda" or "cpu"
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    cpu_name: str = ""
    ram_total_mb: int = 0
    ram_available_mb: int = 0


@dataclass
class QualityPreset:
    """Quality preset configuration for OCR."""
    name: str
    max_image_size: int        # Max dimension for resize
    det_db_thresh: float       # Text detection threshold
    det_db_box_thresh: float   # Box score threshold
    use_angle_cls: bool        # Enable angle classification
    rec_batch_num: int         # Recognition batch size


# Presets optimized for different speed/quality tradeoffs
QUALITY_PRESETS = {
    "fast": QualityPreset(
        name="fast",
        max_image_size=960,
        det_db_thresh=0.3,
        det_db_box_thresh=0.6,
        use_angle_cls=False,
        rec_batch_num=16,
    ),
    "balanced": QualityPreset(
        name="balanced",
        max_image_size=1280,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        use_angle_cls=True,
        rec_batch_num=8,
    ),
    "accurate": QualityPreset(
        name="accurate",
        max_image_size=1920,
        det_db_thresh=0.2,
        det_db_box_thresh=0.4,
        use_angle_cls=True,
        rec_batch_num=4,
    ),
}


class DeviceManager:
    """Manage device selection and quality presets.

    Example:
        >>> dm = DeviceManager(device="auto", mode="balanced")
        >>> print(dm.device_info.device)  # "cuda" or "cpu"
        >>> print(dm.preset.max_image_size)  # 1280
    """

    def __init__(self, device: str = "auto", mode: str = "balanced") -> None:
        self.requested_device = device.lower()
        self.mode = mode.lower()
        self.device_info = self._detect_device()
        self.preset = QUALITY_PRESETS.get(self.mode, QUALITY_PRESETS["balanced"])

        logger.info("=" * 55)
        logger.info("🖥️  Device: %s | Mode: %s", self.device_info.device.upper(), self.preset.name)
        if self.device_info.gpu_name:
            logger.info("🎮 GPU: %s (%d MB)", self.device_info.gpu_name, self.device_info.gpu_memory_mb or 0)
        logger.info("💾 RAM: %d MB available / %d MB total",
                     self.device_info.ram_available_mb, self.device_info.ram_total_mb)
        logger.info("=" * 55)

    def _detect_device(self) -> DeviceInfo:
        """Detect available hardware and select device."""
        mem = psutil.virtual_memory()
        ram_total = int(mem.total / 1024 / 1024)
        ram_avail = int(mem.available / 1024 / 1024)
        cpu_name = platform.processor() or "Unknown CPU"

        info = DeviceInfo(
            device="cpu",
            cpu_name=cpu_name,
            ram_total_mb=ram_total,
            ram_available_mb=ram_avail,
        )

        if self.requested_device == "cpu":
            logger.info("📌 CPU mode forced by user")
            return info

        # Check GPU
        try:
            import paddle
            gpu_available = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            if gpu_available:
                info.device = "cuda"
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info.gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(info.gpu_name, bytes):
                        info.gpu_name = info.gpu_name.decode()
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info.gpu_memory_mb = int(mem_info.total / 1024 / 1024)
                    pynvml.nvmlShutdown()
                except Exception:
                    info.gpu_name = "CUDA GPU"
                    info.gpu_memory_mb = 0
            elif self.requested_device == "cuda":
                logger.warning("⚠️  CUDA requested but not available, falling back to CPU")
        except Exception:
            if self.requested_device == "cuda":
                logger.warning("⚠️  PaddlePaddle not compiled with CUDA, using CPU")

        return info

    @property
    def use_gpu(self) -> bool:
        return self.device_info.device == "cuda"

    def get_paddle_kwargs(self) -> dict:
        """Get PaddleOCR constructor kwargs for current device/mode."""
        return {
            "use_gpu": self.use_gpu,
            "use_angle_cls": self.preset.use_angle_cls,
            "det_db_thresh": self.preset.det_db_thresh,
            "det_db_box_thresh": self.preset.det_db_box_thresh,
            "rec_batch_num": self.preset.rec_batch_num,
        }

    def log_status(self, label: str = "Current") -> None:
        """Log current memory usage."""
        mem = psutil.virtual_memory()
        logger.info(
            "[%s] 💾 RAM: %d/%d MB (%.1f%%)",
            label,
            int(mem.used / 1024 / 1024),
            int(mem.total / 1024 / 1024),
            mem.percent,
        )

"""
Utility functions — Logging, timing, file helpers.
"""

import functools
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# ============================================================================
# LOGGING
# ============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger with consistent format."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger."""
    return logging.getLogger(f"docx.{name.split('.')[-1]}")


# ============================================================================
# TIMING DECORATOR
# ============================================================================


def timer(func):
    """Decorator to log execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("profiling")
        logger.info("⏱️  Starting: %s", func.__qualname__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("✅ Completed: %s in %.3fs", func.__qualname__, elapsed)
        return result
    return wrapper


# ============================================================================
# FILE HELPERS
# ============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files in a directory (sorted)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_supported_files(directory: Path) -> List[Path]:
    """Get all supported files (images + PDF) in a directory."""
    supported = IMAGE_EXTENSIONS | PDF_EXTENSIONS
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in supported
    )


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """Human-readable time formatting."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"


def format_size(bytes_val: int) -> str:
    """Human-readable size formatting."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"

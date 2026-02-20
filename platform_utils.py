"""
Platform detection and GPU configuration utilities.

Provides cross-platform support for:
- Windows: CUDA GPU acceleration (if available)
- macOS: Apple Silicon MPS acceleration (M1/M2/M3) or CPU fallback
- Linux: CUDA or CPU fallback
"""

import os
import platform
import sys
import logging
from typing import TYPE_CHECKING, Any, Optional

# Lazy import for torch to avoid heavy import at module level
if TYPE_CHECKING:
    pass

_torch: Optional[Any] = None

# Inherit logger from the main application (configured in server.py)
logger = logging.getLogger("plesk_unified")


def _get_torch() -> Any:
    """Lazy load torch to avoid import overhead."""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            logger.warning(
                "PyTorch import failed. GPU acceleration will be unavailable."
            )
            raise
    return _torch


def get_platform_info() -> dict:
    """
    Returns a dictionary with detailed platform information.
    """
    info: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }

    # Try to get PyTorch info
    try:
        torch = _get_torch()
        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = torch.cuda.is_available()
        # Check MPS availability on macOS
        info["mps_available"] = (
            torch.backends.mps.is_available()
            if platform.system() == "Darwin" and hasattr(torch.backends, "mps")
            else False
        )

        if torch.cuda.is_available():
            info["cuda_version"] = str(torch.version.cuda)
            info["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info["gpu_name"] = str(torch.cuda.get_device_name(0))

    except ImportError:
        info["torch_available"] = False
    except Exception as e:
        info["torch_error"] = str(e)
        logger.debug("Error gathering detailed platform info: %s", e)

    return info


def get_optimal_device() -> str:
    """
    Determine the optimal compute device based on platform and hardware.
    Priority: Environment Variable -> MPS (macOS) -> CUDA (Win/Linux) -> CPU
    """
    # Check for forced device via environment variable
    forced_device = os.environ.get("FORCE_DEVICE", "").lower().strip()
    if forced_device in ("cuda", "mps", "cpu"):
        logger.info("Device forced via env var: %s", forced_device)
        return forced_device

    system = platform.system()

    # macOS: Check for Apple Silicon MPS
    if system == "Darwin":
        try:
            torch = _get_torch()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # Windows/Linux: Check for CUDA
    if system in ("Windows", "Linux"):
        try:
            torch = _get_torch()
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass

    return "cpu"


def get_device_config() -> dict:
    """
    Get comprehensive device configuration for model initialization.
    """
    device = get_optimal_device()
    config: dict[str, Any] = {
        "device": device,
        "precision": "float32",
    }

    try:
        if device == "cuda":
            torch = _get_torch()
            if torch.cuda.is_available():
                config["precision"] = "float16"
                config["torch_dtype"] = "float16"
        elif device == "mps":
            config["precision"] = "float16"
            config["torch_dtype"] = "float16"
    except Exception as e:
        logger.debug("Failed to set precision config: %s", e)

    return config


def log_platform_info() -> None:
    """Logs platform information to the shared logger."""
    info = get_platform_info()
    device = get_optimal_device()

    # Construct a concise summary for the log
    summary = [
        f"OS: {info.get('system')} {info.get('machine')}",
        f"Python: {info.get('python_version')}",
        f"Device: {device.upper()}",
    ]

    if info.get("cuda_available"):
        summary.append(
            f"GPU: {info.get('gpu_name', 'Unknown')} (CUDA {info.get('cuda_version')})"
        )
    elif info.get("mps_available"):
        summary.append("GPU: Apple Silicon (MPS)")

    logger.info("Platform Check: " + " | ".join(summary))


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def get_model_cache_dir() -> str:
    """
    Get the appropriate model cache directory for the platform.
    """
    import tempfile

    if is_windows():
        base = os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
        return os.path.join(base, "huggingface", "hub")
    elif is_macos():
        return os.path.expanduser("~/Library/Caches/huggingface/hub")
    else:
        return os.path.expanduser("~/.cache/huggingface/hub")

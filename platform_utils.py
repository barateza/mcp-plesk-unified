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
from typing import TYPE_CHECKING, Any, Optional

# Lazy import for torch to avoid heavy import at module level
# For type checking, we use TYPE_CHECKING to avoid runtime import
if TYPE_CHECKING:
    pass

_torch: Optional[Any] = None


def _get_torch() -> Any:
    """Lazy load torch to avoid import overhead."""
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def get_platform_info() -> dict:
    """
    Returns a dictionary with detailed platform information.

    Returns:
        dict: Platform details including OS, machine, Python version,
              and available compute devices.
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
        info["torch_version"] = str(torch.__version__)  # Ensure string type
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = (
            torch.backends.mps.is_available()
            if platform.system() == "Darwin"
            else False
        )

        if torch.cuda.is_available():
            info["cuda_version"] = str(torch.version.cuda)  # Ensure string type
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = (
                str(torch.cuda.get_device_name(0))
                if torch.cuda.device_count() > 0
                else "Unknown"
            )
    except ImportError:
        info["torch_available"] = False
    except Exception as e:
        info["torch_error"] = str(e)

    return info


def get_optimal_device() -> str:
    """
    Determine the optimal compute device based on platform and hardware.

    Priority order:
    1. macOS with Apple Silicon (M1/M2/M3) -> MPS
    2. Windows/Linux with NVIDIA GPU -> CUDA
    3. Otherwise -> CPU

    Returns:
        str: Device identifier ("cuda", "mps", or "cpu")
    """
    # Check for forced device via environment variable
    forced_device = os.environ.get("FORCE_DEVICE", "").lower().strip()
    if forced_device in ("cuda", "mps", "cpu"):
        return forced_device

    system = platform.system()

    # macOS: Check for Apple Silicon MPS
    if system == "Darwin":
        try:
            torch = _get_torch()
            if torch.backends.mps.is_available():
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

    # Fallback to CPU
    return "cpu"


def get_device_config() -> dict:
    """
    Get comprehensive device configuration for model initialization.

    Returns:
        dict: Configuration options for embedding/reranking models
              including device, precision, and optimization settings.
    """
    device = get_optimal_device()
    config: dict[str, Any] = {
        "device": device,
        "precision": "float32",
    }

    # Add CUDA-specific settings
    if device == "cuda":
        try:
            torch = _get_torch()
            # Check if half precision is available
            if torch.cuda.is_available():
                config["precision"] = "float16"  # Faster inference on modern GPUs
                config["torch_dtype"] = "float16"
        except Exception:
            pass

    # MPS supports float16 on Apple Silicon
    elif device == "mps":
        config["precision"] = "float16"
        config["torch_dtype"] = "float16"

    return config


def print_platform_info() -> None:
    """Print formatted platform information to stderr."""
    info = get_platform_info()

    print("=" * 50, file=sys.stderr)
    print("  Platform Information", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"  OS: {info.get('system')} {info.get('machine')}", file=sys.stderr)
    print(f"  Python: {info.get('python_version')}", file=sys.stderr)

    if "torch_version" in info:
        print(f"  PyTorch: {info.get('torch_version')}", file=sys.stderr)
        print(f"  CUDA Available: {info.get('cuda_available', False)}", file=sys.stderr)

        if info.get("cuda_available"):
            print(
                f"  CUDA Version: {info.get('cuda_version', 'unknown')}",
                file=sys.stderr,
            )
            print(f"  GPU Count: {info.get('gpu_count', 0)}", file=sys.stderr)
            if info.get("gpu_name"):
                print(f"  GPU Name: {info.get('gpu_name')}", file=sys.stderr)

        if info.get("mps_available"):
            print("  MPS (Apple Silicon): Available", file=sys.stderr)

    device = get_optimal_device()
    print(f"\n  Selected Device: {device.upper()}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


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

    Returns:
        str: Path to the model cache directory
    """
    import tempfile

    # Use platform-specific default locations
    if is_windows():
        base = os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
        return os.path.join(base, "huggingface", "hub")
    elif is_macos():
        return os.path.expanduser("~/Library/Caches/huggingface/hub")
    else:
        # Linux and others
        return os.path.expanduser("~/.cache/huggingface/hub")

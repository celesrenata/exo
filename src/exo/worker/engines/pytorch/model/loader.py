"""HuggingFace safetensors model loading.

Loads model weights from safetensors files into PyTorch tensors and moves
them to the target GPU device. Accepts model specifications in upstream's
BoundInstance format and reports loading progress.

Requirements: 10.1, 10.2, 10.3, 10.5, 10.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when model loading fails with a descriptive error message.

    Includes context about what went wrong — memory constraints, missing
    files, or format issues — so the caller can report the failure clearly.
    """


@dataclass(frozen=True)
class LoadProgress:
    """Progress report for model loading.

    Attributes:
        loaded_bytes: Number of bytes loaded so far.
        total_bytes: Total bytes to load across all safetensors files.
        percent: Percentage complete (0.0 to 100.0).
        current_file: Name of the safetensors file currently being loaded.
        files_loaded: Number of files fully loaded so far.
        total_files: Total number of safetensors files to load.
    """

    loaded_bytes: int
    total_bytes: int
    percent: float
    current_file: str
    files_loaded: int
    total_files: int


def _discover_safetensors_files(model_dir: Path) -> list[Path]:
    """Find all .safetensors files in a model directory.

    Args:
        model_dir: Path to the directory containing model files.

    Returns:
        Sorted list of safetensors file paths.

    Raises:
        ModelLoadError: If no safetensors files are found.
    """
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ModelLoadError(
            f"No .safetensors files found in {model_dir}. "
            f"Ensure the model has been downloaded in safetensors format."
        )
    return files


def _get_available_memory(device_type: Literal["cuda", "xpu"], device_index: int) -> int:
    """Query available GPU memory in bytes.

    For shared-memory architectures (Intel iGPU), uses system RAM availability.
    For discrete GPUs, uses the device's free memory.

    Args:
        device_type: Either "cuda" or "xpu".
        device_index: Device index.

    Returns:
        Available memory in bytes.
    """
    try:
        import torch
    except ImportError:
        return 0

    if device_type == "cuda":
        torch.cuda.set_device(device_index)
        free, _total = torch.cuda.mem_get_info(device_index)
        return free
    elif device_type == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            # For XPU shared-memory devices, use system RAM as proxy
            try:
                import psutil

                mem = psutil.virtual_memory()
                return mem.available
            except ImportError:
                # Fallback: try torch.xpu memory stats
                try:
                    props = torch.xpu.get_device_properties(device_index)
                    return props.total_memory
                except Exception:
                    return 0
        return 0
    return 0


def _estimate_model_size(safetensors_files: list[Path]) -> int:
    """Estimate total model size from safetensors file sizes on disk.

    This is a conservative estimate — the actual GPU memory usage may be
    slightly higher due to alignment and metadata, but safetensors files
    store raw tensor data with minimal overhead.

    Args:
        safetensors_files: List of safetensors file paths.

    Returns:
        Total size in bytes.
    """
    return sum(f.stat().st_size for f in safetensors_files)


def _check_memory_capacity(
    model_size_bytes: int,
    device_type: Literal["cuda", "xpu"],
    device_index: int,
) -> None:
    """Verify the model fits in available GPU memory.

    Raises ModelLoadError with a descriptive message if the model exceeds
    available capacity.

    Args:
        model_size_bytes: Estimated model size in bytes.
        device_type: Target device type.
        device_index: Target device index.

    Raises:
        ModelLoadError: If model exceeds available GPU memory.
    """
    available = _get_available_memory(device_type, device_index)
    if available <= 0:
        # Cannot determine available memory — proceed optimistically
        logger.warning(
            "Could not determine available %s:%d memory, proceeding without check",
            device_type,
            device_index,
        )
        return

    if model_size_bytes > available:
        model_gib = model_size_bytes / (1024**3)
        available_gib = available / (1024**3)
        raise ModelLoadError(
            f"Model requires approximately {model_gib:.2f} GiB but only "
            f"{available_gib:.2f} GiB is available on {device_type}:{device_index}. "
            f"Consider using pipeline parallelism to shard the model across "
            f"multiple nodes, or free memory on the target device."
        )

    logger.info(
        "Memory check passed: model ~%.2f GiB, available %.2f GiB on %s:%d",
        model_size_bytes / (1024**3),
        available / (1024**3),
        device_type,
        device_index,
    )


def load_safetensors_to_device(
    model_dir: Path,
    device_type: Literal["cuda", "xpu"],
    device_index: int = 0,
) -> Generator[LoadProgress | dict[str, "torch.Tensor"], None, None]:
    """Load all safetensors files from a directory onto a GPU device.

    This is a generator that yields LoadProgress reports as files are loaded,
    and yields the final state_dict as the last value. The caller should
    iterate through progress reports and collect the final dict.

    The generator protocol:
    - Yields LoadProgress instances during loading (for progress reporting).
    - The final yield is the complete state_dict (dict[str, torch.Tensor]).

    Args:
        model_dir: Path to directory containing .safetensors files.
        device_type: Target device type ("cuda" or "xpu").
        device_index: Target device index (default 0).

    Yields:
        LoadProgress during loading, then the final state_dict.

    Raises:
        ModelLoadError: If files are missing, memory is insufficient, or
            loading fails for any reason.
    """
    import torch
    from safetensors.torch import load_file

    # Discover safetensors files
    safetensors_files = _discover_safetensors_files(model_dir)
    total_files = len(safetensors_files)

    logger.info(
        "Found %d safetensors file(s) in %s",
        total_files,
        model_dir,
    )

    # Estimate model size and check memory
    total_bytes = _estimate_model_size(safetensors_files)
    _check_memory_capacity(total_bytes, device_type, device_index)

    # Load tensors file by file
    device_str = f"{device_type}:{device_index}"
    state_dict: dict[str, torch.Tensor] = {}
    loaded_bytes = 0

    for file_idx, safetensors_path in enumerate(safetensors_files):
        file_name = safetensors_path.name
        file_size = safetensors_path.stat().st_size

        logger.debug("Loading %s (%d/%d)", file_name, file_idx + 1, total_files)

        # Yield progress before loading this file
        percent = (loaded_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
        yield LoadProgress(
            loaded_bytes=loaded_bytes,
            total_bytes=total_bytes,
            percent=percent,
            current_file=file_name,
            files_loaded=file_idx,
            total_files=total_files,
        )

        try:
            # Load tensors directly to the target device
            file_tensors = load_file(str(safetensors_path), device=device_str)
            state_dict.update(file_tensors)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load {file_name}: {e}. "
                f"The file may be corrupted or in an unsupported format."
            ) from e

        loaded_bytes += file_size

    # Final progress report (100%)
    yield LoadProgress(
        loaded_bytes=total_bytes,
        total_bytes=total_bytes,
        percent=100.0,
        current_file=safetensors_files[-1].name,
        files_loaded=total_files,
        total_files=total_files,
    )

    # Yield the complete state dict as the final value
    logger.info(
        "Model loading complete: %d tensors, %.2f GiB on %s",
        len(state_dict),
        total_bytes / (1024**3),
        device_str,
    )
    yield state_dict


def load_model_from_bound_instance(
    model_dir: Path,
    device_type: Literal["cuda", "xpu"],
    device_index: int = 0,
) -> Generator[LoadProgress | dict[str, "torch.Tensor"], None, None]:
    """Load model weights accepting specifications in BoundInstance format.

    This is the primary entry point for model loading in the unified PyTorch
    engine. It accepts a model directory path (derived from the BoundInstance's
    shard_metadata.model_card.model_id) and loads all safetensors weights onto
    the target device.

    The model_dir is typically resolved from:
        BoundInstance.bound_shard.model_card.model_id → build_model_path(model_id)

    Args:
        model_dir: Path to the model directory (contains .safetensors files).
            Derived from the BoundInstance's model_card.model_id via
            build_model_path().
        device_type: Target device type ("cuda" or "xpu").
        device_index: Target device index (default 0).

    Yields:
        LoadProgress during loading, then the final state_dict.

    Raises:
        ModelLoadError: If the model directory doesn't exist, contains no
            safetensors files, exceeds available memory, or loading fails.
    """
    if not model_dir.exists():
        raise ModelLoadError(
            f"Model directory does not exist: {model_dir}. "
            f"Ensure the model has been downloaded before loading."
        )

    if not model_dir.is_dir():
        raise ModelLoadError(
            f"Model path is not a directory: {model_dir}. "
            f"Expected a directory containing .safetensors files."
        )

    yield from load_safetensors_to_device(model_dir, device_type, device_index)


# Type import for annotations only
import typing

if typing.TYPE_CHECKING:
    import torch

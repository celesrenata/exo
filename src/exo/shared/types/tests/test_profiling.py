"""Unit tests for GpuMemoryInfo and extended MemoryUsage with gpu_info field."""

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import GpuMemoryInfo, MemoryUsage


def _make_gpu_info(
    device_type: str = "cuda",
    memory_architecture: str = "Discrete",
    total_bytes: int = 12 * 1024**3,
    available_bytes: int = 10 * 1024**3,
) -> GpuMemoryInfo:
    """Helper to construct a GpuMemoryInfo instance."""
    return GpuMemoryInfo(
        device_type=device_type,
        memory_architecture=memory_architecture,
        gpu_total_memory=Memory.from_bytes(total_bytes),
        gpu_available_memory=Memory.from_bytes(available_bytes),
    )


def _make_memory_usage(
    gpu_info: GpuMemoryInfo | None = None,
) -> MemoryUsage:
    """Helper to construct a MemoryUsage with fixed RAM/swap values."""
    return MemoryUsage(
        ram_total=Memory.from_bytes(32 * 1024**3),
        ram_available=Memory.from_bytes(24 * 1024**3),
        swap_total=Memory.from_bytes(8 * 1024**3),
        swap_available=Memory.from_bytes(6 * 1024**3),
        gpu_info=gpu_info,
    )


# --- Construction tests ---


def test_memory_usage_without_gpu_info() -> None:
    """MemoryUsage can be constructed with gpu_info=None (backward compat)."""
    usage = _make_memory_usage(gpu_info=None)
    assert usage.gpu_info is None
    assert usage.ram_total.in_bytes == 32 * 1024**3


def test_memory_usage_with_gpu_info_cuda_discrete() -> None:
    """MemoryUsage can be constructed with a CUDA Discrete GpuMemoryInfo."""
    gpu = _make_gpu_info(device_type="cuda", memory_architecture="Discrete")
    usage = _make_memory_usage(gpu_info=gpu)

    assert usage.gpu_info is not None
    assert usage.gpu_info.device_type == "cuda"
    assert usage.gpu_info.memory_architecture == "Discrete"
    assert usage.gpu_info.gpu_total_memory.in_bytes == 12 * 1024**3
    assert usage.gpu_info.gpu_available_memory.in_bytes == 10 * 1024**3


def test_memory_usage_with_gpu_info_xpu_shared() -> None:
    """MemoryUsage can be constructed with an XPU Shared GpuMemoryInfo."""
    gpu = _make_gpu_info(
        device_type="xpu",
        memory_architecture="Shared",
        total_bytes=32 * 1024**3,
        available_bytes=28 * 1024**3,
    )
    usage = _make_memory_usage(gpu_info=gpu)

    assert usage.gpu_info is not None
    assert usage.gpu_info.device_type == "xpu"
    assert usage.gpu_info.memory_architecture == "Shared"


def test_memory_usage_with_gpu_info_cpu() -> None:
    """MemoryUsage can be constructed with device_type='cpu'."""
    gpu = _make_gpu_info(device_type="cpu", memory_architecture="Discrete")
    usage = _make_memory_usage(gpu_info=gpu)

    assert usage.gpu_info is not None
    assert usage.gpu_info.device_type == "cpu"


# --- Serialization round-trip tests ---


def test_serialization_roundtrip_without_gpu_info() -> None:
    """model_dump_json → model_validate_json preserves MemoryUsage when gpu_info is None."""
    usage = _make_memory_usage(gpu_info=None)
    json_str = usage.model_dump_json()
    restored = MemoryUsage.model_validate_json(json_str)

    assert restored.gpu_info is None
    assert restored.ram_total.in_bytes == usage.ram_total.in_bytes
    assert restored.ram_available.in_bytes == usage.ram_available.in_bytes
    assert restored.swap_total.in_bytes == usage.swap_total.in_bytes
    assert restored.swap_available.in_bytes == usage.swap_available.in_bytes


def test_serialization_roundtrip_with_gpu_info() -> None:
    """model_dump_json → model_validate_json preserves MemoryUsage with gpu_info present."""
    gpu = _make_gpu_info(device_type="xpu", memory_architecture="Shared")
    usage = _make_memory_usage(gpu_info=gpu)
    json_str = usage.model_dump_json()
    restored = MemoryUsage.model_validate_json(json_str)

    assert restored.gpu_info is not None
    assert restored.gpu_info.device_type == "xpu"
    assert restored.gpu_info.memory_architecture == "Shared"
    assert restored.gpu_info.gpu_total_memory.in_bytes == gpu.gpu_total_memory.in_bytes
    assert (
        restored.gpu_info.gpu_available_memory.in_bytes
        == gpu.gpu_available_memory.in_bytes
    )


def test_model_dump_uses_camel_case() -> None:
    """CamelCaseModel serializes field names to camelCase."""
    gpu = _make_gpu_info()
    usage = _make_memory_usage(gpu_info=gpu)
    dumped = usage.model_dump(by_alias=True)

    # Top-level fields should be camelCase
    assert "ramTotal" in dumped
    assert "ramAvailable" in dumped
    assert "swapTotal" in dumped
    assert "swapAvailable" in dumped
    assert "gpuInfo" in dumped

    # Nested GpuMemoryInfo fields should also be camelCase
    gpu_dict = dumped["gpuInfo"]
    assert gpu_dict is not None
    assert "deviceType" in gpu_dict
    assert "memoryArchitecture" in gpu_dict
    assert "gpuTotalMemory" in gpu_dict
    assert "gpuAvailableMemory" in gpu_dict


def test_model_dump_gpu_info_none_excluded() -> None:
    """When gpu_info is None, model_dump includes it as None (not omitted)."""
    usage = _make_memory_usage(gpu_info=None)
    dumped = usage.model_dump(by_alias=True)
    assert "gpuInfo" in dumped
    assert dumped["gpuInfo"] is None


# --- from_psutil test ---


def test_from_psutil_returns_none_gpu_info() -> None:
    """from_psutil creates MemoryUsage with gpu_info=None (no GPU detection in psutil path)."""
    usage = MemoryUsage.from_psutil(override_memory=None)
    assert usage.gpu_info is None
    assert usage.ram_total.in_bytes > 0
    assert usage.ram_available.in_bytes > 0


def test_from_psutil_with_override_memory() -> None:
    """from_psutil with override_memory sets ram_available and leaves gpu_info=None."""
    override = 4 * 1024**3
    usage = MemoryUsage.from_psutil(override_memory=override)
    assert usage.gpu_info is None
    assert usage.ram_available.in_bytes == override

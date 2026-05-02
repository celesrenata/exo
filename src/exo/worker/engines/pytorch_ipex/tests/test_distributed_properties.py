# Feature: distributed-gpu-sharding, Properties 1, 2, 3: Distributed communicator property tests
"""
Property-based tests for the distributed communicator module.

Uses Hypothesis to verify:
- Property 3: CPU tensor staging round-trip preserves dtype and shape
- Property 1: Process group initialization correctness
- Property 2: Communication failure reporting completeness

**Validates: Requirements 1.1, 1.3, 1.5, 2.1, 2.2, 2.3, 2.6, 10.2**
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_PATH = _THIS_DIR.parent / "distributed.py"


def _load_distributed() -> types.ModuleType:
    """Load distributed.py directly from file, avoiding __init__.py."""
    module_name = "distributed_props_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed()
ProcessGroupConfig = _mod.ProcessGroupConfig
CpuStagedTensor = _mod.CpuStagedTensor
stage_to_cpu = _mod.stage_to_cpu
unstage_from_cpu = _mod.unstage_from_cpu
init_process_group = _mod.init_process_group
send_activation = _mod.send_activation
recv_activation = _mod.recv_activation


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_torch_mocks() -> tuple[MagicMock, MagicMock]:
    """Create paired torch and torch.distributed mocks.

    Returns (mock_torch, mock_dist) where mock_torch.distributed IS mock_dist,
    ensuring that both `sys.modules["torch.distributed"]` and
    `sys.modules["torch"].distributed` resolve to the same object.
    """
    mock_dist = MagicMock()
    mock_torch = MagicMock()
    mock_torch.distributed = mock_dist
    return mock_torch, mock_dist


def _patch_torch(mock_torch: MagicMock, mock_dist: MagicMock):  # noqa: ANN201
    """Context manager to patch torch and torch.distributed in sys.modules."""
    return patch.dict(sys.modules, {
        "torch": mock_torch,
        "torch.distributed": mock_dist,
    })


# ---------------------------------------------------------------------------
# Mock tensor class for CPU staging tests
# ---------------------------------------------------------------------------


class MockDtype:
    """Mock for torch.dtype."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MockDtype):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


FLOAT16 = MockDtype("float16")
BFLOAT16 = MockDtype("bfloat16")
FLOAT32 = MockDtype("float32")

DTYPE_MAP = {
    "float16": FLOAT16,
    "bfloat16": BFLOAT16,
    "float32": FLOAT32,
}


class MockTensor:
    """Mock tensor that behaves like torch.Tensor for staging tests.

    Supports .to(device), .dtype, .shape properties needed by
    stage_to_cpu and unstage_from_cpu.
    """

    def __init__(self, shape: tuple[int, ...], dtype: MockDtype, device: str) -> None:
        self._shape = shape
        self._dtype = dtype
        self._device = device

    @property
    def dtype(self) -> MockDtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> str:
        return self._device

    def to(self, target: str) -> "MockTensor":
        """Move tensor to target device, preserving dtype and shape."""
        return MockTensor(shape=self._shape, dtype=self._dtype, device=target)

    def __repr__(self) -> str:
        return f"MockTensor(shape={self._shape}, dtype={self._dtype}, device={self._device})"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Tensor shapes: 1-4 dimensions, each dimension 1-64
_tensor_shapes = st.lists(
    st.integers(min_value=1, max_value=64),
    min_size=1,
    max_size=4,
).map(tuple)

# Supported dtypes
_dtype_names = st.sampled_from(["float16", "bfloat16", "float32"])

# Source devices (GPU devices that would be staged to CPU)
_source_devices = st.sampled_from(["cuda:0", "cuda:1", "xpu:0", "xpu:1"])

# Target devices for unstaging
_target_devices = st.sampled_from(["cuda:0", "cuda:1", "xpu:0", "xpu:1"])

# Valid (rank, world_size) pairs: 0 <= rank < world_size
_rank_world = st.integers(min_value=2, max_value=64).flatmap(
    lambda ws: st.tuples(st.integers(min_value=0, max_value=ws - 1), st.just(ws))
)

# Random IP addresses (valid IPv4)
_ip_addresses = st.tuples(
    st.integers(min_value=1, max_value=254),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=1, max_value=254),
).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

# Random ports (valid ephemeral range)
_ports = st.integers(min_value=1024, max_value=65535)


# ---------------------------------------------------------------------------
# Property 3: CPU tensor staging round-trip preserves dtype and shape
# ---------------------------------------------------------------------------


class TestCpuTensorStagingRoundTrip:
    """Property 3: CPU tensor staging round-trip preserves dtype and shape.

    For any tensor with arbitrary shape and dtype (float16, bfloat16, float32),
    staging to CPU via .to("cpu") and then unstaging to a target device SHALL
    produce a tensor with identical dtype and shape to the original. The staged
    CPU tensor SHALL always reside on the CPU device.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(
        shape=_tensor_shapes,
        dtype_name=_dtype_names,
        source_device=_source_devices,
        target_device=_target_devices,
    )
    @settings(max_examples=100)
    def test_staging_round_trip_preserves_dtype_and_shape(
        self,
        shape: tuple[int, ...],
        dtype_name: str,
        source_device: str,
        target_device: str,
    ) -> None:
        """unstage_from_cpu(stage_to_cpu(tensor), device) produces tensor with
        identical dtype and shape to the original.

        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        dtype = DTYPE_MAP[dtype_name]
        original = MockTensor(shape=shape, dtype=dtype, device=source_device)

        staged = stage_to_cpu(original)

        # Staged tensor must reside on CPU
        assert staged.cpu_tensor.device == "cpu", (
            f"Staged tensor should be on CPU, got {staged.cpu_tensor.device}"
        )

        # Staged metadata must match original
        assert staged.original_dtype == dtype
        assert staged.original_shape == shape

        # Round-trip: unstage back to target device
        result = unstage_from_cpu(staged, target_device)

        # Result must have identical dtype and shape
        assert result.dtype == dtype, (
            f"Expected dtype {dtype}, got {result.dtype}"
        )
        assert tuple(result.shape) == shape, (
            f"Expected shape {shape}, got {tuple(result.shape)}"
        )

    @given(
        shape=_tensor_shapes,
        dtype_name=_dtype_names,
        source_device=_source_devices,
    )
    @settings(max_examples=100)
    def test_staged_tensor_always_on_cpu(
        self,
        shape: tuple[int, ...],
        dtype_name: str,
        source_device: str,
    ) -> None:
        """The staged CPU tensor SHALL always reside on the CPU device.

        **Validates: Requirements 2.1, 2.2**
        """
        dtype = DTYPE_MAP[dtype_name]
        original = MockTensor(shape=shape, dtype=dtype, device=source_device)

        staged = stage_to_cpu(original)

        assert staged.cpu_tensor.device == "cpu", (
            f"Staged tensor must be on CPU regardless of source device "
            f"({source_device}), got {staged.cpu_tensor.device}"
        )

    @given(
        shape=_tensor_shapes,
        dtype_name=_dtype_names,
        source_device=_source_devices,
    )
    @settings(max_examples=100)
    def test_staged_metadata_matches_original(
        self,
        shape: tuple[int, ...],
        dtype_name: str,
        source_device: str,
    ) -> None:
        """CpuStagedTensor preserves original_dtype and original_shape metadata.

        **Validates: Requirements 2.2, 2.3**
        """
        dtype = DTYPE_MAP[dtype_name]
        original = MockTensor(shape=shape, dtype=dtype, device=source_device)

        staged = stage_to_cpu(original)

        assert staged.original_dtype is dtype
        assert staged.original_shape == shape


# ---------------------------------------------------------------------------
# Property 1: Process group initialization correctness
# ---------------------------------------------------------------------------


class TestProcessGroupInitCorrectness:
    """Property 1: Process group initialization correctness.

    For any valid (rank, world_size) pair where 0 <= rank < world_size and any
    valid PyTorchIPEXRingInstance with hosts_by_node containing ethernet IPs,
    initializing the process group SHALL always use backend="gloo", the correct
    rank and world_size, and init_method="env://".

    **Validates: Requirements 1.1, 1.3**
    """

    @given(
        rank_world=_rank_world,
        ip_addr=_ip_addresses,
        port=_ports,
    )
    @settings(max_examples=100)
    def test_init_calls_gloo_with_correct_params(
        self,
        rank_world: tuple[int, int],
        ip_addr: str,
        port: int,
    ) -> None:
        """init_process_group SHALL call torch.distributed.init_process_group
        with backend="gloo", correct rank, world_size, and init_method="env://".

        **Validates: Requirements 1.1, 1.3**
        """
        rank, world_size = rank_world

        config = ProcessGroupConfig(
            rank=rank,
            world_size=world_size,
            master_addr=ip_addr,
            master_port=port,
        )

        mock_torch, mock_dist = _make_torch_mocks()

        with _patch_torch(mock_torch, mock_dist):
            init_process_group(config)

        # Verify init_process_group was called with correct parameters
        mock_dist.init_process_group.assert_called_once()
        kw = mock_dist.init_process_group.call_args.kwargs
        assert kw["backend"] == "gloo"
        assert kw["rank"] == rank
        assert kw["world_size"] == world_size
        assert kw["init_method"] == "env://"

    @given(
        rank_world=_rank_world,
        ip_addr=_ip_addresses,
        port=_ports,
    )
    @settings(max_examples=100)
    def test_env_vars_set_correctly(
        self,
        rank_world: tuple[int, int],
        ip_addr: str,
        port: int,
    ) -> None:
        """MASTER_ADDR and MASTER_PORT environment variables SHALL be set
        correctly before init_process_group is called.

        **Validates: Requirements 1.1, 1.3**
        """
        rank, world_size = rank_world

        config = ProcessGroupConfig(
            rank=rank,
            world_size=world_size,
            master_addr=ip_addr,
            master_port=port,
        )

        mock_torch, mock_dist = _make_torch_mocks()

        orig_addr = os.environ.get("MASTER_ADDR")
        orig_port = os.environ.get("MASTER_PORT")

        try:
            with _patch_torch(mock_torch, mock_dist):
                init_process_group(config)

            assert os.environ.get("MASTER_ADDR") == ip_addr, (
                f"Expected MASTER_ADDR={ip_addr}, got {os.environ.get('MASTER_ADDR')}"
            )
            assert os.environ.get("MASTER_PORT") == str(port), (
                f"Expected MASTER_PORT={port}, got {os.environ.get('MASTER_PORT')}"
            )
        finally:
            if orig_addr is not None:
                os.environ["MASTER_ADDR"] = orig_addr
            elif "MASTER_ADDR" in os.environ:
                del os.environ["MASTER_ADDR"]
            if orig_port is not None:
                os.environ["MASTER_PORT"] = orig_port
            elif "MASTER_PORT" in os.environ:
                del os.environ["MASTER_PORT"]


# ---------------------------------------------------------------------------
# Property 2: Communication failure reporting completeness
# ---------------------------------------------------------------------------


class TestCommunicationFailureReporting:
    """Property 2: Communication failure reporting completeness.

    For any torch.distributed operation failure (init timeout, send timeout,
    recv timeout, peer disconnect), the resulting error message SHALL contain
    the local rank, the peer rank (if applicable), and the backend identifier
    ("gloo"). For send/recv failures, the error message SHALL additionally
    contain the tensor shape.

    **Validates: Requirements 1.5, 2.6, 10.2**
    """

    @given(
        rank_world=_rank_world,
        ip_addr=_ip_addresses,
        port=_ports,
    )
    @settings(max_examples=100)
    def test_init_failure_contains_rank_worldsize_gloo(
        self,
        rank_world: tuple[int, int],
        ip_addr: str,
        port: int,
    ) -> None:
        """Init timeout error message SHALL contain rank, world_size, and "gloo".

        **Validates: Requirements 1.5, 10.2**
        """
        rank, world_size = rank_world

        config = ProcessGroupConfig(
            rank=rank,
            world_size=world_size,
            master_addr=ip_addr,
            master_port=port,
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.init_process_group.side_effect = RuntimeError("Connection timed out")

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                init_process_group(config)

        error_msg = str(exc_info.value)
        assert f"rank={rank}" in error_msg, (
            f"Error message should contain rank={rank}: {error_msg}"
        )
        assert f"world_size={world_size}" in error_msg, (
            f"Error message should contain world_size={world_size}: {error_msg}"
        )
        assert "gloo" in error_msg, (
            f"Error message should contain 'gloo': {error_msg}"
        )

    @given(
        src_rank=st.integers(min_value=0, max_value=63),
        dst_rank=st.integers(min_value=0, max_value=63),
        shape=_tensor_shapes,
        dtype_name=_dtype_names,
    )
    @settings(max_examples=100)
    def test_send_failure_contains_ranks_and_shape(
        self,
        src_rank: int,
        dst_rank: int,
        shape: tuple[int, ...],
        dtype_name: str,
    ) -> None:
        """Send timeout error message SHALL contain src rank, dst rank, and tensor shape.

        **Validates: Requirements 2.6, 10.2**
        """
        dtype = DTYPE_MAP[dtype_name]
        tensor = MockTensor(shape=shape, dtype=dtype, device="cuda:0")

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.send.side_effect = RuntimeError("Send timed out")
        mock_dist.get_rank.return_value = src_rank

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                send_activation(tensor, dst_rank)

        error_msg = str(exc_info.value)
        assert f"src_rank={src_rank}" in error_msg, (
            f"Error message should contain src_rank={src_rank}: {error_msg}"
        )
        assert f"dst_rank={dst_rank}" in error_msg, (
            f"Error message should contain dst_rank={dst_rank}: {error_msg}"
        )
        assert str(shape) in error_msg, (
            f"Error message should contain tensor_shape={shape}: {error_msg}"
        )

    @given(
        src_rank=st.integers(min_value=0, max_value=63),
        local_rank=st.integers(min_value=0, max_value=63),
        shape=_tensor_shapes,
        dtype_name=_dtype_names,
    )
    @settings(max_examples=100)
    def test_recv_failure_contains_ranks_and_shape(
        self,
        src_rank: int,
        local_rank: int,
        shape: tuple[int, ...],
        dtype_name: str,
    ) -> None:
        """Recv timeout error message SHALL contain src rank, local rank, and tensor shape.

        **Validates: Requirements 2.6, 10.2**
        """
        dtype = DTYPE_MAP[dtype_name]

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.recv.side_effect = RuntimeError("Recv timed out")
        mock_dist.get_rank.return_value = local_rank

        # Mock torch.empty to return a mock tensor
        mock_buffer = MockTensor(shape=shape, dtype=dtype, device="cpu")
        mock_torch.empty.return_value = mock_buffer

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                recv_activation(shape, dtype, src_rank, "xpu:0")

        error_msg = str(exc_info.value)
        assert f"src_rank={src_rank}" in error_msg, (
            f"Error message should contain src_rank={src_rank}: {error_msg}"
        )
        assert f"local_rank={local_rank}" in error_msg, (
            f"Error message should contain local_rank={local_rank}: {error_msg}"
        )
        assert str(shape) in error_msg, (
            f"Error message should contain tensor_shape={shape}: {error_msg}"
        )

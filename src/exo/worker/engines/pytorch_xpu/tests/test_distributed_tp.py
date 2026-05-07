# Feature: tensor-parallelism-xpu, Task 4.5: Unit tests for tensor-parallel process group
"""
Unit tests for the tensor-parallel process group management in distributed.py.

Tests cover:
- init_tensor_parallel_group sets correct environment variables (GLOO_SOCKET_IFNAME,
  MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE)
- verify_tensor_parallel_group returns True when all_reduce returns expected sum
- verify_tensor_parallel_group returns False when all_reduce returns wrong value
- verify_tensor_parallel_group returns False on exception
- Separate process groups for TP (TB4) and PP (ethernet) coexistence
- TensorParallelGroupConfig defaults (backend="gloo", init_timeout=60, allreduce_timeout=30)
- Initialization timeout behavior

**Validates: Requirements 4.1, 4.6, 9.1, 9.3, 9.5, 9.6**
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_PATH = _THIS_DIR.parent / "distributed.py"


def _load_distributed() -> types.ModuleType:
    """Load distributed.py directly from file, avoiding __init__.py."""
    module_name = "distributed_tp_unit_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed()
TensorParallelGroupConfig = _mod.TensorParallelGroupConfig
init_tensor_parallel_group = _mod.init_tensor_parallel_group
verify_tensor_parallel_group = _mod.verify_tensor_parallel_group
get_tensor_parallel_group = _mod.get_tensor_parallel_group


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_torch_mocks() -> tuple[MagicMock, MagicMock]:
    """Create paired torch and torch.distributed mocks.

    Returns (mock_torch, mock_dist) where mock_torch.distributed IS mock_dist.
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


def _save_env_vars() -> dict[str, str | None]:
    """Save environment variables that init_tensor_parallel_group modifies."""
    keys = ["GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    return {k: os.environ.get(k) for k in keys}


def _restore_env_vars(saved: dict[str, str | None]) -> None:
    """Restore environment variables to their saved state."""
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


# ---------------------------------------------------------------------------
# Tests: TensorParallelGroupConfig defaults
# ---------------------------------------------------------------------------


class TestTensorParallelGroupConfigDefaults:
    """TensorParallelGroupConfig has correct default values.

    **Validates: Requirements 4.1, 9.1, 9.5**
    """

    def test_backend_defaults_to_gloo(self) -> None:
        """Backend field defaults to 'gloo'."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )
        assert config.backend == "gloo"

    def test_init_timeout_defaults_to_60(self) -> None:
        """init_timeout_seconds defaults to 60 (shorter than ethernet — TB4 is local).

        **Validates: Requirement 9.5**
        """
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )
        assert config.init_timeout_seconds == 60

    def test_allreduce_timeout_defaults_to_30(self) -> None:
        """allreduce_timeout_seconds defaults to 30."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )
        assert config.allreduce_timeout_seconds == 30

    def test_config_is_frozen(self) -> None:
        """TensorParallelGroupConfig is immutable (frozen dataclass)."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            config.rank = 1  # type: ignore[misc]

    def test_config_stores_all_fields(self) -> None:
        """All fields are stored correctly."""
        config = TensorParallelGroupConfig(
            rank=2,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )
        assert config.rank == 2
        assert config.world_size == 4
        assert config.master_addr == "10.4.0.1"
        assert config.master_port == 29600
        assert config.tb4_interface_name == "thunderbolt0"


# ---------------------------------------------------------------------------
# Tests: init_tensor_parallel_group sets correct environment variables
# ---------------------------------------------------------------------------


class TestInitTensorParallelGroupEnvVars:
    """init_tensor_parallel_group sets correct environment variables.

    **Validates: Requirements 4.1, 9.1, 9.6**
    """

    def test_sets_gloo_socket_ifname_to_tb4_interface(self) -> None:
        """GLOO_SOCKET_IFNAME is set to the TB4 interface name."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.group.WORLD = MagicMock()

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            assert os.environ["GLOO_SOCKET_IFNAME"] == "thunderbolt0"
            assert os.environ["TP_SOCKET_IFNAME"] == "thunderbolt0"
        finally:
            _restore_env_vars(saved)
            # Reset module-level state
            _mod._tp_process_group = None

    def test_sets_master_addr_and_port(self) -> None:
        """MASTER_ADDR and MASTER_PORT are set from config."""
        config = TensorParallelGroupConfig(
            rank=1,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.group.WORLD = MagicMock()

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            assert os.environ["MASTER_ADDR"] == "10.4.0.1"
            assert os.environ["MASTER_PORT"] == "29600"
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_sets_rank_and_world_size(self) -> None:
        """RANK and WORLD_SIZE environment variables are set from config."""
        config = TensorParallelGroupConfig(
            rank=3,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.group.WORLD = MagicMock()

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            assert os.environ["RANK"] == "3"
            assert os.environ["WORLD_SIZE"] == "4"
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_calls_init_process_group_when_not_initialized(self) -> None:
        """When no default group exists, calls dist.init_process_group."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.group.WORLD = MagicMock()

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            mock_dist.init_process_group.assert_called_once()
            call_kwargs = mock_dist.init_process_group.call_args.kwargs
            assert call_kwargs["backend"] == "gloo"
            assert call_kwargs["rank"] == 0
            assert call_kwargs["world_size"] == 4
            assert call_kwargs["init_method"] == "env://"
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None


# ---------------------------------------------------------------------------
# Tests: Separate process groups for TP (TB4) and PP (ethernet) coexistence
# ---------------------------------------------------------------------------


class TestTPAndPPCoexistence:
    """Tensor-parallel and pipeline-parallel process groups coexist.

    **Validates: Requirements 4.6, 9.6**
    """

    def test_creates_new_group_when_default_already_initialized(self) -> None:
        """When a default group exists (PP), creates a new group for TP."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = True  # PP group already exists
        mock_tp_group = MagicMock()
        mock_dist.new_group.return_value = mock_tp_group

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            # Should call new_group, NOT init_process_group
            mock_dist.new_group.assert_called_once()
            mock_dist.init_process_group.assert_not_called()

            # Verify new_group was called with correct ranks
            call_kwargs = mock_dist.new_group.call_args.kwargs
            assert call_kwargs["ranks"] == [0, 1, 2, 3]
            assert call_kwargs["backend"] == "gloo"
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_stores_tp_group_handle_when_coexisting(self) -> None:
        """The TP group handle is stored in module-level variable."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = True
        mock_tp_group = MagicMock(name="tp_group")
        mock_dist.new_group.return_value = mock_tp_group

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            assert get_tensor_parallel_group() is mock_tp_group
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_stores_world_group_when_no_default_exists(self) -> None:
        """When no default group exists, stores dist.group.WORLD as TP group."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_world = MagicMock(name="WORLD")
        mock_dist.group.WORLD = mock_world

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            assert get_tensor_parallel_group() is mock_world
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None


# ---------------------------------------------------------------------------
# Tests: verify_tensor_parallel_group with expected sum formula
# ---------------------------------------------------------------------------


class TestVerifyTensorParallelGroup:
    """verify_tensor_parallel_group validates connectivity via all-reduce.

    **Validates: Requirement 9.3**
    """

    def test_returns_true_when_allreduce_returns_expected_sum(self) -> None:
        """Returns True when all_reduce result equals world_size*(world_size-1)/2.

        For world_size=4, expected sum = 4*3/2 = 6 (sum of ranks 0+1+2+3).
        """
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.get_rank.return_value = 0

        # Create a mock tensor that returns 6 (expected sum for world_size=4)
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 6
        mock_torch.tensor.return_value = mock_tensor

        # Set the TP group so get_tensor_parallel_group() returns it
        mock_tp_group = MagicMock()
        _mod._tp_process_group = mock_tp_group

        try:
            with _patch_torch(mock_torch, mock_dist):
                result = verify_tensor_parallel_group(world_size=4)

            assert result is True
            mock_dist.all_reduce.assert_called_once()
        finally:
            _mod._tp_process_group = None

    def test_returns_true_for_world_size_2(self) -> None:
        """Returns True for world_size=2 when sum equals 1 (0+1)."""
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.get_rank.return_value = 0

        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 1  # 2*(2-1)/2 = 1
        mock_torch.tensor.return_value = mock_tensor

        _mod._tp_process_group = MagicMock()

        try:
            with _patch_torch(mock_torch, mock_dist):
                result = verify_tensor_parallel_group(world_size=2)

            assert result is True
        finally:
            _mod._tp_process_group = None

    def test_returns_false_when_allreduce_returns_wrong_value(self) -> None:
        """Returns False when all_reduce result does not match expected sum."""
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.get_rank.return_value = 0

        # Return wrong value (expected 6 for world_size=4, return 5)
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 5
        mock_torch.tensor.return_value = mock_tensor

        _mod._tp_process_group = MagicMock()

        try:
            with _patch_torch(mock_torch, mock_dist):
                result = verify_tensor_parallel_group(world_size=4)

            assert result is False
        finally:
            _mod._tp_process_group = None

    def test_returns_false_on_exception(self) -> None:
        """Returns False when all_reduce raises an exception."""
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.get_rank.return_value = 0
        mock_dist.all_reduce.side_effect = RuntimeError("Gloo transport error")

        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor

        _mod._tp_process_group = MagicMock()

        try:
            with _patch_torch(mock_torch, mock_dist):
                result = verify_tensor_parallel_group(world_size=4)

            assert result is False
        finally:
            _mod._tp_process_group = None

    def test_uses_tp_process_group_for_allreduce(self) -> None:
        """all_reduce is called with the tensor-parallel process group."""
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.get_rank.return_value = 1

        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 6
        mock_torch.tensor.return_value = mock_tensor

        mock_tp_group = MagicMock(name="tp_group")
        _mod._tp_process_group = mock_tp_group

        try:
            with _patch_torch(mock_torch, mock_dist):
                verify_tensor_parallel_group(world_size=4)

            call_kwargs = mock_dist.all_reduce.call_args.kwargs
            assert call_kwargs["group"] is mock_tp_group
        finally:
            _mod._tp_process_group = None


# ---------------------------------------------------------------------------
# Tests: Initialization timeout behavior
# ---------------------------------------------------------------------------


class TestInitTensorParallelGroupTimeout:
    """init_tensor_parallel_group handles timeout correctly.

    **Validates: Requirements 9.5, 9.6**
    """

    def test_timeout_raises_runtime_error_with_context(self) -> None:
        """Timeout during init raises RuntimeError with rank, world_size, and interface."""
        config = TensorParallelGroupConfig(
            rank=2,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
            init_timeout_seconds=60,
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.init_process_group.side_effect = RuntimeError(
            "Connection timed out after 60 seconds"
        )

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                with pytest.raises(RuntimeError) as exc_info:
                    init_tensor_parallel_group(config)

            error_msg = str(exc_info.value)
            assert "rank=2" in error_msg
            assert "world_size=4" in error_msg
            assert "thunderbolt0" in error_msg
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_timeout_preserves_original_exception(self) -> None:
        """The original timeout exception is chained via __cause__."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        original_error = RuntimeError("Peer unreachable on TB4")
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.init_process_group.side_effect = original_error

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                with pytest.raises(RuntimeError) as exc_info:
                    init_tensor_parallel_group(config)

            assert exc_info.value.__cause__ is original_error
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_new_group_failure_raises_runtime_error(self) -> None:
        """Failure when creating new group (PP already exists) raises RuntimeError."""
        config = TensorParallelGroupConfig(
            rank=1,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
        )

        original_error = RuntimeError("new_group timed out")
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = True
        mock_dist.new_group.side_effect = original_error

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                with pytest.raises(RuntimeError) as exc_info:
                    init_tensor_parallel_group(config)

            error_msg = str(exc_info.value)
            assert "rank=1" in error_msg
            assert "world_size=4" in error_msg
            assert "thunderbolt0" in error_msg
            assert exc_info.value.__cause__ is original_error
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

    def test_uses_configured_timeout_value(self) -> None:
        """The configured init_timeout_seconds is passed to init_process_group."""
        config = TensorParallelGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.4.0.1",
            master_port=29600,
            tb4_interface_name="thunderbolt0",
            init_timeout_seconds=30,
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.is_initialized.return_value = False
        mock_dist.group.WORLD = MagicMock()

        saved = _save_env_vars()
        try:
            with _patch_torch(mock_torch, mock_dist):
                init_tensor_parallel_group(config)

            call_kwargs = mock_dist.init_process_group.call_args.kwargs
            # The timeout should be a timedelta of 30 seconds
            from datetime import timedelta
            assert call_kwargs["timeout"] == timedelta(seconds=30)
        finally:
            _restore_env_vars(saved)
            _mod._tp_process_group = None

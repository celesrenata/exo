"""
Unit tests for PipelineParallelShard microbatch interface.

Covers:
- validate_microbatch_compatibility returns empty list for valid microbatch
- validate_microbatch_compatibility identifies requests with missing caches
- forward_microbatch accepts a DecodeMicrobatch (type compatibility)
- forward_microbatch raises ValueError for zero active slots

Requirements: 3.1, 3.2, 3.3, 3.9
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Load modules with torch mocked out
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_SAMPLING_PATH = _ENGINE_DIR / "sampling.py"
_BATCHING_PATH = _ENGINE_DIR / "continuous_batching.py"
_CACHE_PATH = _ENGINE_DIR / "gated_deltanet_cache.py"
_STATE_PATH = _ENGINE_DIR / "gated_deltanet_state.py"
_SHARD_PATH = _ENGINE_DIR / "pipeline_parallel_shard.py"


def _ensure_torch_mock() -> None:
    """Ensure a minimal torch mock is in sys.modules."""
    if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "Tensor"):
        torch_mock = types.ModuleType("torch")
        torch_mock.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        torch_mock.device = type("device", (), {"__init__": lambda self, *a: None})  # type: ignore[attr-defined]
        torch_mock.float32 = "float32"  # type: ignore[attr-defined]
        torch_mock.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
        torch_mock.long = "long"  # type: ignore[attr-defined]

        # Mock torch.zeros to return a mock tensor
        def _mock_zeros(*args: object, **kwargs: object) -> MagicMock:
            tensor = MagicMock()
            tensor.data_ptr.return_value = id(tensor)
            tensor.zero_ = MagicMock(return_value=tensor)
            return tensor

        torch_mock.zeros = _mock_zeros  # type: ignore[attr-defined]

        # Mock torch.no_grad context manager
        class _NoGrad:
            def __enter__(self) -> None:
                pass

            def __exit__(self, *args: object) -> None:
                pass

        torch_mock.no_grad = _NoGrad  # type: ignore[attr-defined]

        # Mock torch.tensor
        def _mock_tensor(*args: object, **kwargs: object) -> MagicMock:
            tensor = MagicMock()
            tensor.shape = (1, 1)
            tensor.device = "cpu"
            return tensor

        torch_mock.tensor = _mock_tensor  # type: ignore[attr-defined]

        # Mock torch.argmax
        def _mock_argmax(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.item.return_value = 42
            return result

        torch_mock.argmax = _mock_argmax  # type: ignore[attr-defined]

        # Mock torch.arange
        def _mock_arange(*args: object, **kwargs: object) -> MagicMock:
            tensor = MagicMock()
            tensor.unsqueeze = MagicMock(return_value=tensor)
            tensor.expand = MagicMock(return_value=tensor)
            return tensor

        torch_mock.arange = _mock_arange  # type: ignore[attr-defined]

        # Mock torch.compile
        torch_mock.compile = MagicMock(side_effect=RuntimeError("no compile"))  # type: ignore[attr-defined]

        # Mock torch.nn
        nn_mock = types.ModuleType("torch.nn")
        nn_mock.ModuleList = list  # type: ignore[attr-defined]
        nn_mock.Module = type("Module", (), {})  # type: ignore[attr-defined]
        nn_mock.Embedding = type("Embedding", (nn_mock.Module,), {})  # type: ignore[attr-defined]
        nn_mock.Linear = type("Linear", (nn_mock.Module,), {})  # type: ignore[attr-defined]
        torch_mock.nn = nn_mock  # type: ignore[attr-defined]
        sys.modules["torch.nn"] = nn_mock

        sys.modules["torch"] = torch_mock
    else:
        # Torch mock already exists — ensure it has the attributes we need
        torch_mock = sys.modules["torch"]
        if not hasattr(torch_mock, "tensor"):

            def _mock_tensor(*args: object, **kwargs: object) -> MagicMock:
                tensor = MagicMock()
                tensor.shape = (1, 1)
                tensor.device = "cpu"
                return tensor

            torch_mock.tensor = _mock_tensor  # type: ignore[attr-defined]

        if not hasattr(torch_mock, "argmax"):

            def _mock_argmax(*args: object, **kwargs: object) -> MagicMock:
                result = MagicMock()
                result.item.return_value = 42
                return result

            torch_mock.argmax = _mock_argmax  # type: ignore[attr-defined]

        if not hasattr(torch_mock, "long"):
            torch_mock.long = "long"  # type: ignore[attr-defined]

        if not hasattr(torch_mock, "no_grad"):

            class _NoGrad:
                def __enter__(self) -> None:
                    pass

                def __exit__(self, *args: object) -> None:
                    pass

            torch_mock.no_grad = _NoGrad  # type: ignore[attr-defined]

        if not hasattr(torch_mock, "arange"):

            def _mock_arange(*args: object, **kwargs: object) -> MagicMock:
                tensor = MagicMock()
                tensor.unsqueeze = MagicMock(return_value=tensor)
                tensor.expand = MagicMock(return_value=tensor)
                return tensor

            torch_mock.arange = _mock_arange  # type: ignore[attr-defined]

        if not hasattr(torch_mock, "compile"):
            torch_mock.compile = MagicMock(side_effect=RuntimeError("no compile"))  # type: ignore[attr-defined]

        if not hasattr(torch_mock, "nn"):
            nn_mock = types.ModuleType("torch.nn")
            nn_mock.ModuleList = list  # type: ignore[attr-defined]
            nn_mock.Module = type("Module", (), {})  # type: ignore[attr-defined]
            nn_mock.Embedding = type("Embedding", (nn_mock.Module,), {})  # type: ignore[attr-defined]
            nn_mock.Linear = type("Linear", (nn_mock.Module,), {})  # type: ignore[attr-defined]
            torch_mock.nn = nn_mock  # type: ignore[attr-defined]
            sys.modules["torch.nn"] = nn_mock


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Load a module by file path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ensure_torch_mock()

# Ensure transformers mock is available
if "transformers" not in sys.modules:
    transformers_mock = types.ModuleType("transformers")
    sys.modules["transformers"] = transformers_mock
if "transformers.cache_utils" not in sys.modules:
    cache_utils_mock = types.ModuleType("transformers.cache_utils")
    cache_utils_mock.DynamicCache = MagicMock  # type: ignore[attr-defined]
    sys.modules["transformers.cache_utils"] = cache_utils_mock

# Load sampling first (dependency of continuous_batching)
_sampling_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.sampling", _SAMPLING_PATH
)

# Load gated_deltanet_state (dependency of gated_deltanet_cache)
_state_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.gated_deltanet_state", _STATE_PATH
)

# Load gated_deltanet_cache
_cache_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.gated_deltanet_cache", _CACHE_PATH
)

# Load continuous_batching
_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
PerRequestCacheManager = _batching_mod.PerRequestCacheManager
DecodeMicrobatch = _batching_mod.DecodeMicrobatch
BatchSlotState = _batching_mod.BatchSlotState
TokenResultBatch = _batching_mod.TokenResultBatch

# Load pipeline_parallel_shard
_shard_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.pipeline_parallel_shard", _SHARD_PATH
)
PipelineParallelShard = _shard_mod.PipelineParallelShard
PipelineStageConfig = _shard_mod.PipelineStageConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_cache() -> MagicMock:
    """Create a mock DynamicCache."""
    cache = MagicMock()
    cache.__len__ = MagicMock(return_value=0)
    return cache


def _make_shard(rank: int = 0, is_last: bool = False) -> PipelineParallelShard:
    """Create a minimal PipelineParallelShard for testing.

    Uses mocked layers and modules to avoid needing real torch operations.
    """
    world_size = 4
    num_layers = 2
    start_layer = rank * num_layers
    end_layer = start_layer + num_layers

    config = PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=256,
        vocab_size=32000,
        num_layers=8,
        device="cpu",
    )

    # Create mock layers
    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        # Layer returns (hidden_states, None) tuple
        output_tensor = MagicMock()
        output_tensor.shape = (1, 1, 256)
        output_tensor.dim.return_value = 3
        output_tensor.dtype = "bfloat16"
        output_tensor.is_contiguous.return_value = True
        layer.return_value = (output_tensor, None)
        layers.append(layer)

    # Mock ModuleList behavior
    mock_layers = MagicMock()
    mock_layers.__len__ = MagicMock(return_value=num_layers)
    mock_layers.__iter__ = MagicMock(return_value=iter(layers))
    mock_layers.__getitem__ = MagicMock(side_effect=lambda i: layers[i])

    embed_tokens = MagicMock() if rank == 0 else None
    lm_head = MagicMock() if is_last else None
    final_norm = MagicMock() if is_last else None

    # Create optimization config that disables GatedDeltaNet persistent state
    # to avoid complex initialization in tests
    opt_config = MagicMock()
    opt_config.enable_gated_deltanet_persistent_state = False

    shard = PipelineParallelShard(
        layers=mock_layers,
        config=config,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
        rotary_emb=None,
        text_model_config=None,
        performance_recorder=None,
        optimization_config=opt_config,
    )

    return shard


def _make_microbatch(
    active_count: int = 2,
    request_ids: list[str] | None = None,
    input_tokens: list[int] | None = None,
) -> DecodeMicrobatch:
    """Create a DecodeMicrobatch for testing."""
    if request_ids is None:
        request_ids = [f"req-{i}" for i in range(active_count)]
    if input_tokens is None:
        input_tokens = [100 + i for i in range(active_count)]

    slot_states = tuple(
        BatchSlotState(
            slot_index=i,
            request_id=request_ids[i],
            slot_generation=1,
            is_active=True,
        )
        for i in range(active_count)
    )

    return DecodeMicrobatch(
        active_slot_count=active_count,
        max_batch_size=8,
        slot_states=slot_states,
        input_token_ids=tuple(input_tokens),
    )


# ---------------------------------------------------------------------------
# Test: validate_microbatch_compatibility returns empty list for valid microbatch
# ---------------------------------------------------------------------------


class TestValidateMicrobatchCompatibilityValid:
    """Tests that validate_microbatch_compatibility returns empty list when all caches exist."""

    def test_all_requests_have_caches(self) -> None:
        """Returns empty list when all active requests have caches."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        # Create caches for all requests
        cache_manager.create_cache("req-0", _make_dynamic_cache())
        cache_manager.create_cache("req-1", _make_dynamic_cache())

        microbatch = _make_microbatch(
            active_count=2, request_ids=["req-0", "req-1"]
        )

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert result == []

    def test_single_request_valid(self) -> None:
        """Returns empty list for a single valid request."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)
        cache_manager.create_cache("req-solo", _make_dynamic_cache())

        microbatch = _make_microbatch(
            active_count=1, request_ids=["req-solo"], input_tokens=[42]
        )

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert result == []

    def test_many_requests_all_valid(self) -> None:
        """Returns empty list for many concurrent valid requests."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        request_ids = [f"req-{i}" for i in range(4)]
        for rid in request_ids:
            cache_manager.create_cache(rid, _make_dynamic_cache())

        microbatch = _make_microbatch(active_count=4, request_ids=request_ids)

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert result == []


# ---------------------------------------------------------------------------
# Test: validate_microbatch_compatibility identifies missing caches
# ---------------------------------------------------------------------------


class TestValidateMicrobatchCompatibilityMissing:
    """Tests that validate_microbatch_compatibility identifies requests with missing caches."""

    def test_all_missing(self) -> None:
        """Returns all request IDs when no caches exist."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        microbatch = _make_microbatch(
            active_count=2, request_ids=["req-0", "req-1"]
        )

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert set(result) == {"req-0", "req-1"}

    def test_partial_missing(self) -> None:
        """Returns only the request IDs that are missing caches."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        # Only create cache for req-0, not req-1
        cache_manager.create_cache("req-0", _make_dynamic_cache())

        microbatch = _make_microbatch(
            active_count=2, request_ids=["req-0", "req-1"]
        )

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert result == ["req-1"]

    def test_inactive_slots_ignored(self) -> None:
        """Inactive slots are not checked for cache validity."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)
        cache_manager.create_cache("req-active", _make_dynamic_cache())

        # Create a microbatch with one active and one inactive slot
        slot_states = (
            BatchSlotState(
                slot_index=0,
                request_id="req-active",
                slot_generation=1,
                is_active=True,
            ),
            BatchSlotState(
                slot_index=1,
                request_id="req-inactive-no-cache",
                slot_generation=1,
                is_active=False,
            ),
        )

        microbatch = DecodeMicrobatch(
            active_slot_count=1,
            max_batch_size=8,
            slot_states=slot_states,
            input_token_ids=(100,),
        )

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert result == []

    def test_none_request_id_ignored(self) -> None:
        """Slots with None request_id are skipped."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        slot_states = (
            BatchSlotState(
                slot_index=0,
                request_id=None,
                slot_generation=1,
                is_active=True,
            ),
        )

        microbatch = DecodeMicrobatch(
            active_slot_count=1,
            max_batch_size=8,
            slot_states=slot_states,
            input_token_ids=(100,),
        )

        result = shard.validate_microbatch_compatibility(microbatch, cache_manager)
        assert result == []


# ---------------------------------------------------------------------------
# Test: forward_microbatch accepts DecodeMicrobatch (type compatibility)
# ---------------------------------------------------------------------------


class TestForwardMicrobatchTypeCompatibility:
    """Tests that forward_microbatch accepts a DecodeMicrobatch and returns TokenResultBatch."""

    def test_accepts_decode_microbatch(self) -> None:
        """forward_microbatch accepts a DecodeMicrobatch and returns TokenResultBatch."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)
        cache_manager.create_cache("req-0", _make_dynamic_cache())

        microbatch = _make_microbatch(
            active_count=1, request_ids=["req-0"], input_tokens=[100]
        )

        # Mock the forward method to avoid needing real tensor operations
        mock_output = MagicMock()
        mock_output.dim.return_value = 2
        mock_output.shape = (1, 1, 256)
        shard.forward = MagicMock(return_value=(mock_output, []))  # type: ignore[method-assign]

        result = shard.forward_microbatch(microbatch, cache_manager)

        assert isinstance(result, TokenResultBatch)
        assert len(result.token_ids) == 1
        assert len(result.slot_indices) == 1
        assert len(result.slot_generations) == 1

    def test_returns_correct_slot_metadata(self) -> None:
        """TokenResultBatch contains correct slot indices and generations."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)
        cache_manager.create_cache("req-a", _make_dynamic_cache())
        cache_manager.create_cache("req-b", _make_dynamic_cache())

        slot_states = (
            BatchSlotState(
                slot_index=2,
                request_id="req-a",
                slot_generation=3,
                is_active=True,
            ),
            BatchSlotState(
                slot_index=5,
                request_id="req-b",
                slot_generation=7,
                is_active=True,
            ),
        )

        microbatch = DecodeMicrobatch(
            active_slot_count=2,
            max_batch_size=8,
            slot_states=slot_states,
            input_token_ids=(100, 200),
        )

        # Mock forward to avoid real tensor ops
        mock_output = MagicMock()
        mock_output.dim.return_value = 2
        mock_output.shape = (1, 1, 256)
        shard.forward = MagicMock(return_value=(mock_output, []))  # type: ignore[method-assign]

        result = shard.forward_microbatch(microbatch, cache_manager)

        assert result.slot_indices == (2, 5)
        assert result.slot_generations == (3, 7)
        assert len(result.token_ids) == 2

    def test_raises_on_zero_active_slots(self) -> None:
        """forward_microbatch raises ValueError for zero active slots."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        microbatch = DecodeMicrobatch(
            active_slot_count=0,
            max_batch_size=8,
            slot_states=(),
            input_token_ids=(),
        )

        with pytest.raises(ValueError, match="zero active slots"):
            shard.forward_microbatch(microbatch, cache_manager)

    def test_skips_requests_without_cache(self) -> None:
        """forward_microbatch skips requests that have no cache entry."""
        shard = _make_shard(rank=0)
        cache_manager = PerRequestCacheManager(max_requests=8, num_layers=8)

        # Only create cache for req-0, not req-1
        cache_manager.create_cache("req-0", _make_dynamic_cache())

        microbatch = _make_microbatch(
            active_count=2, request_ids=["req-0", "req-1"]
        )

        # Mock forward to avoid real tensor ops
        mock_output = MagicMock()
        mock_output.dim.return_value = 2
        mock_output.shape = (1, 1, 256)
        shard.forward = MagicMock(return_value=(mock_output, []))  # type: ignore[method-assign]

        result = shard.forward_microbatch(microbatch, cache_manager)

        # Only req-0 should produce a result (req-1 has no cache)
        assert len(result.token_ids) == 1
        assert result.slot_indices == (0,)

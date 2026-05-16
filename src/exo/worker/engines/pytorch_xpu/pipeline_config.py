"""
Pipeline configuration for XPU inference optimization.

Provides immutable Pydantic configuration models for pipeline-parallel inference,
including layer distribution, chunked prefill settings, and optimization feature flags.

All configuration classes use Pydantic models with ``frozen=True`` and ``strict=True``
to enforce immutability and type safety at runtime.
"""

from __future__ import annotations

from typing import final

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# PipelineLayerDistribution — layer assignment across pipeline ranks
# ---------------------------------------------------------------------------

_DEFAULT_TOTAL_LAYER_COUNT: int = 64
_DEFAULT_RANK_COUNT: int = 4


@final
class PipelineLayerDistribution(BaseModel):
    """
    Immutable representation of how transformer layers are distributed across
    pipeline-parallel ranks.

    Each element of ``layers_per_rank`` specifies the number of contiguous layers
    assigned to that rank. The sum must equal the total layer count and the tuple
    length must equal the rank count.

    Examples for four ranks with 64 layers:
        - Uniform: (16, 16, 16, 16)
        - Final lighter: (17, 17, 16, 14)
        - Attention aware: (15, 17, 17, 15)
    """

    model_config = ConfigDict(frozen=True, strict=True)

    layers_per_rank: tuple[int, ...]
    """Number of layers assigned to each rank, ordered by rank index."""

    total_layer_count: int = _DEFAULT_TOTAL_LAYER_COUNT
    """Total number of transformer layers in the model."""

    rank_count: int = _DEFAULT_RANK_COUNT
    """Number of pipeline-parallel ranks."""

    @field_validator("layers_per_rank")
    @classmethod
    def validate_layers_per_rank_not_empty(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Validate that layers_per_rank is not empty."""
        if len(value) == 0:
            raise ValueError("layers_per_rank must not be empty")
        for count in value:
            if count <= 0:
                raise ValueError(
                    f"Each rank must have at least one layer, got {count}"
                )
        return value

    @field_validator("total_layer_count")
    @classmethod
    def validate_total_layer_count_positive(cls, value: int) -> int:
        """Validate that total_layer_count is positive."""
        if value <= 0:
            raise ValueError(
                f"total_layer_count must be positive, got {value}"
            )
        return value

    @field_validator("rank_count")
    @classmethod
    def validate_rank_count_positive(cls, value: int) -> int:
        """Validate that rank_count is positive."""
        if value <= 0:
            raise ValueError(f"rank_count must be positive, got {value}")
        return value

    def model_post_init(self, __context: object) -> None:
        """Validate cross-field constraints after all fields are set."""
        if len(self.layers_per_rank) != self.rank_count:
            raise ValueError(
                f"layers_per_rank has {len(self.layers_per_rank)} elements "
                f"but rank_count is {self.rank_count}"
            )
        layer_sum = sum(self.layers_per_rank)
        if layer_sum != self.total_layer_count:
            raise ValueError(
                f"layers_per_rank sums to {layer_sum} "
                f"but total_layer_count is {self.total_layer_count}"
            )


# ---------------------------------------------------------------------------
# ChunkedGatedDeltaNetPrefillConfiguration — chunked prefill settings
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE: int = 64


@final
class ChunkedGatedDeltaNetPrefillConfiguration(BaseModel):
    """
    Configuration for chunked GatedDeltaNet prefill optimization.

    When enabled, GatedDeltaNet prefill uses a WY-style decomposition or
    equivalent associative block composition instead of strictly sequential
    token-by-token processing.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    enabled: bool = False
    """Whether chunked prefill is active. Defaults to disabled for safety."""

    chunk_size: int = _DEFAULT_CHUNK_SIZE
    """Number of tokens per chunk during chunked prefill."""

    fallback_on_unsupported_shape: bool = True
    """
    Whether to fall back to sequential prefill when an unsupported shape,
    dtype, or layer parameter combination is encountered. When False,
    unsupported configurations raise a structured error.
    """

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size_positive(cls, value: int) -> int:
        """Validate that chunk_size is positive."""
        if value <= 0:
            raise ValueError(f"chunk_size must be positive, got {value}")
        return value


# ---------------------------------------------------------------------------
# PytorchXpuOptimizationConfiguration — top-level optimization flags
# ---------------------------------------------------------------------------


@final
class PytorchXpuOptimizationConfiguration(BaseModel):
    """
    Top-level configuration for PyTorch XPU pipeline-parallel inference optimization.

    Defaults prioritize correctness and incremental rollout. Continuous batching
    and chunked prefill are initially opt-in until tests and benchmarks confirm
    correctness.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    # --- Performance instrumentation ---

    enable_performance_instrumentation: bool = True
    """Whether to record performance metrics via the PerformanceRecorder."""

    enable_detailed_tracing: bool = False
    """
    Whether to enable detailed event tracing with per-layer and per-operation
    granularity. Increases instrumentation overhead but provides fine-grained
    timing data for optimization analysis.
    """

    enable_xpu_synchronization_timing: bool = False
    """
    Whether to insert XPU synchronization barriers before timing measurements.
    Required for accurate per-operation timing on XPU but adds synchronization
    overhead that reduces throughput. Intended for profiling sessions only.
    """

    # --- Communication optimization ---

    enable_decode_fast_path: bool = True
    """
    Whether to use the decode fast path that avoids per-token shape metadata
    transfers after protocol negotiation.
    """

    # --- Sampling optimization ---

    enable_fast_sampling: bool = True
    """
    Whether to use optimized sampling paths (greedy argmax, top-k without
    full-vocabulary sort) instead of the baseline full-sort path.
    """

    # --- GatedDeltaNet optimization ---

    enable_gated_deltanet_persistent_state: bool = True
    """
    Whether to store GatedDeltaNet recurrent state persistently in fp32
    to avoid per-token dtype promotion and allocation.
    """

    # --- Model loading ---

    enable_local_shard_loading: bool = True
    """
    Whether each rank loads only its assigned pipeline stage tensors
    instead of materializing the full model.
    """

    # --- Batching ---

    enable_continuous_batching: bool = False
    """
    Whether to enable continuous batching for concurrent request processing.
    Opt-in until correctness is confirmed by tests and benchmarks.
    """

    enable_chunked_gated_deltanet_prefill: bool = False
    """
    Whether to enable chunked GatedDeltaNet prefill optimization.
    Opt-in until correctness is confirmed by tests and benchmarks.
    """

    # --- Numeric parameters ---

    maximum_decode_microbatch_size: int = 8
    """Maximum number of requests in a single decode microbatch."""

    decode_protocol_version: int = 1
    """Version of the decode fast-path communication protocol."""

    # --- Nested configuration ---

    pipeline_layer_distribution: PipelineLayerDistribution = (
        PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=_DEFAULT_TOTAL_LAYER_COUNT,
            rank_count=_DEFAULT_RANK_COUNT,
        )
    )
    """Layer distribution across pipeline-parallel ranks."""

    chunked_prefill_configuration: ChunkedGatedDeltaNetPrefillConfiguration = (
        ChunkedGatedDeltaNetPrefillConfiguration()
    )
    """Configuration for chunked GatedDeltaNet prefill."""

    @field_validator("maximum_decode_microbatch_size")
    @classmethod
    def validate_maximum_decode_microbatch_size(cls, value: int) -> int:
        """Validate that maximum_decode_microbatch_size is positive."""
        if value <= 0:
            raise ValueError(
                f"maximum_decode_microbatch_size must be positive, got {value}"
            )
        return value

    @field_validator("decode_protocol_version")
    @classmethod
    def validate_decode_protocol_version(cls, value: int) -> int:
        """Validate that decode_protocol_version is positive."""
        if value <= 0:
            raise ValueError(
                f"decode_protocol_version must be positive, got {value}"
            )
        return value

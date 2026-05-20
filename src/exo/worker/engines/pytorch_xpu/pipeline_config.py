"""
Pipeline configuration for XPU inference optimization.

Provides immutable Pydantic configuration models for pipeline-parallel inference,
including layer distribution, chunked prefill settings, and optimization feature flags.

All configuration classes use Pydantic models with ``frozen=True`` and ``strict=True``
to enforce immutability and type safety at runtime.
"""

from __future__ import annotations

from typing import Literal, final

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# PipelineStageAssignment — single rank's layer assignment
# ---------------------------------------------------------------------------


@final
class PipelineStageAssignment(BaseModel):
    """
    Immutable description of a single rank's layer assignment in the pipeline.

    Defines which contiguous range of transformer layers a rank owns, and
    whether it is responsible for the token embedding (rank 0) or the final
    normalization and language model head (last rank).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    rank: int
    """Pipeline stage rank (0-indexed)."""

    start_layer: int
    """First layer index assigned to this stage (inclusive)."""

    end_layer: int
    """Last layer index assigned to this stage (exclusive)."""

    owns_embedding: bool
    """Whether this rank owns the token embedding (rank 0)."""

    owns_lm_head: bool
    """Whether this rank owns the final normalization and lm_head (last rank)."""

    @field_validator("rank")
    @classmethod
    def validate_rank_nonnegative(cls, value: int) -> int:
        """Validate that rank is non-negative."""
        if value < 0:
            raise ValueError(f"rank must be non-negative, got {value}")
        return value

    @field_validator("end_layer")
    @classmethod
    def validate_end_layer_positive(cls, value: int) -> int:
        """Validate that end_layer is positive (at least 1 layer assigned)."""
        if value <= 0:
            raise ValueError(f"end_layer must be positive, got {value}")
        return value

    def model_post_init(self, __context: object) -> None:
        """Validate that start_layer < end_layer."""
        if self.start_layer >= self.end_layer:
            raise ValueError(
                f"start_layer ({self.start_layer}) must be less than "
                f"end_layer ({self.end_layer})"
            )

    @property
    def num_local_layers(self) -> int:
        """Number of transformer layers assigned to this stage."""
        return self.end_layer - self.start_layer


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

    def get_stage_assignment(self, rank: int) -> PipelineStageAssignment:
        """Compute the stage assignment for a given rank.

        Derives the contiguous layer range from the cumulative sum of
        ``layers_per_rank`` and determines embedding/lm_head ownership
        based on rank position.

        Args:
            rank: Pipeline stage rank (0-indexed). Must be in
                ``[0, rank_count)``.

        Returns:
            A ``PipelineStageAssignment`` describing the rank's layer range
            and auxiliary module ownership.

        Raises:
            ValueError: If rank is out of range.
        """
        if rank < 0 or rank >= self.rank_count:
            raise ValueError(
                f"rank must be in [0, {self.rank_count}), got {rank}"
            )
        start_layer = sum(self.layers_per_rank[:rank])
        end_layer = start_layer + self.layers_per_rank[rank]
        return PipelineStageAssignment(
            rank=rank,
            start_layer=start_layer,
            end_layer=end_layer,
            owns_embedding=(rank == 0),
            owns_lm_head=(rank == self.rank_count - 1),
        )

    def validate_contiguous(self) -> bool:
        """Verify that layer ranges are contiguous with no gaps or overlaps.

        Checks that the stage assignments derived from this distribution
        form a complete, non-overlapping partition of ``[0, total_layer_count)``.

        Returns:
            True if the distribution is contiguous and covers all layers.
        """
        expected_start = 0
        for rank in range(self.rank_count):
            assignment = self.get_stage_assignment(rank)
            if assignment.start_layer != expected_start:
                return False
            expected_start = assignment.end_layer
        return expected_start == self.total_layer_count


# ---------------------------------------------------------------------------
# Helper functions for layer distribution creation and parsing
# ---------------------------------------------------------------------------


def default_layer_distribution(
    total_layers: int, world_size: int
) -> PipelineLayerDistribution:
    """Create a balanced layer distribution using divmod.

    The first ``remainder`` ranks receive ``(base + 1)`` layers, and the
    remaining ranks receive ``base`` layers. This matches the logic in
    ``compute_layer_assignment``.

    Args:
        total_layers: Total number of transformer layers in the model.
        world_size: Number of pipeline-parallel ranks.

    Returns:
        A validated ``PipelineLayerDistribution`` with balanced assignment.

    Raises:
        ValueError: If total_layers < 1 or world_size < 1.
    """
    if total_layers < 1:
        raise ValueError(
            f"total_layers must be at least 1, got {total_layers}"
        )
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}")

    base = total_layers // world_size
    remainder = total_layers % world_size

    layers_per_rank = tuple(
        base + 1 if rank < remainder else base
        for rank in range(world_size)
    )

    return PipelineLayerDistribution(
        layers_per_rank=layers_per_rank,
        total_layer_count=total_layers,
        rank_count=world_size,
    )


def parse_layer_distribution_arg(
    arg: str, total_layers: int, world_size: int
) -> PipelineLayerDistribution:
    """Parse a comma-separated layer distribution string into a validated model.

    Accepts strings like ``"16,16,16,16"`` or ``"17,17,16,14"`` and creates
    a ``PipelineLayerDistribution`` with full validation.

    Args:
        arg: Comma-separated string of integers (e.g. ``"16,16,16,16"``).
        total_layers: Total number of transformer layers in the model.
        world_size: Number of pipeline-parallel ranks.

    Returns:
        A validated ``PipelineLayerDistribution``.

    Raises:
        ValueError: If the string cannot be parsed, contains non-integer
            values, or the resulting distribution fails validation (wrong
            sum, wrong rank count, zero-layer stages).
    """
    parts = arg.strip().split(",")
    try:
        layers_per_rank = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse layer distribution '{arg}': "
            f"all values must be integers. Error: {exc}"
        ) from exc

    if len(layers_per_rank) != world_size:
        raise ValueError(
            f"Layer distribution has {len(layers_per_rank)} elements "
            f"but world_size is {world_size}. "
            f"Expected {world_size} comma-separated integers."
        )

    layer_sum = sum(layers_per_rank)
    if layer_sum != total_layers:
        raise ValueError(
            f"Layer distribution sums to {layer_sum} "
            f"but total_layers is {total_layers}. "
            f"Values must sum to {total_layers}."
        )

    return PipelineLayerDistribution(
        layers_per_rank=layers_per_rank,
        total_layer_count=total_layers,
        rank_count=world_size,
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
    def validate_chunk_size(cls, value: int) -> int:
        """Validate that chunk_size is >= 16 and a power of 2."""
        if value < 16:
            raise ValueError(
                f"chunk_size must be >= 16, got {value}"
            )
        if value & (value - 1) != 0:
            raise ValueError(
                f"chunk_size must be a power of 2, got {value}"
            )
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

    # --- Kernel launch optimization flags ---

    enable_static_kv_cache: bool = False
    """Use pre-allocated static KV cache instead of DynamicCache."""

    enable_torch_compile: bool = False
    """Compile decode path with torch.compile(backend='inductor')."""

    enable_packed_projections: bool = False
    """Pack QKV and gate/up projections into single matmuls."""

    enable_fused_kernels: bool = False
    """Use fused pointwise kernels (RMSNorm+residual, SiLU*gate, rotary)."""

    enable_on_device_sampling: bool = False
    """Perform sampling on XPU without transferring logits to CPU."""

    enable_async_output: bool = False
    """Decouple token output from decode loop via async queue."""

    enable_sync_removal: bool = False
    """Remove all implicit synchronization from the decode hot path."""

    # --- Compile configuration ---

    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    """torch.compile optimization mode for XPU Inductor."""

    static_cache_max_seq_len: int = 2048
    """Maximum sequence length for static KV cache pre-allocation."""

    async_output_max_pending: int = 32
    """Maximum pending tokens before backpressure activates."""

    async_output_resume_threshold: int = 16
    """Queue depth at which generation resumes after backpressure."""

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

    chunked_gated_deltanet_prefill: ChunkedGatedDeltaNetPrefillConfiguration = (
        ChunkedGatedDeltaNetPrefillConfiguration()
    )
    """Configuration for chunked GatedDeltaNet prefill optimization."""

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

    @field_validator("torch_compile_mode")
    @classmethod
    def validate_torch_compile_mode(
        cls, value: str
    ) -> str:
        """Validate that torch_compile_mode is one of the allowed values."""
        allowed = {"default", "reduce-overhead", "max-autotune"}
        if value not in allowed:
            raise ValueError(
                f"torch_compile_mode must be one of {allowed}, got '{value}'"
            )
        return value

    @field_validator("static_cache_max_seq_len")
    @classmethod
    def validate_static_cache_max_seq_len(cls, value: int) -> int:
        """Validate that static_cache_max_seq_len is positive."""
        if value <= 0:
            raise ValueError(
                f"static_cache_max_seq_len must be positive, got {value}"
            )
        return value

    @field_validator("async_output_max_pending")
    @classmethod
    def validate_async_output_max_pending(cls, value: int) -> int:
        """Validate that async_output_max_pending is positive."""
        if value <= 0:
            raise ValueError(
                f"async_output_max_pending must be positive, got {value}"
            )
        return value

    @field_validator("async_output_resume_threshold")
    @classmethod
    def validate_async_output_resume_threshold(cls, value: int) -> int:
        """Validate that async_output_resume_threshold is positive."""
        if value <= 0:
            raise ValueError(
                f"async_output_resume_threshold must be positive, got {value}"
            )
        return value

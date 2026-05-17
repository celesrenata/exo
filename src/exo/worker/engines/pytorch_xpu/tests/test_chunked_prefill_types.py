"""Tests for chunked prefill type definitions and validation.

Verifies:
- ChunkTransform stores tensors with correct shapes
- validate_chunk_transform_shapes passes for valid transforms
- validate_chunk_transform_shapes raises for mismatched shapes or wrong dtype
- ChunkOutput is frozen (immutable)

**Validates: Requirements 7.1, 7.5, 7.6, 7.7**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_CHUNKED_PREFILL_PATH = _THIS_DIR.parent / "chunked_prefill.py"


def _load_chunked_prefill_module() -> types.ModuleType:
    """Load chunked_prefill.py directly from file, avoiding __init__.py."""
    module_name = "chunked_prefill_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _CHUNKED_PREFILL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_chunked_prefill_module()
ChunkTransform = _mod.ChunkTransform
ChunkOutput = _mod.ChunkOutput
ChunkTransformValidationError = _mod.ChunkTransformValidationError
validate_chunk_transform_shapes = _mod.validate_chunk_transform_shapes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_chunk_transform(
    batch_size: int = 2,
    num_heads: int = 4,
    chunk_size: int = 8,
    key_dim: int = 16,
    value_dim: int = 32,
) -> "ChunkTransform":
    """Create a valid ChunkTransform with consistent shapes and fp32 dtype."""
    return ChunkTransform(
        cumulative_log_decay=torch.zeros(batch_size, num_heads, dtype=torch.float32),
        correction_keys=torch.zeros(batch_size, num_heads, chunk_size, key_dim, dtype=torch.float32),
        correction_weights=torch.zeros(batch_size, num_heads, chunk_size, dtype=torch.float32),
        additive_term=torch.zeros(batch_size, num_heads, key_dim, value_dim, dtype=torch.float32),
        chunk_size=chunk_size,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )


# ---------------------------------------------------------------------------
# ChunkTransform shape tests
# ---------------------------------------------------------------------------


class TestChunkTransformShapes:
    """Verify ChunkTransform stores tensors with correct shapes."""

    def test_valid_transform_has_expected_shapes(self) -> None:
        batch_size, num_heads, chunk_size, key_dim, value_dim = 2, 4, 8, 16, 32
        transform = _make_valid_chunk_transform(batch_size, num_heads, chunk_size, key_dim, value_dim)

        assert transform.cumulative_log_decay.shape == (batch_size, num_heads)
        assert transform.correction_keys.shape == (batch_size, num_heads, chunk_size, key_dim)
        assert transform.correction_weights.shape == (batch_size, num_heads, chunk_size)
        assert transform.additive_term.shape == (batch_size, num_heads, key_dim, value_dim)

    def test_metadata_fields_match_tensor_shapes(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=1, num_heads=8, chunk_size=64, key_dim=128, value_dim=128)

        assert transform.chunk_size == 64
        assert transform.num_heads == 8
        assert transform.key_dim == 128
        assert transform.value_dim == 128

    def test_all_tensors_are_fp32(self) -> None:
        transform = _make_valid_chunk_transform()

        assert transform.cumulative_log_decay.dtype == torch.float32
        assert transform.correction_keys.dtype == torch.float32
        assert transform.correction_weights.dtype == torch.float32
        assert transform.additive_term.dtype == torch.float32

    def test_single_batch_single_head(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=1, num_heads=1, chunk_size=4, key_dim=8, value_dim=8)

        assert transform.cumulative_log_decay.shape == (1, 1)
        assert transform.correction_keys.shape == (1, 1, 4, 8)
        assert transform.correction_weights.shape == (1, 1, 4)
        assert transform.additive_term.shape == (1, 1, 8, 8)


# ---------------------------------------------------------------------------
# validate_chunk_transform_shapes — valid cases
# ---------------------------------------------------------------------------


class TestValidateChunkTransformShapesValid:
    """Verify validation passes for correctly shaped transforms."""

    def test_passes_for_valid_transform(self) -> None:
        transform = _make_valid_chunk_transform()
        # Should not raise
        validate_chunk_transform_shapes(transform)

    def test_passes_for_various_valid_sizes(self) -> None:
        for batch_size in (1, 4):
            for num_heads in (1, 8, 64):
                for chunk_size in (1, 16, 64):
                    for key_dim in (8, 128):
                        for value_dim in (8, 128):
                            transform = _make_valid_chunk_transform(
                                batch_size=batch_size,
                                num_heads=num_heads,
                                chunk_size=chunk_size,
                                key_dim=key_dim,
                                value_dim=value_dim,
                            )
                            validate_chunk_transform_shapes(transform)


# ---------------------------------------------------------------------------
# validate_chunk_transform_shapes — dtype errors
# ---------------------------------------------------------------------------


class TestValidateChunkTransformShapesDtypeErrors:
    """Verify validation raises for wrong dtypes."""

    def test_rejects_bf16_cumulative_log_decay(self) -> None:
        transform = _make_valid_chunk_transform()
        transform.cumulative_log_decay = transform.cumulative_log_decay.to(torch.bfloat16)

        with pytest.raises(ChunkTransformValidationError, match="cumulative_log_decay must be fp32"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_fp16_correction_keys(self) -> None:
        transform = _make_valid_chunk_transform()
        transform.correction_keys = transform.correction_keys.to(torch.float16)

        with pytest.raises(ChunkTransformValidationError, match="correction_keys must be fp32"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_bf16_correction_weights(self) -> None:
        transform = _make_valid_chunk_transform()
        transform.correction_weights = transform.correction_weights.to(torch.bfloat16)

        with pytest.raises(ChunkTransformValidationError, match="correction_weights must be fp32"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_fp16_additive_term(self) -> None:
        transform = _make_valid_chunk_transform()
        transform.additive_term = transform.additive_term.to(torch.float16)

        with pytest.raises(ChunkTransformValidationError, match="additive_term must be fp32"):
            validate_chunk_transform_shapes(transform)


# ---------------------------------------------------------------------------
# validate_chunk_transform_shapes — shape mismatch errors
# ---------------------------------------------------------------------------


class TestValidateChunkTransformShapesMismatchErrors:
    """Verify validation raises for mismatched shapes."""

    def test_rejects_wrong_num_heads_in_decay(self) -> None:
        transform = _make_valid_chunk_transform(num_heads=4)
        # Replace with wrong H dimension
        transform.cumulative_log_decay = torch.zeros(2, 8, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="cumulative_log_decay H dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_wrong_batch_in_correction_keys(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, chunk_size=8, key_dim=16)
        # Replace with wrong batch dimension
        transform.correction_keys = torch.zeros(3, 4, 8, 16, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="correction_keys batch dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_wrong_chunk_size_in_correction_keys(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, chunk_size=8, key_dim=16)
        # Replace with wrong C dimension
        transform.correction_keys = torch.zeros(2, 4, 12, 16, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="correction_keys C dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_wrong_key_dim_in_correction_keys(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, chunk_size=8, key_dim=16)
        # Replace with wrong d_k dimension
        transform.correction_keys = torch.zeros(2, 4, 8, 32, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="correction_keys d_k dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_wrong_chunk_size_in_correction_weights(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, chunk_size=8)
        # Replace with wrong C dimension
        transform.correction_weights = torch.zeros(2, 4, 16, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="correction_weights C dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_wrong_key_dim_in_additive_term(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, key_dim=16, value_dim=32)
        # Replace with wrong d_k dimension
        transform.additive_term = torch.zeros(2, 4, 64, 32, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="additive_term d_k dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_wrong_value_dim_in_additive_term(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, key_dim=16, value_dim=32)
        # Replace with wrong d_v dimension
        transform.additive_term = torch.zeros(2, 4, 16, 64, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="additive_term d_v dimension"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_1d_cumulative_log_decay(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4)
        # Replace with wrong dimensionality
        transform.cumulative_log_decay = torch.zeros(8, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="cumulative_log_decay must be 2D"):
            validate_chunk_transform_shapes(transform)

    def test_rejects_3d_correction_keys(self) -> None:
        transform = _make_valid_chunk_transform(batch_size=2, num_heads=4, chunk_size=8, key_dim=16)
        # Replace with wrong dimensionality (3D instead of 4D)
        transform.correction_keys = torch.zeros(2, 4, 128, dtype=torch.float32)

        with pytest.raises(ChunkTransformValidationError, match="correction_keys must be 4D"):
            validate_chunk_transform_shapes(transform)


# ---------------------------------------------------------------------------
# ChunkOutput — frozen immutability
# ---------------------------------------------------------------------------


class TestChunkOutputFrozen:
    """Verify ChunkOutput is frozen (immutable)."""

    def test_cannot_assign_activations(self) -> None:
        output = ChunkOutput(
            activations=torch.zeros(1, 8, 4, 32),
            chunk_index=0,
            chunk_size=8,
        )
        with pytest.raises(AttributeError):
            output.activations = torch.ones(1, 8, 4, 32)  # type: ignore[misc]

    def test_cannot_assign_chunk_index(self) -> None:
        output = ChunkOutput(
            activations=torch.zeros(1, 8, 4, 32),
            chunk_index=0,
            chunk_size=8,
        )
        with pytest.raises(AttributeError):
            output.chunk_index = 5  # type: ignore[misc]

    def test_cannot_assign_chunk_size(self) -> None:
        output = ChunkOutput(
            activations=torch.zeros(1, 8, 4, 32),
            chunk_index=0,
            chunk_size=8,
        )
        with pytest.raises(AttributeError):
            output.chunk_size = 16  # type: ignore[misc]

    def test_stores_correct_values(self) -> None:
        activations = torch.randn(2, 16, 8, 64)
        output = ChunkOutput(
            activations=activations,
            chunk_index=3,
            chunk_size=16,
        )

        assert torch.equal(output.activations, activations)
        assert output.chunk_index == 3
        assert output.chunk_size == 16

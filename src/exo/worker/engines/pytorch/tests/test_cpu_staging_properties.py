"""Property-based tests for CPU-staged tensor transfer round-trip.

Uses Hypothesis to verify that stage_to_cpu() followed by unstage_from_cpu()
preserves tensor data, shape, and dtype for arbitrary tensors.

**Validates: Requirements 6.7**
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.distributed.cpu_staging import (
    stage_to_cpu,
    unstage_from_cpu,
)


# --- Strategies ---

# Tensor shapes: tuples of 1-4 positive integers (each 1-32)
tensor_shape_st = st.tuples(
    st.integers(min_value=1, max_value=32),
).flatmap(
    lambda _: st.lists(
        st.integers(min_value=1, max_value=32), min_size=1, max_size=4
    ).map(tuple)
)

# Dtypes: sampled from the three supported floating-point types
tensor_dtype_st = st.sampled_from([torch.float16, torch.bfloat16, torch.float32])


class TestCpuStagingRoundTripProperty:
    """Property 6: CPU staging round-trip preserves tensor data.

    *For any* tensor with arbitrary shape, dtype (float16, bfloat16, float32),
    and values, staging to CPU via stage_to_cpu() and then unstaging back to
    the original device via unstage_from_cpu() SHALL produce a tensor that is
    element-wise equal to the original, with the same shape and dtype.

    **Validates: Requirements 6.7**
    """

    @settings(max_examples=100)
    @given(shape=tensor_shape_st, dtype=tensor_dtype_st)
    def test_roundtrip_preserves_data(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """unstage_from_cpu(stage_to_cpu(tensor), "cpu") is element-wise equal to original.

        Since we cannot test with real GPU devices in the test environment,
        we use CPU as both source and target. The logic is the same — the
        functions use .cpu() and .to(device).

        **Validates: Requirements 6.7**
        """
        original = torch.randn(shape, dtype=dtype)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")

        assert torch.equal(unstaged, original), (
            f"Round-trip failed: original and unstaged tensors differ. "
            f"Shape={shape}, dtype={dtype}"
        )

    @settings(max_examples=100)
    @given(shape=tensor_shape_st, dtype=tensor_dtype_st)
    def test_roundtrip_preserves_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """Round-trip preserves tensor shape.

        **Validates: Requirements 6.7**
        """
        original = torch.randn(shape, dtype=dtype)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")

        assert unstaged.shape == original.shape, (
            f"Shape mismatch: expected {original.shape}, got {unstaged.shape}"
        )

    @settings(max_examples=100)
    @given(shape=tensor_shape_st, dtype=tensor_dtype_st)
    def test_roundtrip_preserves_dtype(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """Round-trip preserves tensor dtype.

        **Validates: Requirements 6.7**
        """
        original = torch.randn(shape, dtype=dtype)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")

        assert unstaged.dtype == original.dtype, (
            f"Dtype mismatch: expected {original.dtype}, got {unstaged.dtype}"
        )

    @settings(max_examples=100)
    @given(shape=tensor_shape_st, dtype=tensor_dtype_st)
    def test_staged_tensor_is_cpu_and_contiguous(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """The staged tensor is always on CPU and contiguous.

        **Validates: Requirements 6.7**
        """
        original = torch.randn(shape, dtype=dtype)
        staged = stage_to_cpu(original)

        assert staged.device.type == "cpu", (
            f"Staged tensor should be on CPU, got {staged.device}"
        )
        assert staged.is_contiguous(), (
            "Staged tensor should be contiguous"
        )

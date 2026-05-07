"""Unit tests for distributed/cpu_staging.py — CPU-staged tensor transfer.

Tests stage_to_cpu and unstage_from_cpu functions for correctness,
edge cases (already on target device), and preservation of dtype/shape.
Uses CPU tensors to simulate GPU behavior since tests run without GPU hardware.
"""

from __future__ import annotations

import torch

from exo.worker.engines.pytorch.distributed.cpu_staging import (
    stage_to_cpu,
    unstage_from_cpu,
)


class TestStageToCpu:
    """Tests for stage_to_cpu function."""

    def test_cpu_tensor_is_noop(self) -> None:
        """If tensor is already on CPU, stage_to_cpu returns it as-is."""
        tensor = torch.randn(4, 8)
        result = stage_to_cpu(tensor)
        assert result.device.type == "cpu"
        assert torch.equal(result, tensor)

    def test_preserves_shape(self) -> None:
        """Staged tensor preserves original shape."""
        tensor = torch.randn(2, 16, 64)
        result = stage_to_cpu(tensor)
        assert result.shape == (2, 16, 64)

    def test_preserves_dtype_float32(self) -> None:
        """Staged tensor preserves float32 dtype."""
        tensor = torch.randn(3, 5, dtype=torch.float32)
        result = stage_to_cpu(tensor)
        assert result.dtype == torch.float32

    def test_preserves_dtype_float16(self) -> None:
        """Staged tensor preserves float16 dtype."""
        tensor = torch.randn(3, 5).to(torch.float16)
        result = stage_to_cpu(tensor)
        assert result.dtype == torch.float16

    def test_preserves_dtype_bfloat16(self) -> None:
        """Staged tensor preserves bfloat16 dtype."""
        tensor = torch.randn(3, 5).to(torch.bfloat16)
        result = stage_to_cpu(tensor)
        assert result.dtype == torch.bfloat16

    def test_preserves_dtype_int64(self) -> None:
        """Staged tensor preserves int64 dtype."""
        tensor = torch.randint(0, 100, (4, 4), dtype=torch.int64)
        result = stage_to_cpu(tensor)
        assert result.dtype == torch.int64

    def test_result_is_contiguous(self) -> None:
        """Staged tensor is always contiguous."""
        tensor = torch.randn(4, 8).t()  # Transpose makes it non-contiguous
        assert not tensor.is_contiguous()
        result = stage_to_cpu(tensor)
        assert result.is_contiguous()

    def test_preserves_data_values(self) -> None:
        """Staged tensor preserves exact data values."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = stage_to_cpu(tensor)
        assert torch.equal(result, tensor)

    def test_non_contiguous_cpu_tensor_becomes_contiguous(self) -> None:
        """A non-contiguous CPU tensor is made contiguous."""
        tensor = torch.randn(8, 4).t()  # Non-contiguous CPU tensor
        assert tensor.device.type == "cpu"
        assert not tensor.is_contiguous()
        result = stage_to_cpu(tensor)
        assert result.is_contiguous()
        assert torch.equal(result, tensor)

    def test_scalar_tensor(self) -> None:
        """Handles scalar (0-dim) tensors."""
        tensor = torch.tensor(42.0)
        result = stage_to_cpu(tensor)
        assert result.device.type == "cpu"
        assert result.item() == 42.0

    def test_empty_tensor(self) -> None:
        """Handles empty tensors (zero elements)."""
        tensor = torch.empty(0, 4)
        result = stage_to_cpu(tensor)
        assert result.device.type == "cpu"
        assert result.shape == (0, 4)


class TestUnstageFromCpu:
    """Tests for unstage_from_cpu function."""

    def test_cpu_to_cpu_is_noop(self) -> None:
        """If target_device is cpu and tensor is on cpu, returns as-is."""
        tensor = torch.randn(4, 8)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.device.type == "cpu"
        assert torch.equal(result, tensor)

    def test_preserves_shape(self) -> None:
        """Unstaged tensor preserves original shape."""
        tensor = torch.randn(2, 16, 64)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.shape == (2, 16, 64)

    def test_preserves_dtype_float32(self) -> None:
        """Unstaged tensor preserves float32 dtype."""
        tensor = torch.randn(3, 5, dtype=torch.float32)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.dtype == torch.float32

    def test_preserves_dtype_float16(self) -> None:
        """Unstaged tensor preserves float16 dtype."""
        tensor = torch.randn(3, 5).to(torch.float16)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.dtype == torch.float16

    def test_preserves_dtype_bfloat16(self) -> None:
        """Unstaged tensor preserves bfloat16 dtype."""
        tensor = torch.randn(3, 5).to(torch.bfloat16)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.dtype == torch.bfloat16

    def test_preserves_data_values(self) -> None:
        """Unstaged tensor preserves exact data values."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = unstage_from_cpu(tensor, "cpu")
        assert torch.equal(result, tensor)

    def test_scalar_tensor(self) -> None:
        """Handles scalar (0-dim) tensors."""
        tensor = torch.tensor(42.0)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.device.type == "cpu"
        assert result.item() == 42.0

    def test_empty_tensor(self) -> None:
        """Handles empty tensors (zero elements)."""
        tensor = torch.empty(0, 4)
        result = unstage_from_cpu(tensor, "cpu")
        assert result.device.type == "cpu"
        assert result.shape == (0, 4)


class TestRoundTrip:
    """Tests for stage_to_cpu → unstage_from_cpu round-trip on CPU."""

    def test_roundtrip_preserves_data(self) -> None:
        """Round-trip through CPU staging preserves tensor data."""
        original = torch.randn(8, 16, dtype=torch.float32)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")
        assert torch.equal(unstaged, original)

    def test_roundtrip_preserves_float16(self) -> None:
        """Round-trip preserves float16 data."""
        original = torch.randn(4, 4).to(torch.float16)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")
        assert torch.equal(unstaged, original)
        assert unstaged.dtype == torch.float16

    def test_roundtrip_preserves_bfloat16(self) -> None:
        """Round-trip preserves bfloat16 data."""
        original = torch.randn(4, 4).to(torch.bfloat16)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")
        assert torch.equal(unstaged, original)
        assert unstaged.dtype == torch.bfloat16

    def test_roundtrip_preserves_int_tensor(self) -> None:
        """Round-trip preserves integer tensor data."""
        original = torch.randint(0, 1000, (5, 10), dtype=torch.int64)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")
        assert torch.equal(unstaged, original)
        assert unstaged.dtype == torch.int64

    def test_roundtrip_large_tensor(self) -> None:
        """Round-trip works for larger tensors."""
        original = torch.randn(128, 256, dtype=torch.float32)
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")
        assert torch.equal(unstaged, original)

    def test_roundtrip_non_contiguous(self) -> None:
        """Round-trip handles non-contiguous input correctly."""
        original = torch.randn(8, 4).t()  # Non-contiguous
        staged = stage_to_cpu(original)
        unstaged = unstage_from_cpu(staged, "cpu")
        assert staged.is_contiguous()
        assert torch.equal(unstaged, original)

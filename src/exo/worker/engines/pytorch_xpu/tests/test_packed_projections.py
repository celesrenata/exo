"""Unit tests for packed projection utilities.

Verifies that weight packing and output unpacking produce numerically
equivalent results to separate projections, and that model layer
detection works correctly.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**
"""

from __future__ import annotations

import pytest
import torch

from exo.worker.engines.pytorch_xpu.packed_projections import (
    pack_gate_up_weights,
    pack_layer_gate_up_from_model,
    pack_layer_qkv_from_model,
    pack_qkv_weights,
    unpack_gate_up_output,
    unpack_qkv_output,
)

# ---------------------------------------------------------------------------
# Qwen3.5-4B dimensions
# ---------------------------------------------------------------------------

_HIDDEN_SIZE: int = 2560
_NUM_HEADS: int = 32
_HEAD_DIM: int = 80
_NUM_KV_HEADS: int = 4
_Q_DIM: int = _NUM_HEADS * _HEAD_DIM  # 2560
_KV_DIM: int = _NUM_KV_HEADS * _HEAD_DIM  # 320
_INTERMEDIATE_SIZE: int = 9728


class TestPackQkvWeights:
    """Tests for pack_qkv_weights."""

    def test_output_shape_qwen35(self) -> None:
        """Packed QKV weight has shape [3200, 2560] for Qwen3.5-4B."""
        q_weight = torch.randn(_Q_DIM, _HIDDEN_SIZE)
        k_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE)
        v_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE)

        packed = pack_qkv_weights(q_weight, k_weight, v_weight)

        assert packed.shape == (_Q_DIM + _KV_DIM + _KV_DIM, _HIDDEN_SIZE)
        assert packed.shape == (3200, 2560)

    def test_content_is_concatenation(self) -> None:
        """Packed weight is the exact concatenation of Q, K, V along dim=0."""
        q_weight = torch.randn(_Q_DIM, _HIDDEN_SIZE)
        k_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE)
        v_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE)

        packed = pack_qkv_weights(q_weight, k_weight, v_weight)

        assert torch.equal(packed[:_Q_DIM], q_weight)
        assert torch.equal(packed[_Q_DIM:_Q_DIM + _KV_DIM], k_weight)
        assert torch.equal(packed[_Q_DIM + _KV_DIM:], v_weight)

    def test_incompatible_hidden_size_raises(self) -> None:
        """Raises ValueError when hidden dimensions don't match."""
        q_weight = torch.randn(_Q_DIM, _HIDDEN_SIZE)
        k_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE + 1)
        v_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE)

        with pytest.raises(ValueError, match="hidden_size"):
            pack_qkv_weights(q_weight, k_weight, v_weight)

    def test_preserves_dtype(self) -> None:
        """Packed weight preserves the input dtype (bf16)."""
        q_weight = torch.randn(_Q_DIM, _HIDDEN_SIZE, dtype=torch.bfloat16)
        k_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE, dtype=torch.bfloat16)
        v_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE, dtype=torch.bfloat16)

        packed = pack_qkv_weights(q_weight, k_weight, v_weight)

        assert packed.dtype == torch.bfloat16


class TestUnpackQkvOutput:
    """Tests for unpack_qkv_output."""

    def test_roundtrip_equivalence(self) -> None:
        """Packed matmul + unpack produces same result as separate matmuls."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        q_weight = torch.randn(_Q_DIM, _HIDDEN_SIZE, dtype=torch.bfloat16)
        k_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE, dtype=torch.bfloat16)
        v_weight = torch.randn(_KV_DIM, _HIDDEN_SIZE, dtype=torch.bfloat16)
        input_tensor = torch.randn(1, 1, _HIDDEN_SIZE, dtype=torch.bfloat16)

        # Separate projections
        q_separate = torch.nn.functional.linear(input_tensor, q_weight)
        k_separate = torch.nn.functional.linear(input_tensor, k_weight)
        v_separate = torch.nn.functional.linear(input_tensor, v_weight)

        # Packed projection
        packed_weight = pack_qkv_weights(q_weight, k_weight, v_weight)
        packed_output = torch.nn.functional.linear(input_tensor, packed_weight)
        q_packed, k_packed, v_packed = unpack_qkv_output(
            packed_output, _Q_DIM, _KV_DIM, _KV_DIM
        )

        assert torch.equal(q_separate, q_packed)
        assert torch.equal(k_separate, k_packed)
        assert torch.equal(v_separate, v_packed)

    def test_output_shapes(self) -> None:
        """Unpacked outputs have correct individual shapes."""
        packed_output = torch.randn(1, 1, _Q_DIM + _KV_DIM + _KV_DIM)
        q, k, v = unpack_qkv_output(packed_output, _Q_DIM, _KV_DIM, _KV_DIM)

        assert q.shape == (1, 1, _Q_DIM)
        assert k.shape == (1, 1, _KV_DIM)
        assert v.shape == (1, 1, _KV_DIM)


class TestPackGateUpWeights:
    """Tests for pack_gate_up_weights."""

    def test_output_shape_qwen35(self) -> None:
        """Packed gate/up weight has shape [19456, 2560] for Qwen3.5-4B."""
        gate_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
        up_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE)

        packed = pack_gate_up_weights(gate_weight, up_weight)

        assert packed.shape == (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
        assert packed.shape == (19456, 2560)

    def test_content_is_concatenation(self) -> None:
        """Packed weight is the exact concatenation of gate and up along dim=0."""
        gate_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
        up_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE)

        packed = pack_gate_up_weights(gate_weight, up_weight)

        assert torch.equal(packed[:_INTERMEDIATE_SIZE], gate_weight)
        assert torch.equal(packed[_INTERMEDIATE_SIZE:], up_weight)

    def test_incompatible_shapes_raises(self) -> None:
        """Raises ValueError when gate and up shapes don't match."""
        gate_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
        up_weight = torch.randn(_INTERMEDIATE_SIZE + 1, _HIDDEN_SIZE)

        with pytest.raises(ValueError, match="same shape"):
            pack_gate_up_weights(gate_weight, up_weight)

    def test_preserves_dtype(self) -> None:
        """Packed weight preserves the input dtype (bf16)."""
        gate_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, dtype=torch.bfloat16)
        up_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, dtype=torch.bfloat16)

        packed = pack_gate_up_weights(gate_weight, up_weight)

        assert packed.dtype == torch.bfloat16


class TestUnpackGateUpOutput:
    """Tests for unpack_gate_up_output."""

    def test_roundtrip_equivalence(self) -> None:
        """Packed matmul + unpack produces same result as separate matmuls."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        gate_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, dtype=torch.bfloat16)
        up_weight = torch.randn(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, dtype=torch.bfloat16)
        input_tensor = torch.randn(1, 1, _HIDDEN_SIZE, dtype=torch.bfloat16)

        # Separate projections
        gate_separate = torch.nn.functional.linear(input_tensor, gate_weight)
        up_separate = torch.nn.functional.linear(input_tensor, up_weight)

        # Packed projection
        packed_weight = pack_gate_up_weights(gate_weight, up_weight)
        packed_output = torch.nn.functional.linear(input_tensor, packed_weight)
        gate_packed, up_packed = unpack_gate_up_output(packed_output, _INTERMEDIATE_SIZE)

        assert torch.equal(gate_separate, gate_packed)
        assert torch.equal(up_separate, up_packed)

    def test_output_shapes(self) -> None:
        """Unpacked outputs have correct individual shapes."""
        packed_output = torch.randn(1, 1, 2 * _INTERMEDIATE_SIZE)
        gate, up = unpack_gate_up_output(packed_output, _INTERMEDIATE_SIZE)

        assert gate.shape == (1, 1, _INTERMEDIATE_SIZE)
        assert up.shape == (1, 1, _INTERMEDIATE_SIZE)


class TestPackLayerQkvFromModel:
    """Tests for pack_layer_qkv_from_model."""

    def test_detects_qwen35_layer_structure(self) -> None:
        """Detects and packs Q/K/V from a Qwen3.5-style layer."""
        # Build a mock layer with the expected structure
        layer = torch.nn.Module()
        self_attn = torch.nn.Module()
        self_attn.q_proj = torch.nn.Linear(_HIDDEN_SIZE, _Q_DIM, bias=False)
        self_attn.k_proj = torch.nn.Linear(_HIDDEN_SIZE, _KV_DIM, bias=False)
        self_attn.v_proj = torch.nn.Linear(_HIDDEN_SIZE, _KV_DIM, bias=False)
        layer.self_attn = self_attn

        packed = pack_layer_qkv_from_model(layer)

        assert packed is not None
        assert packed.shape == (_Q_DIM + _KV_DIM + _KV_DIM, _HIDDEN_SIZE)

    def test_returns_none_without_self_attn(self) -> None:
        """Returns None when layer has no self_attn submodule."""
        layer = torch.nn.Module()
        assert pack_layer_qkv_from_model(layer) is None

    def test_returns_none_with_missing_projections(self) -> None:
        """Returns None when self_attn is missing a projection."""
        layer = torch.nn.Module()
        self_attn = torch.nn.Module()
        self_attn.q_proj = torch.nn.Linear(_HIDDEN_SIZE, _Q_DIM, bias=False)
        self_attn.k_proj = torch.nn.Linear(_HIDDEN_SIZE, _KV_DIM, bias=False)
        # Missing v_proj
        layer.self_attn = self_attn

        assert pack_layer_qkv_from_model(layer) is None


class TestPackLayerGateUpFromModel:
    """Tests for pack_layer_gate_up_from_model."""

    def test_detects_qwen35_layer_structure(self) -> None:
        """Detects and packs gate/up from a Qwen3.5-style layer."""
        layer = torch.nn.Module()
        mlp = torch.nn.Module()
        mlp.gate_proj = torch.nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)
        mlp.up_proj = torch.nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)
        layer.mlp = mlp

        packed = pack_layer_gate_up_from_model(layer)

        assert packed is not None
        assert packed.shape == (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)

    def test_returns_none_without_mlp(self) -> None:
        """Returns None when layer has no mlp submodule."""
        layer = torch.nn.Module()
        assert pack_layer_gate_up_from_model(layer) is None

    def test_returns_none_with_missing_projections(self) -> None:
        """Returns None when mlp is missing a projection."""
        layer = torch.nn.Module()
        mlp = torch.nn.Module()
        mlp.gate_proj = torch.nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)
        # Missing up_proj
        layer.mlp = mlp

        assert pack_layer_gate_up_from_model(layer) is None

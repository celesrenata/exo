"""Bug condition exploration test for native linear attention layer device mismatch.

This test surfaces counterexamples demonstrating that native linear attention layers
remain on CPU after TensorParallelShard initialization. The test encodes the EXPECTED
behavior — it will validate the fix when it passes after implementation.

**Validates: Requirements 1.1, 2.1, 2.2, 2.3**

Bug Condition: native_linear_attn_layers[layer_idx].parameters_device != self.device

NOTE: Since we cannot use XPU in unit tests, we verify the bug by:
1. Creating the model on CPU
2. Marking native layer parameters with a sentinel (moving them to a known state)
3. Verifying that TPS __init__ does NOT call .to(device) on native layers
   (the structural defect that causes the bug on real hardware)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from unittest.mock import patch, MagicMock

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TPShardConfig,
    TensorParallelShard,
)


# --- Mock HuggingFace Model Components ---


class MockGatedDeltaNet(nn.Module):
    """Mock Qwen3_5GatedDeltaNet module with nn.Linear submodules.

    Simulates the native HuggingFace linear attention layer with the same
    parameter structure as the real Qwen3_5GatedDeltaNet.
    """

    def __init__(self, hidden_size: int = 2560) -> None:
        super().__init__()
        # Key projections matching real Qwen3_5GatedDeltaNet structure
        qkv_dim = hidden_size  # Simplified: q_dim + k_dim + v_dim
        self.in_proj_qkv = nn.Linear(hidden_size, qkv_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, 32, bias=False)  # num_v_heads
        self.in_proj_b = nn.Linear(hidden_size, 32, bias=False)  # num_v_heads
        self.in_proj_z = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=qkv_dim,
            out_channels=qkv_dim,
            kernel_size=4,
            groups=qkv_dim,
            padding=0,
        )
        self.out_proj = nn.Linear(hidden_size // 2, hidden_size, bias=False)
        self.norm = nn.RMSNorm(hidden_size // 2)
        # Learnable parameters
        self.A_log = nn.Parameter(torch.randn(32))
        self.dt_bias = nn.Parameter(torch.randn(32))
        # Track whether .to() was called on this module
        self._device_move_count = 0

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Override to track device moves."""
        self._device_move_count += 1
        return super().to(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simplified forward that exercises the linear projections."""
        batch, seq_len, hidden = hidden_states.shape
        # Exercise the projections to trigger device mismatch
        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        # Return a tensor of the correct shape
        output = self.out_proj(z)
        return output


class MockDecoderLayer(nn.Module):
    """Mock transformer decoder layer with optional linear_attn attribute."""

    def __init__(self, has_linear_attn: bool = True, hidden_size: int = 2560) -> None:
        super().__init__()
        if has_linear_attn:
            self.linear_attn = MockGatedDeltaNet(hidden_size)
        self.self_attn = nn.Module()  # Placeholder
        self.mlp = nn.Module()  # Placeholder
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)


class MockLanguageModel(nn.Module):
    """Mock language model with layers attribute."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 2560) -> None:
        super().__init__()
        # Create layers: 75% linear attention, 25% full attention (every 4th)
        layer_list = []
        for idx in range(num_layers):
            has_linear = (idx % 4 != 3)  # Every 4th layer is full attention
            layer_list.append(MockDecoderLayer(has_linear_attn=has_linear, hidden_size=hidden_size))
        self.layers = nn.ModuleList(layer_list)
        self.norm = nn.RMSNorm(hidden_size)
        self.embed_tokens = nn.Embedding(1000, hidden_size)


class MockInnerModel(nn.Module):
    """Mock model.model with language_model attribute."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 2560) -> None:
        super().__init__()
        self.language_model = MockLanguageModel(num_layers, hidden_size)


class MockHFModel(nn.Module):
    """Mock HuggingFace model with model.model.language_model.layers structure."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 2560) -> None:
        super().__init__()
        self.model = MockInnerModel(num_layers, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 1000, bias=False)


def create_tps(num_layers: int = 4, hidden_size: int = 2560) -> TensorParallelShard:
    """Create a TensorParallelShard with a mock model for testing."""
    model = MockHFModel(num_layers=num_layers, hidden_size=hidden_size)
    config = TPShardConfig(
        rank=0,
        world_size=2,
        hidden_size=hidden_size,
        num_attention_heads=32,
        head_dim=80,  # 2560 / 32
        intermediate_size=hidden_size * 4,
        num_key_value_heads=8,
    )
    return TensorParallelShard(model, config, device="cpu")


# --- Property-Based Tests ---


@given(layer_idx=st.integers(min_value=0, max_value=2))
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_native_layer_explicitly_moved_to_device(layer_idx: int) -> None:
    """Property 1: Bug Condition - Native Linear Attention Layer Device Mismatch.

    **Validates: Requirements 1.1, 2.1, 2.2, 2.3**

    For all layer_idx in _native_linear_attn_layers, the native layer module
    SHALL have been explicitly moved to self.device via .to(device) during
    TPS initialization.

    The bug is that _extract_native_linear_attn_layers() saves references to
    nn.Module instances BEFORE shard_weights() runs, and after shard_weights()
    completes, no code calls .to(self.device) on the native layer modules.
    On real XPU hardware, this means native layer parameters stay on CPU while
    self.device is "xpu:0".

    This test verifies the structural fix: .to(device) MUST be called on each
    native layer after shard_weights() completes.
    """
    tps = create_tps()

    # The mock model has linear_attn at layers 0, 1, 2 (not 3)
    assert layer_idx in tps._native_linear_attn_layers, (
        f"Layer {layer_idx} should be in _native_linear_attn_layers but found: "
        f"{list(tps._native_linear_attn_layers.keys())}"
    )

    native_layer = tps._native_linear_attn_layers[layer_idx]

    # Property assertion: The native layer MUST have had .to() called on it
    # during TPS initialization (after shard_weights completes).
    # In the unfixed code, _device_move_count will be 0 because no code
    # calls .to(device) on the extracted native layers.
    assert hasattr(native_layer, '_device_move_count'), (
        f"Native layer at index {layer_idx} is not a MockGatedDeltaNet"
    )
    assert native_layer._device_move_count > 0, (
        f"Bug confirmed: layer {layer_idx} native module was NEVER moved to "
        f"target device '{tps.device}'. _device_move_count = 0. "
        f"The _extract_native_linear_attn_layers() method saves references to "
        f"nn.Module instances before shard_weights() runs, but no subsequent "
        f"code calls .to(self.device) on these modules. On XPU hardware, this "
        f"means native layer parameters remain on CPU while hidden_states are "
        f"on XPU, causing device mismatch errors."
    )


@given(
    layer_idx=st.integers(min_value=0, max_value=2),
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=1, max_value=4),
)
@settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_native_layer_in_eval_mode_after_init(
    layer_idx: int, batch_size: int, seq_len: int
) -> None:
    """Property 1 (Secondary): Native layers SHALL be in eval mode after init.

    **Validates: Requirements 2.1, 2.2, 2.3**

    After TPS initialization, native linear attention layers SHALL be set to
    eval mode (.eval()) to disable dropout and ensure deterministic inference.

    Additionally, calling _forward_linear_attn_layer(hidden_states, layer_idx)
    SHALL produce output of shape (batch, seq_len, hidden_size) on the correct
    device without raising RuntimeError.
    """
    hidden_size = 2560
    tps = create_tps(num_layers=4, hidden_size=hidden_size)

    assert layer_idx in tps._native_linear_attn_layers

    native_layer = tps._native_linear_attn_layers[layer_idx]

    # Property: native layer SHALL be in eval mode after TPS init
    assert not native_layer.training, (
        f"Bug confirmed: layer {layer_idx} native module is still in training mode. "
        f"Native layers must be set to eval mode (.eval()) during TPS initialization "
        f"to disable dropout and ensure deterministic inference behavior."
    )

    # Secondary: forward pass SHALL produce valid output
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=tps.device)
    output = tps._forward_linear_attn_layer(hidden_states, layer_idx)

    assert output.shape == (batch_size, seq_len, hidden_size), (
        f"Expected output shape ({batch_size}, {seq_len}, {hidden_size}), "
        f"got {output.shape}"
    )
    assert output.device == torch.device(tps.device), (
        f"Expected output on {tps.device}, got {output.device}"
    )

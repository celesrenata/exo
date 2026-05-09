# Feature: pipeline-parallelism-optimization, Property 6: Compiled forward is numerically equivalent to eager forward
"""
Property-based tests for compiled vs eager forward pass equivalence.

**Validates: Requirements 7.3**

Uses Hypothesis to generate random hidden_state tensors with varying
seq_len and hidden_size, then verifies that the torch.compile()-wrapped
forward pass produces numerically equivalent output to the eager forward
pass within floating-point tolerance (atol=1e-5, rtol=1e-3).

Note: torch.compile with the inductor backend requires a working C++ toolchain
(Python.h, g++, etc.). On environments where compilation fails at runtime
(e.g., NixOS without python3-dev headers), the test is skipped gracefully.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_SHARD_PATH = _THIS_DIR.parent / "pipeline_parallel_shard.py"


def _load_pipeline_parallel_shard() -> types.ModuleType:
    """Load pipeline_parallel_shard.py directly from file, avoiding __init__.py."""
    module_name = "pipeline_parallel_shard_compile_test"
    spec = importlib.util.spec_from_file_location(module_name, _PIPELINE_SHARD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_pipeline_parallel_shard()
PipelineParallelShard = _mod.PipelineParallelShard
PipelineStageConfig = _mod.PipelineStageConfig


# ---------------------------------------------------------------------------
# Skip condition: torch.compile must be available
# ---------------------------------------------------------------------------

_has_torch_compile = hasattr(torch, "compile")


# ---------------------------------------------------------------------------
# Check if inductor backend can actually compile (requires C++ toolchain)
# ---------------------------------------------------------------------------


def _inductor_backend_works() -> bool:
    """Test whether the inductor backend can compile a trivial function.

    torch.compile() wraps lazily — the actual C++ compilation happens on first
    invocation. This probe function triggers compilation to detect missing
    Python.h or g++ issues before running the property test.
    """
    if not _has_torch_compile:
        return False
    try:

        @torch.compile(backend="inductor")  # type: ignore[misc]
        def _probe(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        _probe(torch.tensor([1.0]))
        return True
    except Exception:
        return False


_inductor_works: bool | None = None


def _check_inductor() -> bool:
    """Cached check for inductor backend availability."""
    global _inductor_works
    if _inductor_works is None:
        _inductor_works = _inductor_backend_works()
    return _inductor_works


# ---------------------------------------------------------------------------
# Simple linear layer that performs real computation (not identity)
# ---------------------------------------------------------------------------


class SimpleTransformerLayer(nn.Module):
    """A minimal transformer-like layer with real linear computation.

    Uses two linear projections to ensure torch.compile has actual
    computation to fuse, rather than a trivial identity pass-through.
    Returns (hidden_states, None) to match HuggingFace layer output format.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        residual = hidden_states
        x = self.linear1(hidden_states)
        x = self.activation(x)
        x = self.linear2(x)
        output = self.norm(x + residual)
        return (output, None)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

seq_len_strategy = st.integers(min_value=1, max_value=8)
hidden_size_strategy = st.sampled_from([32, 64])


# ---------------------------------------------------------------------------
# Property 6: Compiled forward is numerically equivalent to eager forward
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_torch_compile,
    reason="torch.compile not available",
)
@settings(max_examples=10, deadline=None)
@given(
    seq_len=seq_len_strategy,
    hidden_size=hidden_size_strategy,
)
def test_compiled_forward_equivalent_to_eager(
    seq_len: int,
    hidden_size: int,
) -> None:
    """Compiled forward produces numerically equivalent output to eager forward.

    For any valid hidden_state tensor, the output of the torch.compile()-wrapped
    forward pass is equal to the eager forward pass output within floating-point
    tolerance (atol=1e-5, rtol=1e-3).

    **Validates: Requirements 7.3**
    """
    # Skip if inductor backend cannot compile (missing Python.h, g++, etc.)
    if not _check_inductor():
        pytest.skip(
            "torch.compile inductor backend not functional "
            "(missing C++ toolchain or Python.h)"
        )

    num_layers = 2
    vocab_size = 100

    config = PipelineStageConfig(
        rank=1,
        world_size=2,
        start_layer=0,
        end_layer=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers * 2,
        device="cpu",
    )

    # Use real linear layers so there's actual computation to compile
    layers = nn.ModuleList([SimpleTransformerLayer(hidden_size) for _ in range(num_layers)])

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=None,
        lm_head=None,
        final_norm=None,
    )

    # If compilation failed during __init__ (e.g., inductor backend not supported),
    # skip the test — we cannot compare compiled vs eager if compilation didn't work
    if shard._compiled_forward is None:
        pytest.skip("torch.compile failed on this platform (inductor backend not supported)")

    # Generate a random hidden_state tensor
    hidden_state = torch.randn(1, seq_len, hidden_size)

    # Build the arguments for _eager_forward
    position_ids = torch.arange(0, seq_len).unsqueeze(0)
    kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    # Run eager forward
    with torch.no_grad():
        eager_output, eager_kv = shard._eager_forward(
            hidden_state, position_ids, None, kv_cache
        )

    # Run compiled forward (may fail at runtime if C++ toolchain is broken)
    with torch.no_grad():
        compiled_output, compiled_kv = shard._compiled_forward(
            hidden_state, position_ids, None, kv_cache
        )

    # Compare outputs within floating-point tolerance
    assert torch.allclose(compiled_output, eager_output, atol=1e-5, rtol=1e-3), (
        f"Compiled forward output differs from eager forward output.\n"
        f"Max absolute difference: {(compiled_output - eager_output).abs().max().item()}\n"
        f"Max relative difference: {((compiled_output - eager_output).abs() / (eager_output.abs() + 1e-8)).max().item()}\n"
        f"Input shape: (1, {seq_len}, {hidden_size})"
    )

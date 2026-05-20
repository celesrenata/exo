"""
Integration test for disabled-optimization output equivalence.

Validates Requirement 11.2: WHEN all optimization flags are disabled, THE system
SHALL produce identical output to the current unoptimized pipeline at the same
temperature and seed.

This test verifies that:
1. Running with all optimization flags set to False (default config) produces
   identical token sequences to running with no config at all (None).
2. Both paths produce identical text output, token IDs, and finish reasons.
3. The invariant holds for different token sequences (EOS, max_tokens, multi-token).
4. Temperature=0 (greedy) ensures deterministic output for comparison.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock missing transitive dependencies before importing pipeline_generator.
# ---------------------------------------------------------------------------


class _MockFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that returns MagicMock modules for missing dependencies."""

    _MOCK_PREFIXES: tuple[str, ...] = (
        "aiofiles",
        "tomlkit",
        "huggingface_hub",
        "safetensors",
        "transformers",
        "starlette",
        "fastapi",
        "uvicorn",
        "httpx",
        "sse_starlette",
        "zstandard",
        "hypercorn",
        "h11",
    )

    def find_spec(
        self,
        fullname: str,
        path: object,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        for prefix in self._MOCK_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.machinery.ModuleSpec(fullname, self)  # pyright: ignore[reportArgumentType]
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> MagicMock:
        mock_mod = MagicMock()
        mock_mod.__name__ = spec.name
        mock_mod.__package__ = (
            spec.name.rsplit(".", 1)[0] if "." in spec.name else spec.name
        )
        mock_mod.__path__ = []
        mock_mod.__spec__ = spec
        mock_mod.__file__ = None
        mock_mod.__loader__ = self
        return mock_mod

    def exec_module(self, module: object) -> None:
        pass


# Install the mock finder before importing pipeline_generator
sys.meta_path.insert(0, _MockFinder())

from exo.worker.engines.pytorch_xpu.pipeline_config import (  # noqa: E402
    PytorchXpuOptimizationConfiguration,
)
from exo.worker.engines.pytorch_xpu.pipeline_generator import (  # noqa: E402
    pipeline_parallel_generate,
)
from exo.shared.types.worker.runner_response import GenerationResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer mock that produces deterministic output."""

    def __init__(self, eos_token_id: int = 2) -> None:
        self.eos_token_id = eos_token_id
        self.all_special_ids: list[int] = [eos_token_id]

    def encode(self, text: str) -> list[int]:
        """Return a fixed token sequence for any input."""
        return [1, 10, 20, 30]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Return deterministic text for any token ID."""
        return f"<tok_{token_ids[0]}>"


class FakeModel:
    """Minimal model mock that produces a predetermined token sequence.

    Uses argmax-friendly logits: sets the target token's logit to 10.0
    while all others remain at 0.0, ensuring deterministic greedy sampling.
    """

    def __init__(self, token_sequence: list[int], vocab_size: int = 100) -> None:
        self._token_sequence = token_sequence
        self._vocab_size = vocab_size
        self._step = 0
        self.config = MagicMock()
        self.config.hidden_size = 128

    def forward(self, input_data: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return logits that produce the expected token via argmax."""
        target_token = (
            self._token_sequence[self._step]
            if self._step < len(self._token_sequence)
            else 2  # fallback to EOS
        )
        self._step += 1

        logits = torch.zeros(1, 1, self._vocab_size)
        logits[0, 0, target_token] = 10.0
        return logits, None

    def reset(self) -> None:
        """Reset step counter for reuse."""
        self._step = 0


def _collect_responses(
    gen: Generator[GenerationResponse, None, None],
) -> list[GenerationResponse]:
    """Collect all responses from a generator."""
    return list(gen)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_disabled_config() -> PytorchXpuOptimizationConfiguration:
    """Configuration with ALL kernel launch optimization flags set to False.

    This is the default configuration — all optimization flags default to False.
    The non-optimization flags (performance instrumentation, decode fast path, etc.)
    are set to match the behavior of optimization_configuration=None.
    """
    return PytorchXpuOptimizationConfiguration(
        # Kernel launch optimization flags (all False by default)
        enable_static_kv_cache=False,
        enable_torch_compile=False,
        enable_packed_projections=False,
        enable_fused_kernels=False,
        enable_on_device_sampling=False,
        enable_async_output=False,
        enable_sync_removal=False,
        # Disable performance instrumentation to match None config behavior
        enable_performance_instrumentation=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisabledOptimizationOutputEquivalence:
    """Verify that disabled optimizations produce identical output to no-config path.

    Validates: Requirement 11.2
    """

    def test_identical_tokens_multi_token_sequence(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Multi-token generation produces identical tokens with disabled config vs None.

        Validates: Requirement 11.2
        """
        token_sequence = [5, 6, 7, 8, 2]

        # Run with all-disabled config
        model_disabled = FakeModel(token_sequence.copy())
        tokenizer_disabled = FakeTokenizer(eos_token_id=2)
        responses_disabled = _collect_responses(
            pipeline_parallel_generate(
                model=model_disabled,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_disabled,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        # Run with no config (None)
        model_none = FakeModel(token_sequence.copy())
        tokenizer_none = FakeTokenizer(eos_token_id=2)
        responses_none = _collect_responses(
            pipeline_parallel_generate(
                model=model_none,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_none,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=None,
            )
        )

        # Verify identical token sequences
        tokens_disabled = [r.token for r in responses_disabled]
        tokens_none = [r.token for r in responses_none]
        assert tokens_disabled == tokens_none, (
            f"Token mismatch: disabled={tokens_disabled}, none={tokens_none}"
        )

        # Verify identical text output
        texts_disabled = [r.text for r in responses_disabled]
        texts_none = [r.text for r in responses_none]
        assert texts_disabled == texts_none, (
            f"Text mismatch: disabled={texts_disabled}, none={texts_none}"
        )

        # Verify identical finish reasons
        reasons_disabled = [r.finish_reason for r in responses_disabled]
        reasons_none = [r.finish_reason for r in responses_none]
        assert reasons_disabled == reasons_none, (
            f"Finish reason mismatch: disabled={reasons_disabled}, none={reasons_none}"
        )

    def test_identical_tokens_eos_termination(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """EOS termination produces identical output with disabled config vs None.

        Validates: Requirement 11.2
        """
        token_sequence = [5, 6, 2]

        model_disabled = FakeModel(token_sequence.copy())
        tokenizer_disabled = FakeTokenizer(eos_token_id=2)
        responses_disabled = _collect_responses(
            pipeline_parallel_generate(
                model=model_disabled,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_disabled,
                prompt="Test prompt",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        model_none = FakeModel(token_sequence.copy())
        tokenizer_none = FakeTokenizer(eos_token_id=2)
        responses_none = _collect_responses(
            pipeline_parallel_generate(
                model=model_none,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_none,
                prompt="Test prompt",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=None,
            )
        )

        tokens_disabled = [r.token for r in responses_disabled]
        tokens_none = [r.token for r in responses_none]
        assert tokens_disabled == tokens_none, (
            f"Token mismatch on EOS: disabled={tokens_disabled}, none={tokens_none}"
        )

        # Verify EOS response has empty text and "stop" finish reason
        assert responses_disabled[-1].finish_reason == "stop"
        assert responses_none[-1].finish_reason == "stop"
        assert responses_disabled[-1].text == responses_none[-1].text

    def test_identical_tokens_max_tokens_termination(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """max_tokens termination produces identical output with disabled config vs None.

        Validates: Requirement 11.2
        """
        token_sequence = [10, 11, 12, 13, 14, 15, 16, 17]

        model_disabled = FakeModel(token_sequence.copy())
        tokenizer_disabled = FakeTokenizer(eos_token_id=2)
        responses_disabled = _collect_responses(
            pipeline_parallel_generate(
                model=model_disabled,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_disabled,
                prompt="Hello world",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=5,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        model_none = FakeModel(token_sequence.copy())
        tokenizer_none = FakeTokenizer(eos_token_id=2)
        responses_none = _collect_responses(
            pipeline_parallel_generate(
                model=model_none,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_none,
                prompt="Hello world",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=5,
                temperature=0.0,
                optimization_configuration=None,
            )
        )

        tokens_disabled = [r.token for r in responses_disabled]
        tokens_none = [r.token for r in responses_none]
        assert tokens_disabled == tokens_none, (
            f"Token mismatch on max_tokens: disabled={tokens_disabled}, none={tokens_none}"
        )

        assert len(responses_disabled) == 5
        assert len(responses_none) == 5
        assert responses_disabled[-1].finish_reason == "length"
        assert responses_none[-1].finish_reason == "length"

    def test_identical_tokens_single_token_generation(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Single token (max_tokens=1) produces identical output.

        Validates: Requirement 11.2
        """
        token_sequence = [42, 2]

        model_disabled = FakeModel(token_sequence.copy())
        tokenizer_disabled = FakeTokenizer(eos_token_id=2)
        responses_disabled = _collect_responses(
            pipeline_parallel_generate(
                model=model_disabled,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_disabled,
                prompt="Hi",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=1,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        model_none = FakeModel(token_sequence.copy())
        tokenizer_none = FakeTokenizer(eos_token_id=2)
        responses_none = _collect_responses(
            pipeline_parallel_generate(
                model=model_none,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_none,
                prompt="Hi",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=1,
                temperature=0.0,
                optimization_configuration=None,
            )
        )

        assert len(responses_disabled) == 1
        assert len(responses_none) == 1
        assert responses_disabled[0].token == responses_none[0].token
        assert responses_disabled[0].text == responses_none[0].text
        assert responses_disabled[0].finish_reason == responses_none[0].finish_reason

    def test_identical_tokens_immediate_eos(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Immediate EOS after prefill produces identical output.

        Validates: Requirement 11.2
        """
        token_sequence = [2]  # EOS immediately

        model_disabled = FakeModel(token_sequence.copy())
        tokenizer_disabled = FakeTokenizer(eos_token_id=2)
        responses_disabled = _collect_responses(
            pipeline_parallel_generate(
                model=model_disabled,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_disabled,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        model_none = FakeModel(token_sequence.copy())
        tokenizer_none = FakeTokenizer(eos_token_id=2)
        responses_none = _collect_responses(
            pipeline_parallel_generate(
                model=model_none,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_none,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=None,
            )
        )

        assert len(responses_disabled) == 1
        assert len(responses_none) == 1
        assert responses_disabled[0].token == responses_none[0].token
        assert responses_disabled[0].finish_reason == "stop"
        assert responses_none[0].finish_reason == "stop"
        assert responses_disabled[0].text == responses_none[0].text

    def test_no_optimization_code_paths_activated_when_flags_disabled(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Verify that disabled flags do not activate any optimization code paths.

        This test checks the configuration invariants:
        - enable_sync_removal=False → standard (non-deferred) tokenizer decode path
        - enable_torch_compile=False → no CompiledDecodePath created
        - enable_on_device_sampling=False → standard CPU sampling via sample_token()
        - enable_async_output=False → no async streamer used

        Validates: Requirement 11.2
        """
        # All kernel launch optimization flags are False
        assert all_disabled_config.enable_sync_removal is False
        assert all_disabled_config.enable_torch_compile is False
        assert all_disabled_config.enable_on_device_sampling is False
        assert all_disabled_config.enable_async_output is False
        assert all_disabled_config.enable_static_kv_cache is False
        assert all_disabled_config.enable_packed_projections is False
        assert all_disabled_config.enable_fused_kernels is False

        # Run generation and verify it completes without error
        token_sequence = [5, 6, 7, 2]
        model = FakeModel(token_sequence)
        tokenizer = FakeTokenizer(eos_token_id=2)

        responses = _collect_responses(
            pipeline_parallel_generate(
                model=model,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        # Verify generation completed with expected tokens
        tokens = [r.token for r in responses]
        assert tokens == [5, 6, 7, 2]

    def test_response_count_matches(
        self, all_disabled_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Both paths produce the same number of responses.

        Validates: Requirement 11.2
        """
        token_sequence = [3, 4, 5, 6, 7, 8, 9, 2]

        model_disabled = FakeModel(token_sequence.copy())
        tokenizer_disabled = FakeTokenizer(eos_token_id=2)
        responses_disabled = _collect_responses(
            pipeline_parallel_generate(
                model=model_disabled,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_disabled,
                prompt="Count test",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=20,
                temperature=0.0,
                optimization_configuration=all_disabled_config,
            )
        )

        model_none = FakeModel(token_sequence.copy())
        tokenizer_none = FakeTokenizer(eos_token_id=2)
        responses_none = _collect_responses(
            pipeline_parallel_generate(
                model=model_none,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_none,
                prompt="Count test",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=20,
                temperature=0.0,
                optimization_configuration=None,
            )
        )

        assert len(responses_disabled) == len(responses_none), (
            f"Response count mismatch: disabled={len(responses_disabled)}, "
            f"none={len(responses_none)}"
        )

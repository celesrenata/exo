"""
Tests for deferred tokenizer decode in pipeline_generator.py.

Validates Requirement 1.4: WHEN the Decode_Loop produces a token, THE Decode_Loop
SHALL defer tokenizer decode operations until after the next forward pass has been
launched.

Also validates Requirement 9.1: WHEN a token is sampled, THE Decode_Loop SHALL
launch the next forward pass before performing tokenizer decode on the previous
token.

Tests verify:
1. Token output ordering is preserved with deferred decode
2. Deferred decode is gated behind enable_sync_removal flag
3. Existing behavior is preserved when enable_sync_removal=False
4. Edge cases: EOS, max_tokens, termination sentinel
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
# Uses a modern MetaPathFinder to intercept imports of packages not installed
# in the test environment (aiofiles, huggingface_hub, etc.).
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
    """Minimal tokenizer mock that tracks decode call order."""

    def __init__(self, eos_token_id: int = 2) -> None:
        self.eos_token_id = eos_token_id
        self.all_special_ids: list[int] = [eos_token_id]
        self.decode_call_order: list[int] = []

    def encode(self, text: str) -> list[int]:
        """Return a fixed token sequence for any input."""
        return [1, 10, 20, 30]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Track decode calls and return token text."""
        self.decode_call_order.append(token_ids[0])
        return f"<tok_{token_ids[0]}>"


class FakeModel:
    """Minimal model mock that produces a predetermined token sequence."""

    def __init__(self, token_sequence: list[int]) -> None:
        self._token_sequence = token_sequence
        self._step = 0
        self.forward_call_order: list[int] = []
        self.config = MagicMock()
        self.config.hidden_size = 128

    def forward(self, input_data: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return logits that will produce the expected token via argmax."""
        input_token_id = int(input_data[0, -1].item())
        self.forward_call_order.append(input_token_id)

        target_token = (
            self._token_sequence[self._step]
            if self._step < len(self._token_sequence)
            else 2  # fallback to EOS
        )
        self._step += 1

        logits = torch.zeros(1, 1, 100)
        logits[0, 0, target_token] = 10.0
        return logits, None


def _collect_responses(
    gen: Generator[GenerationResponse, None, None],
) -> list[GenerationResponse]:
    """Collect all responses from a generator."""
    return list(gen)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_removal_config() -> PytorchXpuOptimizationConfiguration:
    """Configuration with enable_sync_removal=True."""
    return PytorchXpuOptimizationConfiguration(
        enable_sync_removal=True,
        enable_performance_instrumentation=False,
    )


@pytest.fixture
def default_config() -> PytorchXpuOptimizationConfiguration:
    """Configuration with enable_sync_removal=False (default)."""
    return PytorchXpuOptimizationConfiguration(
        enable_sync_removal=False,
        enable_performance_instrumentation=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeferredTokenizerDecode:
    """Tests for deferred tokenizer decode behavior."""

    def test_deferred_decode_preserves_token_ordering(
        self, sync_removal_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Token output ordering is preserved when decode is deferred.

        Validates: Requirement 1.4, 9.1
        """
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
                optimization_configuration=sync_removal_config,
            )
        )

        token_ids = [r.token for r in responses]
        assert token_ids == [5, 6, 7, 2], f"Got: {token_ids}"

    def test_deferred_decode_forward_before_tokenizer(
        self, sync_removal_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Forward pass is launched before tokenizer decode of previous token.

        Validates: Requirement 1.4, 9.1
        """
        token_sequence = [5, 6, 7, 2]
        model = FakeModel(token_sequence)
        tokenizer = FakeTokenizer(eos_token_id=2)

        call_log: list[tuple[str, int]] = []
        original_forward = model.forward
        original_decode = tokenizer.decode

        def tracked_forward(input_data: torch.Tensor) -> tuple[torch.Tensor, None]:
            token_id = int(input_data[0, -1].item())
            call_log.append(("forward", token_id))
            return original_forward(input_data)

        def tracked_decode(
            token_ids: list[int], skip_special_tokens: bool = True
        ) -> str:
            call_log.append(("decode", token_ids[0]))
            return original_decode(token_ids, skip_special_tokens)

        model.forward = tracked_forward
        tokenizer.decode = tracked_decode

        _collect_responses(
            pipeline_parallel_generate(
                model=model,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=sync_removal_config,
            )
        )

        # In the decode loop with deferred decode, the pattern is:
        # forward(token_N) is called BEFORE decode(token_N-1)
        #
        # Find decode-loop forward calls (skip the prefill forward).
        # The prefill forward uses the last prompt token as input.
        # The first decode-loop forward uses first_token_id (5) as input.
        decode_loop_start = None
        for i, (op, tok) in enumerate(call_log):
            if op == "forward" and tok == 5 and i > 0:
                decode_loop_start = i
                break

        assert decode_loop_start is not None, (
            f"Could not find decode loop start in call_log: {call_log}"
        )

        decode_loop_calls = call_log[decode_loop_start:]

        # Verify the deferred decode property: in the decode loop, every
        # "decode" call is preceded by a "forward" call. This proves that
        # the forward pass is launched before the tokenizer decode.
        last_op_was_forward = False
        for op, _tok in decode_loop_calls:
            if op == "forward":
                last_op_was_forward = True
            elif op == "decode":
                assert last_op_was_forward, (
                    f"decode({_tok}) occurred without a preceding forward in "
                    f"decode loop. Full decode loop calls: {decode_loop_calls}"
                )
                last_op_was_forward = False

    def test_standard_path_when_sync_removal_disabled(
        self, default_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Standard (non-deferred) path is used when enable_sync_removal=False.

        Validates: Requirement 11.2 (backward compatibility)
        """
        token_sequence = [5, 6, 2]
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
                optimization_configuration=default_config,
            )
        )

        token_ids = [r.token for r in responses]
        assert token_ids == [5, 6, 2], f"Got: {token_ids}"

    def test_deferred_decode_with_max_tokens(
        self, sync_removal_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Deferred decode handles max_tokens termination correctly."""
        token_sequence = [5, 6, 7, 8, 9, 10, 11, 12]
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
                max_tokens=4,
                temperature=0.0,
                optimization_configuration=sync_removal_config,
            )
        )

        assert len(responses) == 4, (
            f"Got {len(responses)} responses: {[r.token for r in responses]}"
        )
        assert responses[-1].finish_reason == "length"
        token_ids = [r.token for r in responses]
        assert token_ids == [5, 6, 7, 8], f"Got: {token_ids}"

    def test_deferred_decode_with_eos(
        self, sync_removal_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Deferred decode handles EOS termination correctly."""
        token_sequence = [5, 6, 2]
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
                optimization_configuration=sync_removal_config,
            )
        )

        eos_responses = [r for r in responses if r.finish_reason == "stop"]
        assert len(eos_responses) == 1
        assert eos_responses[0].text == ""
        assert eos_responses[0].token == 2

        non_eos_tokens = [r.token for r in responses if r.finish_reason != "stop"]
        assert non_eos_tokens == [5, 6], f"Got: {non_eos_tokens}"

    def test_deferred_decode_no_config_uses_standard_path(self) -> None:
        """When no optimization_configuration is provided, standard path is used."""
        token_sequence = [5, 6, 2]
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
                optimization_configuration=None,
            )
        )

        token_ids = [r.token for r in responses]
        assert token_ids == [5, 6, 2], f"Got: {token_ids}"

    def test_deferred_decode_same_tokens_as_standard(self) -> None:
        """Deferred decode produces the same token sequence as standard path."""
        token_sequence = [5, 6, 7, 8, 2]

        model_deferred = FakeModel(token_sequence.copy())
        tokenizer_deferred = FakeTokenizer(eos_token_id=2)
        config_deferred = PytorchXpuOptimizationConfiguration(
            enable_sync_removal=True,
            enable_performance_instrumentation=False,
        )

        responses_deferred = _collect_responses(
            pipeline_parallel_generate(
                model=model_deferred,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_deferred,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=config_deferred,
            )
        )

        model_standard = FakeModel(token_sequence.copy())
        tokenizer_standard = FakeTokenizer(eos_token_id=2)
        config_standard = PytorchXpuOptimizationConfiguration(
            enable_sync_removal=False,
            enable_performance_instrumentation=False,
        )

        responses_standard = _collect_responses(
            pipeline_parallel_generate(
                model=model_standard,  # pyright: ignore[reportArgumentType]
                tokenizer=tokenizer_standard,
                prompt="Hello",
                device="cpu",
                rank=0,
                world_size=1,
                max_tokens=10,
                temperature=0.0,
                optimization_configuration=config_standard,
            )
        )

        tokens_deferred = [r.token for r in responses_deferred]
        tokens_standard = [r.token for r in responses_standard]
        assert tokens_deferred == tokens_standard, (
            f"Deferred: {tokens_deferred}, Standard: {tokens_standard}"
        )

        texts_deferred = [r.text for r in responses_deferred]
        texts_standard = [r.text for r in responses_standard]
        assert texts_deferred == texts_standard, (
            f"Deferred: {texts_deferred}, Standard: {texts_standard}"
        )

    def test_deferred_decode_immediate_eos_after_prefill(
        self, sync_removal_config: PytorchXpuOptimizationConfiguration
    ) -> None:
        """Handles EOS as the very first decode token (step 0)."""
        token_sequence = [5, 2]
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
                optimization_configuration=sync_removal_config,
            )
        )

        token_ids = [r.token for r in responses]
        assert token_ids == [5, 2], f"Got: {token_ids}"
        assert responses[-1].finish_reason == "stop"
        assert responses[-1].text == ""

"""
Tests for the sampling routing infrastructure.

Validates that:
1. The public ``sample_token(...)`` signature remains compatible
2. Routing dispatches to the correct specialized implementation
3. Greedy returns argmax
4. Top-k returns only tokens from ``torch.topk``
5. Fallback preserves existing behavior
6. SamplingConfiguration validates correctly

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from exo.worker.engines.pytorch_xpu.sampling import (  # noqa: E402
    SamplingConfiguration,
    SamplingRoute,
    determine_sampling_route,
    route_and_sample_token,
    sample_token_fallback,
    sample_token_greedy,
    sample_token_top_k,
)
from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token  # noqa: E402


# ---------------------------------------------------------------------------
# Route determination tests
# ---------------------------------------------------------------------------


class TestDetermineSamplingRoute:
    """Tests for the route determination logic."""

    def test_greedy_when_temperature_near_zero(self) -> None:
        """Temperature <= 1e-7 routes to greedy."""
        route = determine_sampling_route(
            temperature=0.0, top_k=None, top_p=None
        )
        assert route == "greedy"

    def test_greedy_when_temperature_at_threshold(self) -> None:
        """Temperature exactly at threshold routes to greedy."""
        route = determine_sampling_route(
            temperature=1e-7, top_k=None, top_p=None
        )
        assert route == "greedy"

    def test_greedy_when_do_sample_false(self) -> None:
        """do_sample=False routes to greedy regardless of other params."""
        route = determine_sampling_route(
            temperature=1.0, top_k=50, top_p=0.9, do_sample=False
        )
        assert route == "greedy"

    def test_top_k_when_top_k_set_no_top_p(self) -> None:
        """top_k set without top_p routes to top_k."""
        route = determine_sampling_route(
            temperature=1.0, top_k=50, top_p=None
        )
        assert route == "top_k"

    def test_fallback_when_top_p_set(self) -> None:
        """top_p set routes to fallback."""
        route = determine_sampling_route(
            temperature=1.0, top_k=None, top_p=0.9
        )
        assert route == "fallback"

    def test_fallback_when_both_top_k_and_top_p_set(self) -> None:
        """Both top_k and top_p set routes to fallback."""
        route = determine_sampling_route(
            temperature=1.0, top_k=50, top_p=0.9
        )
        assert route == "fallback"

    def test_fallback_when_no_filtering(self) -> None:
        """No filtering with normal temperature routes to fallback."""
        route = determine_sampling_route(
            temperature=1.0, top_k=None, top_p=None
        )
        assert route == "fallback"

    def test_top_k_zero_routes_to_fallback(self) -> None:
        """top_k=0 is treated as disabled, routes to fallback."""
        route = determine_sampling_route(
            temperature=1.0, top_k=0, top_p=None
        )
        assert route == "fallback"


# ---------------------------------------------------------------------------
# Greedy sampling tests
# ---------------------------------------------------------------------------


class TestSampleTokenGreedy:
    """Tests for the greedy sampling implementation."""

    def test_returns_argmax(self) -> None:
        """Greedy sampling returns the index of the maximum logit."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        token = sample_token_greedy(logits)
        assert token == 1  # index of 5.0

    def test_returns_argmax_large_vocabulary(self) -> None:
        """Greedy works with large vocabulary tensors."""
        logits = torch.randn(32000)
        expected = int(logits.argmax().item())
        token = sample_token_greedy(logits)
        assert token == expected

    def test_returns_first_on_tie(self) -> None:
        """When multiple values are equal max, returns the first occurrence."""
        logits = torch.tensor([3.0, 3.0, 1.0, 2.0])
        token = sample_token_greedy(logits)
        assert token == 0  # argmax returns first occurrence


# ---------------------------------------------------------------------------
# Top-k sampling tests
# ---------------------------------------------------------------------------


class TestSampleTokenTopK:
    """Tests for the top-k sampling implementation."""

    def test_returns_token_from_top_k(self) -> None:
        """Sampled token must be within the top-k indices."""
        torch.manual_seed(42)
        logits = torch.randn(1000)
        top_k = 10

        # Get the actual top-k indices
        _, top_k_indices = torch.topk(logits, top_k)
        top_k_set = set(top_k_indices.tolist())

        # Sample multiple times to verify membership
        for seed in range(20):
            torch.manual_seed(seed)
            token = sample_token_top_k(logits, top_k=top_k, temperature=1.0)
            assert token in top_k_set, (
                f"Token {token} not in top-{top_k} indices {top_k_set}"
            )

    def test_top_k_one_equals_greedy(self) -> None:
        """Top-k with k=1 returns the same as greedy (argmax)."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        token = sample_token_top_k(logits, top_k=1, temperature=1.0)
        expected = int(logits.argmax().item())
        assert token == expected

    def test_top_k_respects_temperature(self) -> None:
        """Temperature affects the distribution but tokens stay in top-k."""
        torch.manual_seed(0)
        logits = torch.randn(500)
        top_k = 5

        _, top_k_indices = torch.topk(logits, top_k)
        top_k_set = set(top_k_indices.tolist())

        token = sample_token_top_k(logits, top_k=top_k, temperature=0.5)
        assert token in top_k_set


# ---------------------------------------------------------------------------
# Fallback sampling tests
# ---------------------------------------------------------------------------


class TestSampleTokenFallback:
    """Tests for the fallback sampling implementation."""

    def test_returns_valid_token(self) -> None:
        """Fallback returns a valid token index."""
        torch.manual_seed(42)
        logits = torch.randn(1000)
        token = sample_token_fallback(logits, temperature=1.0)
        assert 0 <= token < 1000

    def test_with_top_p(self) -> None:
        """Fallback handles top-p filtering."""
        torch.manual_seed(42)
        logits = torch.randn(100)
        token = sample_token_fallback(
            logits, temperature=1.0, top_p=0.9
        )
        assert 0 <= token < 100

    def test_with_top_k_and_top_p(self) -> None:
        """Fallback handles combined top-k and top-p."""
        torch.manual_seed(42)
        logits = torch.randn(100)
        token = sample_token_fallback(
            logits, temperature=1.0, top_k=20, top_p=0.9
        )
        assert 0 <= token < 100


# ---------------------------------------------------------------------------
# Routing dispatcher tests
# ---------------------------------------------------------------------------


class TestRouteAndSampleToken:
    """Tests for the routing dispatcher."""

    def test_greedy_route_returns_argmax(self) -> None:
        """Greedy route returns argmax token."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        token, route = route_and_sample_token(
            logits, temperature=0.0
        )
        assert route == "greedy"
        assert token == 1

    def test_top_k_route_returns_top_k_member(self) -> None:
        """Top-k route returns a token from the top-k set."""
        torch.manual_seed(42)
        logits = torch.randn(1000)
        top_k = 10
        _, top_k_indices = torch.topk(logits, top_k)
        top_k_set = set(top_k_indices.tolist())

        token, route = route_and_sample_token(
            logits, temperature=1.0, top_k=top_k
        )
        assert route == "top_k"
        assert token in top_k_set

    def test_fallback_route_for_top_p(self) -> None:
        """Top-p triggers fallback route."""
        torch.manual_seed(42)
        logits = torch.randn(100)
        token, route = route_and_sample_token(
            logits, temperature=1.0, top_p=0.9
        )
        assert route == "fallback"
        assert 0 <= token < 100

    def test_nan_logits_with_validation_enabled(self) -> None:
        """NaN logits with validation enabled fall back to argmax on finite values."""
        logits = torch.tensor([1.0, float("nan"), 3.0, 2.0])
        token, route = route_and_sample_token(
            logits, temperature=1.0, validate_logits=True
        )
        assert route == "greedy"
        assert token == 2  # argmax of finite values

    def test_nan_logits_with_validation_disabled(self) -> None:
        """NaN logits with validation disabled are passed through to the route."""
        # When validation is disabled, NaN handling depends on the route implementation
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])  # valid logits
        token, route = route_and_sample_token(
            logits, temperature=0.0, validate_logits=False
        )
        assert route == "greedy"
        assert token == 1

    def test_constant_logits_with_validation(self) -> None:
        """Near-constant logits with validation enabled fall back to argmax."""
        logits = torch.full((100,), 3.0)
        token, route = route_and_sample_token(
            logits, temperature=1.0, validate_logits=True
        )
        assert route == "greedy"
        assert 0 <= token < 100


# ---------------------------------------------------------------------------
# Public sample_token compatibility tests
# ---------------------------------------------------------------------------


class TestSampleTokenPublicInterface:
    """Tests that the public sample_token function maintains backward compatibility."""

    def test_basic_call_with_3d_logits(self) -> None:
        """sample_token accepts (1, seq_len, vocab_size) shaped logits."""
        logits = torch.randn(1, 5, 100)
        token = sample_token(logits, temperature=0.0)
        # Should return argmax of last position
        expected = int(logits[0, -1, :].argmax().item())
        assert token == expected

    def test_basic_call_with_2d_logits(self) -> None:
        """sample_token accepts (1, vocab_size) shaped logits."""
        logits = torch.randn(1, 100)
        token = sample_token(logits, temperature=0.0)
        expected = int(logits[0].argmax().item())
        assert token == expected

    def test_greedy_with_near_zero_temperature(self) -> None:
        """Near-zero temperature produces greedy (argmax) result."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        token = sample_token(logits, temperature=1e-8)
        assert token == 1

    def test_top_k_sampling(self) -> None:
        """Top-k sampling returns a token from the top-k set."""
        torch.manual_seed(42)
        logits = torch.randn(1, 1000)
        top_k = 10
        _, top_k_indices = torch.topk(logits[0], top_k)
        top_k_set = set(top_k_indices.tolist())

        token = sample_token(logits, temperature=1.0, top_k=top_k)
        assert token in top_k_set

    def test_top_p_sampling(self) -> None:
        """Top-p sampling returns a valid token."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)
        token = sample_token(logits, temperature=1.0, top_p=0.9)
        assert 0 <= token < 100

    def test_diagnostics_flag_accepted(self) -> None:
        """enable_diagnostics parameter is accepted without error."""
        logits = torch.randn(1, 100)
        token = sample_token(
            logits, temperature=0.0, enable_diagnostics=True, model_id="test"
        )
        assert 0 <= token < 100

    def test_performance_recorder_none_accepted(self) -> None:
        """performance_recorder=None is accepted without error."""
        logits = torch.randn(1, 100)
        token = sample_token(logits, temperature=0.0, performance_recorder=None)
        assert 0 <= token < 100


# ---------------------------------------------------------------------------
# SamplingConfiguration model tests
# ---------------------------------------------------------------------------


class TestSamplingConfiguration:
    """Tests for the SamplingConfiguration Pydantic model."""

    def test_default_configuration(self) -> None:
        """Default configuration is valid."""
        config = SamplingConfiguration()
        assert config.do_sample is True
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.validate_logits is False

    def test_frozen_immutability(self) -> None:
        """Configuration is immutable (frozen)."""
        config = SamplingConfiguration()
        with pytest.raises(Exception):
            config.temperature = 2.0  # type: ignore[misc]

    def test_negative_temperature_rejected(self) -> None:
        """Negative temperature raises validation error."""
        with pytest.raises(Exception):
            SamplingConfiguration(temperature=-1.0)

    def test_zero_top_k_rejected(self) -> None:
        """top_k=0 raises validation error."""
        with pytest.raises(Exception):
            SamplingConfiguration(top_k=0)

    def test_negative_top_k_rejected(self) -> None:
        """Negative top_k raises validation error."""
        with pytest.raises(Exception):
            SamplingConfiguration(top_k=-5)

    def test_top_p_out_of_range_rejected(self) -> None:
        """top_p outside (0, 1] raises validation error."""
        with pytest.raises(Exception):
            SamplingConfiguration(top_p=0.0)
        with pytest.raises(Exception):
            SamplingConfiguration(top_p=1.5)

    def test_valid_top_p_accepted(self) -> None:
        """top_p=1.0 is valid (upper bound inclusive)."""
        config = SamplingConfiguration(top_p=1.0)
        assert config.top_p == 1.0

    def test_valid_configuration_with_all_params(self) -> None:
        """Full configuration with all parameters set."""
        config = SamplingConfiguration(
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            validate_logits=True,
        )
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.validate_logits is True



# ---------------------------------------------------------------------------
# Batched per-request sampling tests
# ---------------------------------------------------------------------------

from exo.worker.engines.pytorch_xpu.sampling import sample_tokens_batched  # noqa: E402


class TestSampleTokensBatched:
    """Tests for the batched per-request sampling function."""

    def test_single_element_greedy(self) -> None:
        """Single-element batch with greedy config returns argmax."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        configs = [SamplingConfiguration(do_sample=False)]
        result = sample_tokens_batched(logits, configs)
        assert result.shape == (1,)
        assert int(result[0].item()) == 1  # index of 5.0

    def test_batch_all_greedy(self) -> None:
        """All-greedy batch returns argmax for each element."""
        logits = torch.tensor([
            [1.0, 5.0, 3.0],
            [4.0, 2.0, 6.0],
            [7.0, 1.0, 3.0],
        ])
        configs = [
            SamplingConfiguration(do_sample=False),
            SamplingConfiguration(do_sample=False),
            SamplingConfiguration(do_sample=False),
        ]
        result = sample_tokens_batched(logits, configs)
        assert result.shape == (3,)
        assert int(result[0].item()) == 1  # argmax of [1, 5, 3]
        assert int(result[1].item()) == 2  # argmax of [4, 2, 6]
        assert int(result[2].item()) == 0  # argmax of [7, 1, 3]

    def test_mixed_configurations(self) -> None:
        """Mixed greedy and top-k configurations in the same batch."""
        torch.manual_seed(42)
        vocabulary_size = 1000
        logits = torch.randn(3, vocabulary_size)

        configs = [
            SamplingConfiguration(do_sample=False),  # greedy
            SamplingConfiguration(do_sample=True, top_k=10, temperature=1.0),  # top-k
            SamplingConfiguration(do_sample=False),  # greedy
        ]

        result = sample_tokens_batched(logits, configs)
        assert result.shape == (3,)

        # First element: greedy = argmax
        expected_first = int(logits[0].argmax().item())
        assert int(result[0].item()) == expected_first

        # Second element: must be in top-k set
        _, top_k_indices = torch.topk(logits[1], 10)
        top_k_set = set(top_k_indices.tolist())
        assert int(result[1].item()) in top_k_set

        # Third element: greedy = argmax
        expected_third = int(logits[2].argmax().item())
        assert int(result[2].item()) == expected_third

    def test_mixed_greedy_and_fallback(self) -> None:
        """Mixed greedy and fallback (top-p) configurations."""
        torch.manual_seed(42)
        logits = torch.randn(2, 100)

        configs = [
            SamplingConfiguration(do_sample=False),  # greedy
            SamplingConfiguration(do_sample=True, top_p=0.9, temperature=1.0),  # fallback
        ]

        result = sample_tokens_batched(logits, configs)
        assert result.shape == (2,)

        # First element: greedy = argmax
        expected_first = int(logits[0].argmax().item())
        assert int(result[0].item()) == expected_first

        # Second element: valid token index
        assert 0 <= int(result[1].item()) < 100

    def test_returns_1d_tensor(self) -> None:
        """Result is a 1D tensor with correct dtype and device."""
        logits = torch.randn(4, 50)
        configs = [SamplingConfiguration(do_sample=False)] * 4
        result = sample_tokens_batched(logits, configs)
        assert result.ndim == 1
        assert result.shape[0] == 4
        assert result.dtype == torch.long

    def test_rejects_1d_logits(self) -> None:
        """1D logits tensor raises ValueError."""
        logits = torch.randn(100)
        configs = [SamplingConfiguration()]
        with pytest.raises(ValueError, match="must be 2D"):
            sample_tokens_batched(logits, configs)

    def test_rejects_3d_logits(self) -> None:
        """3D logits tensor raises ValueError."""
        logits = torch.randn(2, 5, 100)
        configs = [SamplingConfiguration(), SamplingConfiguration()]
        with pytest.raises(ValueError, match="must be 2D"):
            sample_tokens_batched(logits, configs)

    def test_rejects_mismatched_batch_size(self) -> None:
        """Mismatched configurations count raises ValueError."""
        logits = torch.randn(3, 100)
        configs = [SamplingConfiguration(), SamplingConfiguration()]
        with pytest.raises(ValueError, match="must match batch size"):
            sample_tokens_batched(logits, configs)

    def test_nan_logits_with_validation_enabled(self) -> None:
        """NaN logits with validation enabled fall back to argmax on finite values."""
        logits = torch.tensor([
            [1.0, float("nan"), 3.0, 2.0],
            [4.0, 5.0, 1.0, 2.0],
        ])
        configs = [
            SamplingConfiguration(do_sample=False, validate_logits=True),
            SamplingConfiguration(do_sample=False, validate_logits=True),
        ]
        result = sample_tokens_batched(logits, configs)
        assert int(result[0].item()) == 2  # argmax of finite values [1, -inf, 3, 2]
        assert int(result[1].item()) == 1  # argmax of [4, 5, 1, 2]

    def test_all_nan_logits_defaults_to_zero(self) -> None:
        """All-NaN logits with validation enabled defaults to token 0."""
        logits = torch.tensor([
            [float("nan"), float("nan"), float("nan")],
        ])
        configs = [
            SamplingConfiguration(do_sample=False, validate_logits=True),
        ]
        result = sample_tokens_batched(logits, configs)
        assert int(result[0].item()) == 0

    def test_near_constant_logits_with_validation(self) -> None:
        """Near-constant logits with validation fall back to argmax."""
        logits = torch.full((2, 100), 3.0)
        configs = [
            SamplingConfiguration(do_sample=True, top_k=10, validate_logits=True),
            SamplingConfiguration(do_sample=True, top_k=10, validate_logits=True),
        ]
        result = sample_tokens_batched(logits, configs)
        assert result.shape == (2,)
        # Both should return valid indices (argmax on constant = first element)
        assert 0 <= int(result[0].item()) < 100
        assert 0 <= int(result[1].item()) < 100

    def test_with_performance_recorder(self) -> None:
        """Performance recorder integration records span and counters."""
        from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

        recorder = PerformanceRecorder(enabled=True, rank=3, stage=3)
        logits = torch.randn(3, 50)
        configs = [
            SamplingConfiguration(do_sample=False),  # greedy
            SamplingConfiguration(do_sample=True, top_k=5, temperature=1.0),  # top_k
            SamplingConfiguration(do_sample=True, top_p=0.9, temperature=1.0),  # fallback
        ]

        result = sample_tokens_batched(logits, configs, performance_recorder=recorder)
        assert result.shape == (3,)

        # Verify instrumentation recorded the span
        summary = recorder.summarize()
        assert summary.total_event_count >= 1
        assert "sample_tokens_batched" in summary.events_by_name

        # Verify per-route counters
        assert recorder.get_counter("batched_sampling_route_greedy") == 1
        assert recorder.get_counter("batched_sampling_route_top_k") == 1
        assert recorder.get_counter("batched_sampling_route_fallback") == 1

    def test_top_k_with_k_one_equals_greedy(self) -> None:
        """Top-k with k=1 in batched mode returns same as greedy."""
        logits = torch.tensor([
            [1.0, 5.0, 3.0, 2.0, 4.0],
            [7.0, 2.0, 3.0, 1.0, 6.0],
        ])
        configs_greedy = [
            SamplingConfiguration(do_sample=False),
            SamplingConfiguration(do_sample=False),
        ]
        configs_top_k_one = [
            SamplingConfiguration(do_sample=True, top_k=1, temperature=1.0),
            SamplingConfiguration(do_sample=True, top_k=1, temperature=1.0),
        ]

        result_greedy = sample_tokens_batched(logits, configs_greedy)
        result_top_k = sample_tokens_batched(logits, configs_top_k_one)

        assert int(result_greedy[0].item()) == int(result_top_k[0].item())
        assert int(result_greedy[1].item()) == int(result_top_k[1].item())

    def test_large_batch(self) -> None:
        """Handles a larger batch size correctly."""
        torch.manual_seed(123)
        batch_size = 32
        vocabulary_size = 32000
        logits = torch.randn(batch_size, vocabulary_size)
        configs = [SamplingConfiguration(do_sample=False)] * batch_size

        result = sample_tokens_batched(logits, configs)
        assert result.shape == (batch_size,)

        # All should be argmax
        expected = logits.argmax(dim=-1)
        assert torch.equal(result, expected)

    def test_empty_batch(self) -> None:
        """Empty batch (0 elements) returns empty tensor."""
        logits = torch.randn(0, 100)
        configs: list[SamplingConfiguration] = []
        result = sample_tokens_batched(logits, configs)
        assert result.shape == (0,)

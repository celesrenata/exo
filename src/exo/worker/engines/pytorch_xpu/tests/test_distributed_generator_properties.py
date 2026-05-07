# Feature: distributed-generation-pipeline, Property 1: Sampling uses last position only
"""
Property-based tests for the distributed generation pipeline.

Uses Hypothesis to verify:
- Property 1: Sampling uses last position only

**Validates: Requirements 2.2**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_GENERATOR_PATH = _THIS_DIR.parent / "distributed_generator.py"


def _load_distributed_generator() -> types.ModuleType:
    """Load distributed_generator.py directly from file, avoiding __init__.py."""
    module_name = "distributed_generator_props_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed_generator()
sample_token = _mod.sample_token


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Sequence length: at least 1, up to 10
_seq_len = st.integers(min_value=1, max_value=10)

# Vocabulary size: at least 2, up to 100
_vocab_size = st.integers(min_value=2, max_value=100)

# Random seed for reproducibility within each test case
_seed = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Property 1: Sampling uses last position only
# ---------------------------------------------------------------------------


class TestSamplingUsesLastPositionOnly:
    """Property 1: Sampling uses last position only.

    For any logits tensor of shape (1, seq_len, vocab_size) where seq_len >= 1,
    the sample_token function SHALL produce the same result as if called with
    logits[:, -1:, :] — i.e., only the last position's logits influence the
    sampled token.

    **Validates: Requirements 2.2**
    """

    @given(
        seq_len=_seq_len,
        vocab_size=_vocab_size,
        seed=_seed,
    )
    @settings(max_examples=100)
    def test_sample_token_uses_last_position_only(
        self,
        seq_len: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        """sample_token(full_logits) produces the same result as
        sample_token(last_position_logits) when using the same random seed.

        **Validates: Requirements 2.2**
        """
        # Generate random logits tensor of shape (1, seq_len, vocab_size)
        torch.manual_seed(seed)
        full_logits = torch.randn(1, seq_len, vocab_size)

        # Extract last position only: shape (1, 1, vocab_size)
        last_position_logits = full_logits[:, -1:, :]

        # Sample from full logits with a fixed seed
        torch.manual_seed(seed)
        token_from_full = sample_token(full_logits)

        # Sample from last-position-only logits with the same fixed seed
        torch.manual_seed(seed)
        token_from_last = sample_token(last_position_logits)

        assert token_from_full == token_from_last, (
            f"sample_token should only use last position logits. "
            f"Full tensor (1, {seq_len}, {vocab_size}) produced token {token_from_full}, "
            f"but last-position tensor (1, 1, {vocab_size}) produced token {token_from_last}."
        )


# Feature: distributed-generation-pipeline, Property 4: Temperature scaling is division
# ---------------------------------------------------------------------------
# Property 4: Temperature scaling is division
# ---------------------------------------------------------------------------


class TestTemperatureScalingIsDivision:
    """Property 4: Temperature scaling is division.

    For any logits tensor and temperature value T > 0, applying temperature
    scaling SHALL produce a tensor equal to logits / T. When T == 1.0, the
    logits SHALL be unchanged.

    **Validates: Requirements 3.1**
    """

    @given(
        vocab_size=st.integers(min_value=2, max_value=50),
        temperature=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_temperature_scaling_equivalence(
        self,
        vocab_size: int,
        temperature: float,
        seed: int,
    ) -> None:
        """sample_token(logits * T, temperature=T) produces the same token as
        sample_token(logits, temperature=1.0) when using the same random seed.

        This verifies that temperature scaling is equivalent to dividing logits
        by T, because (logits * T) / T == logits.

        **Validates: Requirements 3.1**
        """
        # Generate random logits tensor of shape (1, vocab_size)
        torch.manual_seed(seed)
        logits = torch.randn(1, vocab_size)

        # Scale logits by T — when sample_token divides by T, we get back original logits
        scaled_logits = logits * temperature

        # Sample from scaled logits with temperature=T (effectively: (logits * T) / T = logits)
        torch.manual_seed(seed + 1)
        token_with_temp = sample_token(scaled_logits.clone(), temperature=temperature)

        # Sample from original logits with temperature=1.0 (no scaling applied)
        torch.manual_seed(seed + 1)
        token_no_temp = sample_token(logits.clone(), temperature=1.0)

        assert token_with_temp == token_no_temp, (
            f"Temperature scaling should be equivalent to division. "
            f"sample_token(logits * {temperature}, temperature={temperature}) produced token {token_with_temp}, "
            f"but sample_token(logits, temperature=1.0) produced token {token_no_temp}. "
            f"vocab_size={vocab_size}, seed={seed}"
        )

    @given(
        vocab_size=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_temperature_one_leaves_logits_unchanged(
        self,
        vocab_size: int,
        seed: int,
    ) -> None:
        """When T == 1.0, sample_token produces the same result regardless of
        whether temperature is explicitly passed or defaulted.

        This verifies that temperature=1.0 leaves logits unchanged.

        **Validates: Requirements 3.1**
        """
        # Generate random logits tensor of shape (1, vocab_size)
        torch.manual_seed(seed)
        logits = torch.randn(1, vocab_size)

        # Sample with explicit temperature=1.0
        torch.manual_seed(seed + 1)
        token_explicit = sample_token(logits.clone(), temperature=1.0)

        # Sample with default temperature (should also be 1.0)
        torch.manual_seed(seed + 1)
        token_default = sample_token(logits.clone())

        assert token_explicit == token_default, (
            f"Temperature=1.0 should leave logits unchanged (same as default). "
            f"Explicit T=1.0 produced token {token_explicit}, "
            f"default produced token {token_default}. "
            f"vocab_size={vocab_size}, seed={seed}"
        )


# Feature: distributed-generation-pipeline, Property 5: Top-k retains exactly k values
# ---------------------------------------------------------------------------
# Property 5: Top-k retains exactly k values
# ---------------------------------------------------------------------------


class TestTopKRetainsExactlyKValues:
    """Property 5: Top-k retains exactly k values.

    For any logits tensor of vocabulary size V and top_k value k where 0 < k ≤ V,
    after top-k filtering, exactly k logit values SHALL be finite (not negative
    infinity), and these SHALL be the k largest values from the original tensor.

    **Validates: Requirements 3.2**
    """

    @given(
        vocab_size=st.integers(min_value=5, max_value=50),
        data=st.data(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_sampled_token_is_within_top_k(
        self,
        vocab_size: int,
        data: st.DataObject,
        seed: int,
    ) -> None:
        """With top_k=k, the sampled token SHALL always be one of the k positions
        with the highest logit values.

        **Validates: Requirements 3.2**
        """
        top_k = data.draw(st.integers(min_value=1, max_value=vocab_size), label="top_k")

        # Generate logits with DISTINCT values to avoid ties at the threshold
        # Use a permutation of a range to guarantee uniqueness
        torch.manual_seed(seed)
        logits_1d = torch.randperm(vocab_size).float()
        logits = logits_1d.unsqueeze(0)  # Shape: (1, vocab_size)

        # Determine the top-k token positions (the k largest values)
        _, top_k_indices = torch.topk(logits_1d, top_k)
        top_k_set = set(top_k_indices.tolist())

        # Sample with top_k — the result must be within the top-k set
        torch.manual_seed(seed)
        token_id = sample_token(logits.clone(), temperature=1.0, top_k=top_k)

        assert token_id in top_k_set, (
            f"Sampled token {token_id} is not in the top-{top_k} set {top_k_set}. "
            f"vocab_size={vocab_size}, logits={logits_1d.tolist()}"
        )

    @given(
        vocab_size=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_top_k_equals_one_is_greedy(
        self,
        vocab_size: int,
        seed: int,
    ) -> None:
        """With top_k=1, sample_token SHALL always return the argmax token,
        since only the single highest logit value is retained.

        **Validates: Requirements 3.2**
        """
        # Generate logits with DISTINCT values to avoid ties
        torch.manual_seed(seed)
        logits_1d = torch.randperm(vocab_size).float()
        logits = logits_1d.unsqueeze(0)  # Shape: (1, vocab_size)

        expected_token = int(logits_1d.argmax().item())

        # With top_k=1, only the max value survives — sampling must return argmax
        torch.manual_seed(seed)
        token_id = sample_token(logits.clone(), temperature=1.0, top_k=1)

        assert token_id == expected_token, (
            f"With top_k=1, expected argmax token {expected_token} but got {token_id}. "
            f"vocab_size={vocab_size}, logits={logits_1d.tolist()}"
        )


# Feature: distributed-generation-pipeline, Property 6: Top-p respects cumulative probability threshold
# ---------------------------------------------------------------------------
# Property 6: Top-p respects cumulative probability threshold
# ---------------------------------------------------------------------------


class TestTopPRespectsCumulativeProbabilityThreshold:
    """Property 6: Top-p respects cumulative probability threshold.

    For any logits tensor and top_p value p where 0 < p < 1, after top-p filtering
    and softmax, the cumulative probability of all retained tokens SHALL be ≥ p,
    and removing any single retained token (other than the highest-probability one)
    would make the cumulative probability < p.

    **Validates: Requirements 3.3**
    """

    @given(
        vocab_size=st.integers(min_value=5, max_value=50),
        top_p=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_sampled_token_is_within_top_p_set(
        self,
        vocab_size: int,
        top_p: float,
        seed: int,
    ) -> None:
        """The sampled token SHALL always be among the tokens retained by top-p
        filtering — i.e., within the minimal set whose cumulative probability ≥ top_p.

        **Validates: Requirements 3.3**
        """
        # Generate logits with DISTINCT values to avoid ties
        torch.manual_seed(seed)
        logits_1d = torch.randperm(vocab_size).float()
        logits = logits_1d.unsqueeze(0)  # Shape: (1, vocab_size)

        # Manually compute the top-p retained set:
        # 1. Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits_1d, descending=True)
        # 2. Compute softmax on sorted logits
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        # 3. Compute cumulative sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 4. Determine retained tokens: keep all tokens up to and including
        #    the first one that makes cumulative >= top_p
        #    (The mask in sample_token keeps index 0 always, and index i if cumulative[i-1] < top_p)
        retained_indices = []
        for i in range(vocab_size):
            retained_indices.append(int(sorted_indices[i].item()))
            if cumulative_probs[i].item() >= top_p:
                break

        top_p_set = set(retained_indices)

        # Sample with top_p — the result must be within the top-p set
        torch.manual_seed(seed)
        token_id = sample_token(logits.clone(), temperature=1.0, top_p=top_p)

        assert token_id in top_p_set, (
            f"Sampled token {token_id} is not in the top-p set {top_p_set}. "
            f"top_p={top_p}, vocab_size={vocab_size}, "
            f"sorted_indices={sorted_indices.tolist()[:10]}, "
            f"cumulative_probs={cumulative_probs.tolist()[:10]}"
        )

    @given(
        vocab_size=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_very_low_top_p_is_greedy(
        self,
        vocab_size: int,
        seed: int,
    ) -> None:
        """With top_p very small (approaching 0), only the highest-probability
        token should survive, so sample_token SHALL return the argmax.

        **Validates: Requirements 3.3**
        """
        # Generate logits with DISTINCT values to avoid ties
        torch.manual_seed(seed)
        logits_1d = torch.randperm(vocab_size).float()
        logits = logits_1d.unsqueeze(0)  # Shape: (1, vocab_size)

        # The argmax token — with very low top_p, only this should survive
        expected_token = int(logits_1d.argmax().item())

        # Use a very small top_p value — the highest-probability token alone
        # should exceed this threshold after softmax, so only it survives
        # We use 0.01 which is smaller than any single token's probability
        # would be in a uniform-ish distribution, but the argmax token in a
        # permutation-based logits will dominate after softmax
        torch.manual_seed(seed)
        token_id = sample_token(logits.clone(), temperature=1.0, top_p=0.01)

        assert token_id == expected_token, (
            f"With top_p=0.01, expected argmax token {expected_token} but got {token_id}. "
            f"vocab_size={vocab_size}, logits={logits_1d.tolist()}"
        )


# Feature: distributed-generation-pipeline, Property 12: Sampling pipeline order
# ---------------------------------------------------------------------------
# Property 12: Sampling pipeline order
# ---------------------------------------------------------------------------


class TestSamplingPipelineOrder:
    """Property 12: Sampling pipeline order.

    For any logits tensor with temperature ≠ 1.0, top_k set, and top_p set,
    the sample_token function SHALL produce results consistent with applying
    operations in the exact order: temperature scaling → top-k filtering →
    top-p filtering → softmax → multinomial sampling.

    **Validates: Requirements 3.5**
    """

    @given(
        vocab_size=st.integers(min_value=5, max_value=30),
        temperature=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
        data=st.data(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_pipeline_order_matches_manual_application(
        self,
        vocab_size: int,
        temperature: float,
        data: st.DataObject,
        seed: int,
    ) -> None:
        """sample_token with all parameters (temperature, top_k, top_p) produces
        the same result as manually applying the pipeline steps in order:
        1. Divide logits by temperature
        2. Apply top-k (keep only top-k values, set rest to -inf)
        3. Apply top-p (sort by probability, compute cumulative, mask above threshold)
        4. Apply softmax
        5. Multinomial sample

        **Validates: Requirements 3.5**
        """
        top_k = data.draw(st.integers(min_value=1, max_value=vocab_size), label="top_k")
        top_p = data.draw(
            st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
            label="top_p",
        )

        # Generate logits with DISTINCT values to avoid threshold ties in top-k
        torch.manual_seed(seed)
        logits_1d = torch.randperm(vocab_size).float()
        logits = logits_1d.unsqueeze(0)  # Shape: (1, vocab_size)

        # --- Call sample_token with all parameters ---
        torch.manual_seed(seed + 1)
        token_from_sample = sample_token(
            logits.clone(), temperature=temperature, top_k=top_k, top_p=top_p
        )

        # --- Manually apply the pipeline in the exact order ---
        manual_logits = logits_1d.clone()

        # Step 1: Temperature scaling (divide by temperature)
        manual_logits = manual_logits / temperature

        # Step 2: Top-k filtering (keep only top-k values, set rest to -inf)
        if top_k < vocab_size:
            top_k_values, _ = torch.topk(manual_logits, top_k)
            threshold = top_k_values[-1]
            manual_logits = manual_logits.masked_fill(manual_logits < threshold, float("-inf"))

        # Step 3: Top-p filtering (sort by probability, compute cumulative, mask above threshold)
        sorted_logits, sorted_indices = torch.sort(manual_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift right: keep the token that crosses the threshold
        sorted_mask = torch.zeros_like(cumulative_probs, dtype=torch.bool)
        sorted_mask[1:] = cumulative_probs[:-1] >= top_p

        # Set masked logits to -inf in sorted order
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # Scatter back to original positions
        manual_logits = torch.zeros_like(manual_logits).scatter(0, sorted_indices, sorted_logits)

        # Step 4: Softmax
        probs = torch.softmax(manual_logits, dim=-1)

        # Step 5: Multinomial sample (with same seed as sample_token)
        torch.manual_seed(seed + 1)
        token_from_manual = int(
            torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze().item()
        )

        assert token_from_sample == token_from_manual, (
            f"Pipeline order mismatch! sample_token produced token {token_from_sample}, "
            f"but manual pipeline produced token {token_from_manual}. "
            f"vocab_size={vocab_size}, temperature={temperature}, "
            f"top_k={top_k}, top_p={top_p}, seed={seed}"
        )


# Feature: distributed-generation-pipeline, Property 2: EOS token terminates with "stop"
# ---------------------------------------------------------------------------
# Property 2: EOS token terminates with "stop"
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch


class TestEOSTokenTerminatesWithStop:
    """Property 2: EOS token terminates with "stop".

    For any logits tensor that, after sampling, produces a token ID matching any
    configured EOS token ID, the generation loop SHALL terminate and the final
    GenerationResponse SHALL have finish_reason == "stop".

    **Validates: Requirements 2.5**
    """

    @given(
        eos_token_id=st.integers(min_value=0, max_value=99),
        vocab_size=st.integers(min_value=100, max_value=200),
        prompt_length=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_eos_token_produces_stop_finish_reason(
        self,
        eos_token_id: int,
        vocab_size: int,
        prompt_length: int,
    ) -> None:
        """When the sampled token matches the EOS token ID, distributed_generate()
        SHALL terminate and the final GenerationResponse SHALL have
        finish_reason == "stop".

        **Validates: Requirements 2.5**
        """
        # Load distributed_generate from the module
        distributed_generate = _mod.distributed_generate

        # --- Create mock model ---
        mock_model = MagicMock()
        # model.lm_head.weight.shape[0] returns vocab_size
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        # model.forward() returns (hidden_states, past_key_values)
        # hidden_states shape: (1, 1, hidden_size) — doesn't matter for rank 0 logic
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Create mock tokenizer ---
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(prompt_length))
        mock_tokenizer.decode.return_value = ""
        mock_tokenizer.eos_token_id = eos_token_id
        # Remove attributes that would add extra EOS tokens
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that will deterministically sample to eos_token_id ---
        # Set the EOS token position to a very high value, rest to very low
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, eos_token_id] = 100.0  # This will dominate after softmax

        # --- Patch send_activation and recv_activation ---
        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            # recv_activation returns our rigged logits (simulating last rank sending logits)
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            # Call distributed_generate and collect all responses
            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=50,  # High limit — should terminate via EOS first
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # The generation should have produced at least one response
        assert len(responses) >= 1, (
            f"Expected at least 1 response, got {len(responses)}"
        )

        # The last (and in this case only) response should have finish_reason == "stop"
        last_response = responses[-1]
        assert last_response.finish_reason == "stop", (
            f"Expected finish_reason='stop' but got '{last_response.finish_reason}'. "
            f"eos_token_id={eos_token_id}, vocab_size={vocab_size}, "
            f"token={last_response.token}, responses={len(responses)}"
        )

        # The token in the final response should be the EOS token
        assert last_response.token == eos_token_id, (
            f"Expected final token to be EOS ({eos_token_id}) but got {last_response.token}"
        )

    @given(
        eos_token_id=st.integers(min_value=0, max_value=99),
        vocab_size=st.integers(min_value=100, max_value=200),
        non_eos_steps=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_eos_after_multiple_tokens_still_terminates_with_stop(
        self,
        eos_token_id: int,
        vocab_size: int,
        non_eos_steps: int,
    ) -> None:
        """When the EOS token appears after several non-EOS tokens, the generation
        loop SHALL still terminate with finish_reason == "stop" on the EOS token.

        This tests that EOS detection works in the decode loop (not just prefill).

        **Validates: Requirements 2.5**
        """
        distributed_generate = _mod.distributed_generate

        # Pick a non-EOS token that's different from eos_token_id
        non_eos_token_id = (eos_token_id + 1) % vocab_size

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Create mock tokenizer ---
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]  # 3-token prompt
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = eos_token_id
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits sequences ---
        # First `non_eos_steps` calls return logits that sample to non_eos_token_id
        # Then the next call returns logits that sample to eos_token_id
        non_eos_logits = torch.full((1, 1, vocab_size), -100.0)
        non_eos_logits[0, 0, non_eos_token_id] = 100.0

        eos_logits = torch.full((1, 1, vocab_size), -100.0)
        eos_logits[0, 0, eos_token_id] = 100.0

        # Build the sequence of logits to return from recv_activation
        call_count = [0]
        total_recv_calls = non_eos_steps + 1  # non_eos_steps non-EOS + 1 EOS

        def recv_side_effect(*args, **kwargs):  # noqa: ANN002, ANN003
            idx = call_count[0]
            call_count[0] += 1
            if idx < non_eos_steps:
                return non_eos_logits.clone()
            else:
                return eos_logits.clone()

        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.side_effect = recv_side_effect
            mock_send.return_value = None

            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=50,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # Should have non_eos_steps + 1 responses (non-EOS tokens + final EOS)
        assert len(responses) == non_eos_steps + 1, (
            f"Expected {non_eos_steps + 1} responses but got {len(responses)}. "
            f"non_eos_steps={non_eos_steps}"
        )

        # The last response should have finish_reason == "stop"
        last_response = responses[-1]
        assert last_response.finish_reason == "stop", (
            f"Expected finish_reason='stop' but got '{last_response.finish_reason}'. "
            f"eos_token_id={eos_token_id}, non_eos_steps={non_eos_steps}"
        )

        # The final token should be the EOS token
        assert last_response.token == eos_token_id, (
            f"Expected final token to be EOS ({eos_token_id}) but got {last_response.token}"
        )

        # All non-final responses should NOT have finish_reason == "stop"
        for i, resp in enumerate(responses[:-1]):
            assert resp.finish_reason is None, (
                f"Response {i} should have finish_reason=None but got '{resp.finish_reason}'"
            )


# Feature: distributed-generation-pipeline, Property 3: Max tokens terminates with "length"
# ---------------------------------------------------------------------------
# Property 3: Max tokens terminates with "length"
# ---------------------------------------------------------------------------


class TestMaxTokensTerminatesWithLength:
    """Property 3: Max tokens terminates with "length".

    For any configured max_tokens value N, after exactly N tokens have been
    generated (without encountering EOS), the generation loop SHALL terminate
    and the final GenerationResponse SHALL have finish_reason == "length".

    **Validates: Requirements 2.6**
    """

    @given(
        max_tokens=st.integers(min_value=1, max_value=10),
        vocab_size=st.integers(min_value=100, max_value=200),
        non_eos_token_id=st.integers(min_value=50, max_value=99),
    )
    @settings(max_examples=100)
    def test_max_tokens_produces_length_finish_reason(
        self,
        max_tokens: int,
        vocab_size: int,
        non_eos_token_id: int,
    ) -> None:
        """When max_tokens is reached without encountering EOS, distributed_generate()
        SHALL terminate and the final GenerationResponse SHALL have
        finish_reason == "length".

        **Validates: Requirements 2.6**
        """
        distributed_generate = _mod.distributed_generate

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Create mock tokenizer ---
        # eos_token_id=0 so non_eos_token_id (50-99) will never match EOS
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]  # 3-token prompt
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = 0
        # Remove attributes that would add extra EOS tokens
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that always sample to non_eos_token_id ---
        # Set non_eos_token_id position to +100, rest to -100
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, non_eos_token_id] = 100.0

        # --- Patch send_activation and recv_activation ---
        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            # Call distributed_generate and collect all responses
            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # Exactly max_tokens responses should be produced
        assert len(responses) == max_tokens, (
            f"Expected {max_tokens} responses but got {len(responses)}. "
            f"max_tokens={max_tokens}, non_eos_token_id={non_eos_token_id}"
        )

        # The last response should have finish_reason == "length"
        last_response = responses[-1]
        assert last_response.finish_reason == "length", (
            f"Expected finish_reason='length' but got '{last_response.finish_reason}'. "
            f"max_tokens={max_tokens}, non_eos_token_id={non_eos_token_id}, "
            f"responses={len(responses)}"
        )

        # All non-final responses should have finish_reason == None
        for i, resp in enumerate(responses[:-1]):
            assert resp.finish_reason is None, (
                f"Response {i} should have finish_reason=None but got '{resp.finish_reason}'. "
                f"max_tokens={max_tokens}"
            )

    @given(
        max_tokens=st.integers(min_value=2, max_value=10),
        vocab_size=st.integers(min_value=100, max_value=200),
        non_eos_token_id=st.integers(min_value=50, max_value=99),
    )
    @settings(max_examples=100)
    def test_all_tokens_are_non_eos_before_length_termination(
        self,
        max_tokens: int,
        vocab_size: int,
        non_eos_token_id: int,
    ) -> None:
        """When generation terminates due to max_tokens, all generated tokens
        SHALL be non-EOS tokens (confirming termination was due to length, not EOS).

        **Validates: Requirements 2.6**
        """
        distributed_generate = _mod.distributed_generate

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Create mock tokenizer ---
        # eos_token_id=0 so non_eos_token_id (50-99) will never match EOS
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = 0
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that always sample to non_eos_token_id ---
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, non_eos_token_id] = 100.0

        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # All tokens should be non_eos_token_id (not the EOS token 0)
        eos_token_id = 0
        for i, resp in enumerate(responses):
            assert resp.token != eos_token_id, (
                f"Response {i} has token {resp.token} which matches EOS ({eos_token_id}). "
                f"Generation should have terminated via max_tokens, not EOS. "
                f"max_tokens={max_tokens}, non_eos_token_id={non_eos_token_id}"
            )
            assert resp.token == non_eos_token_id, (
                f"Response {i} has unexpected token {resp.token}, expected {non_eos_token_id}. "
                f"max_tokens={max_tokens}"
            )


# Feature: distributed-generation-pipeline, Property 7: GenerationResponse contains all required fields
# ---------------------------------------------------------------------------
# Property 7: GenerationResponse contains all required fields
# ---------------------------------------------------------------------------


class TestGenerationResponseContainsAllRequiredFields:
    """Property 7: GenerationResponse contains all required fields.

    For any token generated during the decode phase, the yielded GenerationResponse
    SHALL have: non-None usage with prompt_tokens > 0 and completion_tokens > 0,
    a valid token ID >= 0, and text that is the tokenizer's decoding of that token
    ID (or empty string for stop tokens).

    **Validates: Requirements 4.1**
    """

    @given(
        max_tokens=st.integers(min_value=1, max_value=10),
        vocab_size=st.integers(min_value=100, max_value=200),
        non_eos_token_id=st.integers(min_value=50, max_value=99),
        prompt_length=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_all_responses_have_required_fields(
        self,
        max_tokens: int,
        vocab_size: int,
        non_eos_token_id: int,
        prompt_length: int,
    ) -> None:
        """Every yielded GenerationResponse SHALL have non-None usage with
        prompt_tokens > 0 and completion_tokens > 0, token >= 0, non-None stats,
        and text matching what tokenizer.decode returns for non-stop responses.

        **Validates: Requirements 4.1**
        """
        distributed_generate = _mod.distributed_generate

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Create mock tokenizer ---
        # eos_token_id=0 so non_eos_token_id (50-99) will never match EOS
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(prompt_length))
        # Return a predictable decode string based on the token ID
        decode_text = f"token_{non_eos_token_id}"
        mock_tokenizer.decode.return_value = decode_text
        mock_tokenizer.eos_token_id = 0
        # Remove attributes that would add extra EOS tokens
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that always sample to non_eos_token_id ---
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, non_eos_token_id] = 100.0

        # --- Patch send_activation and recv_activation ---
        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            # Call distributed_generate and collect all responses
            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # Should have exactly max_tokens responses (terminated by length)
        assert len(responses) == max_tokens, (
            f"Expected {max_tokens} responses but got {len(responses)}"
        )

        # Verify every response has all required fields
        for i, resp in enumerate(responses):
            # usage is not None
            assert resp.usage is not None, (
                f"Response {i}: usage is None. "
                f"max_tokens={max_tokens}, prompt_length={prompt_length}"
            )

            # usage.prompt_tokens > 0
            assert resp.usage.prompt_tokens > 0, (
                f"Response {i}: usage.prompt_tokens={resp.usage.prompt_tokens}, expected > 0. "
                f"prompt_length={prompt_length}"
            )

            # usage.completion_tokens > 0
            assert resp.usage.completion_tokens > 0, (
                f"Response {i}: usage.completion_tokens={resp.usage.completion_tokens}, expected > 0. "
                f"max_tokens={max_tokens}"
            )

            # token >= 0
            assert resp.token >= 0, (
                f"Response {i}: token={resp.token}, expected >= 0. "
                f"non_eos_token_id={non_eos_token_id}"
            )

            # stats is not None
            assert resp.stats is not None, (
                f"Response {i}: stats is None. "
                f"max_tokens={max_tokens}"
            )

            # For non-stop responses: text equals what tokenizer.decode returns
            if resp.finish_reason != "stop":
                assert resp.text == decode_text, (
                    f"Response {i}: text='{resp.text}', expected '{decode_text}'. "
                    f"non_eos_token_id={non_eos_token_id}, finish_reason={resp.finish_reason}"
                )

    @given(
        vocab_size=st.integers(min_value=100, max_value=200),
        eos_token_id=st.integers(min_value=50, max_value=99),
        non_eos_steps=st.integers(min_value=1, max_value=5),
        prompt_length=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_stop_response_has_required_fields_with_empty_text(
        self,
        vocab_size: int,
        eos_token_id: int,
        non_eos_steps: int,
        prompt_length: int,
    ) -> None:
        """When generation terminates with EOS (stop), the final response SHALL
        have non-None usage with prompt_tokens > 0 and completion_tokens > 0,
        token >= 0, non-None stats, and text == "" (empty for stop tokens).

        **Validates: Requirements 4.1**
        """
        distributed_generate = _mod.distributed_generate

        # Pick a non-EOS token that's different from eos_token_id
        non_eos_token_id = (eos_token_id + 50) % vocab_size
        # Ensure non_eos_token_id != eos_token_id
        if non_eos_token_id == eos_token_id:
            non_eos_token_id = (eos_token_id + 1) % vocab_size

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Create mock tokenizer ---
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(prompt_length))
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = eos_token_id
        # Remove attributes that would add extra EOS tokens
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits sequences ---
        non_eos_logits = torch.full((1, 1, vocab_size), -100.0)
        non_eos_logits[0, 0, non_eos_token_id] = 100.0

        eos_logits = torch.full((1, 1, vocab_size), -100.0)
        eos_logits[0, 0, eos_token_id] = 100.0

        call_count = [0]

        def recv_side_effect(*args, **kwargs):  # noqa: ANN002, ANN003
            idx = call_count[0]
            call_count[0] += 1
            if idx < non_eos_steps:
                return non_eos_logits.clone()
            else:
                return eos_logits.clone()

        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.side_effect = recv_side_effect
            mock_send.return_value = None

            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=50,  # High limit — should terminate via EOS
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # Should have non_eos_steps + 1 responses
        assert len(responses) == non_eos_steps + 1, (
            f"Expected {non_eos_steps + 1} responses but got {len(responses)}"
        )

        # Verify the final (stop) response
        last_response = responses[-1]
        assert last_response.finish_reason == "stop", (
            f"Expected finish_reason='stop' but got '{last_response.finish_reason}'"
        )

        # Stop response: usage is not None
        assert last_response.usage is not None, (
            f"Stop response: usage is None"
        )

        # Stop response: usage.prompt_tokens > 0
        assert last_response.usage.prompt_tokens > 0, (
            f"Stop response: usage.prompt_tokens={last_response.usage.prompt_tokens}, expected > 0"
        )

        # Stop response: usage.completion_tokens > 0
        assert last_response.usage.completion_tokens > 0, (
            f"Stop response: usage.completion_tokens={last_response.usage.completion_tokens}, expected > 0"
        )

        # Stop response: token >= 0
        assert last_response.token >= 0, (
            f"Stop response: token={last_response.token}, expected >= 0"
        )

        # Stop response: stats is not None
        assert last_response.stats is not None, (
            f"Stop response: stats is None"
        )

        # Stop response: text is empty string for stop tokens
        assert last_response.text == "", (
            f"Stop response: text='{last_response.text}', expected '' (empty for stop tokens)"
        )

        # Also verify all non-stop responses have required fields
        for i, resp in enumerate(responses[:-1]):
            assert resp.usage is not None, f"Response {i}: usage is None"
            assert resp.usage.prompt_tokens > 0, f"Response {i}: prompt_tokens not > 0"
            assert resp.usage.completion_tokens > 0, f"Response {i}: completion_tokens not > 0"
            assert resp.token >= 0, f"Response {i}: token < 0"
            assert resp.stats is not None, f"Response {i}: stats is None"


# Feature: distributed-generation-pipeline, Property 10: Only rank 0 produces output
# ---------------------------------------------------------------------------
# Property 10: Only rank 0 produces output
# ---------------------------------------------------------------------------

distributed_worker_loop = _mod.distributed_worker_loop
TERMINATION_SENTINEL = _mod.TERMINATION_SENTINEL


class TestOnlyRank0ProducesOutput:
    """Property 10: Only rank 0 produces output.

    For any rank r where r != 0, the distributed_worker_loop SHALL produce zero
    GenerationResponse objects and SHALL not invoke any token decoding
    (tokenizer.decode) operations.

    **Validates: Requirements 9.1, 9.2, 9.4**
    """

    @given(
        rank=st.integers(min_value=1, max_value=3),
        world_size=st.integers(min_value=2, max_value=4),
        hidden_size=st.sampled_from([64, 128, 256]),
    )
    @settings(max_examples=100)
    def test_worker_loop_returns_none_not_generator(
        self,
        rank: int,
        world_size: int,
        hidden_size: int,
    ) -> None:
        """distributed_worker_loop() returns None (not a generator), so it
        structurally CANNOT yield GenerationResponse objects. Additionally,
        it does not accept a tokenizer parameter, so it CANNOT invoke
        tokenizer.decode.

        **Validates: Requirements 9.1, 9.2, 9.4**
        """
        import inspect

        # Ensure rank < world_size
        if rank >= world_size:
            world_size = rank + 1

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (hidden_size,)  # vocab_size for last rank
        mock_model.model.config.vocab_size = hidden_size
        mock_hidden_states = torch.randn(1, 1, hidden_size)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]
        mock_model.forward.return_value = (mock_hidden_states, mock_past_kv)

        # --- Set up recv_activation to return appropriate tensors then SENTINEL ---
        # Worker loop protocol for prefill:
        #   1. recv seq_len metadata (shape (1,), int64) from prev rank
        #   2. recv hidden_states (shape (1, seq_len, hidden_size)) from prev rank
        #   3. model.forward()
        #   4. send output to next rank (or logits to rank 0 if last)
        #   5. recv token from rank 0 (shape (1,), int64) — we send SENTINEL here
        seq_len_tensor = torch.tensor([1], dtype=torch.int64)
        hidden_states_tensor = torch.randn(1, 1, hidden_size)
        sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)

        recv_call_count = [0]

        def recv_side_effect(*args, **kwargs):  # noqa: ANN002, ANN003
            idx = recv_call_count[0]
            recv_call_count[0] += 1
            if idx == 0:
                # First recv: seq_len metadata
                return seq_len_tensor.clone()
            elif idx == 1:
                # Second recv: hidden_states for prefill
                return hidden_states_tensor.clone()
            else:
                # Third recv: token from rank 0 — send SENTINEL to terminate
                return sentinel_tensor.clone()

        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.side_effect = recv_side_effect
            mock_send.return_value = None

            # Call distributed_worker_loop — it should return None
            result = distributed_worker_loop(
                model=mock_model,
                device_type="cpu",
                device_id=0,
                rank=rank,
                world_size=world_size,
                hidden_size=hidden_size,
                dtype=torch.float32,
            )

        # Property assertion 1: The function returns None (not a generator)
        assert result is None, (
            f"distributed_worker_loop should return None, got {type(result)}. "
            f"rank={rank}, world_size={world_size}"
        )

        # Property assertion 2: The return type is NOT a generator
        assert not inspect.isgenerator(result), (
            f"distributed_worker_loop should not be a generator function. "
            f"rank={rank}, world_size={world_size}"
        )

        # Property assertion 3: The function signature does NOT accept a tokenizer
        # parameter, so it structurally cannot call tokenizer.decode
        sig = inspect.signature(distributed_worker_loop)
        param_names = set(sig.parameters.keys())
        assert "tokenizer" not in param_names, (
            f"distributed_worker_loop should not accept a 'tokenizer' parameter. "
            f"Parameters: {param_names}"
        )


# Feature: distributed-generation-pipeline, Property 9: Input tensor shape reflects generation phase
# ---------------------------------------------------------------------------
# Property 9: Input tensor shape reflects generation phase
# ---------------------------------------------------------------------------


class TestInputTensorShapeReflectsGenerationPhase:
    """Property 9: Input tensor shape reflects generation phase.

    For any generation run with a prompt of length N, the first forward pass
    input to each rank's TransformerShard SHALL have seq_len == N (prefill),
    and all subsequent forward pass inputs SHALL have seq_len == 1 (decode).

    **Validates: Requirements 8.1, 8.2, 8.5**
    """

    @given(
        prompt_length=st.integers(min_value=1, max_value=10),
        max_tokens=st.integers(min_value=2, max_value=5),
        vocab_size=st.integers(min_value=100, max_value=200),
    )
    @settings(max_examples=100)
    def test_prefill_uses_full_prompt_decode_uses_single_token(
        self,
        prompt_length: int,
        max_tokens: int,
        vocab_size: int,
    ) -> None:
        """The first model.forward call SHALL receive input_data with
        shape[1] == prompt_length (prefill), and all subsequent calls SHALL
        receive input_data with shape[1] == 1 (decode).

        **Validates: Requirements 8.1, 8.2, 8.5**
        """
        distributed_generate = _mod.distributed_generate

        # Track input_data shapes passed to model.forward
        forward_input_shapes: list[tuple[int, ...]] = []

        # --- Create mock model that records input_data shapes ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]

        def forward_side_effect(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
            # Record the shape of input_data
            input_data = kwargs.get("input_data")
            if input_data is None and len(args) > 0:
                input_data = args[0]
            if input_data is not None:
                forward_input_shapes.append(tuple(input_data.shape))
            return (mock_hidden_states, mock_past_kv)

        mock_model.forward.side_effect = forward_side_effect

        # --- Create mock tokenizer ---
        # Use a non-EOS token (50+) so generation doesn't stop early
        non_eos_token_id = 50
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(prompt_length))
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = 0  # EOS is 0, we'll sample 50+
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that always sample to non_eos_token_id ---
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, non_eos_token_id] = 100.0

        # --- Patch send_activation and recv_activation ---
        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            # Call distributed_generate and consume all responses
            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # We should have max_tokens responses (terminated by length)
        assert len(responses) == max_tokens, (
            f"Expected {max_tokens} responses but got {len(responses)}"
        )

        # model.forward should have been called max_tokens times:
        # 1 prefill + (max_tokens - 1) decode steps
        expected_forward_calls = max_tokens
        assert len(forward_input_shapes) == expected_forward_calls, (
            f"Expected {expected_forward_calls} forward calls but got {len(forward_input_shapes)}. "
            f"prompt_length={prompt_length}, max_tokens={max_tokens}"
        )

        # First call (prefill): input_data shape should be (1, prompt_length)
        prefill_shape = forward_input_shapes[0]
        assert prefill_shape == (1, prompt_length), (
            f"Prefill forward call should have input shape (1, {prompt_length}) "
            f"but got {prefill_shape}. prompt_length={prompt_length}"
        )

        # All subsequent calls (decode): input_data shape should be (1, 1)
        for i, shape in enumerate(forward_input_shapes[1:], start=1):
            assert shape == (1, 1), (
                f"Decode forward call {i} should have input shape (1, 1) "
                f"but got {shape}. prompt_length={prompt_length}, max_tokens={max_tokens}"
            )


# Feature: distributed-generation-pipeline, Property 8: KV cache grows correctly across phases
# ---------------------------------------------------------------------------
# Property 8: KV cache grows correctly across phases
# ---------------------------------------------------------------------------

from typing import Any


class TestKVCacheGrowsCorrectlyAcrossPhases:
    """Property 8: KV cache grows correctly across phases.

    For any prompt of length N tokens, after the prefill phase completes, the KV
    cache on each rank SHALL contain key-value pairs covering N positions. After
    each subsequent decode step, the KV cache length SHALL increase by exactly 1.

    **Validates: Requirements 5.2, 5.3, 8.3, 8.4**
    """

    @given(
        prompt_length=st.integers(min_value=1, max_value=10),
        max_tokens=st.integers(min_value=2, max_value=5),
        vocab_size=st.integers(min_value=100, max_value=200),
    )
    @settings(max_examples=100)
    def test_kv_cache_grows_by_one_each_decode_step(
        self,
        prompt_length: int,
        max_tokens: int,
        vocab_size: int,
    ) -> None:
        """Mock model.forward to return KV caches of increasing size. Track what
        past_key_values are passed TO model.forward on each call. Verify:
        - First call (prefill): past_key_values=None
        - Second call (first decode): past_key_values has seq_len == prompt_length
        - Nth call: past_key_values has seq_len == prompt_length + (N-2)

        **Validates: Requirements 5.2, 5.3, 8.3, 8.4**
        """
        distributed_generate = _mod.distributed_generate

        # Track past_key_values passed TO model.forward on each call
        past_kv_passed_to_forward: list[Any] = []

        # Counter to track how many times forward has been called
        forward_call_count = [0]

        def forward_side_effect(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
            """Mock model.forward that returns growing KV caches."""
            # Record what past_key_values was passed in
            past_kv = kwargs.get("past_key_values")
            past_kv_passed_to_forward.append(past_kv)

            call_idx = forward_call_count[0]
            forward_call_count[0] += 1

            # Compute the KV cache seq_len for the OUTPUT of this call:
            # - Prefill (call 0): output KV has seq_len == prompt_length
            # - Decode call N (call N): output KV has seq_len == prompt_length + N
            if call_idx == 0:
                # Prefill: KV cache covers all prompt positions
                kv_seq_len = prompt_length
            else:
                # Decode: KV cache grows by 1 each step
                kv_seq_len = prompt_length + call_idx

            # Create mock KV cache with the correct seq_len
            # Shape: list of (key, value) tuples where key/value have shape
            # (batch, num_heads, seq_len, head_dim)
            num_heads = 4
            head_dim = 16
            mock_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
                (
                    torch.randn(1, num_heads, kv_seq_len, head_dim),
                    torch.randn(1, num_heads, kv_seq_len, head_dim),
                )
            ]

            # Return hidden_states and new KV cache
            mock_hidden_states = torch.randn(1, 1, 64)
            return (mock_hidden_states, mock_kv)

        # --- Create mock model ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_model.forward.side_effect = forward_side_effect

        # --- Create mock tokenizer ---
        non_eos_token_id = 50
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(prompt_length))
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = 0  # EOS is 0, we'll sample 50+
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that always sample to non_eos_token_id ---
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, non_eos_token_id] = 100.0

        # --- Patch send_activation and recv_activation ---
        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            # Call distributed_generate and consume all responses
            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # Should have max_tokens responses (terminated by length)
        assert len(responses) == max_tokens, (
            f"Expected {max_tokens} responses but got {len(responses)}"
        )

        # model.forward should have been called max_tokens times:
        # 1 prefill + (max_tokens - 1) decode steps
        expected_forward_calls = max_tokens
        assert len(past_kv_passed_to_forward) == expected_forward_calls, (
            f"Expected {expected_forward_calls} forward calls but got "
            f"{len(past_kv_passed_to_forward)}. "
            f"prompt_length={prompt_length}, max_tokens={max_tokens}"
        )

        # --- Verify KV cache growth pattern ---

        # First call (prefill): past_key_values should be None
        assert past_kv_passed_to_forward[0] is None, (
            f"First forward call (prefill) should receive past_key_values=None, "
            f"but got {type(past_kv_passed_to_forward[0])}. "
            f"prompt_length={prompt_length}"
        )

        # Subsequent calls (decode): past_key_values should have growing seq_len
        for i in range(1, len(past_kv_passed_to_forward)):
            past_kv = past_kv_passed_to_forward[i]

            # past_key_values should not be None for decode steps
            assert past_kv is not None, (
                f"Forward call {i} (decode) should receive non-None past_key_values, "
                f"but got None. prompt_length={prompt_length}, max_tokens={max_tokens}"
            )

            # Extract seq_len from the KV cache
            # KV cache is list of (key, value) tuples with shape (batch, heads, seq_len, head_dim)
            key_tensor = past_kv[0][0]  # First layer's key tensor
            kv_seq_len = key_tensor.shape[2]  # seq_len dimension

            # Expected seq_len: prompt_length + (i - 1)
            # - Call 1 (first decode): gets KV from prefill output → seq_len == prompt_length
            # - Call 2 (second decode): gets KV from first decode output → seq_len == prompt_length + 1
            # - Call N: seq_len == prompt_length + (N - 1)
            expected_seq_len = prompt_length + (i - 1)

            assert kv_seq_len == expected_seq_len, (
                f"Forward call {i} (decode step {i}): expected KV cache seq_len "
                f"== {expected_seq_len} (prompt_length={prompt_length} + {i - 1}), "
                f"but got {kv_seq_len}. max_tokens={max_tokens}"
            )


# Feature: distributed-generation-pipeline, Property 11: All ranks execute equal iterations
# ---------------------------------------------------------------------------
# Property 11: All ranks execute equal iterations
# ---------------------------------------------------------------------------


class TestAllRanksExecuteEqualIterations:
    """Property 11: All ranks execute equal iterations.

    For any generation run that produces K tokens, every participating rank SHALL
    execute exactly K+1 forward passes (1 prefill + K decode steps).

    In our implementation, max_tokens=N produces N tokens via N forward passes
    (1 prefill + N-1 decode). The key property is that ALL ranks execute the
    same number of forward passes — rank 0 and every worker rank must be in
    lockstep.

    **Validates: Requirements 6.5**
    """

    @given(
        max_tokens=st.integers(min_value=1, max_value=10),
        vocab_size=st.integers(min_value=100, max_value=200),
    )
    @settings(max_examples=100)
    def test_rank_0_executes_max_tokens_forward_passes(
        self,
        max_tokens: int,
        vocab_size: int,
    ) -> None:
        """Rank 0 (distributed_generate) SHALL execute exactly max_tokens
        forward passes when generating max_tokens tokens without EOS.

        This is 1 prefill + (max_tokens - 1) decode = max_tokens total.

        **Validates: Requirements 6.5**
        """
        distributed_generate = _mod.distributed_generate

        # Track model.forward call count
        forward_call_count = [0]

        # --- Create mock model that counts forward calls ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (vocab_size,)
        mock_model.model.config.vocab_size = vocab_size
        mock_hidden_states = torch.randn(1, 1, 64)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]

        def forward_side_effect(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
            forward_call_count[0] += 1
            return (mock_hidden_states, mock_past_kv)

        mock_model.forward.side_effect = forward_side_effect

        # --- Create mock tokenizer (no EOS hit) ---
        non_eos_token_id = 50
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "tok"
        mock_tokenizer.eos_token_id = 0  # EOS is 0, we sample 50
        del mock_tokenizer.additional_special_tokens_ids
        del mock_tokenizer.all_special_ids

        # --- Create logits that always sample to non_eos_token_id ---
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, non_eos_token_id] = 100.0

        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.return_value = logits.clone()
            mock_send.return_value = None

            responses = list(
                distributed_generate(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    prompt="test prompt",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    model_id="test-model",
                )
            )

        # Verify K tokens produced
        assert len(responses) == max_tokens, (
            f"Expected {max_tokens} tokens but got {len(responses)}"
        )

        # Verify rank 0 executed exactly max_tokens forward passes
        # (1 prefill + max_tokens-1 decode = max_tokens total)
        assert forward_call_count[0] == max_tokens, (
            f"Rank 0 should execute exactly {max_tokens} forward passes "
            f"(1 prefill + {max_tokens - 1} decode), but executed {forward_call_count[0]}. "
            f"max_tokens={max_tokens}"
        )

    @given(
        max_tokens=st.integers(min_value=1, max_value=10),
        hidden_size=st.sampled_from([64, 128]),
    )
    @settings(max_examples=100)
    def test_worker_loop_executes_same_iterations_as_rank_0(
        self,
        max_tokens: int,
        hidden_size: int,
    ) -> None:
        """The worker loop (non-rank-0) SHALL execute the same number of forward
        passes as rank 0 for the same generation run. For max_tokens tokens
        produced, the worker executes max_tokens forward passes (1 prefill +
        max_tokens-1 decode).

        **Validates: Requirements 6.5**
        """
        # Track model.forward call count on the worker
        forward_call_count = [0]

        # --- Create mock model that counts forward calls ---
        mock_model = MagicMock()
        mock_model.lm_head.weight.shape = (hidden_size,)  # vocab_size for last rank
        mock_model.model.config.vocab_size = hidden_size
        mock_hidden_states = torch.randn(1, 1, hidden_size)
        mock_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        ]

        def forward_side_effect(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
            forward_call_count[0] += 1
            return (mock_hidden_states, mock_past_kv)

        mock_model.forward.side_effect = forward_side_effect

        # --- Set up recv_activation to simulate the protocol ---
        # Worker loop protocol:
        #   Prefill:
        #     1. recv seq_len metadata (shape (1,), int64)
        #     2. recv hidden_states (shape (1, seq_len, hidden_size))
        #     3. model.forward()
        #     4. send output
        #     5. recv token from rank 0 (shape (1,), int64)
        #   Decode (repeats max_tokens-1 times):
        #     1. recv hidden_states (shape (1, 1, hidden_size))
        #     2. model.forward()
        #     3. send output
        #     4. recv token from rank 0 (shape (1,), int64)
        #   Final:
        #     After last decode, recv SENTINEL to terminate

        seq_len_tensor = torch.tensor([3], dtype=torch.int64)  # prefill seq_len=3
        prefill_hidden = torch.randn(1, 3, hidden_size)
        decode_hidden = torch.randn(1, 1, hidden_size)
        non_eos_token = torch.tensor([50], dtype=torch.int64)
        sentinel_tensor = torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)

        recv_call_count = [0]

        def recv_side_effect(*args: Any, **kwargs: Any) -> torch.Tensor:
            idx = recv_call_count[0]
            recv_call_count[0] += 1

            # Prefill phase:
            # idx 0: seq_len metadata
            # idx 1: hidden_states for prefill
            # idx 2: token from rank 0 (non-EOS or SENTINEL if max_tokens==1)
            if idx == 0:
                return seq_len_tensor.clone()
            elif idx == 1:
                return prefill_hidden.clone()
            elif idx == 2:
                # After prefill: if max_tokens==1, send SENTINEL; else send token
                if max_tokens <= 1:
                    return sentinel_tensor.clone()
                return non_eos_token.clone()
            else:
                # Decode phase: alternates between hidden_states and token
                # idx 3: decode hidden_states (decode step 1)
                # idx 4: token (or SENTINEL if this was the last decode step)
                # idx 5: decode hidden_states (decode step 2)
                # idx 6: token (or SENTINEL)
                # ...
                decode_idx = idx - 3  # 0-based index into decode phase
                if decode_idx % 2 == 0:
                    # Even: hidden_states for decode
                    return decode_hidden.clone()
                else:
                    # Odd: token from rank 0
                    # Decode step number (1-based): (decode_idx // 2) + 1
                    decode_step = (decode_idx // 2) + 1
                    # Total decode steps needed: max_tokens - 1
                    # After the last decode step, send SENTINEL
                    if decode_step >= max_tokens - 1:
                        return sentinel_tensor.clone()
                    return non_eos_token.clone()

        with patch(
            "distributed_generator_props_isolated.send_activation"
        ) as mock_send, patch(
            "distributed_generator_props_isolated.recv_activation"
        ) as mock_recv:
            mock_recv.side_effect = recv_side_effect
            mock_send.return_value = None

            # Run worker loop (rank 1, last rank in a 2-rank setup)
            distributed_worker_loop(
                model=mock_model,
                device_type="cpu",
                device_id=0,
                rank=1,
                world_size=2,
                hidden_size=hidden_size,
                dtype=torch.float32,
            )

        # Worker should execute exactly max_tokens forward passes
        # (1 prefill + max_tokens-1 decode = max_tokens total)
        # This matches what rank 0 does, ensuring all ranks are in lockstep
        assert forward_call_count[0] == max_tokens, (
            f"Worker (rank 1) should execute exactly {max_tokens} forward passes "
            f"(1 prefill + {max_tokens - 1} decode), but executed {forward_call_count[0]}. "
            f"max_tokens={max_tokens}, hidden_size={hidden_size}"
        )

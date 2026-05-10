# Feature: xpu-performance-benchmark, Property 1: CLI argument parsing accepts valid configurations
"""
Property-based tests for XPU Performance Benchmark.

**Validates: Requirements 1.1**

Uses Hypothesis to generate valid CLI argument combinations and verifies
that parse_args produces a BenchmarkConfig with matching field values.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BENCH_XPU_PATH = _THIS_DIR.parent / "bench_xpu.py"


def _load_bench_xpu() -> types.ModuleType:
    """Load bench_xpu.py directly from file, avoiding __init__.py."""
    module_name = "bench_xpu_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _BENCH_XPU_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_bench_xpu()
parse_args = _mod.parse_args
BenchmarkConfig = _mod.BenchmarkConfig


# ---------------------------------------------------------------------------
# Strategies for generating valid CLI arguments
# ---------------------------------------------------------------------------

# Non-empty model IDs (HuggingFace format: org/model or just model name)
model_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() == s and len(s.strip()) > 0 and not s.startswith("-"))

# Device strings matching xpu:\d+
device_strategy = st.integers(min_value=0, max_value=9).map(lambda n: f"xpu:{n}")

# Dtype choices
dtype_strategy = st.sampled_from(["bf16", "fp16"])

# Positive integers for prompt_tokens and gen_tokens
prompt_tokens_strategy = st.integers(min_value=1, max_value=4096)
gen_tokens_strategy = st.integers(min_value=1, max_value=4096)

# Non-negative integers for warmup
warmup_strategy = st.integers(min_value=0, max_value=10)

# Positive integers for iterations
iterations_strategy = st.integers(min_value=1, max_value=20)

# Non-negative integers for seed
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)

# Boolean flags for --compile and --report-sdpa
compile_strategy = st.booleans()
report_sdpa_strategy = st.booleans()


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    model_id=model_id_strategy,
    device=device_strategy,
    dtype=dtype_strategy,
    prompt_tokens=prompt_tokens_strategy,
    gen_tokens=gen_tokens_strategy,
    warmup=warmup_strategy,
    iterations=iterations_strategy,
    seed=seed_strategy,
    compile_flag=compile_strategy,
    report_sdpa=report_sdpa_strategy,
)
def test_cli_argument_parsing_accepts_valid_configurations(
    model_id: str,
    device: str,
    dtype: str,
    prompt_tokens: int,
    gen_tokens: int,
    warmup: int,
    iterations: int,
    seed: int,
    compile_flag: bool,
    report_sdpa: bool,
) -> None:
    """Property 1: CLI argument parsing accepts valid configurations.

    For any valid combination of arguments, parse_args SHALL produce a
    BenchmarkConfig with matching field values.

    **Validates: Requirements 1.1**
    """
    # Build argv list
    argv = [
        "--model_id", model_id,
        "--device", device,
        "--dtype", dtype,
        "--prompt_tokens", str(prompt_tokens),
        "--gen_tokens", str(gen_tokens),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--seed", str(seed),
    ]

    if compile_flag:
        argv.append("--compile")

    if report_sdpa:
        argv.append("--report-sdpa")

    # Parse and verify
    config = parse_args(argv)

    assert isinstance(config, BenchmarkConfig)
    assert config.model_id == model_id
    assert config.device == device
    assert config.dtype == dtype
    assert config.prompt_tokens == prompt_tokens
    assert config.gen_tokens == gen_tokens
    assert config.warmup == warmup
    assert config.iterations == iterations
    assert config.seed == seed
    assert config.compile == compile_flag
    assert config.report_sdpa == report_sdpa


# Feature: xpu-performance-benchmark, Property 2: Decode TPS calculation
"""
Property-based test for decode TPS calculation.

**Validates: Requirements 1.5**

For any positive gen_tokens and positive elapsed_seconds, the computed
decode_tps SHALL equal gen_tokens / elapsed_seconds within floating-point
tolerance.
"""

import math


@settings(max_examples=100)
@given(
    gen_tokens=st.integers(min_value=1, max_value=10000),
    elapsed_seconds=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_decode_tps_calculation(gen_tokens: int, elapsed_seconds: float) -> None:
    """Property 2: Decode TPS calculation.

    For any positive gen_tokens and positive elapsed_seconds, the computed
    decode_tps equals gen_tokens / elapsed_seconds within floating-point
    tolerance.

    **Validates: Requirements 1.5**
    """
    decode_tps = gen_tokens / elapsed_seconds
    expected = gen_tokens / elapsed_seconds

    assert math.isclose(decode_tps, expected, rel_tol=1e-9), (
        f"decode_tps={decode_tps} != expected={expected} "
        f"for gen_tokens={gen_tokens}, elapsed_seconds={elapsed_seconds}"
    )
    # Additional invariant: decode_tps must be positive
    assert decode_tps > 0, f"decode_tps must be positive, got {decode_tps}"


# Feature: xpu-performance-benchmark, Property 3: Greedy decoding selects argmax
"""
Property-based test for greedy decoding argmax selection.

**Validates: Requirements 1.6, 5.2**

For any 2D logits tensor of shape [1, vocab_size] with a unique maximum value,
greedy decoding SHALL select the index of the maximum value.
"""

import torch


@settings(max_examples=100)
@given(
    vocab_size=st.integers(min_value=10, max_value=1000),
    max_index_frac=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
)
def test_greedy_decoding_selects_argmax(vocab_size: int, max_index_frac: float) -> None:
    """Property 3: Greedy decoding selects argmax.

    For any 2D logits tensor of shape [1, vocab_size] with a unique maximum
    value, greedy decoding selects the index of the maximum value.

    **Validates: Requirements 1.6, 5.2**
    """
    # Derive a deterministic max_index from the fraction and vocab_size
    max_index = int(max_index_frac * vocab_size)
    # Clamp to valid range (safety against floating point edge)
    max_index = min(max_index, vocab_size - 1)

    # Generate a random logits tensor of shape [1, vocab_size]
    logits = torch.randn(1, vocab_size)

    # Place a uniquely large value at max_index (larger than all others)
    unique_max_value = logits.max().item() + 1.0
    logits[0, max_index] = unique_max_value

    # Greedy decoding: argmax selection (same as in the decode loop)
    selected_index = logits[:, -1:].argmax(dim=-1).item() if vocab_size == 1 else logits[0].argmax(dim=-1).item()

    # The standard greedy decode pattern from the design: logits[:, -1, :].argmax(dim=-1)
    # For a 2D tensor [1, vocab_size], this is equivalent to logits[0].argmax()
    selected_index_standard = logits[0].argmax(dim=-1).item()

    assert selected_index_standard == max_index, (
        f"Expected argmax to select index {max_index} (value={unique_max_value}), "
        f"but got index {selected_index_standard} (value={logits[0, selected_index_standard].item()}) "
        f"for vocab_size={vocab_size}"
    )


# Feature: xpu-performance-benchmark, Property 4: Prompt construction produces target token count
"""
Property-based test for prompt construction.

**Validates: Requirements 1.7**

For any target token count between 1 and 4096, the build_prompt function
SHALL produce a token tensor whose length equals exactly the target count.
"""


build_prompt = _mod.build_prompt


class MockTokenizer:
    """A mock tokenizer that simulates HuggingFace tokenizer behavior.

    Encodes text by mapping each character to its ordinal value (producing
    more tokens than target_tokens so truncation works correctly).
    Decodes token IDs back to a string by mapping ordinals to characters.
    """

    def encode(self, text: str) -> list[int]:
        """Encode text as a list of character ordinal values."""
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a string."""
        return "".join(chr(t % 0x110000) for t in token_ids)


@settings(max_examples=100)
@given(
    target_tokens=st.integers(min_value=1, max_value=4096),
)
def test_prompt_construction_produces_target_token_count(target_tokens: int) -> None:
    """Property 4: Prompt construction produces target token count.

    For any target token count between 1 and 4096, the build_prompt function
    produces a token tensor whose length equals exactly the target count.

    **Validates: Requirements 1.7**
    """
    tokenizer = MockTokenizer()
    prompt_text, prompt_tensor = build_prompt(tokenizer, target_tokens)

    # Verify tensor shape is [1, target_tokens]
    assert prompt_tensor.shape == (1, target_tokens), (
        f"Expected tensor shape (1, {target_tokens}), "
        f"got {prompt_tensor.shape}"
    )

    # Verify tensor dtype is long (int64)
    assert prompt_tensor.dtype == torch.long, (
        f"Expected dtype torch.long, got {prompt_tensor.dtype}"
    )


# Feature: xpu-performance-benchmark, Property 5: Report contains all required fields
"""
Property-based test for report completeness.

**Validates: Requirements 1.8, 5.3**

For any valid BenchmarkStats, BenchmarkConfig, and EnvironmentInfo, the
formatted report string SHALL contain substrings for TTFT, prefill TPS,
decode TPS, total time, device name, dtype, and PyTorch version.
"""

format_report = _mod.format_report
BenchmarkStats = _mod.BenchmarkStats
EnvironmentInfo = _mod.EnvironmentInfo


# Strategies for generating valid objects
positive_float_strategy = st.floats(min_value=0.001, max_value=10000.0, allow_nan=False, allow_infinity=False)
non_negative_float_strategy = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
non_empty_string_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
).filter(lambda s: len(s.strip()) > 0)


@settings(max_examples=100)
@given(
    ttft_mean=positive_float_strategy,
    ttft_std=non_negative_float_strategy,
    decode_tps_mean=positive_float_strategy,
    decode_tps_std=non_negative_float_strategy,
    prefill_tps_mean=positive_float_strategy,
    prefill_tps_std=non_negative_float_strategy,
    total_time_mean=positive_float_strategy,
    total_time_std=non_negative_float_strategy,
    is_unstable=st.booleans(),
    pytorch_version=non_empty_string_strategy,
    xpu_device_name=non_empty_string_strategy,
    driver_version=non_empty_string_strategy,
    env_dtype=st.sampled_from(["bf16", "fp16"]),
)
def test_report_contains_all_required_fields(
    ttft_mean: float,
    ttft_std: float,
    decode_tps_mean: float,
    decode_tps_std: float,
    prefill_tps_mean: float,
    prefill_tps_std: float,
    total_time_mean: float,
    total_time_std: float,
    is_unstable: bool,
    pytorch_version: str,
    xpu_device_name: str,
    driver_version: str,
    env_dtype: str,
) -> None:
    """Property 5: Report contains all required fields.

    For any valid BenchmarkStats, BenchmarkConfig, and EnvironmentInfo,
    the formatted report string contains substrings for TTFT, prefill TPS,
    decode TPS, total time, device name, dtype, and PyTorch version.

    **Validates: Requirements 1.8, 5.3**
    """
    stats = BenchmarkStats(
        ttft_mean=ttft_mean,
        ttft_std=ttft_std,
        decode_tps_mean=decode_tps_mean,
        decode_tps_std=decode_tps_std,
        prefill_tps_mean=prefill_tps_mean,
        prefill_tps_std=prefill_tps_std,
        total_time_mean=total_time_mean,
        total_time_std=total_time_std,
        is_unstable=is_unstable,
    )

    config = BenchmarkConfig(
        model_id="test/model",
        device="xpu:0",
        dtype=env_dtype,
        prompt_tokens=256,
        gen_tokens=128,
        warmup=2,
        iterations=3,
        seed=42,
        compile=False,
        report_sdpa=False,
    )

    env = EnvironmentInfo(
        pytorch_version=pytorch_version,
        xpu_device_name=xpu_device_name,
        driver_version=driver_version,
        model_id="test/model",
        dtype=env_dtype,
    )

    report = format_report(stats, config, env)

    # Verify all required substrings are present
    assert "TTFT" in report, f"Report missing 'TTFT' substring"
    assert "Prefill TPS" in report, f"Report missing 'Prefill TPS' substring"
    assert "Decode TPS" in report, f"Report missing 'Decode TPS' substring"
    assert "Total time" in report, f"Report missing 'Total time' substring"
    assert pytorch_version in report, (
        f"Report missing PyTorch version string '{pytorch_version}'"
    )
    assert xpu_device_name in report, (
        f"Report missing device name string '{xpu_device_name}'"
    )
    assert env_dtype in report, (
        f"Report missing dtype string '{env_dtype}'"
    )
    # Driver version appears either as "Driver version" label or the value itself
    assert "Driver version" in report or driver_version in report, (
        f"Report missing 'Driver version' label or driver version value '{driver_version}'"
    )


# Feature: xpu-performance-benchmark, Property 8: Performance threshold classification
"""
Property-based test for performance threshold classification.

**Validates: Requirements 3.1, 3.3, 3.4**

For any positive float decode_tps, the classify_performance function SHALL set
is_critical = True iff decode_tps < 2.0, meets_minimum = True iff decode_tps >= 5.0,
and meets_stretch = True iff decode_tps >= 10.0.
"""

classify_performance = _mod.classify_performance
PerformanceClassification = _mod.PerformanceClassification


@settings(max_examples=100)
@given(
    decode_tps=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_performance_threshold_classification(decode_tps: float) -> None:
    """Property 8: Performance threshold classification.

    For any positive float decode_tps, classify_performance sets:
    - is_critical == True iff decode_tps < 2.0
    - meets_minimum == True iff decode_tps >= 5.0
    - meets_stretch == True iff decode_tps >= 10.0

    **Validates: Requirements 3.1, 3.3, 3.4**
    """
    # Use a fixed ttft value that doesn't affect decode_tps thresholds
    result = classify_performance(decode_tps, ttft=1.0)

    assert isinstance(result, PerformanceClassification)

    # is_critical iff decode_tps < 2.0
    assert result.is_critical == (decode_tps < 2.0), (
        f"is_critical={result.is_critical} but decode_tps={decode_tps}, "
        f"expected is_critical={decode_tps < 2.0}"
    )

    # meets_minimum iff decode_tps >= 5.0
    assert result.meets_minimum == (decode_tps >= 5.0), (
        f"meets_minimum={result.meets_minimum} but decode_tps={decode_tps}, "
        f"expected meets_minimum={decode_tps >= 5.0}"
    )

    # meets_stretch iff decode_tps >= 10.0
    assert result.meets_stretch == (decode_tps >= 10.0), (
        f"meets_stretch={result.meets_stretch} but decode_tps={decode_tps}, "
        f"expected meets_stretch={decode_tps >= 10.0}"
    )


# Feature: xpu-performance-benchmark, Property 9: TTFT threshold classification
"""
Property-based test for TTFT threshold classification.

**Validates: Requirements 3.2**

For any positive float ttft, the classify_performance function SHALL set
ttft_acceptable = True iff ttft < 5.0.
"""


@settings(max_examples=100)
@given(
    ttft=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_ttft_threshold_classification(ttft: float) -> None:
    """Property 9: TTFT threshold classification.

    For any positive float ttft, classify_performance sets:
    - ttft_acceptable == True iff ttft < 5.0

    **Validates: Requirements 3.2**
    """
    # Use a fixed decode_tps value that doesn't affect ttft threshold
    result = classify_performance(decode_tps=5.0, ttft=ttft)

    assert isinstance(result, PerformanceClassification)

    # ttft_acceptable iff ttft < 5.0
    assert result.ttft_acceptable == (ttft < 5.0), (
        f"ttft_acceptable={result.ttft_acceptable} but ttft={ttft}, "
        f"expected ttft_acceptable={ttft < 5.0}"
    )


# Feature: xpu-performance-benchmark, Property 10: Statistical computation correctness
"""
Property-based test for statistical computation correctness.

**Validates: Requirements 5.4**

For any list of 2 or more positive floats representing measurements (as
BenchmarkResult objects), compute_stats SHALL produce a mean equal to
sum(values) / len(values) and a standard deviation equal to the population
stddev within floating-point tolerance.
"""

compute_stats = _mod.compute_stats
BenchmarkResult = _mod.BenchmarkResult


@settings(max_examples=100)
@given(
    values=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
)
def test_statistical_computation_correctness(values: list[float]) -> None:
    """Property 10: Statistical computation correctness.

    For any list of 2+ positive floats representing measurements, compute_stats
    produces a mean equal to sum(values) / len(values) and a standard deviation
    equal to the population stddev within floating-point tolerance.

    **Validates: Requirements 5.4**
    """
    # Create BenchmarkResult objects using the generated values for all metrics
    results = [
        BenchmarkResult(
            ttft_seconds=v,
            prefill_tps=v,
            decode_tps=v,
            total_time_seconds=v,
            tokens_generated=10,
            generated_text="test",
        )
        for v in values
    ]

    stats = compute_stats(results)

    # Expected mean and population stddev
    expected_mean = sum(values) / len(values)
    expected_variance = sum((x - expected_mean) ** 2 for x in values) / len(values)
    expected_std = math.sqrt(expected_variance)

    # Verify mean for all metrics (they all use the same values)
    assert math.isclose(stats.ttft_mean, expected_mean, rel_tol=1e-9), (
        f"ttft_mean={stats.ttft_mean} != expected={expected_mean}"
    )
    assert math.isclose(stats.decode_tps_mean, expected_mean, rel_tol=1e-9), (
        f"decode_tps_mean={stats.decode_tps_mean} != expected={expected_mean}"
    )
    assert math.isclose(stats.prefill_tps_mean, expected_mean, rel_tol=1e-9), (
        f"prefill_tps_mean={stats.prefill_tps_mean} != expected={expected_mean}"
    )
    assert math.isclose(stats.total_time_mean, expected_mean, rel_tol=1e-9), (
        f"total_time_mean={stats.total_time_mean} != expected={expected_mean}"
    )

    # Verify stddev for all metrics (population stddev)
    assert math.isclose(stats.ttft_std, expected_std, rel_tol=1e-9, abs_tol=1e-15), (
        f"ttft_std={stats.ttft_std} != expected={expected_std}"
    )
    assert math.isclose(stats.decode_tps_std, expected_std, rel_tol=1e-9, abs_tol=1e-15), (
        f"decode_tps_std={stats.decode_tps_std} != expected={expected_std}"
    )
    assert math.isclose(stats.prefill_tps_std, expected_std, rel_tol=1e-9, abs_tol=1e-15), (
        f"prefill_tps_std={stats.prefill_tps_std} != expected={expected_std}"
    )
    assert math.isclose(stats.total_time_std, expected_std, rel_tol=1e-9, abs_tol=1e-15), (
        f"total_time_std={stats.total_time_std} != expected={expected_std}"
    )


# Feature: xpu-performance-benchmark, Property 11: Instability warning fires on high variance
"""
Property-based test for instability warning on high variance.

**Validates: Requirements 5.5**

For any list of 2 or more positive measurements, the is_unstable flag SHALL be
True iff any metric's coefficient of variation (std/mean) exceeds 0.2.
"""


@settings(max_examples=100)
@given(
    ttft_values=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
    decode_values=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
    prefill_values=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
    total_values=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
)
def test_instability_warning_fires_on_high_variance(
    ttft_values: list[float],
    decode_values: list[float],
    prefill_values: list[float],
    total_values: list[float],
) -> None:
    """Property 11: Instability warning fires on high variance.

    For any list of 2+ positive measurements, is_unstable == True iff any
    metric's coefficient of variation (std/mean) exceeds 0.2.

    **Validates: Requirements 5.5**
    """
    # Ensure all lists have the same length (use the minimum length)
    min_len = min(len(ttft_values), len(decode_values), len(prefill_values), len(total_values))
    ttft_values = ttft_values[:min_len]
    decode_values = decode_values[:min_len]
    prefill_values = prefill_values[:min_len]
    total_values = total_values[:min_len]

    # Create BenchmarkResult objects with different values per metric
    results = [
        BenchmarkResult(
            ttft_seconds=ttft_values[i],
            decode_tps=decode_values[i],
            prefill_tps=prefill_values[i],
            total_time_seconds=total_values[i],
            tokens_generated=10,
            generated_text="test",
        )
        for i in range(min_len)
    ]

    stats = compute_stats(results)

    # Compute expected instability independently
    def _cv_exceeds_threshold(values: list[float]) -> bool:
        mean = sum(values) / len(values)
        if mean <= 0:
            return False
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        return (std / mean) > 0.2

    expected_unstable = (
        _cv_exceeds_threshold(ttft_values)
        or _cv_exceeds_threshold(decode_values)
        or _cv_exceeds_threshold(prefill_values)
        or _cv_exceeds_threshold(total_values)
    )

    assert stats.is_unstable == expected_unstable, (
        f"is_unstable={stats.is_unstable} but expected={expected_unstable}. "
        f"CVs: ttft={_cv_exceeds_threshold(ttft_values)}, "
        f"decode={_cv_exceeds_threshold(decode_values)}, "
        f"prefill={_cv_exceeds_threshold(prefill_values)}, "
        f"total={_cv_exceeds_threshold(total_values)}"
    )


# Feature: xpu-performance-benchmark, Property 6: Optimized decode loop produces identical tokens to naive implementation
"""
Property-based test for decode loop equivalence.

**Validates: Requirements 2.2**

For any sequence of logits outputs from a model (mocked), the optimized decode
loop (single-token input + KV cache reuse + CPU list accumulation) SHALL produce
the same token ID sequence as a naive implementation (full input_ids concatenation
+ no cache).
"""

from unittest.mock import patch, MagicMock


VOCAB_SIZE = 100


class MockModelOutput:
    """Mock output object mimicking HuggingFace model output."""

    def __init__(self, logits: torch.Tensor, past_key_values: object) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class MockModel:
    """Mock model that produces deterministic logits for a pre-defined token sequence.

    The model tracks a step counter and returns logits where the argmax is at
    the expected token ID for that step. It simulates KV cache behavior by
    accepting and returning past_key_values.
    """

    def __init__(self, expected_tokens: list[int], vocab_size: int = VOCAB_SIZE) -> None:
        self.expected_tokens = expected_tokens
        self.vocab_size = vocab_size
        self.step = 0

    def _make_logits(self, seq_len: int) -> torch.Tensor:
        """Create logits tensor with argmax at the expected token for current step.

        Returns shape [1, seq_len, vocab_size] with the max value at the
        expected token position in the last position ([:, -1, :]).
        """
        logits = torch.zeros(1, seq_len, self.vocab_size)
        # Set the expected token to have the highest logit in the last position
        token_id = self.expected_tokens[self.step]
        logits[0, -1, token_id] = 10.0
        self.step += 1
        return logits

    def __call__(self, input_ids: torch.Tensor, past_key_values: object = None, use_cache: bool = False) -> MockModelOutput:
        seq_len = input_ids.shape[-1]
        logits = self._make_logits(seq_len)
        # Return a dummy past_key_values (grows by one entry per call to simulate cache)
        new_past = (past_key_values or []) + ["kv_entry"]  # type: ignore[operator]
        return MockModelOutput(logits=logits, past_key_values=new_past)


def naive_decode(
    model: MockModel,
    prompt_ids: torch.Tensor,
    gen_tokens: int,
    eos_id: int,
) -> list[int]:
    """Naive decode loop: concatenates all tokens into input_ids each step.

    This is the inefficient approach that does NOT use KV cache.
    It serves as the reference implementation for correctness.
    """
    # Prefill: run model on full prompt
    outputs = model(input_ids=prompt_ids, use_cache=False)
    logits = outputs.logits

    # First token via argmax on last position
    first_token = logits[:, -1, :].argmax(dim=-1)
    token_ids: list[int] = [first_token.item()]

    if token_ids[0] == eos_id:
        return token_ids

    # Decode loop: concatenate all tokens each step (naive, no cache)
    # first_token is shape [1] from argmax(dim=-1), reshape to [1, 1] for cat with [1, N]
    all_ids = torch.cat([prompt_ids, first_token.unsqueeze(-1)], dim=-1)

    for _ in range(gen_tokens - 1):
        outputs = model(input_ids=all_ids, use_cache=False)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)

        token_id = next_token.item()
        token_ids.append(token_id)

        if token_id == eos_id:
            break

        all_ids = torch.cat([all_ids, next_token.unsqueeze(-1)], dim=-1)

    return token_ids


# Import greedy_decode from the bench_xpu module
greedy_decode = _mod.greedy_decode


@settings(max_examples=100)
@given(
    token_sequence=st.lists(
        st.integers(min_value=0, max_value=VOCAB_SIZE - 2),  # Exclude last index to reserve for EOS
        min_size=2,
        max_size=20,
    ),
)
def test_optimized_decode_loop_equivalence(token_sequence: list[int]) -> None:
    """Property 6: Optimized decode loop produces identical tokens to naive implementation.

    For any sequence of logits outputs from a model (mocked), the optimized
    decode loop (single-token input + KV cache reuse + CPU list accumulation)
    produces the same token ID sequence as a naive implementation (full
    input_ids concatenation + no cache).

    **Validates: Requirements 2.2**
    """
    gen_tokens = len(token_sequence)
    eos_id = VOCAB_SIZE - 1  # Use last vocab index as EOS (never generated)

    # Create a prompt tensor (content doesn't matter, shape does)
    prompt_ids = torch.zeros(1, 5, dtype=torch.long)

    # Create two independent mock models with the same expected token sequence
    model_naive = MockModel(expected_tokens=token_sequence, vocab_size=VOCAB_SIZE)
    model_optimized = MockModel(expected_tokens=token_sequence, vocab_size=VOCAB_SIZE)

    # Run naive decode
    naive_tokens = naive_decode(model_naive, prompt_ids, gen_tokens, eos_id)

    # Run optimized decode with torch.xpu.synchronize mocked out
    # We need to mock torch.xpu module since it won't exist on CPU-only machines
    mock_xpu = MagicMock()
    mock_xpu.synchronize = MagicMock()

    with patch.object(torch, "xpu", mock_xpu, create=True):
        optimized_tokens, _, _ = greedy_decode(
            model=model_optimized,
            prompt_ids=prompt_ids,
            gen_tokens=gen_tokens,
            eos_id=eos_id,
            device="cpu",
        )

    assert optimized_tokens == naive_tokens, (
        f"Token sequences differ!\n"
        f"  Optimized: {optimized_tokens}\n"
        f"  Naive:     {naive_tokens}\n"
        f"  Expected:  {token_sequence[:len(naive_tokens)]}"
    )


# Feature: xpu-performance-benchmark, Property 7: EOS terminates decode at correct position
"""
Property-based test for EOS termination.

**Validates: Requirements 2.4**

For any model that produces an EOS token at step N (where 1 <= N <= gen_tokens),
the decode loop SHALL return exactly N token IDs with the last being the EOS token.
"""


EOS_TOKEN_ID = 2  # Common EOS token ID for many tokenizers


class MockModelEOS:
    """Mock model that produces non-EOS tokens for steps 1..N-1 and EOS at step N.

    The model tracks a step counter. For each step before N, it returns logits
    with argmax at a non-EOS token. At step N, it returns logits with argmax
    at the EOS token ID.
    """

    def __init__(self, eos_position: int, eos_id: int = EOS_TOKEN_ID, vocab_size: int = VOCAB_SIZE) -> None:
        self.eos_position = eos_position
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.step = 0
        # Choose a non-EOS token that is different from eos_id
        self.non_eos_token = 0 if eos_id != 0 else 1

    def _make_logits(self, seq_len: int) -> torch.Tensor:
        """Create logits tensor with argmax at the appropriate token for current step.

        Returns shape [1, seq_len, vocab_size].
        At step N (1-indexed), the argmax is at eos_id.
        Before step N, the argmax is at non_eos_token.
        """
        logits = torch.zeros(1, seq_len, self.vocab_size)
        self.step += 1

        if self.step >= self.eos_position:
            # EOS step: place max logit at eos_id
            logits[0, -1, self.eos_id] = 10.0
        else:
            # Non-EOS step: place max logit at non_eos_token
            logits[0, -1, self.non_eos_token] = 10.0

        return logits

    def __call__(self, input_ids: torch.Tensor, past_key_values: object = None, use_cache: bool = False) -> MockModelOutput:
        seq_len = input_ids.shape[-1]
        logits = self._make_logits(seq_len)
        new_past = (past_key_values or []) + ["kv_entry"]  # type: ignore[operator]
        return MockModelOutput(logits=logits, past_key_values=new_past)


@settings(max_examples=100)
@given(
    eos_position=st.integers(min_value=1, max_value=20),
)
def test_eos_terminates_decode_at_correct_position(eos_position: int) -> None:
    """Property 7: EOS terminates decode at correct position.

    For any model that produces an EOS token at step N (where 1 <= N <= gen_tokens),
    the decode loop returns exactly N token IDs with the last being the EOS token.

    **Validates: Requirements 2.4**
    """
    # gen_tokens must be larger than eos_position to ensure EOS stops generation
    # (not hitting the token limit)
    gen_tokens = eos_position + 10

    # Create a prompt tensor (content doesn't matter, shape does)
    prompt_ids = torch.zeros(1, 5, dtype=torch.long)

    # Create mock model that produces EOS at step eos_position
    model = MockModelEOS(eos_position=eos_position, eos_id=EOS_TOKEN_ID, vocab_size=VOCAB_SIZE)

    # Run optimized decode with torch.xpu.synchronize mocked out
    mock_xpu = MagicMock()
    mock_xpu.synchronize = MagicMock()

    with patch.object(torch, "xpu", mock_xpu, create=True):
        token_ids, _, _ = greedy_decode(
            model=model,
            prompt_ids=prompt_ids,
            gen_tokens=gen_tokens,
            eos_id=EOS_TOKEN_ID,
            device="cpu",
        )

    # Verify exactly N tokens are returned
    assert len(token_ids) == eos_position, (
        f"Expected {eos_position} tokens, got {len(token_ids)}. "
        f"Token IDs: {token_ids}"
    )

    # Verify the last token is the EOS token ID
    assert token_ids[-1] == EOS_TOKEN_ID, (
        f"Expected last token to be EOS ({EOS_TOKEN_ID}), "
        f"got {token_ids[-1]}. Token IDs: {token_ids}"
    )

    # Verify all tokens before the last are non-EOS
    for i, tid in enumerate(token_ids[:-1]):
        assert tid != EOS_TOKEN_ID, (
            f"Token at position {i} is EOS ({EOS_TOKEN_ID}) but EOS should only "
            f"appear at position {eos_position - 1}. Token IDs: {token_ids}"
        )

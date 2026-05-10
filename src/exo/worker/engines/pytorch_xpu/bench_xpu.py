"""XPU Performance Benchmark for Qwen3.5-4B inference on Intel Arc iGPUs.

This module provides dataclasses and pure logic functions for benchmarking
inference performance. It is importable without XPU hardware — only the
orchestration functions (implemented in later tasks) require a real device.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run, parsed from CLI arguments."""

    model_id: str = "Qwen/Qwen3.5-4B"
    device: str = "xpu:0"
    dtype: str = "bf16"
    prompt_tokens: int = 256
    gen_tokens: int = 128
    warmup: int = 2
    iterations: int = 3
    seed: int = 42
    compile: bool = False
    report_sdpa: bool = False


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing results from a single benchmark iteration."""

    ttft_seconds: float
    prefill_tps: float
    decode_tps: float
    total_time_seconds: float
    tokens_generated: int
    generated_text: str


@dataclass(frozen=True)
class BenchmarkStats:
    """Aggregated statistics across multiple benchmark iterations."""

    ttft_mean: float
    ttft_std: float
    decode_tps_mean: float
    decode_tps_std: float
    prefill_tps_mean: float
    prefill_tps_std: float
    total_time_mean: float
    total_time_std: float
    is_unstable: bool  # True if any CV > 0.2


@dataclass(frozen=True)
class PerformanceClassification:
    """Threshold-based classification of benchmark results."""

    meets_minimum: bool  # decode_tps >= 5
    meets_stretch: bool  # decode_tps >= 10
    is_critical: bool  # decode_tps < 2
    ttft_acceptable: bool  # ttft < 5.0 (for 256-token prompt)


@dataclass(frozen=True)
class EnvironmentInfo:
    """Environment metadata for reproducibility."""

    pytorch_version: str
    xpu_device_name: str
    driver_version: str
    model_id: str
    dtype: str
    compile_status: str | None = None
    sdpa_backend: str | None = None


# ---------------------------------------------------------------------------
# CLI Parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    """Parse command-line arguments into a BenchmarkConfig.

    Args:
        argv: Argument list (defaults to sys.argv[1:] if None).

    Returns:
        A frozen BenchmarkConfig with all fields populated.
    """
    parser = argparse.ArgumentParser(
        description="XPU Performance Benchmark for LLM inference"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--device", type=str, default="xpu:0", help="XPU device string"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Numeric precision (bf16 or fp16)",
    )
    parser.add_argument(
        "--prompt_tokens",
        type=int,
        default=256,
        help="Target prompt length in tokens",
    )
    parser.add_argument(
        "--gen_tokens",
        type=int,
        default=128,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Warmup iteration count"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Timed iteration count"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply torch.compile optimization",
    )
    parser.add_argument(
        "--report-sdpa",
        action="store_true",
        default=False,
        help="Report SDPA backend info",
    )

    args = parser.parse_args(argv)

    return BenchmarkConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        compile=args.compile,
        report_sdpa=args.report_sdpa,
    )


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------


def build_prompt(
    tokenizer: object, target_tokens: int
) -> tuple[str, torch.Tensor]:
    """Build a prompt of exactly target_tokens length.

    Strategy: Repeat a fixed sentence, encode, truncate to target_tokens,
    then decode back to get the actual prompt text.

    Args:
        tokenizer: A HuggingFace tokenizer with encode/decode methods.
        target_tokens: Desired number of tokens in the prompt.

    Returns:
        A tuple of (prompt_text, prompt_tensor) where prompt_tensor has
        shape [1, target_tokens].
    """
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    # Over-generate then truncate
    repeated = base_sentence * (target_tokens // 5 + 10)
    token_ids = tokenizer.encode(repeated)  # type: ignore[union-attr]
    truncated = token_ids[:target_tokens]
    prompt_text = tokenizer.decode(truncated)  # type: ignore[union-attr]
    prompt_tensor = torch.tensor([truncated], dtype=torch.long)
    return prompt_text, prompt_tensor


# ---------------------------------------------------------------------------
# Statistics Computation
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean of a list of floats."""
    return sum(values) / len(values)


def _stddev(values: list[float], mean: float) -> float:
    """Compute population standard deviation given a precomputed mean."""
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def compute_stats(results: list[BenchmarkResult]) -> BenchmarkStats:
    """Compute mean/stddev statistics across benchmark iterations.

    Args:
        results: List of BenchmarkResult from timed iterations (len >= 1).

    Returns:
        BenchmarkStats with mean, stddev, and instability flag.
        The is_unstable flag is True if any metric's coefficient of
        variation (std/mean) exceeds 0.2.
    """
    ttft_values = [r.ttft_seconds for r in results]
    decode_values = [r.decode_tps for r in results]
    prefill_values = [r.prefill_tps for r in results]
    total_values = [r.total_time_seconds for r in results]

    ttft_mean = _mean(ttft_values)
    ttft_std = _stddev(ttft_values, ttft_mean)

    decode_tps_mean = _mean(decode_values)
    decode_tps_std = _stddev(decode_values, decode_tps_mean)

    prefill_tps_mean = _mean(prefill_values)
    prefill_tps_std = _stddev(prefill_values, prefill_tps_mean)

    total_time_mean = _mean(total_values)
    total_time_std = _stddev(total_values, total_time_mean)

    # Check instability: CV > 0.2 for any metric with non-zero mean
    is_unstable = False
    for mean_val, std_val in [
        (ttft_mean, ttft_std),
        (decode_tps_mean, decode_tps_std),
        (prefill_tps_mean, prefill_tps_std),
        (total_time_mean, total_time_std),
    ]:
        if mean_val > 0 and (std_val / mean_val) > 0.2:
            is_unstable = True
            break

    return BenchmarkStats(
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


# ---------------------------------------------------------------------------
# Performance Classification
# ---------------------------------------------------------------------------


def classify_performance(
    decode_tps: float, ttft: float
) -> PerformanceClassification:
    """Classify benchmark results against performance targets.

    Args:
        decode_tps: Decode tokens per second (mean).
        ttft: Time to first token in seconds (mean).

    Returns:
        PerformanceClassification with threshold flags.
    """
    return PerformanceClassification(
        meets_minimum=decode_tps >= 5.0,
        meets_stretch=decode_tps >= 10.0,
        is_critical=decode_tps < 2.0,
        ttft_acceptable=ttft < 5.0,
    )


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------


def format_report(
    stats: BenchmarkStats, config: BenchmarkConfig, env: EnvironmentInfo
) -> str:
    """Format benchmark results into a structured text report.

    Args:
        stats: Aggregated statistics from benchmark iterations.
        config: The benchmark configuration used.
        env: Environment metadata.

    Returns:
        A multi-line string report suitable for terminal output.
    """
    classification = classify_performance(stats.decode_tps_mean, stats.ttft_mean)

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("XPU PERFORMANCE BENCHMARK REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Timing results
    lines.append("--- Timing Results ---")
    lines.append(
        f"TTFT:        {stats.ttft_mean:.3f} ± {stats.ttft_std:.3f} s"
    )
    lines.append(
        f"Prefill TPS: {stats.prefill_tps_mean:.1f} ± {stats.prefill_tps_std:.1f} tok/s"
    )
    lines.append(
        f"Decode TPS:  {stats.decode_tps_mean:.1f} ± {stats.decode_tps_std:.1f} tok/s"
    )
    lines.append(
        f"Total time:  {stats.total_time_mean:.3f} ± {stats.total_time_std:.3f} s"
    )
    lines.append("")

    # Performance classification
    lines.append("--- Performance Classification ---")
    if classification.is_critical:
        lines.append(
            "⛔ CRITICAL: decode_tps < 2.0 — investigate kernel fallbacks or memory bandwidth issues"
        )
    if classification.meets_stretch:
        lines.append("✅ Stretch target met: decode_tps >= 10.0")
    elif classification.meets_minimum:
        lines.append("✅ Minimum target met: decode_tps >= 5.0")
    else:
        lines.append("❌ Below minimum target: decode_tps < 5.0")

    if classification.ttft_acceptable:
        lines.append("✅ TTFT acceptable: < 5.0 s")
    else:
        lines.append("❌ TTFT too high: >= 5.0 s")
    lines.append("")

    # Environment info
    lines.append("--- Environment ---")
    lines.append(f"PyTorch version: {env.pytorch_version}")
    lines.append(f"XPU device:      {env.xpu_device_name}")
    lines.append(f"Driver version:  {env.driver_version}")
    lines.append(f"Model:           {env.model_id}")
    lines.append(f"Dtype:           {env.dtype}")
    if env.compile_status is not None:
        lines.append(f"torch.compile:   {env.compile_status}")
    if env.sdpa_backend is not None:
        lines.append(f"SDPA backend:    {env.sdpa_backend}")
    lines.append("")

    # Configuration
    lines.append("--- Configuration ---")
    lines.append(f"Prompt tokens:   {config.prompt_tokens}")
    lines.append(f"Gen tokens:      {config.gen_tokens}")
    lines.append(f"Warmup:          {config.warmup}")
    lines.append(f"Iterations:      {config.iterations}")
    lines.append(f"Seed:            {config.seed}")
    lines.append("")

    # Instability warning
    if stats.is_unstable:
        lines.append(
            "⚠ WARNING: High variance detected (CV > 20%). "
            "Consider increasing --warmup count."
        )
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimized Greedy Decode Loop
# ---------------------------------------------------------------------------


@torch.inference_mode()
def greedy_decode(
    model: object,
    prompt_ids: torch.Tensor,
    gen_tokens: int,
    eos_id: int,
    device: str,
) -> tuple[list[int], float, float]:
    """Optimized greedy decode loop with KV cache reuse.

    Returns (generated_token_ids, prefill_seconds, decode_seconds).

    Key optimizations:
    1. No torch.cat — only the single current token tensor is on device
    2. Token IDs accumulated in a Python list (CPU, zero-cost)
    3. No tokenizer.decode() during timed loop
    4. KV cache (past_key_values) reused across steps via use_cache=True
    5. torch.inference_mode() eliminates autograd overhead
    6. .item() only for EOS check (unavoidable sync point)
    """
    # Move prompt to device
    prompt_ids = prompt_ids.to(device)

    # --- Prefill ---
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    outputs = model(input_ids=prompt_ids, use_cache=True)  # type: ignore[operator]
    torch.xpu.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = outputs.past_key_values
    logits = outputs.logits

    # First token via argmax
    first_token = logits[:, -1, :].argmax(dim=-1)
    token_ids: list[int] = [first_token.item()]

    if token_ids[0] == eos_id:
        return token_ids, prefill_time, 0.0

    # --- Decode loop ---
    cur_token = first_token.unsqueeze(0)  # shape: [1, 1]

    torch.xpu.synchronize()
    t1 = time.perf_counter()

    for _ in range(gen_tokens - 1):
        outputs = model(  # type: ignore[operator]
            input_ids=cur_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)

        token_id = next_token.item()  # sync point for EOS check
        token_ids.append(token_id)

        if token_id == eos_id:
            break

        cur_token = next_token.unsqueeze(0)

    torch.xpu.synchronize()
    decode_time = time.perf_counter() - t1

    return token_ids, prefill_time, decode_time


# ---------------------------------------------------------------------------
# Benchmark Iteration
# ---------------------------------------------------------------------------


def run_benchmark_iteration(
    model: object,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run a single timed benchmark iteration.

    Args:
        model: The loaded HuggingFace model on XPU device.
        tokenizer: The tokenizer for decoding output.
        prompt_ids: Pre-tokenized prompt tensor of shape [1, prompt_tokens].
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing measurements for this iteration.
    """
    eos_id = tokenizer.eos_token_id  # type: ignore[union-attr]
    if eos_id is None:
        eos_id = 2  # Fallback EOS token ID

    # Run the optimized decode loop
    token_ids, prefill_time, decode_time = greedy_decode(
        model=model,
        prompt_ids=prompt_ids,
        gen_tokens=config.gen_tokens,
        eos_id=eos_id,
        device=config.device,
    )

    # Compute metrics
    tokens_generated = len(token_ids)
    total_time = prefill_time + decode_time

    # Prefill TPS: prompt_tokens / prefill_time
    prefill_tps = config.prompt_tokens / prefill_time if prefill_time > 0 else 0.0

    # Decode TPS: tokens generated in decode phase / decode time
    # The first token comes from prefill, so decode tokens = tokens_generated - 1
    decode_tokens = max(tokens_generated - 1, 0)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0

    # TTFT: prefill time (time to produce the first token)
    ttft_seconds = prefill_time

    # Decode the generated text (outside timed loop)
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)  # type: ignore[union-attr]

    return BenchmarkResult(
        ttft_seconds=ttft_seconds,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        total_time_seconds=total_time,
        tokens_generated=tokens_generated,
        generated_text=generated_text,
    )


# ---------------------------------------------------------------------------
# Device Detection and Environment Info
# ---------------------------------------------------------------------------


def detect_xpu_devices() -> list[str]:
    """Detect available XPU devices and return their names.

    Exits with code 1 and clear error message if no XPU devices are available.
    """
    if not torch.xpu.is_available():
        print(
            "ERROR: No XPU devices detected. Ensure Intel GPU drivers and PyTorch XPU are installed.",
            file=sys.stderr,
        )
        sys.exit(1)

    device_count = torch.xpu.device_count()
    devices = []
    for i in range(device_count):
        name = torch.xpu.get_device_name(i)
        devices.append(name)
        print(f"  XPU device {i}: {name}")

    return devices


def collect_environment_info(
    config: BenchmarkConfig,
    compile_status: str | None = None,
    sdpa_backend: str | None = None,
) -> EnvironmentInfo:
    """Collect environment metadata for the benchmark report."""
    pytorch_version = torch.__version__

    # Get device name
    device_idx = int(config.device.split(":")[-1]) if ":" in config.device else 0
    xpu_device_name = torch.xpu.get_device_name(device_idx)

    # Try to get driver version from Level Zero
    driver_version = "unknown"
    try:
        props = torch.xpu.get_device_properties(device_idx)
        if hasattr(props, "driver_version"):
            driver_version = str(props.driver_version)
    except Exception:
        pass

    return EnvironmentInfo(
        pytorch_version=pytorch_version,
        xpu_device_name=xpu_device_name,
        driver_version=driver_version,
        model_id=config.model_id,
        dtype=config.dtype,
        compile_status=compile_status,
        sdpa_backend=sdpa_backend,
    )


# ---------------------------------------------------------------------------
# torch.compile Support
# ---------------------------------------------------------------------------


def apply_torch_compile(
    model: object, config: BenchmarkConfig
) -> tuple[object, str | None]:
    """Apply torch.compile if --compile flag is set.

    Returns (model, compile_status) where compile_status is:
    - "success" if compilation succeeded
    - "fallback" if compilation failed and fell back to eager
    - None if --compile was not requested
    """
    if not config.compile:
        return model, None

    try:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")  # type: ignore[assignment]
        print(
            "torch.compile applied successfully (backend=inductor, mode=reduce-overhead)"
        )
        return model, "success"
    except Exception as e:
        print(
            f"WARNING: torch.compile failed ({e}), falling back to eager mode.",
            file=sys.stderr,
        )
        return model, "fallback"


# ---------------------------------------------------------------------------
# SDPA Backend Detection
# ---------------------------------------------------------------------------


def detect_sdpa_backend(device: str) -> str | None:
    """Run a diagnostic SDPA operation and report which backend is active.

    Returns the backend name ("flash", "math", "efficient") or None on failure.
    """
    try:
        import torch.nn.functional as F

        # Create small test tensors for SDPA diagnostic
        batch, heads, seq_len, head_dim = 1, 1, 16, 64
        q = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.float16
        )
        v = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.float16
        )

        # Try each backend to see which one works
        with torch.inference_mode():
            # Try flash attention
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                ):
                    F.scaled_dot_product_attention(q, k, v)
                    return "flash"
            except Exception:
                pass

            # Try efficient attention
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=False, enable_mem_efficient=True
                ):
                    F.scaled_dot_product_attention(q, k, v)
                    return "efficient"
            except Exception:
                pass

            # Fall back to math
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    F.scaled_dot_product_attention(q, k, v)
                    return "math"
            except Exception:
                pass

        # If none of the specific backends work, just run default
        F.scaled_dot_product_attention(q, k, v)
        return "default"

    except Exception as e:
        print(f"WARNING: SDPA backend detection failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the XPU performance benchmark."""
    # Parse CLI arguments
    config = parse_args()

    print("=" * 60)
    print("XPU Performance Benchmark")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    print(f"Prompt tokens: {config.prompt_tokens}")
    print(f"Gen tokens: {config.gen_tokens}")
    print(f"Warmup: {config.warmup}")
    print(f"Iterations: {config.iterations}")
    print(f"Seed: {config.seed}")
    print()

    # Detect XPU devices
    print("Detecting XPU devices...")
    devices = detect_xpu_devices()
    print(f"Found {len(devices)} XPU device(s)")
    print()

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(config.seed)

    # Determine dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[config.dtype]

    # Load model and tokenizer
    print(f"Loading model {config.model_id} in {config.dtype}...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            device_map=config.device,
        )
    except Exception as e:
        print(
            f"ERROR: Failed to load model '{config.model_id}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Model loaded successfully")
    print()

    # Apply torch.compile if requested
    model, compile_status = apply_torch_compile(model, config)

    # Detect SDPA backend if requested
    sdpa_backend = None
    if config.report_sdpa:
        print("Detecting SDPA backend...")
        sdpa_backend = detect_sdpa_backend(config.device)
        if sdpa_backend:
            print(f"SDPA backend: {sdpa_backend}")
        print()

    # Build prompt
    print(f"Building prompt ({config.prompt_tokens} tokens)...")
    prompt_text, prompt_ids = build_prompt(tokenizer, config.prompt_tokens)
    print(f"Prompt: {prompt_text[:80]}...")
    print()

    # Warmup iterations
    if config.warmup > 0:
        print(f"Running {config.warmup} warmup iteration(s)...")
        for i in range(config.warmup):
            try:
                run_benchmark_iteration(model, tokenizer, prompt_ids, config)
                print(f"  Warmup {i + 1}/{config.warmup} complete")
            except Exception as e:
                print(
                    f"ERROR: Model forward pass failed during warmup: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
        print()

    # Timed iterations
    print(f"Running {config.iterations} timed iteration(s)...")
    results: list[BenchmarkResult] = []
    for i in range(config.iterations):
        try:
            result = run_benchmark_iteration(model, tokenizer, prompt_ids, config)
            results.append(result)
            print(
                f"  Iteration {i + 1}/{config.iterations}: "
                f"TTFT={result.ttft_seconds:.3f}s, "
                f"Decode={result.decode_tps:.1f} tok/s, "
                f"Tokens={result.tokens_generated}"
            )
        except Exception as e:
            print(
                f"WARNING: Iteration {i + 1} failed: {e}. "
                f"Reporting results from {len(results)} successful iterations.",
                file=sys.stderr,
            )
            break

    if not results:
        print(
            "ERROR: No successful iterations. Cannot produce report.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()

    # Compute statistics and generate report
    stats = compute_stats(results)
    env = collect_environment_info(config, compile_status, sdpa_backend)
    report = format_report(stats, config, env)

    print(report)


if __name__ == "__main__":
    main()

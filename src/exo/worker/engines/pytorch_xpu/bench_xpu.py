"""XPU Performance Benchmark for Qwen3.5-4B inference on Intel Arc iGPUs.

This module provides dataclasses and pure logic functions for benchmarking
inference performance. It is importable without XPU hardware — only the
orchestration functions (implemented in later tasks) require a real device.

Supports benchmark modes:
- single-request-decode: Single request greedy decode (default)
- prefill: Prefill-only timing for TTFT analysis
- continuous-batching: Concurrent request throughput measurement

Outputs pipeline performance metrics including tokens per second, TTFT,
inter-token latency statistics, per-rank utilization, communication time
ratio, and pipeline bubble estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

from exo.worker.engines.pytorch_xpu.instrumentation import (
    PerformanceEvent,
    PerformanceRecorder,
)
from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PipelineLayerDistribution,
    default_layer_distribution,
    parse_layer_distribution_arg,
)
from exo.worker.engines.pytorch_xpu.stage_balancing import (
    PerStageTimingSummary,
    export_per_layer_timing,
    export_per_stage_timing,
    recommend_pipeline_layer_distribution,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BenchmarkMode = Literal[
    "single-request-decode", "prefill", "continuous-batching"
]
"""Supported benchmark execution modes."""


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
    json_output_path: str | None = None
    benchmark_mode: BenchmarkMode = "single-request-decode"
    pipeline_layer_distribution: PipelineLayerDistribution | None = None
    recommend_layer_distribution: bool = False
    concurrent_requests: int = 4


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
class InterTokenLatencyStats:
    """Inter-token latency statistics across decode steps."""

    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    all_latencies: list[float]


@dataclass(frozen=True)
class PipelineMetrics:
    """Pipeline-level performance metrics from instrumentation."""

    tokens_per_second: float
    time_to_first_token_seconds: float
    inter_token_latency: InterTokenLatencyStats
    per_rank_utilization: dict[int, float]
    communication_time_ratio: float
    pipeline_bubble_estimate: float


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


@dataclass(frozen=True)
class MemoryLoadingBenchmark:
    """Memory metrics from model loading, for benchmark output."""

    peak_rss_mib: float
    rss_before_mib: float
    rss_after_mib: float
    rss_delta_mib: float
    tensor_mib_loaded: float
    loading_duration_seconds: float


@dataclass(frozen=True)
class BenchmarkJsonOutput:
    """Structured JSON output for cross-run comparison."""

    metadata: dict[str, Any]
    metrics: dict[str, Any]
    raw_events: list[dict[str, Any]]


@dataclass(frozen=True)
class ContinuousBatchingBenchmarkResult:
    """Benchmark result for continuous batching throughput measurement.

    Reports aggregate and per-request tokens per second for a given
    concurrency level, along with pipeline utilization metrics.
    """

    concurrent_requests: int
    total_tokens_generated: int
    total_decode_steps: int
    wall_clock_seconds: float
    aggregate_tokens_per_second: float  # total_tokens / wall_clock
    per_request_tokens_per_second: float  # aggregate / concurrent_requests
    average_microbatch_size: float
    pipeline_occupancy: float


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
    parser.add_argument(
        "--json-output-path",
        type=str,
        default=None,
        help="Path to write JSON benchmark results for cross-run comparison",
    )
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default="single-request-decode",
        choices=[
            "single-request-decode",
            "prefill",
            "continuous-batching",
        ],
        help=(
            "Benchmark execution mode: "
            "single-request-decode (default), prefill, or continuous-batching"
        ),
    )
    parser.add_argument(
        "--pipeline-layer-distribution",
        type=str,
        default=None,
        help=(
            "Comma-separated layer distribution across pipeline ranks "
            "(e.g. '16,16,16,16' or '17,17,16,14'). "
            "Overrides the default balanced distribution."
        ),
    )
    parser.add_argument(
        "--recommend-layer-distribution",
        action="store_true",
        default=False,
        help=(
            "After benchmark completes, use per-layer timing to recommend "
            "an optimal pipeline layer distribution and print a comparison "
            "table showing current vs recommended distribution."
        ),
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=4,
        help=(
            "Number of concurrent requests for continuous-batching mode "
            "(default: 4). Only used when --benchmark-mode is "
            "continuous-batching."
        ),
    )

    args = parser.parse_args(argv)

    # Parse pipeline layer distribution if provided
    pipeline_layer_distribution: PipelineLayerDistribution | None = None
    if args.pipeline_layer_distribution is not None:
        # Default to 64 total layers and 4 ranks (Qwen3.5-27B on gremlin cluster)
        # The rank count is inferred from the number of comma-separated values
        parts = args.pipeline_layer_distribution.strip().split(",")
        world_size = len(parts)
        # Infer total layers from the sum of the distribution
        try:
            total_layers = sum(int(p.strip()) for p in parts)
        except ValueError:
            print(
                f"ERROR: Invalid --pipeline-layer-distribution: "
                f"'{args.pipeline_layer_distribution}'. "
                f"Expected comma-separated integers (e.g. '16,16,16,16').",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            pipeline_layer_distribution = parse_layer_distribution_arg(
                args.pipeline_layer_distribution,
                total_layers=total_layers,
                world_size=world_size,
            )
        except ValueError as exc:
            print(
                f"ERROR: Invalid --pipeline-layer-distribution: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

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
        json_output_path=args.json_output_path,
        benchmark_mode=args.benchmark_mode,
        pipeline_layer_distribution=pipeline_layer_distribution,
        recommend_layer_distribution=args.recommend_layer_distribution,
        concurrent_requests=args.concurrent_requests,
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
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[float], mean: float) -> float:
    """Compute population standard deviation given a precomputed mean."""
    if len(values) == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _median(values: list[float]) -> float:
    """Compute median of a list of floats."""
    if len(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    count = len(sorted_values)
    midpoint = count // 2
    if count % 2 == 0:
        return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0
    return sorted_values[midpoint]


def _percentile_95(values: list[float]) -> float:
    """Compute the 95th percentile of a list of floats."""
    if len(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    index = int(math.ceil(0.95 * len(sorted_values))) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def compute_inter_token_latency(
    token_timestamps: list[float],
) -> InterTokenLatencyStats:
    """Compute inter-token latency statistics from per-token timestamps.

    Args:
        token_timestamps: Monotonically increasing timestamps for each
            generated token (from time.perf_counter()).

    Returns:
        InterTokenLatencyStats with mean, median, and p95 latencies.
    """
    if len(token_timestamps) < 2:
        return InterTokenLatencyStats(
            mean_seconds=0.0,
            median_seconds=0.0,
            p95_seconds=0.0,
            all_latencies=[],
        )

    latencies: list[float] = []
    for i in range(1, len(token_timestamps)):
        latencies.append(token_timestamps[i] - token_timestamps[i - 1])

    mean_latency = _mean(latencies)
    median_latency = _median(latencies)
    p95_latency = _percentile_95(latencies)

    return InterTokenLatencyStats(
        mean_seconds=mean_latency,
        median_seconds=median_latency,
        p95_seconds=p95_latency,
        all_latencies=latencies,
    )


def compute_pipeline_metrics(
    recorder: PerformanceRecorder,
    total_tokens_generated: int,
    total_generation_time_seconds: float,
    time_to_first_token_seconds: float,
    token_timestamps: list[float],
    rank_count: int = 4,
) -> PipelineMetrics:
    """Compute pipeline-level performance metrics from instrumentation data.

    Derives per-rank utilization, communication time ratio, and pipeline
    bubble estimate from recorded performance events.

    Args:
        recorder: PerformanceRecorder with collected events.
        total_tokens_generated: Number of tokens produced.
        total_generation_time_seconds: Wall time for full generation.
        time_to_first_token_seconds: Time to first token.
        token_timestamps: Per-token timestamps for latency computation.
        rank_count: Number of pipeline ranks (default 4).

    Returns:
        PipelineMetrics with all derived metrics.
    """
    tokens_per_second = (
        total_tokens_generated / total_generation_time_seconds
        if total_generation_time_seconds > 0.0
        else 0.0
    )

    inter_token_latency = compute_inter_token_latency(token_timestamps)

    # Compute per-rank utilization from events
    per_rank_utilization = _compute_per_rank_utilization(
        recorder.events, total_generation_time_seconds, rank_count
    )

    # Compute communication time ratio
    communication_time_ratio = _compute_communication_time_ratio(
        recorder.events, total_generation_time_seconds
    )

    # Compute pipeline bubble estimate
    pipeline_bubble_estimate = _compute_pipeline_bubble_estimate(
        recorder.events, total_generation_time_seconds, rank_count
    )

    return PipelineMetrics(
        tokens_per_second=tokens_per_second,
        time_to_first_token_seconds=time_to_first_token_seconds,
        inter_token_latency=inter_token_latency,
        per_rank_utilization=per_rank_utilization,
        communication_time_ratio=communication_time_ratio,
        pipeline_bubble_estimate=pipeline_bubble_estimate,
    )


def _compute_per_rank_utilization(
    events: list[PerformanceEvent],
    total_time_seconds: float,
    rank_count: int,
) -> dict[int, float]:
    """Compute per-rank utilization as fraction of time spent computing.

    Utilization is defined as the ratio of compute time to total wall time
    for each rank. Compute events are those not classified as communication
    or idle waiting.

    Args:
        events: List of performance events from the recorder.
        total_time_seconds: Total wall time for the generation.
        rank_count: Number of pipeline ranks.

    Returns:
        Dictionary mapping rank index to utilization fraction (0.0 to 1.0).
    """
    if total_time_seconds <= 0.0:
        return {rank: 0.0 for rank in range(rank_count)}

    # Accumulate compute time per rank from events
    compute_time_by_rank: dict[int, float] = {
        rank: 0.0 for rank in range(rank_count)
    }

    communication_event_names = frozenset({
        "activation_send",
        "activation_receive",
        "metadata_send",
        "metadata_receive",
        "token_send",
        "token_receive",
        "token_synchronization",
    })

    for event in events:
        if event.rank is None:
            continue
        if event.event_name in communication_event_names:
            continue
        # Count non-communication events as compute
        if event.rank in compute_time_by_rank:
            compute_time_by_rank[event.rank] += event.duration

    per_rank_utilization: dict[int, float] = {}
    for rank in range(rank_count):
        compute_time = compute_time_by_rank.get(rank, 0.0)
        utilization = min(compute_time / total_time_seconds, 1.0)
        per_rank_utilization[rank] = utilization

    return per_rank_utilization


def _compute_communication_time_ratio(
    events: list[PerformanceEvent],
    total_time_seconds: float,
) -> float:
    """Compute fraction of total time spent in communication.

    Communication events include activation sends/receives, metadata
    transfers, and token synchronization.

    Args:
        events: List of performance events from the recorder.
        total_time_seconds: Total wall time for the generation.

    Returns:
        Ratio of communication time to total time (0.0 to 1.0).
    """
    if total_time_seconds <= 0.0:
        return 0.0

    communication_event_names = frozenset({
        "activation_send",
        "activation_receive",
        "metadata_send",
        "metadata_receive",
        "token_send",
        "token_receive",
        "token_synchronization",
    })

    total_communication_time = 0.0
    for event in events:
        if event.event_name in communication_event_names:
            total_communication_time += event.duration

    return min(total_communication_time / total_time_seconds, 1.0)


def _compute_pipeline_bubble_estimate(
    events: list[PerformanceEvent],
    total_time_seconds: float,
    rank_count: int,
) -> float:
    """Estimate pipeline bubble fraction.

    Pipeline bubble is the fraction of total pipeline capacity that is idle
    due to pipeline startup/drain or imbalanced stage times. Computed as:

        bubble = 1 - (sum of all rank compute times) / (rank_count * total_time)

    This gives the fraction of the theoretical pipeline capacity that is
    wasted due to bubbles.

    Args:
        events: List of performance events from the recorder.
        total_time_seconds: Total wall time for the generation.
        rank_count: Number of pipeline ranks.

    Returns:
        Pipeline bubble estimate as a fraction (0.0 to 1.0).
        Returns 0.0 if no events or zero total time.
    """
    if total_time_seconds <= 0.0 or rank_count <= 0:
        return 0.0

    total_pipeline_capacity = total_time_seconds * rank_count

    # Sum all event durations (compute + communication) per rank
    total_useful_time = 0.0
    for event in events:
        if event.rank is not None:
            total_useful_time += event.duration

    # Bubble is the unused fraction of total pipeline capacity
    bubble = 1.0 - min(total_useful_time / total_pipeline_capacity, 1.0)
    return max(bubble, 0.0)


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
# JSON Output
# ---------------------------------------------------------------------------


def build_json_output(
    config: BenchmarkConfig,
    stats: BenchmarkStats,
    pipeline_metrics: PipelineMetrics | None,
    env: EnvironmentInfo,
    recorder: PerformanceRecorder | None,
    memory_loading: MemoryLoadingBenchmark | None = None,
) -> BenchmarkJsonOutput:
    """Build structured JSON output for cross-run comparison.

    Args:
        config: Benchmark configuration.
        stats: Aggregated statistics.
        pipeline_metrics: Pipeline-level metrics (None if not computed).
        env: Environment metadata.
        recorder: PerformanceRecorder with raw events (None if unavailable).
        memory_loading: Memory metrics from model loading (None if not measured).

    Returns:
        BenchmarkJsonOutput with metadata, metrics, and raw_events sections.
    """
    metadata: dict[str, Any] = {
        "model": config.model_id,
        "mode": config.benchmark_mode,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "device": config.device,
        "dtype": config.dtype,
        "prompt_tokens": config.prompt_tokens,
        "gen_tokens": config.gen_tokens,
        "warmup_iterations": config.warmup,
        "timed_iterations": config.iterations,
        "seed": config.seed,
        "pytorch_version": env.pytorch_version,
        "xpu_device_name": env.xpu_device_name,
        "driver_version": env.driver_version,
    }
    if env.compile_status is not None:
        metadata["compile_status"] = env.compile_status
    if env.sdpa_backend is not None:
        metadata["sdpa_backend"] = env.sdpa_backend

    metrics: dict[str, Any] = {
        "ttft_mean_seconds": stats.ttft_mean,
        "ttft_std_seconds": stats.ttft_std,
        "decode_tps_mean": stats.decode_tps_mean,
        "decode_tps_std": stats.decode_tps_std,
        "prefill_tps_mean": stats.prefill_tps_mean,
        "prefill_tps_std": stats.prefill_tps_std,
        "total_time_mean_seconds": stats.total_time_mean,
        "total_time_std_seconds": stats.total_time_std,
        "is_unstable": stats.is_unstable,
    }

    if pipeline_metrics is not None:
        metrics["tokens_per_second"] = pipeline_metrics.tokens_per_second
        metrics["time_to_first_token_seconds"] = (
            pipeline_metrics.time_to_first_token_seconds
        )
        metrics["inter_token_latency_mean_seconds"] = (
            pipeline_metrics.inter_token_latency.mean_seconds
        )
        metrics["inter_token_latency_median_seconds"] = (
            pipeline_metrics.inter_token_latency.median_seconds
        )
        metrics["inter_token_latency_p95_seconds"] = (
            pipeline_metrics.inter_token_latency.p95_seconds
        )
        metrics["per_rank_utilization"] = (
            pipeline_metrics.per_rank_utilization
        )
        metrics["communication_time_ratio"] = (
            pipeline_metrics.communication_time_ratio
        )
        metrics["pipeline_bubble_estimate"] = (
            pipeline_metrics.pipeline_bubble_estimate
        )

    if memory_loading is not None:
        metrics["memory_loading"] = {
            "peak_rss_mib": memory_loading.peak_rss_mib,
            "rss_before_mib": memory_loading.rss_before_mib,
            "rss_after_mib": memory_loading.rss_after_mib,
            "rss_delta_mib": memory_loading.rss_delta_mib,
            "tensor_mib_loaded": memory_loading.tensor_mib_loaded,
            "loading_duration_seconds": memory_loading.loading_duration_seconds,
        }

    # Collect raw events from recorder
    raw_events: list[dict[str, Any]] = []
    if recorder is not None:
        for event in recorder.events:
            raw_events.append(event.model_dump())

    return BenchmarkJsonOutput(
        metadata=metadata,
        metrics=metrics,
        raw_events=raw_events,
    )


def write_json_output(output: BenchmarkJsonOutput, path: str) -> None:
    """Write benchmark JSON output to a file.

    Args:
        output: Structured benchmark output.
        path: File path to write JSON to.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata": output.metadata,
        "metrics": output.metrics,
        "raw_events": output.raw_events,
    }

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, allow_nan=False)

    print(f"JSON results written to: {path}")


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------


def format_report(
    stats: BenchmarkStats,
    config: BenchmarkConfig,
    env: EnvironmentInfo,
    pipeline_metrics: PipelineMetrics | None = None,
    memory_loading: MemoryLoadingBenchmark | None = None,
) -> str:
    """Format benchmark results into a structured text report.

    Args:
        stats: Aggregated statistics from benchmark iterations.
        config: The benchmark configuration used.
        env: Environment metadata.
        pipeline_metrics: Pipeline-level metrics (None if not computed).
        memory_loading: Memory metrics from model loading (None if not measured).

    Returns:
        A multi-line string report suitable for terminal output.
    """
    classification = classify_performance(
        stats.decode_tps_mean, stats.ttft_mean
    )

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("XPU PERFORMANCE BENCHMARK REPORT")
    lines.append(f"Mode: {config.benchmark_mode}")
    lines.append("=" * 60)
    lines.append("")

    # Timing results
    lines.append("--- Timing Results ---")
    lines.append(
        f"TTFT:        {stats.ttft_mean:.3f} +/- {stats.ttft_std:.3f} s"
    )
    lines.append(
        f"Prefill TPS: {stats.prefill_tps_mean:.1f}"
        f" +/- {stats.prefill_tps_std:.1f} tok/s"
    )
    lines.append(
        f"Decode TPS:  {stats.decode_tps_mean:.1f}"
        f" +/- {stats.decode_tps_std:.1f} tok/s"
    )
    lines.append(
        f"Total time:  {stats.total_time_mean:.3f}"
        f" +/- {stats.total_time_std:.3f} s"
    )
    lines.append("")

    # Pipeline metrics (if available)
    if pipeline_metrics is not None:
        lines.append("--- Pipeline Metrics ---")
        lines.append(
            f"Tokens/sec:              "
            f"{pipeline_metrics.tokens_per_second:.2f}"
        )
        lines.append(
            f"TTFT:                    "
            f"{pipeline_metrics.time_to_first_token_seconds:.3f} s"
        )
        lines.append(
            f"Inter-token latency mean:   "
            f"{pipeline_metrics.inter_token_latency.mean_seconds * 1000:.1f} ms"
        )
        lines.append(
            f"Inter-token latency median: "
            f"{pipeline_metrics.inter_token_latency.median_seconds * 1000:.1f} ms"
        )
        lines.append(
            f"Inter-token latency p95:    "
            f"{pipeline_metrics.inter_token_latency.p95_seconds * 1000:.1f} ms"
        )
        lines.append("")
        lines.append("Per-rank utilization:")
        for rank, utilization in sorted(
            pipeline_metrics.per_rank_utilization.items()
        ):
            lines.append(
                f"  Rank {rank}: {utilization * 100:.1f}%"
            )
        lines.append("")
        lines.append(
            f"Communication time ratio: "
            f"{pipeline_metrics.communication_time_ratio * 100:.1f}%"
        )
        lines.append(
            f"Pipeline bubble estimate: "
            f"{pipeline_metrics.pipeline_bubble_estimate * 100:.1f}%"
        )
        lines.append("")

    # Performance classification
    lines.append("--- Performance Classification ---")
    if classification.is_critical:
        lines.append(
            "CRITICAL: decode_tps < 2.0"
            " — investigate kernel fallbacks or memory bandwidth issues"
        )
    if classification.meets_stretch:
        lines.append("Stretch target met: decode_tps >= 10.0")
    elif classification.meets_minimum:
        lines.append("Minimum target met: decode_tps >= 5.0")
    else:
        lines.append("Below minimum target: decode_tps < 5.0")

    if classification.ttft_acceptable:
        lines.append("TTFT acceptable: < 5.0 s")
    else:
        lines.append("TTFT too high: >= 5.0 s")
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

    # Memory loading metrics (if available)
    if memory_loading is not None:
        lines.append("--- Memory Loading ---")
        lines.append(
            f"Peak RSS:        {memory_loading.peak_rss_mib:.1f} MiB"
        )
        lines.append(
            f"RSS before:      {memory_loading.rss_before_mib:.1f} MiB"
        )
        lines.append(
            f"RSS after:       {memory_loading.rss_after_mib:.1f} MiB"
        )
        lines.append(
            f"RSS delta:       {memory_loading.rss_delta_mib:.1f} MiB"
        )
        lines.append(
            f"Tensors loaded:  {memory_loading.tensor_mib_loaded:.1f} MiB"
        )
        lines.append(
            f"Load duration:   {memory_loading.loading_duration_seconds:.2f} s"
        )
        lines.append("")

    # Configuration
    lines.append("--- Configuration ---")
    lines.append(f"Benchmark mode:  {config.benchmark_mode}")
    lines.append(f"Prompt tokens:   {config.prompt_tokens}")
    lines.append(f"Gen tokens:      {config.gen_tokens}")
    lines.append(f"Warmup:          {config.warmup}")
    lines.append(f"Iterations:      {config.iterations}")
    lines.append(f"Seed:            {config.seed}")
    lines.append("")

    # Instability warning
    if stats.is_unstable:
        lines.append(
            "WARNING: High variance detected (CV > 20%). "
            "Consider increasing --warmup count."
        )
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distribution Comparison Formatting
# ---------------------------------------------------------------------------


def format_distribution_comparison(
    current_distribution: PipelineLayerDistribution,
    recommended_distribution: PipelineLayerDistribution,
    current_stage_timing: list[PerStageTimingSummary],
    recommended_stage_timing: list[PerStageTimingSummary],
) -> str:
    """Format a comparison table between current and recommended distributions.

    Shows per-stage timing for both distributions and the improvement in
    max stage time (pipeline bottleneck reduction).

    Args:
        current_distribution: The current pipeline layer distribution.
        recommended_distribution: The recommended optimal distribution.
        current_stage_timing: Per-stage timing for the current distribution.
        recommended_stage_timing: Per-stage timing for the recommended distribution.

    Returns:
        A multi-line string with a readable comparison table.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("PIPELINE LAYER DISTRIBUTION COMPARISON")
    lines.append("=" * 60)
    lines.append("")

    # Current distribution
    current_layers_str = ",".join(
        str(n) for n in current_distribution.layers_per_rank
    )
    lines.append(f"Current distribution:     [{current_layers_str}]")

    # Recommended distribution
    recommended_layers_str = ",".join(
        str(n) for n in recommended_distribution.layers_per_rank
    )
    lines.append(f"Recommended distribution: [{recommended_layers_str}]")
    lines.append("")

    # Per-stage timing table header
    lines.append("--- Per-Stage Decode Timing (seconds) ---")
    lines.append(
        f"{'Rank':<6}{'Current Layers':<16}"
        f"{'Current Time':<14}{'Rec. Layers':<14}{'Rec. Time':<14}"
    )
    lines.append("-" * 64)

    # Per-stage rows
    current_max_time = 0.0
    recommended_max_time = 0.0

    for rank in range(current_distribution.rank_count):
        current_timing = current_stage_timing[rank]
        recommended_timing = recommended_stage_timing[rank]

        current_time = current_timing.decode_total_mean_seconds
        recommended_time = recommended_timing.decode_total_mean_seconds

        current_max_time = max(current_max_time, current_time)
        recommended_max_time = max(recommended_max_time, recommended_time)

        current_layer_count = current_timing.layer_count
        recommended_layer_count = recommended_timing.layer_count

        lines.append(
            f"{rank:<6}{current_layer_count:<16}"
            f"{current_time:<14.6f}{recommended_layer_count:<14}"
            f"{recommended_time:<14.6f}"
        )

    lines.append("-" * 64)
    lines.append("")

    # Max stage time comparison
    lines.append(f"Max stage time (current):     {current_max_time:.6f} s")
    lines.append(
        f"Max stage time (recommended): {recommended_max_time:.6f} s"
    )

    # Improvement percentage
    if current_max_time > 0.0:
        improvement = (
            (current_max_time - recommended_max_time) / current_max_time
        ) * 100.0
        lines.append(f"Bottleneck reduction:         {improvement:.1f}%")
    else:
        lines.append("Bottleneck reduction:         N/A (no timing data)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Continuous Batching Report Formatting
# ---------------------------------------------------------------------------


def format_continuous_batching_report(
    result: ContinuousBatchingBenchmarkResult,
) -> str:
    """Format continuous batching benchmark result as a readable report section.

    Displays aggregate throughput, per-request throughput, and pipeline
    utilization metrics for the continuous batching benchmark mode.

    Args:
        result: The continuous batching benchmark result to format.

    Returns:
        A multi-line string report section suitable for terminal output.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("CONTINUOUS BATCHING BENCHMARK RESULTS")
    lines.append("=" * 60)
    lines.append("")

    # Concurrency and workload
    lines.append("--- Workload ---")
    lines.append(f"Concurrent requests:     {result.concurrent_requests}")
    lines.append(f"Total tokens generated:  {result.total_tokens_generated}")
    lines.append(f"Total decode steps:      {result.total_decode_steps}")
    lines.append(f"Wall clock time:         {result.wall_clock_seconds:.3f} s")
    lines.append("")

    # Throughput
    lines.append("--- Throughput ---")
    lines.append(
        f"Aggregate tokens/sec:    "
        f"{result.aggregate_tokens_per_second:.2f} tok/s"
    )
    lines.append(
        f"Per-request tokens/sec:  "
        f"{result.per_request_tokens_per_second:.2f} tok/s"
    )
    lines.append("")

    # Pipeline utilization
    lines.append("--- Pipeline Utilization ---")
    lines.append(
        f"Average microbatch size: {result.average_microbatch_size:.2f}"
    )
    lines.append(
        f"Pipeline occupancy:      {result.pipeline_occupancy * 100:.1f}%"
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


@torch.inference_mode()
def greedy_decode_with_timestamps(
    model: object,
    prompt_ids: torch.Tensor,
    gen_tokens: int,
    eos_id: int,
    device: str,
    recorder: PerformanceRecorder | None = None,
) -> tuple[list[int], float, float, list[float]]:
    """Greedy decode loop that records per-token timestamps.

    Returns (generated_token_ids, prefill_seconds, decode_seconds,
             token_timestamps).

    The token_timestamps list contains a perf_counter timestamp for each
    generated token, enabling inter-token latency computation.
    """
    # Move prompt to device
    prompt_ids = prompt_ids.to(device)

    # --- Prefill ---
    torch.xpu.synchronize()
    t0 = time.perf_counter()

    if recorder is not None:
        with recorder.span("prefill_forward", mode="prefill"):
            outputs = model(input_ids=prompt_ids, use_cache=True)  # type: ignore[operator]
    else:
        outputs = model(input_ids=prompt_ids, use_cache=True)  # type: ignore[operator]

    torch.xpu.synchronize()
    prefill_time = time.perf_counter() - t0

    past_key_values = outputs.past_key_values
    logits = outputs.logits

    # First token via argmax
    first_token = logits[:, -1, :].argmax(dim=-1)
    token_ids: list[int] = [first_token.item()]
    token_timestamps: list[float] = [time.perf_counter()]

    if token_ids[0] == eos_id:
        return token_ids, prefill_time, 0.0, token_timestamps

    # --- Decode loop ---
    cur_token = first_token.unsqueeze(0)  # shape: [1, 1]

    torch.xpu.synchronize()
    t1 = time.perf_counter()

    for step in range(gen_tokens - 1):
        if recorder is not None:
            with recorder.span(
                "decode_step", mode="decode", metadata={"step": step}
            ):
                outputs = model(  # type: ignore[operator]
                    input_ids=cur_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        else:
            outputs = model(  # type: ignore[operator]
                input_ids=cur_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)

        token_id = next_token.item()  # sync point for EOS check
        token_ids.append(token_id)
        token_timestamps.append(time.perf_counter())

        if token_id == eos_id:
            break

        cur_token = next_token.unsqueeze(0)

    torch.xpu.synchronize()
    decode_time = time.perf_counter() - t1

    return token_ids, prefill_time, decode_time, token_timestamps


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
    prefill_tps = (
        config.prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    )

    # Decode TPS: tokens generated in decode phase / decode time
    # The first token comes from prefill, so decode tokens = tokens_generated - 1
    decode_tokens = max(tokens_generated - 1, 0)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0

    # TTFT: prefill time (time to produce the first token)
    ttft_seconds = prefill_time

    # Decode the generated text (outside timed loop)
    generated_text: str = tokenizer.decode(  # type: ignore[union-attr]
        token_ids, skip_special_tokens=True
    )

    return BenchmarkResult(
        ttft_seconds=ttft_seconds,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        total_time_seconds=total_time,
        tokens_generated=tokens_generated,
        generated_text=generated_text,
    )


def run_benchmark_iteration_with_instrumentation(
    model: object,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    recorder: PerformanceRecorder,
) -> tuple[BenchmarkResult, list[float]]:
    """Run a single timed benchmark iteration with instrumentation.

    Records per-token timestamps and performance events for pipeline
    metrics computation.

    Args:
        model: The loaded HuggingFace model on XPU device.
        tokenizer: The tokenizer for decoding output.
        prompt_ids: Pre-tokenized prompt tensor of shape [1, prompt_tokens].
        config: Benchmark configuration.
        recorder: PerformanceRecorder for event collection.

    Returns:
        Tuple of (BenchmarkResult, token_timestamps).
    """
    eos_id = tokenizer.eos_token_id  # type: ignore[union-attr]
    if eos_id is None:
        eos_id = 2  # Fallback EOS token ID

    # Run the instrumented decode loop
    token_ids, prefill_time, decode_time, token_timestamps = (
        greedy_decode_with_timestamps(
            model=model,
            prompt_ids=prompt_ids,
            gen_tokens=config.gen_tokens,
            eos_id=eos_id,
            device=config.device,
            recorder=recorder,
        )
    )

    # Compute metrics
    tokens_generated = len(token_ids)
    total_time = prefill_time + decode_time

    prefill_tps = (
        config.prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    )
    decode_tokens = max(tokens_generated - 1, 0)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0
    ttft_seconds = prefill_time

    generated_text: str = tokenizer.decode(  # type: ignore[union-attr]
        token_ids, skip_special_tokens=True
    )

    result = BenchmarkResult(
        ttft_seconds=ttft_seconds,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        total_time_seconds=total_time,
        tokens_generated=tokens_generated,
        generated_text=generated_text,
    )

    return result, token_timestamps


# ---------------------------------------------------------------------------
# Benchmark Mode Runners
# ---------------------------------------------------------------------------


def run_single_request_decode_benchmark(
    model: object,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    recorder: PerformanceRecorder,
) -> tuple[list[BenchmarkResult], list[float]]:
    """Run single-request decode benchmark mode.

    Executes warmup iterations followed by timed iterations with
    instrumentation, collecting per-token timestamps from the final
    timed iteration for pipeline metrics.

    Args:
        model: The loaded HuggingFace model on XPU device.
        tokenizer: The tokenizer for decoding output.
        prompt_ids: Pre-tokenized prompt tensor.
        config: Benchmark configuration.
        recorder: PerformanceRecorder for event collection.

    Returns:
        Tuple of (results_list, final_token_timestamps).
    """
    # Warmup iterations (no instrumentation)
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

    # Timed iterations with instrumentation
    print(f"Running {config.iterations} timed iteration(s)...")
    results: list[BenchmarkResult] = []
    final_token_timestamps: list[float] = []

    for i in range(config.iterations):
        try:
            result, token_timestamps = (
                run_benchmark_iteration_with_instrumentation(
                    model, tokenizer, prompt_ids, config, recorder
                )
            )
            results.append(result)
            final_token_timestamps = token_timestamps
            print(
                f"  Iteration {i + 1}/{config.iterations}: "
                f"TTFT={result.ttft_seconds:.3f}s, "
                f"Decode={result.decode_tps:.1f} tok/s, "
                f"Tokens={result.tokens_generated}"
            )
        except Exception as e:
            print(
                f"WARNING: Iteration {i + 1} failed: {e}. "
                f"Reporting results from {len(results)} successful"
                " iterations.",
                file=sys.stderr,
            )
            break

    return results, final_token_timestamps


def run_prefill_benchmark(
    model: object,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    recorder: PerformanceRecorder,
) -> tuple[list[BenchmarkResult], list[float]]:
    """Run prefill-only benchmark mode.

    Measures time to first token without running the full decode loop.
    Useful for analyzing prefill performance in isolation.

    Note: This is a stub that runs the full decode but reports only
    prefill metrics. The actual prefill-only mode depends on later tasks
    implementing chunked prefill and prefill isolation.

    Args:
        model: The loaded HuggingFace model on XPU device.
        tokenizer: The tokenizer for decoding output.
        prompt_ids: Pre-tokenized prompt tensor.
        config: Benchmark configuration.
        recorder: PerformanceRecorder for event collection.

    Returns:
        Tuple of (results_list, token_timestamps).
    """
    print("Prefill benchmark mode (TTFT-focused)")
    print(
        "Note: Full decode runs but only prefill metrics are primary."
    )
    print()

    # Use the same execution path as single-request-decode
    # but the report will emphasize TTFT and prefill TPS
    return run_single_request_decode_benchmark(
        model, tokenizer, prompt_ids, config, recorder
    )


def run_continuous_batching_benchmark(
    model: object,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    recorder: PerformanceRecorder,
) -> tuple[list[BenchmarkResult], list[float]]:
    """Run continuous-batching benchmark mode.

    Note: This is a stub. The actual continuous-batching benchmark depends
    on the ContinuousBatchScheduler (Task 7). Currently falls back to
    single-request decode with a notice.

    Args:
        model: The loaded HuggingFace model on XPU device.
        tokenizer: The tokenizer for decoding output.
        prompt_ids: Pre-tokenized prompt tensor.
        config: Benchmark configuration.
        recorder: PerformanceRecorder for event collection.

    Returns:
        Tuple of (results_list, token_timestamps).
    """
    print("Continuous-batching benchmark mode (STUB)")
    print(
        "Note: ContinuousBatchScheduler not yet implemented. "
        "Falling back to single-request decode."
    )
    print(
        "This mode will be fully implemented after Task 7 "
        "(Continuous Batching)."
    )
    print()

    # Fall back to single-request decode until continuous batching is ready
    return run_single_request_decode_benchmark(
        model, tokenizer, prompt_ids, config, recorder
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
            "ERROR: No XPU devices detected. Ensure Intel GPU drivers"
            " and PyTorch XPU are installed.",
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
    device_idx = (
        int(config.device.split(":")[-1]) if ":" in config.device else 0
    )
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
        model = torch.compile(  # type: ignore[assignment]
            model, backend="inductor", mode="reduce-overhead"
        )
        print(
            "torch.compile applied successfully"
            " (backend=inductor, mode=reduce-overhead)"
        )
        return model, "success"
    except Exception as e:
        print(
            f"WARNING: torch.compile failed ({e}),"
            " falling back to eager mode.",
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
        import torch.nn.functional as functional

        # Create small test tensors for SDPA diagnostic
        batch, heads, seq_len, head_dim = 1, 1, 16, 64
        query = torch.randn(
            batch, heads, seq_len, head_dim,
            device=device, dtype=torch.float16,
        )
        key = torch.randn(
            batch, heads, seq_len, head_dim,
            device=device, dtype=torch.float16,
        )
        value = torch.randn(
            batch, heads, seq_len, head_dim,
            device=device, dtype=torch.float16,
        )

        # Try each backend to see which one works
        with torch.inference_mode():
            # Try flash attention
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    functional.scaled_dot_product_attention(
                        query, key, value
                    )
                    return "flash"
            except Exception:
                pass

            # Try efficient attention
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True,
                ):
                    functional.scaled_dot_product_attention(
                        query, key, value
                    )
                    return "efficient"
            except Exception:
                pass

            # Fall back to math
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    functional.scaled_dot_product_attention(
                        query, key, value
                    )
                    return "math"
            except Exception:
                pass

        # If none of the specific backends work, run default
        functional.scaled_dot_product_attention(query, key, value)
        return "default"

    except Exception as e:
        print(
            f"WARNING: SDPA backend detection failed: {e}",
            file=sys.stderr,
        )
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
    print(f"Mode: {config.benchmark_mode}")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    print(f"Prompt tokens: {config.prompt_tokens}")
    print(f"Gen tokens: {config.gen_tokens}")
    print(f"Warmup: {config.warmup}")
    print(f"Iterations: {config.iterations}")
    print(f"Seed: {config.seed}")
    if config.json_output_path is not None:
        print(f"JSON output: {config.json_output_path}")
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
            dtype=torch_dtype,
        )
        model = model.to(config.device)  # type: ignore[union-attr]
        model.eval()  # type: ignore[union-attr]
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

    # Initialize performance recorder
    recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)

    # Run benchmark based on mode
    benchmark_mode = config.benchmark_mode
    if benchmark_mode == "single-request-decode":
        results, token_timestamps = run_single_request_decode_benchmark(
            model, tokenizer, prompt_ids, config, recorder
        )
    elif benchmark_mode == "prefill":
        results, token_timestamps = run_prefill_benchmark(
            model, tokenizer, prompt_ids, config, recorder
        )
    elif benchmark_mode == "continuous-batching":
        results, token_timestamps = run_continuous_batching_benchmark(
            model, tokenizer, prompt_ids, config, recorder
        )
    else:
        print(
            f"ERROR: Unknown benchmark mode: {benchmark_mode}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not results:
        print(
            "ERROR: No successful iterations. Cannot produce report.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()

    # Compute statistics
    stats = compute_stats(results)
    env = collect_environment_info(config, compile_status, sdpa_backend)

    # Compute pipeline metrics from the final iteration
    final_result = results[-1]
    pipeline_metrics = compute_pipeline_metrics(
        recorder=recorder,
        total_tokens_generated=final_result.tokens_generated,
        total_generation_time_seconds=final_result.total_time_seconds,
        time_to_first_token_seconds=final_result.ttft_seconds,
        token_timestamps=token_timestamps,
        rank_count=4,
    )

    # Generate and print report
    report = format_report(stats, config, env, pipeline_metrics)
    print(report)

    # Recommend layer distribution if requested
    if config.recommend_layer_distribution:
        per_layer_timing = export_per_layer_timing(recorder)

        if per_layer_timing:
            # Determine current distribution
            if config.pipeline_layer_distribution is not None:
                current_distribution = config.pipeline_layer_distribution
            else:
                # Default balanced distribution based on layer count from timing
                total_layers = len(per_layer_timing)
                current_distribution = default_layer_distribution(
                    total_layers=total_layers, world_size=4
                )

            world_size = current_distribution.rank_count

            # Get recommendation
            recommended_distribution = (
                recommend_pipeline_layer_distribution(
                    per_layer_timing=per_layer_timing,
                    world_size=world_size,
                )
            )

            # Compute per-stage timing for both distributions
            current_stage_timing = export_per_stage_timing(
                per_layer_timing, current_distribution
            )
            recommended_stage_timing = export_per_stage_timing(
                per_layer_timing, recommended_distribution
            )

            # Print comparison
            comparison = format_distribution_comparison(
                current_distribution=current_distribution,
                recommended_distribution=recommended_distribution,
                current_stage_timing=current_stage_timing,
                recommended_stage_timing=recommended_stage_timing,
            )
            print(comparison)
        else:
            print(
                "\nWARNING: No per-layer timing data available. "
                "Cannot recommend distribution.",
                file=sys.stderr,
            )

    # Write JSON output if requested
    if config.json_output_path is not None:
        json_output = build_json_output(
            config=config,
            stats=stats,
            pipeline_metrics=pipeline_metrics,
            env=env,
            recorder=recorder,
        )
        write_json_output(json_output, config.json_output_path)


if __name__ == "__main__":
    main()

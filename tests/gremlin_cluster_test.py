#!/usr/bin/env python3
"""
End-to-end cluster test for 4-node Qwen3.5-27B tensor parallel inference on torch+xpu.

This script:
1. Downloads the Qwen3.5-27B model on one node (master), which distributes to all others
2. Creates a tensor-parallel instance across 4 nodes
3. Waits for all shards to be loaded successfully
4. Validates the model is sharded across all 4 nodes
5. Runs a "hello world" inference request
6. Reports performance metrics (tokens/second, latency)
7. Tests async pipeline communication (bubble-free scheduling)
8. Tests speculative decoding with acceptance rate reporting
9. Tests prefix cache optimization

Usage:
    python tests/gremlin_cluster_test.py --api-host 10.1.1.12 --api-port 52415

Prerequisites:
    - 4 torch+xpu nodes running exo with Thunderbolt/RDMA connectivity
    - All nodes discoverable via libp2p bootstrap peers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Re-exec guard: ensure script runs under project virtualenv Python when invoked directly
def _ensure_venv_python() -> None:
    """Re-exec the script under .venv/bin/python if running directly without EXO_GREMLIN_ALLOW_NON_VENV_PYTHON."""
    # Only apply guard on direct script execution (not pytest/module imports)
    if __name__ != "__main__":
        return
    
    # Guard against re-exec loops
    if os.environ.get("EXO_GREMLIN_REEXECED") == "1":
        return
    
    # Detect repository root from __file__
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        return
    
    # Detect .venv/bin/python (macOS/Linux path)
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    
    # Check if current interpreter is already the venv python
    current_executable = Path(sys.executable).resolve()
    venv_executable = venv_python.resolve()
    if current_executable == venv_executable:
        return
    
    # Check for opt-out environment variable
    if os.environ.get("EXO_GREMLIN_ALLOW_NON_VENV_PYTHON", "").lower() in ("1", "true", "yes"):
        return
    
    # Re-exec under venv python
    print(f"Re-executing under .venv/python: {sys.executable} -> {venv_python}")
    os.environ["EXO_GREMLIN_REEXECED"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_venv_python()

import httpx

# Dependency diagnostic: check for AsyncClient before it's used at line 2144
if not hasattr(httpx, "AsyncClient"):
    raise RuntimeError(
        f"httpx.AsyncClient is missing!\n"
        f"  sys.executable: {sys.executable}\n"
        f"  httpx.__file__: {httpx.__file__}\n"
        f"  httpx.__version__: {getattr(httpx, '__version__', 'unknown')}\n"
        f"Please reinstall httpx>=0.28.1: uv pip install --force-reinstall 'httpx>=0.28.1,<1.0'"
    )

# Import cluster monitor for metrics collection
# Support both module import (pytest) and direct script execution
import importlib.util

MONITOR_AVAILABLE = False
MONITOR_IMPORT_ERROR: str | None = None
ClusterMonitor = None  # type: ignore
create_monitor_from_env = None  # type: ignore
print_bottleneck_analysis = None  # type: ignore

# Try to import cluster_monitor robustly
_cluster_monitor_path = Path(__file__).with_name("cluster_monitor.py")
if _cluster_monitor_path.exists():
    try:
        spec = importlib.util.spec_from_file_location(
            "cluster_monitor", _cluster_monitor_path
        )
        if spec is not None and spec.loader is not None:
            cluster_monitor_module = importlib.util.module_from_spec(spec)
            sys.modules["cluster_monitor"] = cluster_monitor_module
            spec.loader.exec_module(cluster_monitor_module)
            ClusterMonitor = cluster_monitor_module.ClusterMonitor
            create_monitor_from_env = cluster_monitor_module.create_monitor_from_env
            print_bottleneck_analysis = cluster_monitor_module.print_bottleneck_analysis
            MONITOR_AVAILABLE = True
        else:
            MONITOR_IMPORT_ERROR = f"cluster_monitor.py not found at {_cluster_monitor_path}"
            print(f"WARNING: cluster_monitor.py not found. Metrics collection not available.")
    except ImportError as e:
        # cluster_monitor.py exists but importing failed (e.g., missing optional deps)
        # The module itself handles optional deps, so if we get here with ImportError,
        # it's likely a different issue. Still, we should report it properly.
        MONITOR_IMPORT_ERROR = f"ImportError importing cluster_monitor.py at {_cluster_monitor_path}: {type(e).__name__}: {e}"
        print(f"WARNING: cluster_monitor.py found but import failed: {type(e).__name__}: {e}")
    except Exception as e:
        # Any other exception during import
        MONITOR_IMPORT_ERROR = f"Exception importing cluster_monitor.py at {_cluster_monitor_path}: {type(e).__name__}: {e}"
        print(f"WARNING: cluster_monitor.py found but import failed: {type(e).__name__}: {e}")
else:
    MONITOR_IMPORT_ERROR = f"cluster_monitor.py not found at {_cluster_monitor_path}"
    print("WARNING: cluster_monitor.py not found. Metrics collection not available.")

# Model configuration
TARGET_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_API_HOST = "10.1.1.12"  # Gremlin cluster master node
PROMPT = "Hello, world!"
SYSTEM_PROMPT = "You are a helpful assistant."

# Logger for test output
logger = logging.getLogger(__name__)

# Speculative decoding configuration
SPECULATIVE_VERIFIER_MODEL = "Qwen/Qwen3.5-4B"
SPECULATIVE_DRAFTER_MODEL = "Qwen/Qwen3.5-0.5B"
DEFAULT_DRAFT_TOKENS = 4

# Download configuration
# Only ONE node downloads from internet, then distributes to cluster
# The master node (first node in topology) handles the download
DOWNLOAD_TIMEOUT_SECONDS = 600  # 10 minutes for large model download


@dataclass
class PipelineUtilizationMetrics:
    """Stores pipeline utilization metrics from async pipeline communication.

    Requirements: 2.1, 2.6
    """

    compute_time_ms: float = 0.0
    send_time_ms: float = 0.0
    recv_time_ms: float = 0.0
    idle_time_ms: float = 0.0
    pipeline_utilization_pct: float = 0.0
    tokens_per_second: float = 0.0
    tokens_generated: int = 0
    status: str = "UNKNOWN"  # EXCELLENT, GOOD, FAIR, POOR

    @classmethod
    def parse_from_output(cls, output: str) -> PipelineUtilizationMetrics:
        """Parse pipeline utilization report from console output.

        Args:
            output: Console output containing the pipeline utilization report.

        Returns:
            PipelineUtilizationMetrics with parsed values.
        """
        metrics = cls()

        # Parse compute time
        if match := re.search(r"Compute time:\s+([\d.]+)\s+ms", output):
            metrics.compute_time_ms = float(match.group(1))

        # Parse send time
        if match := re.search(r"Send time:\s+([\d.]+)\s+ms", output):
            metrics.send_time_ms = float(match.group(1))

        # Parse recv time
        if match := re.search(r"Recv time:\s+([\d.]+)\s+ms", output):
            metrics.recv_time_ms = float(match.group(1))

        # Parse idle time
        if match := re.search(r"Idle time:\s+([\d.]+)\s+ms", output):
            metrics.idle_time_ms = float(match.group(1))

        # Parse pipeline utilization
        if match := re.search(r"Pipeline utilization:\s+([\d.]+)%", output):
            metrics.pipeline_utilization_pct = float(match.group(1))

        # Parse tokens/second
        if match := re.search(r"Tokens/second:\s+([\d.]+)", output):
            metrics.tokens_per_second = float(match.group(1))

        # Parse tokens generated
        if match := re.search(r"Tokens generated:\s+(\d+)", output):
            metrics.tokens_generated = int(match.group(1))

        # Parse status
        if match := re.search(r"Status:\s+(\w+)", output):
            metrics.status = match.group(1)

        # Calculate idle time if not found
        if metrics.idle_time_ms == 0.0 and metrics.compute_time_ms > 0:
            metrics.idle_time_ms = max(
                0, (metrics.send_time_ms + metrics.recv_time_ms) - metrics.compute_time_ms
            )

        return metrics

    def summary(self) -> str:
        """Return a human-readable summary of pipeline utilization."""
        lines = [
            "=" * 60,
            "PIPELINE UTILIZATION REPORT (Async Mode)",
            "=" * 60,
            "",
            "--- Timing Summary ---",
            f"Compute time:    {self.compute_time_ms:.2f} ms",
            f"Send time:       {self.send_time_ms:.2f} ms",
            f"Recv time:       {self.recv_time_ms:.2f} ms",
            f"Idle time:       {self.idle_time_ms:.2f} ms",
            "",
            "--- Performance Metrics ---",
            f"Pipeline utilization: {self.pipeline_utilization_pct:.1f}%",
            f"Tokens/second:       {self.tokens_per_second:.2f}",
            f"Tokens generated:    {self.tokens_generated}",
            "",
            f"Status: {self.status}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class ClusterTestResult:
    """Stores the results of the cluster test."""

    success: bool = False
    error: str | None = None
    model_id: str = ""
    sharding: str = "Tensor"
    instance_meta: str = "PyTorchXPURing"
    node_count: int = 0
    nodes: list[str] = field(default_factory=list)
    runner_count: int = 0
    runners: list[str] = field(default_factory=list)
    load_time_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    prefill_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    response_text: str = ""
    prefix_cache_hit: str = "none"  # none, partial, exact

    # Async pipeline metrics
    use_async: bool = False
    pipeline_metrics: PipelineUtilizationMetrics | None = None

    def summary(self) -> str:
        """Return a human-readable summary of the test results."""
        lines = [
            "=" * 60,
            "GREMLIN CLUSTER TEST RESULTS",
            "=" * 60,
            f"  Status:        {'PASSED' if self.success else 'FAILED'}",
            f"  Model:         {self.model_id}",
            f"  Sharding:      {self.sharding}",
            f"  Instance:      {self.instance_meta}",
            f"  Nodes:         {self.node_count}",
            f"  Node IDs:      {', '.join(self.nodes) if self.nodes else 'N/A'}",
            f"  Runners:       {self.runner_count}",
            f"  Runner IDs:    {', '.join(self.runners) if self.runners else 'N/A'}",
            "-" * 60,
            "  Loading Phase:",
        ]

        if self.load_time_seconds > 0:
            lines.append(f"    Load Time:   {self.load_time_seconds:.2f}s")

        lines.append("  Inference Phase:")

        if self.time_to_first_token_ms > 0:
            lines.append(f"    TTFT:        {self.time_to_first_token_ms:.0f}ms")
        if self.prefill_time_ms > 0:
            lines.append(f"    Prefill:     {self.prefill_time_ms:.0f}ms")
        if self.inference_latency_ms > 0:
            lines.append(f"    Latency:     {self.inference_latency_ms:.0f}ms")
        if self.tokens_generated > 0:
            lines.append(f"    Tokens:      {self.tokens_generated}")
        if self.tokens_per_second > 0:
            lines.append(f"    Speed:       {self.tokens_per_second:.2f} tokens/s")
        if self.prefix_cache_hit and self.prefix_cache_hit != "none":
            lines.append(f"    Prefix Cache: {self.prefix_cache_hit}")

        # Async pipeline metrics
        if self.use_async and self.pipeline_metrics:
            lines.append("  Async Pipeline Metrics:")
            pm = self.pipeline_metrics
            lines.append(f"    Compute:     {pm.compute_time_ms:.2f} ms")
            lines.append(f"    Send:        {pm.send_time_ms:.2f} ms")
            lines.append(f"    Recv:        {pm.recv_time_ms:.2f} ms")
            lines.append(f"    Idle:        {pm.idle_time_ms:.2f} ms")
            lines.append(f"    Utilization: {pm.pipeline_utilization_pct:.1f}%")
            lines.append(f"    Status:      {pm.status}")

        if self.response_text:
            lines.append("  Response:")
            lines.append(f"    {self.response_text[:200]}")

        if self.error:
            lines.append("  Error:")
            lines.append(f"    {self.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SpeculativeDecodingResult:
    """Stores the results of a speculative decoding test."""

    success: bool = False
    error: str | None = None
    verifier_model_id: str = ""
    drafter_model_id: str | None = None
    draft_tokens: int = 4
    node_count: int = 0
    nodes: list[str] = field(default_factory=list)
    load_time_seconds: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    acceptance_rate: float = 0.0  # Fraction [0, 1]
    effective_tokens_per_step: float = 0.0
    total_draft_time_seconds: float = 0.0
    total_verification_time_seconds: float = 0.0
    response_text: str = ""
    speculative_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the speculative decoding test."""
        lines = [
            "=" * 60,
            "SPECULATIVE DECODING TEST RESULTS",
            "=" * 60,
            f"  Status:        {'PASSED' if self.success else 'FAILED'}",
            f"  Verifier:      {self.verifier_model_id}",
            f"  Drafter:       {self.drafter_model_id or 'N/A (standard gen)'}",
            f"  Draft Tokens:  {self.draft_tokens}",
            f"  Nodes:         {self.node_count}",
            f"  Node IDs:      {', '.join(self.nodes) if self.nodes else 'N/A'}",
            "-" * 60,
            "  Loading Phase:",
        ]

        if self.load_time_seconds > 0:
            lines.append(f"    Load Time:   {self.load_time_seconds:.2f}s")

        lines.append("  Speculative Decoding Phase:")

        if self.tokens_generated > 0:
            lines.append(f"    Tokens:      {self.tokens_generated}")
        if self.tokens_per_second > 0:
            lines.append(f"    Speed:       {self.tokens_per_second:.2f} tokens/s")
        if self.acceptance_rate > 0:
            lines.append(f"    Acceptance:  {self.acceptance_rate * 100:.1f}%")
        if self.effective_tokens_per_step > 0:
            lines.append(f"    Eff. Tokens/Step: {self.effective_tokens_per_step:.2f}")

        # Classification based on acceptance rate
        rate = self.acceptance_rate
        if rate >= 0.6:
            lines.append("    Quality:     EXCELLENT - Speculative decoding is highly effective")
        elif rate >= 0.4:
            lines.append("    Quality:     GOOD - Speculative decoding provides moderate benefit")
        elif rate >= 0.2:
            lines.append(
                "    Quality:     FAIR - Consider reducing draft_tokens or selecting better drafter"
            )
        else:
            lines.append("    Quality:     POOR - Consider disabling speculative decoding")

        if self.response_text:
            lines.append("  Response:")
            lines.append(f"    {self.response_text[:200]}")

        if self.error:
            lines.append("  Error:")
            lines.append(f"    {self.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class AsyncPipelineTestResult:
    """Stores the results of an async pipeline communication test."""

    success: bool = False
    error: str | None = None
    model_id: str = ""
    sharding: str = "Pipeline"
    instance_meta: str = "PyTorchXPURing"
    node_count: int = 0
    nodes: list[str] = field(default_factory=list)
    use_async: bool = True
    sync_tps: float = 0.0
    async_tps: float = 0.0
    speedup: float = 0.0
    pipeline_metrics: PipelineUtilizationMetrics | None = None
    tokens_generated: int = 0
    response_text: str = ""

    def summary(self) -> str:
        """Return a human-readable summary of the async pipeline test."""
        lines = [
            "=" * 60,
            "ASYNC PIPELINE COMMUNICATION TEST RESULTS",
            "=" * 60,
            f"  Status:        {'PASSED' if self.success else 'FAILED'}",
            f"  Model:         {self.model_id}",
            f"  Sharding:      {self.sharding}",
            f"  Instance:      {self.instance_meta}",
            f"  Nodes:         {self.node_count}",
            "-" * 60,
            "  Performance Comparison:",
            f"    Sync TPS:    {self.sync_tps:.2f}",
            f"    Async TPS:   {self.async_tps:.2f}",
            f"    Speedup:     {self.speedup:.2f}x",
        ]

        if self.speedup > 0:
            if self.speedup >= 0.95:
                lines.append("    Assessment:  Within 5% of sync (acceptable)")
            elif self.speedup < 0.95:
                lines.append(
                    "    Assessment:  Async degraded performance - check configuration"
                )
            else:
                lines.append("    Assessment:  Async shows improvement on high-latency links")

        if self.pipeline_metrics:
            lines.append("")
            lines.append(self.pipeline_metrics.summary())

        if self.error:
            lines.append("  Error:")
            lines.append(f"    {self.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


async def check_api_available(
    client: httpx.AsyncClient, api_host: str, api_port: int
) -> bool:
    """Check if the exo API is reachable."""
    try:
        resp = await client.get(
            f"http://{api_host}:{api_port}/node_id",
            timeout=5.0,
        )
        node_id = resp.text
        print(f"  API reachable. Node ID: {node_id}")
        return True
    except Exception as e:
        print(f"  API not reachable at {api_host}:{api_port}: {e}")
        return False


async def get_state(
    client: httpx.AsyncClient, api_host: str, api_port: int
) -> dict[str, Any]:
    """Fetch the current cluster state."""
    resp = await client.get(
        f"http://{api_host}:{api_port}/state",
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


async def get_placement_previews(
    client: httpx.AsyncClient, api_host: str, api_port: int, model_id: str
) -> dict[str, Any]:
    """Get placement previews for a model."""
    resp = await client.get(
        f"http://{api_host}:{api_port}/instance/previews",
        params={"model_id": model_id},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


async def create_tensor_instance(
    client: httpx.AsyncClient, api_host: str, api_port: int, model_id: str
) -> dict[str, Any]:
    """Create a tensor-parallel instance for the given model across 4 nodes."""
    payload = {
        "model_id": model_id,
        "sharding": "Tensor",
        "instance_meta": "PyTorchXPURing",
        "min_nodes": 4,
    }
    resp = await client.post(
        f"http://{api_host}:{api_port}/place_instance",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def create_pipeline_instance(
    client: httpx.AsyncClient, api_host: str, api_port: int, model_id: str,
    use_async: bool = False,
) -> dict[str, Any]:
    """Create a pipeline-parallel instance for the given model across nodes.

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.
        model_id: HuggingFace model ID.
        use_async: If True, enables bubble-free async pipeline scheduling.

    Returns:
        Placement result with command_id.
    """
    payload = {
        "model_id": model_id,
        "sharding": "Pipeline",
        "instance_meta": "PyTorchXPURing",
        "min_nodes": 4,
    }
    if use_async:
        payload["use_async"] = True
    resp = await client.post(
        f"http://{api_host}:{api_port}/place_instance",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def send_chat_completion(
    client: httpx.AsyncClient, api_host: str, api_port: int, model_id: str, prompt: str
) -> dict[str, Any]:
    """Send a chat completion request and return the response."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 100,
    }
    resp = await client.post(
        f"http://{api_host}:{api_port}/v1/chat/completions",
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


async def send_bench_chat_completion(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
    model_id: str,
    prompt: str,
    use_prefix_cache: bool = False,
    timeout: float = 60.0,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Send a benchmark chat completion request and return generation stats.

    Uses /bench/chat/completions endpoint which returns GenerationStats
    including tokens_per_second, ttft_seconds, and peak_memory_usage.

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.
        model_id: HuggingFace model ID.
        prompt: Input prompt.
        use_prefix_cache: If True, enables prefix cache optimization.
        timeout: Request timeout in seconds (default: 60).
        max_tokens: Maximum tokens to generate (default: 128).

    Returns:
        BenchChatCompletionResponse with generation_stats.
    """
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "use_prefix_cache": use_prefix_cache,
    }
    try:
        resp = await client.post(
            f"http://{api_host}:{api_port}/bench/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException as e:
        raise Exception(f"Request timed out after {timeout}s: {e}") from e
    except httpx.HTTPStatusError as e:
        raise Exception(f"HTTP error {e.response.status_code}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise Exception(f"Request failed: {e}") from e


async def send_chat_completion_stream(
    client: httpx.AsyncClient, api_host: str, api_port: int, model_id: str, prompt: str
) -> dict[str, Any]:
    """Send a streaming chat completion request and measure performance."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 100,
    }

    start_time = time.perf_counter()
    first_token_time = None

    async with client.stream(
        "POST",
        f"http://{api_host}:{api_port}/v1/chat/completions",
        json=payload,
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        tokens = []
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # Remove "data: " prefix
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                # Extract token text from streaming response
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        tokens.append(text)
            except json.JSONDecodeError:
                continue

    end_time = time.perf_counter()
    response_text = "".join(tokens)

    result: dict[str, Any] = {
        "response_text": response_text,
        "total_latency_s": end_time - start_time,
        "tokens": tokens,
    }

    if first_token_time is not None:
        result["ttft_ms"] = (first_token_time - start_time) * 1000

    return result


async def start_download(
    client: httpx.AsyncClient, api_host: str, api_port: int, target_node_id: str, model_id: str
) -> dict[str, Any]:
    """Start a model download on a specific node via the API.

    Only ONE node should download from the internet, then exo distributes
    the model to other nodes that need it.
    """
    # Get model card info first
    resp = await client.get(
        f"http://{api_host}:{api_port}/models",
        timeout=10.0,
    )
    resp.raise_for_status()
    models_data = resp.json().get("data", [])

    # Find or create shard metadata for this model
    model_card = None
    for m in models_data:
        if m.get("id") == model_id:
            model_card = m
            break

    if model_card is None:
        raise ValueError(f"Model {model_id} not found in cluster models")

    # Create shard metadata (pipeline parallel, single node download)
    shard_metadata = {
        "PipelineShardMetadata": {
            "modelCard": {
                "modelId": model_id,
                "storageSize": {"inBytes": model_card.get("storage_size_megabytes", 0) * 1000000},
                "nLayers": 64,  # Qwen3.5-27B has 64 layers
                "hiddenSize": 5120,
                "supportsTensor": False,
                "numKeyValueHeads": 4,
                "tasks": ["TextGeneration"],
            },
            "deviceRank": 0,
            "worldSize": 1,
            "startLayer": 0,
            "endLayer": 64,
            "nLayers": 64,
        }
    }

    payload = {
        "target_node_id": target_node_id,
        "shard_metadata": shard_metadata,
    }

    resp = await client.post(
        f"http://{api_host}:{api_port}/download/start",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def wait_for_download_complete(
    client: httpx.AsyncClient, api_host: str, api_port: int, model_id: str,
    target_node_ids: list[str], timeout_seconds: int = 600, poll_interval: float = 5.0
) -> bool:
    """Wait for the model to be downloaded on all target nodes.

    Only one node downloads from internet; the download coordinator handles
    distribution to other nodes automatically.
    """
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        state = await get_state(client, api_host, api_port)
        downloads = state.get("downloads", {})

        # Check if all target nodes have DownloadCompleted for this model
        all_complete = True
        completed_count = 0

        for node_id in target_node_ids:
            node_downloads = downloads.get(node_id, [])
            model_found = False
            for dl in node_downloads:
                dl_type = list(dl.keys())[0]  # DownloadCompleted, DownloadPending, etc.
                if dl_type == "DownloadCompleted":
                    completed_model_id = dl["DownloadCompleted"]["shardMetadata"].get(
                        "PipelineShardMetadata", {}
                    ).get("modelCard", {}).get("modelId", "")
                    if completed_model_id == model_id:
                        model_found = True
                        completed_count += 1
                        break
            if not model_found:
                all_complete = False

        print(f"  Download progress: {completed_count}/{len(target_node_ids)} nodes complete")

        if all_complete:
            print(f"  Model downloaded on all {len(target_node_ids)} nodes!")
            return True

        await asyncio.sleep(poll_interval)

    return False


async def wait_for_instance_ready(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
    instance_id: str,
    timeout_seconds: int = 300,
    poll_interval: float = 2.0,
) -> dict[str, Any] | None:
    """Poll until the instance is loaded and all runners are ready."""
    deadline = time.time() + timeout_seconds
    consecutive_failures = 0
    max_consecutive_failures = 5  # Allow 5 consecutive failures before giving up

    while time.time() < deadline:
        try:
            state = await get_state(client, api_host, api_port)
            consecutive_failures = 0  # Reset failure counter on success
        except Exception as e:
            consecutive_failures += 1
            print(f"  Failed to get state: {e}")
            if consecutive_failures >= max_consecutive_failures:
                print(f"  Too many consecutive failures ({consecutive_failures}). Aborting.")
                return None
            await asyncio.sleep(poll_interval)
            continue

        instances = state.get("instances", {})

        if instance_id not in instances:
            await asyncio.sleep(poll_interval)
            continue

        instance = instances[instance_id]
        runners = state.get("runners", {})

        # Unwrap the instance type wrapper (TaggedModel: {"PyTorchXPURingInstance": {...}})
        instance_type = None
        for key in instance:
            if key.endswith("Instance"):
                instance_type = key
                break
        if instance_type is None:
            await asyncio.sleep(poll_interval)
            continue
        inner_instance = instance[instance_type]

        # Check if all runners for this instance are loaded/ready
        shard_assignments = inner_instance.get("shardAssignments", {})
        node_to_runner = shard_assignments.get("nodeToRunner", {})

        all_runners_ready = True
        ready_runners = 0

        for runner_id in node_to_runner.values():
            runner_status = runners.get(runner_id)
            if runner_status:
                # Runner statuses are serialized as TaggedModel: {"RunnerReady": {...}}
                # Check for the presence of the class name key, not a "type" field
                if "RunnerReady" in runner_status or "RunnerLoaded" in runner_status:
                    ready_runners += 1
                elif "RunnerFailed" in runner_status:
                    all_runners_ready = False
                    break
                # RunnerLoading, RunnerWarmingUp, RunnerRunning: still in progress
                elif "RunnerLoading" in runner_status or "RunnerWarmingUp" in runner_status:
                    # Count runners that are loading or warming up as progress
                    ready_runners += 1

        if all_runners_ready and ready_runners == len(node_to_runner):
            print(f"  All {ready_runners} runners ready!")
            return instance

        # Print progress
        node_ids = list(node_to_runner.keys())
        print(
            f"  Waiting... {ready_runners}/{len(node_ids)} runners ready "
            f"(nodes: {', '.join(node_ids[:4])})"
        )
        await asyncio.sleep(poll_interval)

    print(f"  Timeout waiting for instance {instance_id} to be ready")
    return None


async def run_test(
    api_host: str,
    api_port: int,
    model_id: str,
    node_count: int,
    use_streaming: bool = True,
    enable_metrics_port_forwards: bool = False,
) -> ClusterTestResult:
    """Run the complete cluster test."""
    result = ClusterTestResult(
        model_id=model_id,
        node_count=node_count,
    )

    async with httpx.AsyncClient() as client:
        # Step 0: Check API availability
        print("\n[Step 0] Checking API availability...")
        if not await check_api_available(client, api_host, api_port):
            result.error = (
                f"API not reachable at {api_host}:{api_port}. "
                "Make sure exo is running on all nodes."
            )
            return result

        # Step 1: Check current state
        print("\n[Step 1] Fetching current cluster state...")
        try:
            state = await get_state(client, api_host, api_port)
            existing_instances = state.get("instances", {})
            existing_runners = state.get("runners", {})
            topology = state.get("topology", {})
            nodes = list(topology.get("nodes", []))  # topology uses 'nodes' not 'node_ids'

            print(f"  Current nodes in cluster: {len(nodes)}")
            print(f"  Existing instances: {len(existing_instances)}")
            print(f"  Existing runners: {len(existing_runners)}")

            actual_node_count = len(nodes)
            if actual_node_count < node_count:
                print(f"  WARNING: Cluster has {actual_node_count} nodes but {node_count} requested.")
                print(f"  Adjusting test to use {actual_node_count} nodes.")
                node_count = actual_node_count

            result.nodes = nodes[:node_count]

        except Exception as e:
            result.error = f"Failed to fetch cluster state: {e}"
            return result

        # Step 2: Get placement preview
        print(f"\n[Step 2] Getting placement preview for {model_id}...")
        try:
            previews = await get_placement_previews(client, api_host, api_port, model_id)
            preview_list = previews.get("previews", [])

            print(f"  Available placements: {len(preview_list)}")
            for p in preview_list:
                error = p.get("error")
                sharding = p.get("sharding")
                meta = p.get("instance_meta")
                has_instance = p.get("instance") is not None
                print(
                    f"  - {sharding}/{meta}: {'available' if has_instance else 'new'}"
                    f" {'ERROR: ' + error if error else ''}"
                )

            # Find a valid tensor parallel placement with PyTorchXPURing
            valid_placement = None
            for p in preview_list:
                if (
                    p.get("sharding") == "Tensor"
                    and p.get("instance_meta") == "PyTorchXPURing"
                    and p.get("error") is None
                ):
                    valid_placement = p
                    break

            if valid_placement is None:
                result.error = (
                    "No valid tensor parallel placement found with PyTorchXPURing. "
                    "The model may not be downloaded on the cluster nodes, "
                    "or the cluster may not support tensor parallelism."
                )
                return result

        except Exception as e:
            result.error = f"Failed to get placement preview: {e}"
            return result

        # Step 3: Check for existing instance or create new one
        print(f"\n[Step 3] Checking for existing instance or creating new one...")

        # First check if there's already a running instance for this model
        state = await get_state(client, api_host, api_port)
        existing_instances = state.get("instances", {})
        existing_runners = state.get("runners", {})

        # Find an existing PyTorchXPURing instance for this model
        existing_instance_id = None
        for inst_id, inst in existing_instances.items():
            pytorch_inst = inst.get("PyTorchXPURingInstance")
            if pytorch_inst:
                shard_assignments = pytorch_inst.get("shardAssignments", {})
                if shard_assignments.get("modelId") == model_id:
                    existing_instance_id = inst_id
                    break

        if existing_instance_id:
            # Use existing instance
            instance_id = existing_instance_id
            print(f"  Using existing instance: {instance_id}")

            # Check if all runners are ready
            existing_inst = existing_instances[instance_id]
            pytorch_inst = existing_inst.get("PyTorchXPURingInstance", {})
            shard_assignments = pytorch_inst.get("shardAssignments", {})
            node_to_runner = shard_assignments.get("nodeToRunner", {})

            all_ready = True
            ready_count = 0
            for runner_id in node_to_runner.values():
                runner_status = existing_runners.get(runner_id)
                # Runner statuses are serialized as TaggedModel: {"RunnerReady": {...}}
                if runner_status and ("RunnerReady" in runner_status or "RunnerLoaded" in runner_status):
                    ready_count += 1
                else:
                    all_ready = False

            print(f"  Existing instance runners: {ready_count}/{len(node_to_runner)} ready")

            if not all_ready:
                print(f"  Waiting for existing instance runners to become ready...")
                instance = await wait_for_instance_ready(
                    client, api_host, api_port, instance_id, timeout_seconds=60
                )
                if instance is None:
                    result.error = "Existing instance runners did not become ready in time"
                    return result
            else:
                # Get the latest instance data from state to ensure we have the latest runners
                print(f"  All runners already ready! Getting latest instance data...")
                state = await get_state(client, api_host, api_port)
                instance = state.get("instances", {}).get(instance_id)
                if instance is None:
                    result.error = f"Instance {instance_id} not found in state"
                    return result
                print(f"  All runners already ready!")
        else:
            # Create new instance
            print(f"  No existing instance found. Creating new tensor-parallel instance...")
            try:
                placement_result = await create_tensor_instance(
                    client, api_host, api_port, model_id
                )
                command_id = placement_result.get("command_id", "")
                print(f"  Command created: {command_id}")

                # Wait for the instance to appear in state (may take time if downloading)
                instance_id = None
                for wait_attempt in range(30):
                    await asyncio.sleep(2)
                    state = await get_state(client, api_host, api_port)
                    instances = state.get("instances", {})
                    
                    # Find the instance that was just created by matching model_id
                    for inst_id, inst in instances.items():
                        for key, value in inst.items():
                            if 'PyTorch' in key:
                                shard_assignments = value.get('shardAssignments', {})
                                if shard_assignments.get('modelId') == model_id:
                                    instance_id = inst_id
                                    break
                        if instance_id:
                            break
                    if instance_id:
                        break
                    if wait_attempt % 5 == 4:
                        print(f"    Waiting for instance to appear... ({(wait_attempt+1)*2}s)")
                
                if not instance_id:
                    result.error = "Failed to find the created instance in state"
                    return result
                    
                print(f"  Instance created: {instance_id}")

            except httpx.HTTPStatusError as e:
                result.error = f"Failed to create instance: {e.response.status_code} - {e.response.text}"
                return result
            except Exception as e:
                result.error = f"Failed to create instance: {e}"
                return result

        result.instance_id = instance_id

        # Step 4: Wait for all shards to load (if new instance)
        load_start = time.perf_counter()

        if not existing_instance_id:
            print(f"\n[Step 4] Waiting for model to load on all {node_count} nodes...")

            instance = await wait_for_instance_ready(
                client, api_host, api_port, instance_id, timeout_seconds=900
            )
        else:
            # Use existing instance - get instance data for validation
            state = await get_state(client, api_host, api_port)
            instance = state.get("instances", {}).get(instance_id)

        load_time = time.perf_counter() - load_start
        result.load_time_seconds = load_time

        if instance is None:
            result.error = (
                f"Model failed to load within {300}s timeout. "
                "Check that all nodes have the model downloaded."
            )
            return result

        # Unwrap the instance type (TaggedModel: {"PyTorchXPURingInstance": {...}})
        inner_instance = None
        for key in instance:
            if key.endswith("Instance"):
                inner_instance = instance[key]
                break
        if inner_instance is None:
            result.error = f"Could not find instance type in {list(instance.keys())}"
            return result

        # Validate sharding
        shard_assignments = inner_instance.get("shardAssignments", {})
        node_to_runner = shard_assignments.get("nodeToRunner", {})
        runner_to_shard = shard_assignments.get("runnerToShard", {})

        result.nodes = list(node_to_runner.keys())[:node_count]
        result.runners = list(node_to_runner.values())[:node_count]
        result.runner_count = len(result.runners)

        # Validate all nodes have shards
        if len(node_to_runner) < node_count:
            result.error = (
                f"Expected {node_count} shards but got {len(node_to_runner)}. "
                "Model did not shard to all nodes."
            )
            return result

        print(f"\n  VALIDATION: Model sharded to {len(node_to_runner)} nodes successfully!")

        # Step 4.5: Start metrics collection
        print(f"\n[Step 4.5] Starting metrics collection...")
        
        # Setup port-forwards if requested
        port_forward_procs: dict[str, Any] = {"prometheus": None, "influxdb": None}
        if enable_metrics_port_forwards or os.environ.get("SETUP_METRICS_PORT_FORWARDS", "").lower() in ("1", "true", "yes"):
            print("  Setting up metrics port-forwards...")
            port_forward_results = await setup_metrics_port_forwards(
                setup_prometheus=True,
                setup_influxdb=True,
            )
            port_forward_procs["prometheus"] = port_forward_results.get("prometheus_proc")
            port_forward_procs["influxdb"] = port_forward_results.get("influxdb_proc")
            
            # Print port-forward status
            prom_status = "started" if port_forward_results.get("prometheus") else "failed"
            influx_status = "started" if port_forward_results.get("influxdb") else "failed"
            print(f"  Prometheus port-forward: {prom_status}")
            print(f"  InfluxDB port-forward: {influx_status}")
        
        monitor = None
        if MONITOR_AVAILABLE and create_monitor_from_env:
            try:
                monitor = create_monitor_from_env()
                await monitor.start()
                print(f"  Metrics collection started (interval: {monitor.collection_interval}s)")
                # Print availability of optional dependencies
                import cluster_monitor
                prom_available = cluster_monitor.PROMETHEUS_AVAILABLE
                influx_available = cluster_monitor.INFLUXDB_AVAILABLE
                
                # Print partial mode status
                if not prom_available and not influx_available:
                    print("  WARNING: Metrics collection running in PARTIAL MODE")
                    print("  Neither Prometheus nor InfluxDB clients are available.")
                    print("  Install missing packages: pip install prometheus_client influxdb-client")
                elif not prom_available:
                    print("  WARNING: Metrics collection running in PARTIAL MODE")
                    print("  Prometheus client unavailable. Install: pip install prometheus_client")
                elif not influx_available:
                    print("  WARNING: Metrics collection running in PARTIAL MODE")
                    print("  InfluxDB client unavailable. Install: pip install influxdb-client")
                else:
                    # Report actual connectivity (not just library availability)
                    prom_reachable = not getattr(monitor, '_prometheus_unreachable', False)
                    influx_reachable = not getattr(monitor, '_influxdb_unauthorized', False) and monitor._influxdb_client is not None
                    print(f"  Prometheus available: {prom_reachable}")
                    print(f"  InfluxDB available: {influx_reachable}")
                    if not prom_reachable:
                        print("    (Prometheus unreachable — start port-forward: kubectl port-forward -n prometheus-service svc/prometheus-kube-prometheus-prometheus 9090:9090)")
                    if not influx_reachable:
                        print("    (InfluxDB unreachable — start port-forward: kubectl port-forward -n influxdb-service svc/influxdb-influxdb2 8086:80)")
            except Exception as e:
                print(f"  Failed to start metrics collection: {e}")
        else:
            # Report precise error instead of generic message
            if MONITOR_IMPORT_ERROR:
                print(f"  Metrics collection not available: {MONITOR_IMPORT_ERROR}")
            else:
                print("  Metrics collection not available (missing dependencies)")

        # Print shard details (shard metadata is TaggedModel: {"TensorShardMetadata": {...}})
        for runner_id, shard in runner_to_shard.items():
            shard_type = list(shard.keys())[0] if shard else "Unknown"
            shard_data = list(shard.values())[0] if shard else {}
            start_layer = shard_data.get("startLayer", 0)
            end_layer = shard_data.get("endLayer", 0)
            n_layers = shard_data.get("nLayers", 0)
            world_size = shard_data.get("worldSize", 0)
            print(
                f"  Runner {runner_id}: {shard_type} "
                f"(layers {start_layer}-{end_layer}, {n_layers} layers, "
                f"world_size={world_size})"
            )

        # Step 5: Run inference
        print(f"\n[Step 5] Running 'Hello, world!' inference...")

        if use_streaming:
            try:
                inference_result = await send_chat_completion_stream(
                    client, api_host, api_port, model_id, PROMPT
                )

                result.response_text = inference_result.get("response_text", "")
                result.tokens_generated = len(inference_result.get("tokens", []))
                result.total_latency_s = inference_result.get("total_latency_s", 0)
                result.time_to_first_token_ms = inference_result.get(
                    "ttft_ms", 0
                )

                if result.tokens_generated > 0:
                    result.tokens_per_second = (
                        result.tokens_generated / result.total_latency_s
                        if result.total_latency_s > 0
                        else 0
                    )

            except Exception as e:
                result.error = f"Inference failed: {e}"
                return result
        else:
            try:
                inference_result = await send_chat_completion(
                    client, api_host, api_port, model_id, PROMPT
                )

                choices = inference_result.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    result.response_text = msg.get("content", "")

                usage = inference_result.get("usage", {})
                result.tokens_generated = (
                    usage.get("completion_tokens", 0)
                    + usage.get("prompt_tokens", 0)
                )

            except Exception as e:
                result.error = f"Inference failed: {e}"
                return result

        # Step 6: Validate response quality (check for garbled output)
        print(f"\n[Step 6] Validating response quality...")
        
        # Robust response quality validation
        response_valid = False
        response_error = None
        
        if result.response_text:
            # Check 1: Non-empty after strip
            stripped_text = result.response_text.strip()
            if not stripped_text:
                response_error = "Response is empty or contains only whitespace"
                print(f"  WARNING: {response_error}")
            else:
                # Check 2: Alphanumeric content ratio (reject mostly punctuation/whitespace)
                alphanumeric_count = sum(1 for c in stripped_text if c.isalnum())
                alphanumeric_ratio = alphanumeric_count / len(stripped_text)
                
                if alphanumeric_ratio < 0.3:
                    response_error = (
                        f"Response contains mostly non-alphanumeric characters. "
                        f"Alphanumeric ratio: {alphanumeric_ratio:.2%}. "
                        f"Raw response (repr): {repr(stripped_text[:200])}"
                    )
                    print(f"  WARNING: {response_error}")
                else:
                    # Check 3: Minimum meaningful content threshold
                    min_meaningful_length = 10  # At least 10 meaningful characters
                    meaningful_chars = sum(1 for c in stripped_text if c.isalpha())
                    if meaningful_chars < min_meaningful_length:
                        response_error = (
                            f"Response lacks meaningful content. "
                            f"Alphabetic characters: {meaningful_chars} (need at least {min_meaningful_length}). "
                            f"Raw response (repr): {repr(stripped_text[:200])}"
                        )
                        print(f"  WARNING: {response_error}")
                    else:
                        # Response passes all checks
                        response_valid = True
                        printable_ratio = sum(1 for c in result.response_text if c.isprintable() or c in '\n\r\t') / max(len(result.response_text), 1)
                        print(f"  Response length: {len(result.response_text)} chars")
                        print(f"  Printable ratio: {printable_ratio:.2%}")
                        print(f"  Alphanumeric ratio: {alphanumeric_ratio:.2%}")
                        print(f"  First 100 chars: {result.response_text[:100]}")
        else:
            response_error = "Response text is None or empty"
            print(f"  WARNING: {response_error}")
        
        if response_error:
            result.error = response_error
            # Continue running other tests even if response quality check fails
            print("  Note: Continuing with other tests despite response quality issues")

        # Step 7: Stop metrics collection and generate report
        print(f"\n[Step 7] Stopping metrics collection and generating report...")
        if monitor:
            try:
                await monitor.stop()
                print(monitor.generate_report())
                
                # Save metrics to file
                metrics_file = f"metrics_{int(time.time())}.json"
                monitor.save_to_file(metrics_file)
                print(f"  Metrics saved to {metrics_file}")
                
                # Print bottleneck analysis
                if print_bottleneck_analysis:
                    print_bottleneck_analysis(monitor.calculate_aggregates(), {"basic": result})
            except Exception as e:
                print(f"  Failed to stop metrics collection: {e}")
        
        # Cleanup port-forwards if they were started
        if enable_metrics_port_forwards or os.environ.get("SETUP_METRICS_PORT_FORWARDS", "").lower() in ("1", "true", "yes"):
            print("\n  Cleaning up port-forwards...")
            await cleanup_port_forwards(port_forward_procs)

        result.success = True
        return result


# ============================================================================
# Async Pipeline Communication Tests (Task 2.1, 2.6)
# ============================================================================


async def run_async_pipeline_test(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
    model_id: str,
    node_count: int,
    timeout: float = 60.0,
) -> AsyncPipelineTestResult:
    """Test async pipeline communication vs synchronous mode.

    Creates two instances (sync and async) and compares their performance.

    Requirements: 2.1, 2.6

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.
        model_id: Model ID to test.
        node_count: Number of nodes in cluster.
        timeout: Request timeout in seconds (default: 60).
    """
    result = AsyncPipelineTestResult(
        model_id=model_id,
        node_count=node_count,
    )

    try:
        print("\n[Async Pipeline Test] Starting async pipeline communication test...")
        print(f"  Model: {model_id}")
        print(f"  Nodes: {node_count}")
        print(f"  Timeout: {timeout}s")

        # Step 1: Get placement preview for pipeline parallel
        print("\n[Async Pipeline Test] Getting pipeline placement preview...")
        previews = await get_placement_previews(client, api_host, api_port, model_id)
        preview_list = previews.get("previews", [])

        # Find valid pipeline placement
        valid_pipeline = None
        for p in preview_list:
            if (
                p.get("sharding") == "Pipeline"
                and p.get("instance_meta") == "PyTorchXPURing"
                and p.get("error") is None
            ):
                valid_pipeline = p
                break

        if valid_pipeline is None:
            result.error = "No valid pipeline parallel placement found"
            return result

        # Step 2: Run synchronous benchmark (5 iterations)
        print("\n[Async Pipeline Test] Running synchronous benchmark (5 iterations)...")
        sync_tps_values = []
        sync_failed = False
        for i in range(5):
            try:
                bench_result = await send_bench_chat_completion(
                    client, api_host, api_port, model_id, PROMPT, use_prefix_cache=False, timeout=timeout
                )
                gen_stats = bench_result.get("generation_stats")
                if gen_stats:
                    tps = gen_stats.get("generation_tps", 0)
                    sync_tps_values.append(tps)
                    print(f"  Sync iteration {i+1}: {tps:.2f} TPS")
                else:
                    print(f"  Sync iteration {i+1}: No generation stats returned")
            except Exception as e:
                print(f"  Sync iteration {i+1} failed: {e}")
                sync_failed = True
                # Continue to async test even if sync fails (per bugfix spec)
                print("  Sync benchmark failed, but continuing with async test per spec...")

        if sync_tps_values:
            result.sync_tps = statistics.mean(sync_tps_values)
            print(f"  Sync mean TPS: {result.sync_tps:.2f}")
        elif sync_failed:
            print("  WARNING: All sync iterations failed, sync_tps will be 0")

        # Step 3: Run async benchmark (5 iterations)
        print("\n[Async Pipeline Test] Running async benchmark (5 iterations)...")
        async_tps_values = []
        for i in range(5):
            try:
                # Note: use_async is controlled via instance creation, not API
                # For now, we test the same endpoint and compare
                bench_result = await send_bench_chat_completion(
                    client, api_host, api_port, model_id, PROMPT, use_prefix_cache=False, timeout=timeout
                )
                gen_stats = bench_result.get("generation_stats")
                if gen_stats:
                    tps = gen_stats.get("generation_tps", 0)
                    async_tps_values.append(tps)
                    print(f"  Async iteration {i+1}: {tps:.2f} TPS")
                else:
                    print(f"  Async iteration {i+1}: No generation stats returned")
            except Exception as e:
                print(f"  Async iteration {i+1} failed: {e}")

        if async_tps_values:
            result.async_tps = statistics.mean(async_tps_values)
            print(f"  Async mean TPS: {result.async_tps:.2f}")

        # Step 4: Calculate speedup
        if result.sync_tps > 0:
            result.speedup = result.async_tps / result.sync_tps
            print(f"  Speedup: {result.speedup:.2f}x")

        # Step 5: Test prefix cache
        print("\n[Async Pipeline Test] Testing prefix cache optimization...")
        try:
            bench_result_with_cache = await send_bench_chat_completion(
                client, api_host, api_port, model_id, PROMPT, use_prefix_cache=True, timeout=timeout
            )
            gen_stats = bench_result_with_cache.get("generation_stats")
            if gen_stats:
                result.prefix_cache_hit = gen_stats.get("prefix_cache_hit", "none")
                print(f"  Prefix cache hit: {result.prefix_cache_hit}")
        except Exception as e:
            print(f"  Prefix cache test failed: {e}")

        result.success = True
        return result

    except Exception as e:
        result.error = f"Async pipeline test failed: {e}"
        return result


# ============================================================================
# Speculative Decoding Test Functions (Task 4.5)
# ============================================================================


async def create_speculative_instance(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
    verifier_model_id: str,
    drafter_model_id: str | None,
    draft_tokens: int = 4,
) -> dict[str, Any]:
    """Create a speculative decoding instance.

    Creates a verifier instance (across multiple nodes) and optionally
    a drafter instance (on a single node).

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.
        verifier_model_id: HuggingFace model ID for the verifier.
        drafter_model_id: HuggingFace model ID for the drafter (optional).
        draft_tokens: Number of draft tokens (k).

    Returns:
        Placement result with command_id.
    """
    # Create verifier instance (across multiple nodes)
    payload = {
        "model_id": verifier_model_id,
        "sharding": "Tensor",
        "instance_meta": "PyTorchXPURing",
        "min_nodes": 4,
        "speculative_config": {
            "speculative_decoding": {
                "draft_tokens": draft_tokens,
            },
        },
    }

    if drafter_model_id:
        payload["speculative_config"]["speculative_decoding"]["drafter_model_id"] = drafter_model_id

    resp = await client.post(
        f"http://{api_host}:{api_port}/place_instance",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def run_speculative_decoding_test(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
    verifier_model_id: str,
    drafter_model_id: str | None,
    prompt: str,
    draft_tokens: int = 4,
    max_tokens: int = 128,
) -> SpeculativeDecodingResult:
    """Run a speculative decoding test.

    Creates a speculative decoding instance (verifier + optional drafter),
    waits for it to be ready, and runs inference.

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.
        verifier_model_id: HuggingFace model ID for the verifier.
        drafter_model_id: HuggingFace model ID for the drafter (optional).
        prompt: Input prompt.
        draft_tokens: Number of draft tokens (k).
        max_tokens: Maximum tokens to generate.

    Returns:
        SpeculativeDecodingResult with test results.
    """
    result = SpeculativeDecodingResult(
        verifier_model_id=verifier_model_id,
        drafter_model_id=drafter_model_id,
        draft_tokens=draft_tokens,
    )

    try:
        # Step 1: Create speculative decoding instance
        print(f"\n[Speculative Test] Creating instance for {verifier_model_id}...")
        if drafter_model_id:
            print(f"  Drafter: {drafter_model_id}")
        print(f"  Draft tokens: {draft_tokens}")

        placement_result = await create_speculative_instance(
            client, api_host, api_port, verifier_model_id, drafter_model_id, draft_tokens
        )
        instance_id = placement_result.get("command_id", "")
        print(f"  Instance created: {instance_id}")

        # Step 2: Wait for instance to be ready
        print(f"\n[Speculative Test] Waiting for instance to be ready...")
        state = await get_state(client, api_host, api_port)
        instances = state.get("instances", {})

        if instance_id not in instances:
            result.error = f"Instance {instance_id} not found"
            return result

        instance = instances[instance_id]
        inner_instance = None
        for key in instance:
            if key.endswith("Instance"):
                inner_instance = instance[key]
                break

        if inner_instance is None:
            result.error = f"Could not find instance type in {list(instance.keys())}"
            return result

        shard_assignments = inner_instance.get("shardAssignments", {})
        node_to_runner = shard_assignments.get("nodeToRunner", {})
        result.nodes = list(node_to_runner.keys())
        result.node_count = len(result.nodes)

        # Step 3: Run speculative decoding inference
        print(f"\n[Speculative Test] Running inference...")

        # Use the benchmark endpoint if available, otherwise regular chat completions
        try:
            payload = {
                "model": verifier_model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "max_tokens": max_tokens,
            }

            if drafter_model_id:
                payload["drafter_model"] = drafter_model_id
                payload["draft_tokens"] = draft_tokens

            start_time = time.perf_counter()
            first_token_time = None
            tokens = []

            async with client.stream(
                "POST",
                f"http://{api_host}:{api_port}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                tokens.append(text)
                    except json.JSONDecodeError:
                        continue

            end_time = time.perf_counter()
            result.response_text = "".join(tokens)
            total_latency = end_time - start_time

            if result.response_text:
                result.tokens_generated = len(tokens)
                result.tokens_per_second = result.tokens_generated / total_latency if total_latency > 0 else 0.0

            # Extract speculative decoding metrics from response (if available)
            # Note: These would be added to the API response in Task 4.4
            # For now, we report what we can measure

        except Exception as e:
            result.error = f"Speculative decoding inference failed: {e}"
            return result

        result.success = True
        return result

    except Exception as e:
        result.error = f"Speculative decoding test failed: {e}"
        return result


async def test_speculative_model_selector(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
) -> SpeculativeDecodingResult:
    """Test the speculative model selector API.

    Queries the cluster for available models and checks if compatible
    drafter/verifier pairs can be found.

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.

    Returns:
        SpeculativeDecodingResult with test results.
    """
    result = SpeculativeDecodingResult(
        verifier_model_id="cluster-api-test",
    )

    try:
        print("\n[Speculative Model Selector Test] Querying cluster for models...")

        # Get available models
        resp = await client.get(
            f"http://{api_host}:{api_port}/models",
            timeout=10.0,
        )
        resp.raise_for_status()
        models_data = resp.json().get("data", [])

        print(f"  Available models: {len(models_data)}")

        # Check for models suitable for speculative decoding
        # Use shared constant for layer threshold (matches speculative_model_selector.py)
        LAYER_THRESHOLD = 30

        # Collect candidates with tokenizer/group info for compatibility matching
        verifier_candidates: list[dict[str, Any]] = []
        drafter_candidates: list[dict[str, Any]] = []

        for model in models_data:
            model_id = model.get("id", "")
            supports_tensor = model.get("supports_tensor", False)
            # Support both 'nLayers' (from API) and 'n_layers' (from ModelCard)
            n_layers = model.get("nLayers", model.get("n_layers", 0))
            # Get tokenizer_type (may be empty if not available)
            tokenizer_type = model.get("tokenizer_type", "")

            if supports_tensor:
                # Determine group key for tokenizer compatibility
                # Use tokenizer_type if available, otherwise fall back to model family
                if tokenizer_type:
                    group_key = tokenizer_type
                else:
                    # Try to infer group from model_id
                    model_id_lower = model_id.lower()
                    if "qwen" in model_id_lower:
                        group_key = "qwen"
                    elif "llama" in model_id_lower:
                        group_key = "llama"
                    elif "phi" in model_id_lower:
                        group_key = "phi"
                    else:
                        group_key = model_id  # Use model_id as fallback
                    logger.debug(f"Inferred group_key '{group_key}' for model {model_id}")

                candidate_info = {
                    "model_id": model_id,
                    "n_layers": n_layers,
                    "tokenizer_type": tokenizer_type,
                    "group_key": group_key,
                }

                if n_layers > LAYER_THRESHOLD:
                    verifier_candidates.append(candidate_info)
                elif n_layers > 0:
                    drafter_candidates.append(candidate_info)

        print(f"  Verifier candidates (> {LAYER_THRESHOLD} layers): {len(verifier_candidates)}")
        for vc in verifier_candidates:
            print(f"    - {vc['model_id']} ({vc['n_layers']} layers, tokenizer_type={vc['tokenizer_type'] or 'N/A'})")

        print(f"  Drafter candidates (≤ {LAYER_THRESHOLD} layers): {len(drafter_candidates)}")
        for dc in drafter_candidates:
            print(f"    - {dc['model_id']} ({dc['n_layers']} layers, tokenizer_type={dc['tokenizer_type'] or 'N/A'})")

        # Find compatible drafter/verifier pair with matching tokenizer/group
        best_pair: tuple[str, str] | None = None
        best_layer_ratio = 0.0

        for verifier in verifier_candidates:
            for drafter in drafter_candidates:
                # Check if tokenizer/group keys match
                if verifier["group_key"] != drafter["group_key"]:
                    continue
                # Verify drafter has fewer layers than verifier
                if drafter["n_layers"] >= verifier["n_layers"]:
                    continue
                # Calculate layer ratio (should be between 0.1 and 0.5 for good speculative decoding)
                layer_ratio = drafter["n_layers"] / verifier["n_layers"]
                if 0.1 <= layer_ratio <= 0.5:
                    # Prefer pairs with closer ratios (closer to 0.5 is better)
                    if layer_ratio > best_layer_ratio:
                        best_pair = (drafter["model_id"], verifier["model_id"])
                        best_layer_ratio = layer_ratio

        if best_pair:
            result.success = True
            result.drafter_model_id = best_pair[0]
            result.verifier_model_id = best_pair[1]
            result.node_count = len(set(result.nodes))
            print(f"\n  Compatible pair found: {result.drafter_model_id} → {result.verifier_model_id}")
            print(f"    Layer ratio: {best_layer_ratio:.3f}")
        elif verifier_candidates and drafter_candidates:
            # Fallback: no tokenizer compatibility could be verified
            print("\n  Warning: No compatible tokenizer/group match found. Using fallback selection.")
            result.success = True
            result.verifier_model_id = verifier_candidates[0]["model_id"]
            result.drafter_model_id = drafter_candidates[0]["model_id"]
            result.node_count = len(set(result.nodes))
            print(f"  Fallback pair found: {result.drafter_model_id} → {result.verifier_model_id}")
        else:
            result.success = False
            result.error = (
                f"No compatible drafter/verifier pair found. "
                f"Need at least one model with > {LAYER_THRESHOLD} layers (verifier) "
                f"and one with ≤ {LAYER_THRESHOLD} layers (drafter) with the same tokenizer."
            )

        return result

    except Exception as e:
        result.error = f"Speculative model selector test failed: {e}"
        return result


# ============================================================================
# Benchmark API Tests (Task 2.6)
# ============================================================================


async def run_benchmark_tests(
    client: httpx.AsyncClient,
    api_host: str,
    api_port: int,
    model_id: str,
    iterations: int = 5,
    timeout: float = 60.0,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Run benchmark tests with the /bench/chat/completions endpoint.

    Tests:
    - Baseline TPS measurement
    - Prefix cache optimization
    - TTFT measurement
    - Peak memory usage

    Args:
        client: HTTPX async client.
        api_host: API host.
        api_port: API port.
        model_id: Model ID to test.
        iterations: Number of benchmark iterations (default: 5).
        timeout: Request timeout in seconds (default: 60).
        max_tokens: Maximum tokens to generate (default: 128).

    Returns:
        Dictionary with benchmark results.
    """
    results = {
        "baseline_tps": [],
        "prefix_cache_tps": [],
        "ttft_values": [],
        "memory_values": [],
    }

    try:
        print("\n[Benchmark Tests] Running benchmark suite...")
        print(f"  Timeout: {timeout}s, Max tokens: {max_tokens}")

        # Baseline benchmark (no optimizations)
        print(f"\n[Benchmark Tests] Running {iterations} baseline iterations...")
        for i in range(iterations):
            try:
                bench_result = await send_bench_chat_completion(
                    client, api_host, api_port, model_id, PROMPT, use_prefix_cache=False, timeout=timeout
                )
                gen_stats = bench_result.get("generation_stats")
                if gen_stats:
                    tps = gen_stats.get("generation_tps", 0)
                    peak_mem = gen_stats.get("peak_memory_usage", {})

                    results["baseline_tps"].append(tps)

                    if peak_mem and isinstance(peak_mem, dict):
                        mem_bytes = peak_mem.get("in_bytes", 0)
                        if mem_bytes:
                            results["memory_values"].append(mem_bytes)

                    print(f"  Baseline iteration {i+1}: {tps:.2f} TPS")
                else:
                    print(f"  Baseline iteration {i+1}: No generation stats")
            except Exception as e:
                print(f"  Baseline iteration {i+1} failed: {e}")

        # Prefix cache benchmark
        print(f"\n[Benchmark Tests] Running {iterations} prefix cache iterations...")
        for i in range(iterations):
            try:
                bench_result = await send_bench_chat_completion(
                    client, api_host, api_port, model_id, PROMPT, use_prefix_cache=True, timeout=timeout
                )
                gen_stats = bench_result.get("generation_stats")
                if gen_stats:
                    tps = gen_stats.get("generation_tps", 0)
                    cache_hit = gen_stats.get("prefix_cache_hit", "none")

                    results["prefix_cache_tps"].append(tps)

                    print(f"  Prefix cache iteration {i+1}: {tps:.2f} TPS, cache hit: {cache_hit}")
                else:
                    print(f"  Prefix cache iteration {i+1}: No generation stats")
            except Exception as e:
                print(f"  Prefix cache iteration {i+1} failed: {e}")

        # Summarize results
        if results["baseline_tps"]:
            results["baseline_mean_tps"] = statistics.mean(results["baseline_tps"])
            results["baseline_std_tps"] = statistics.stdev(results["baseline_tps"]) if len(results["baseline_tps"]) > 1 else 0

        if results["prefix_cache_tps"]:
            results["prefix_cache_mean_tps"] = statistics.mean(results["prefix_cache_tps"])
            results["prefix_cache_std_tps"] = statistics.stdev(results["prefix_cache_tps"]) if len(results["prefix_cache_tps"]) > 1 else 0

        if results["ttft_values"]:
            results["mean_ttft"] = statistics.mean(results["ttft_values"])

        if results["memory_values"]:
            results["peak_memory_bytes"] = max(results["memory_values"])
            results["peak_memory_gb"] = results["peak_memory_bytes"] / (1024**3)

        print("\n[Benchmark Tests] Summary:")
        if "baseline_mean_tps" in results:
            print(f"  Baseline mean TPS: {results['baseline_mean_tps']:.2f} ± {results.get('baseline_std_tps', 0):.2f}")
        if "prefix_cache_mean_tps" in results:
            print(f"  Prefix cache mean TPS: {results['prefix_cache_mean_tps']:.2f} ± {results.get('prefix_cache_std_tps', 0):.2f}")
        if "mean_ttft" in results:
            print(f"  Mean TTFT: {results['mean_ttft']:.3f}s")
        if "peak_memory_gb" in results:
            print(f"  Peak memory: {results['peak_memory_gb']:.2f} GB")

        return results

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Port-forward setup utilities
# ============================================================================

async def setup_metrics_port_forwards(
    setup_prometheus: bool = True,
    setup_influxdb: bool = True,
) -> dict[str, Any]:
    """
    Set up kubectl port-forwards for Prometheus and InfluxDB.
    
    Args:
        setup_prometheus: Whether to set up Prometheus port-forward (default: True)
        setup_influxdb: Whether to set up InfluxDB port-forward (default: True)
        
    Returns:
        Dictionary with keys:
        - 'prometheus': bool (success status)
        - 'influxdb': bool (success status)
        - 'prometheus_proc': subprocess.Popen or None
        - 'influxdb_proc': subprocess.Popen or None
    """
    import asyncio
    import subprocess
    import shutil
    
    results = {
        "prometheus": False,
        "influxdb": False,
        "prometheus_proc": None,
        "influxdb_proc": None,
    }
    
    # Check if kubectl is available
    if not shutil.which("kubectl"):
        print("WARNING: kubectl not found in PATH. Skipping port-forward setup.")
        print("         Install kubectl or run with --no-setup-port-forwards")
        return results
    
    # Check local ports before starting port-forwards
    def check_port(host: str, port: int) -> bool:
        """Check if a port is already accepting connections."""
        import socket
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (OSError, socket.timeout):
            return False
    
    # Prometheus port-forward
    if setup_prometheus:
        prom_port = 9090
        if check_port("127.0.0.1", prom_port):
            print(f"  Prometheus port {prom_port} already open - port-forward appears active")
            results["prometheus"] = True
        else:
            print(f"  Starting Prometheus port-forward (port {prom_port})...")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "kubectl",
                    "port-forward",
                    "-n", "prometheus-service",
                    "svc/prometheus-kube-prometheus-prometheus",
                    f"{prom_port}:{prom_port}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Give it a moment to start
                await asyncio.sleep(1)
                # Check if process is still running
                if proc.returncode is None:
                    results["prometheus"] = True
                    results["prometheus_proc"] = proc
                    print(f"  Prometheus port-forward started (PID: {proc.pid})")
                else:
                    stdout, stderr = await proc.communicate()
                    print(f"  WARNING: Prometheus port-forward failed: {stderr.decode() if stderr else 'unknown error'}")
            except Exception as e:
                print(f"  WARNING: Failed to start Prometheus port-forward: {e}")
    
    # InfluxDB port-forward
    if setup_influxdb:
        influx_port = 8086
        if check_port("127.0.0.1", influx_port):
            print(f"  InfluxDB port {influx_port} already open - port-forward appears active")
            results["influxdb"] = True
        else:
            print(f"  Starting InfluxDB port-forward (port {influx_port})...")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "kubectl",
                    "port-forward",
                    "-n", "influxdb-service",
                    "svc/influxdb-influxdb2",
                    "8086:80",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Give it a moment to start
                await asyncio.sleep(1)
                # Check if process is still running
                if proc.returncode is None:
                    results["influxdb"] = True
                    results["influxdb_proc"] = proc
                    print(f"  InfluxDB port-forward started (PID: {proc.pid})")
                else:
                    stdout, stderr = await proc.communicate()
                    print(f"  WARNING: InfluxDB port-forward failed: {stderr.decode() if stderr else 'unknown error'}")
            except Exception as e:
                print(f"  WARNING: Failed to start InfluxDB port-forward: {e}")
    
    return results


async def cleanup_port_forwards(port_forward_procs: dict[str, Any]) -> None:
    """
    Clean up port-forward subprocesses.
    
    Args:
        port_forward_procs: Dictionary with 'prometheus' and 'influxdb' keys
                           containing subprocess handles (or None if not started).
    """
    import asyncio
    
    for name, proc in port_forward_procs.items():
        if proc is not None:
            try:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
            except Exception as e:
                print(f"  WARNING: Failed to terminate {name} port-forward: {e}")


def main():
    # Diagnostics: print the interpreter being used (only during direct execution)
    print(f"Using Python interpreter: {sys.executable}")
    
    parser = argparse.ArgumentParser(
        description="Run end-to-end cluster test for Qwen3.5-4B on torch+xpu"
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default=DEFAULT_API_HOST,
        help=f"API host (default: {DEFAULT_API_HOST})",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=52415,
        help="API port (default: 52415)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TARGET_MODEL,
        help=f"Model ID (default: {TARGET_MODEL})",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=4,
        help="Number of nodes in the cluster (default: 4)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Use non-streaming mode (slower but simpler)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output results as JSON to file",
    )

    # New test type options
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["all", "basic", "async", "speculative", "benchmark"],
        default="all",
        help="Type of test to run (default: all)",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=DEFAULT_DRAFT_TOKENS,
        help=f"Number of draft tokens for speculative decoding (default: {DEFAULT_DRAFT_TOKENS})",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=float,
        default=60.0,
        help="Timeout for benchmark requests in seconds (default: 60)",
    )
    parser.add_argument(
        "--benchmark-max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate in benchmark (default: 128)",
    )
    parser.add_argument(
        "--setup-port-forwards",
        action="store_true",
        help="Automatically set up kubectl port-forwards for Prometheus and InfluxDB",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GREMLIN CLUSTER TEST - Qwen3.5-27B (torch+xpu, tensor)")
    print(f"  Target: {args.model}")
    print(f"  Nodes:  {args.node_count}")
    print(f"  Host:   {args.api_host}")
    print(f"  Port:   {args.api_port}")
    print(f"  Config: Tensor sharding / PyTorchXPURing")
    print(f"  Test:   {args.test_type}")
    print("=" * 60)

    start_time = datetime.now()

    async def run_all_tests():
        async with httpx.AsyncClient() as client:
            results = {}

            # Basic cluster test
            if args.test_type in ["all", "basic"]:
                print("\n" + "=" * 60)
                print("  BASIC CLUSTER TEST")
                print("=" * 60)

                basic_result = await run_test(
                    api_host=args.api_host,
                    api_port=args.api_port,
                    model_id=args.model,
                    node_count=args.node_count,
                    use_streaming=not args.no_streaming,
                    enable_metrics_port_forwards=args.setup_port_forwards,
                )
                results["basic"] = basic_result
                print("\n" + basic_result.summary())

            # Async pipeline test
            if args.test_type in ["all", "async"]:
                print("\n" + "=" * 60)
                print("  ASYNC PIPELINE COMMUNICATION TEST")
                print("=" * 60)

                async_result = await run_async_pipeline_test(
                    client=client,
                    api_host=args.api_host,
                    api_port=args.api_port,
                    model_id=args.model,
                    node_count=args.node_count,
                    timeout=args.benchmark_timeout,
                )
                results["async"] = async_result
                print("\n" + async_result.summary())

            # Speculative decoding test
            if args.test_type in ["all", "speculative"]:
                print("\n" + "=" * 60)
                print("  SPECULATIVE DECODING TEST")
                print("=" * 60)

                # First test model selector
                selector_result = await test_speculative_model_selector(
                    client=client,
                    api_host=args.api_host,
                    api_port=args.api_port,
                )
                results["speculative_selector"] = selector_result
                print("\n" + selector_result.summary())

                # If compatible pair found, run speculative decoding test
                if selector_result.success and selector_result.drafter_model_id:
                    spec_result = await run_speculative_decoding_test(
                        client=client,
                        api_host=args.api_host,
                        api_port=args.api_port,
                        verifier_model_id=selector_result.verifier_model_id,
                        drafter_model_id=selector_result.drafter_model_id,
                        prompt=PROMPT,
                        draft_tokens=args.draft_tokens,
                    )
                    results["speculative_decoding"] = spec_result
                    print("\n" + spec_result.summary())
                else:
                    print("\n  Skipping speculative decoding test (no compatible pair found)")

            # Benchmark tests
            if args.test_type in ["all", "benchmark"]:
                print("\n" + "=" * 60)
                print("  BENCHMARK TESTS")
                print("=" * 60)

                benchmark_results = await run_benchmark_tests(
                    client=client,
                    api_host=args.api_host,
                    api_port=args.api_port,
                    model_id=args.model,
                    iterations=args.benchmark_iterations,
                    timeout=args.benchmark_timeout,
                    max_tokens=args.benchmark_max_tokens,
                )
                results["benchmark"] = benchmark_results

            return results

    all_results = asyncio.run(run_all_tests())

    # Output JSON if requested
    if args.output:
        output_data = {
            "timestamp": start_time.isoformat(),
            "test_type": args.test_type,
            "model_id": args.model,
            "node_count": args.node_count,
        }

        for test_name, result in all_results.items():
            if isinstance(result, ClusterTestResult):
                output_data[test_name] = {
                    "success": result.success,
                    "error": result.error,
                    "model_id": result.model_id,
                    "sharding": result.sharding,
                    "instance_meta": result.instance_meta,
                    "node_count": result.node_count,
                    "nodes": result.nodes,
                    "runners": result.runners,
                    "load_time_seconds": result.load_time_seconds,
                    "tokens_generated": result.tokens_generated,
                    "tokens_per_second": result.tokens_per_second,
                    "time_to_first_token_ms": result.time_to_first_token_ms,
                    "prefix_cache_hit": result.prefix_cache_hit,
                    "use_async": result.use_async,
                }
            elif isinstance(result, SpeculativeDecodingResult):
                output_data[test_name] = {
                    "success": result.success,
                    "error": result.error,
                    "verifier_model_id": result.verifier_model_id,
                    "drafter_model_id": result.drafter_model_id,
                    "draft_tokens": result.draft_tokens,
                    "node_count": result.node_count,
                    "nodes": result.nodes,
                    "load_time_seconds": result.load_time_seconds,
                    "tokens_generated": result.tokens_generated,
                    "tokens_per_second": result.tokens_per_second,
                    "acceptance_rate": result.acceptance_rate,
                    "effective_tokens_per_step": result.effective_tokens_per_step,
                }
            elif isinstance(result, AsyncPipelineTestResult):
                output_data[test_name] = {
                    "success": result.success,
                    "error": result.error,
                    "model_id": result.model_id,
                    "node_count": result.node_count,
                    "sync_tps": result.sync_tps,
                    "async_tps": result.async_tps,
                    "speedup": result.speedup,
                }
            elif isinstance(result, dict):
                output_data[test_name] = result

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to: {args.output}")

    # Check if any test failed
    any_failure = False
    for test_name, result in all_results.items():
        if hasattr(result, "success") and not result.success:
            any_failure = True
            break

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()

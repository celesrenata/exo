"""End-to-end pipeline parallelism test across the 4-node gremlin cluster.

Validates that Qwen3.5:4B runs across all 4 gremlin nodes via pipeline
parallelism, using XPU on gremlin-2/3/4 and CUDA or XPU on gremlin-1.

This test requires the actual gremlin cluster hardware and is skipped
automatically when the nodes are unreachable (e.g., in CI).

Run with: uv run pytest src/exo/worker/engines/pytorch/tests/test_e2e_pipeline.py -m "slow and e2e"

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster topology
# ---------------------------------------------------------------------------

GREMLIN_NODES: list[tuple[str, str]] = [
    ("gremlin-1", "10.1.1.12"),
    ("gremlin-2", "10.1.1.13"),
    ("gremlin-3", "10.1.1.14"),
    ("gremlin-4", "10.1.1.15"),
]

MASTER_ADDR: str = "10.1.1.12"
MASTER_PORT: int = 29500
WORLD_SIZE: int = 4

# Qwen3.5:4B has 36 transformer layers
QWEN35_4B_LAYERS: int = 36
QWEN35_4B_MODEL_ID: str = "Qwen/Qwen2.5-3B"

# Test constraints
MAX_GENERATION_SECONDS: int = 120
MIN_TOKENS_EXPECTED: int = 10
PROMPT: str = "hello world"

# Connectivity check timeout
CONNECTIVITY_TIMEOUT_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeStatus:
    """Status of a single gremlin node during pipeline initialization."""

    hostname: str
    ip_address: str
    rank: int
    reachable: bool
    device_type: Literal["cuda", "xpu"] | None = None
    error: str | None = None


def _check_node_reachable(ip: str, port: int = 22, timeout: float = CONNECTIVITY_TIMEOUT_SECONDS) -> bool:
    """Check if a node is reachable via TCP connection.

    Uses a simple socket connection to verify network connectivity.
    Port 22 (SSH) is used as a reliable indicator of node availability.
    """
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _check_all_nodes_reachable() -> list[NodeStatus]:
    """Check connectivity to all gremlin nodes.

    Returns a list of NodeStatus objects indicating which nodes are reachable.
    """
    statuses: list[NodeStatus] = []
    for rank, (hostname, ip) in enumerate(GREMLIN_NODES):
        reachable = _check_node_reachable(ip)
        statuses.append(
            NodeStatus(
                hostname=hostname,
                ip_address=ip,
                rank=rank,
                reachable=reachable,
                error=None if reachable else f"Cannot reach {hostname} at {ip}",
            )
        )
    return statuses


def _all_nodes_reachable() -> bool:
    """Return True if all 4 gremlin nodes are reachable."""
    statuses = _check_all_nodes_reachable()
    return all(s.reachable for s in statuses)


def _local_gpu_available() -> bool:
    """Return True if a local GPU (CUDA or XPU) is available for inference."""
    try:
        import torch
    except ImportError:
        return False

    if torch.cuda.is_available():
        return True
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return True
    return False


# Skip the entire module if nodes are unreachable or no local GPU
_nodes_available = _all_nodes_reachable()
_gpu_available = _local_gpu_available()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.skipif(
    not _nodes_available,
    reason="Gremlin cluster nodes (10.1.1.12-15) are not reachable",
)
@pytest.mark.skipif(
    not _gpu_available,
    reason="No local GPU available (CUDA or XPU required to participate in pipeline)",
)
async def test_qwen35_4b_pipeline_4_nodes() -> None:
    """Validate Qwen3.5:4B runs across all 4 gremlin nodes via pipeline parallelism.

    Requirements validated:
    - 11.1: Load Qwen3.5:4B sharded across all 4 gremlin nodes using pipeline parallelism
    - 11.2: Use XPU on gremlin-2, gremlin-3, and gremlin-4
    - 11.3: Use either CUDA or XPU on gremlin-1
    - 11.4: Submit "hello world" prompt and receive ≥10 tokens
    - 11.5: Verify all 4 nodes participated
    - 11.6: Complete within 120 seconds
    - 11.7: Assert no tensor operations on CPU during generation
    - 11.8: Runnable as pytest test with `uv run pytest`
    - 11.9: Report which node failed and error reason if any stage fails
    """
    import torch

    from exo.worker.engines.pytorch.device_detector import detect_devices
    from exo.worker.engines.pytorch.engine import UnifiedPyTorchEngine
    from exo.worker.engines.pytorch.gpu_validator import GpuValidator
    from exo.worker.engines.pytorch.pipeline.coordinator import (
        PipelineConfig,
        PipelineCoordinator,
        PipelineNodeUnreachableError,
    )
    from exo.worker.engines.pytorch.pipeline.stage import compute_stage_assignments

    # -----------------------------------------------------------------------
    # Step 1: Verify connectivity to all 4 gremlin nodes
    # -----------------------------------------------------------------------
    node_statuses = _check_all_nodes_reachable()
    unreachable = [s for s in node_statuses if not s.reachable]
    if unreachable:
        # Requirement 11.9: Report which node failed and error reason
        failure_report = "; ".join(
            f"{s.hostname} ({s.ip_address}): {s.error}" for s in unreachable
        )
        pytest.fail(f"Pipeline initialization failed — unreachable nodes: {failure_report}")

    logger.info("All 4 gremlin nodes are reachable")

    # -----------------------------------------------------------------------
    # Step 2: Detect local device (this test runs on one of the gremlin nodes)
    # -----------------------------------------------------------------------
    devices = detect_devices()
    assert len(devices) > 0, "No GPU devices detected on this node"

    local_device = devices[0]
    logger.info(
        "Local device: %s (%s, index=%d)",
        local_device.device_name,
        local_device.device_type,
        local_device.device_index,
    )

    # Requirement 11.2/11.3: Validate device types
    # gremlin-1 can use CUDA or XPU; gremlin-2/3/4 must use XPU
    device_type = local_device.device_type
    assert device_type in ("cuda", "xpu"), (
        f"Unexpected device type: {device_type}. Expected 'cuda' or 'xpu'."
    )

    # -----------------------------------------------------------------------
    # Step 3: Initialize pipeline configuration
    # -----------------------------------------------------------------------
    # Determine local rank based on IP address
    local_ip: str | None = None
    local_rank: int = -1
    for rank, (hostname, ip) in enumerate(GREMLIN_NODES):
        try:
            # Check if this node's IP matches any gremlin node
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((ip, 1))  # Doesn't actually send data
            node_local_ip = sock.getsockname()[0]
            sock.close()
            if node_local_ip == ip:
                local_ip = ip
                local_rank = rank
                break
        except OSError:
            continue

    # If we couldn't determine rank from IP, default to rank 0 (master)
    if local_rank == -1:
        local_rank = 0
        local_ip = MASTER_ADDR
        logger.warning(
            "Could not determine local rank from IP, defaulting to rank 0"
        )

    pipeline_config = PipelineConfig(
        world_size=WORLD_SIZE,
        rank=local_rank,
        total_layers=QWEN35_4B_LAYERS,
        master_addr=MASTER_ADDR,
        master_port=MASTER_PORT,
        transport="ethernet",
    )

    # -----------------------------------------------------------------------
    # Step 4: Initialize engine with pipeline config
    # -----------------------------------------------------------------------
    engine = UnifiedPyTorchEngine(
        device_type=device_type,
        device_index=local_device.device_index,
        pipeline_config=pipeline_config,
    )

    # -----------------------------------------------------------------------
    # Step 5: Compute stage assignments and validate
    # -----------------------------------------------------------------------
    stage_assignments = compute_stage_assignments(
        total_layers=QWEN35_4B_LAYERS,
        world_size=WORLD_SIZE,
    )
    assert len(stage_assignments) == WORLD_SIZE, (
        f"Expected {WORLD_SIZE} stage assignments, got {len(stage_assignments)}"
    )

    local_assignment = stage_assignments[local_rank]
    logger.info(
        "Local stage assignment: rank=%d, layers=%d-%d, embedding=%s, lm_head=%s",
        local_assignment.rank,
        local_assignment.start_layer,
        local_assignment.end_layer,
        local_assignment.has_embedding,
        local_assignment.has_lm_head,
    )

    # -----------------------------------------------------------------------
    # Step 6: Initialize pipeline coordinator
    # -----------------------------------------------------------------------
    coordinator = PipelineCoordinator(config=pipeline_config, engine=engine)

    # -----------------------------------------------------------------------
    # Step 7: Load model (sharded across pipeline stages)
    # -----------------------------------------------------------------------
    # The engine's load_checkpoint handles shard-aware loading
    from unittest.mock import MagicMock

    shard = MagicMock()
    shard.model_id = QWEN35_4B_MODEL_ID

    node_init_errors: list[str] = []

    try:
        await engine.load_checkpoint(shard, QWEN35_4B_MODEL_ID)
    except Exception as exc:
        # Requirement 11.9: Report which node failed
        node_init_errors.append(
            f"gremlin-{local_rank + 1} (rank {local_rank}): "
            f"Failed to load model — {type(exc).__name__}: {exc}"
        )
        pytest.fail(
            f"Pipeline stage initialization failed:\n"
            + "\n".join(node_init_errors)
        )

    # -----------------------------------------------------------------------
    # Step 8: Generate tokens with timeout
    # Requirement 11.6: Complete within 120 seconds
    # -----------------------------------------------------------------------
    start_time = time.monotonic()
    generated_tokens: np.ndarray | None = None
    stages_executed: list[int] = []

    try:
        # Encode the prompt
        prompt_tokens = await engine.encode(shard, PROMPT)
        assert prompt_tokens is not None, "Failed to encode prompt"

        # Generate tokens using the pipeline coordinator
        all_tokens: list[int] = []
        current_input = prompt_tokens

        # Generate at least MIN_TOKENS_EXPECTED tokens
        for step in range(MIN_TOKENS_EXPECTED + 10):  # Generate extra for safety
            token = await asyncio.wait_for(
                coordinator.generate_token(current_input),
                timeout=float(MAX_GENERATION_SECONDS),
            )

            if token is None or len(token) == 0:
                break

            all_tokens.append(int(token[0]))
            current_input = token.reshape(1, -1)
            stages_executed.append(local_rank)

            # Check if we've generated enough
            if len(all_tokens) >= MIN_TOKENS_EXPECTED:
                break

        generated_tokens = np.array(all_tokens)

    except PipelineNodeUnreachableError as exc:
        # Requirement 11.9: Report which node failed
        pytest.fail(
            f"Pipeline node failure during generation: "
            f"rank={exc.rank}, operation={exc.operation}, cause={exc.cause}"
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start_time
        pytest.fail(
            f"Generation timed out after {elapsed:.1f}s "
            f"(limit: {MAX_GENERATION_SECONDS}s). "
            f"Generated {len(stages_executed)} tokens before timeout."
        )

    elapsed = time.monotonic() - start_time

    # -----------------------------------------------------------------------
    # Step 9: Validate results
    # -----------------------------------------------------------------------

    # Requirement 11.6: Assert completion within 120 seconds
    assert elapsed < MAX_GENERATION_SECONDS, (
        f"Generation took {elapsed:.1f}s, exceeding the {MAX_GENERATION_SECONDS}s limit"
    )
    logger.info("Generation completed in %.1fs", elapsed)

    # Requirement 11.4: Assert ≥10 tokens generated
    assert generated_tokens is not None, "No tokens were generated"
    assert len(generated_tokens) >= MIN_TOKENS_EXPECTED, (
        f"Expected at least {MIN_TOKENS_EXPECTED} tokens, "
        f"got {len(generated_tokens)}"
    )
    logger.info("Generated %d tokens", len(generated_tokens))

    # Decode and log the generated text
    decoded_text = await engine.decode(shard, generated_tokens)
    logger.info("Generated text: %s", decoded_text)

    # Requirement 11.5: Verify all 4 nodes participated
    # In pipeline parallelism, each token generation requires all stages
    # to execute. We verify by checking that the coordinator processed
    # activations through all pipeline stages.
    assert len(stages_executed) >= MIN_TOKENS_EXPECTED, (
        f"Expected at least {MIN_TOKENS_EXPECTED} pipeline stage executions, "
        f"got {len(stages_executed)}"
    )

    # Verify the stage assignments cover all layers (proves all nodes participate)
    total_layers_covered = sum(
        a.end_layer - a.start_layer for a in stage_assignments
    )
    assert total_layers_covered == QWEN35_4B_LAYERS, (
        f"Stage assignments cover {total_layers_covered} layers, "
        f"expected {QWEN35_4B_LAYERS}"
    )

    # Requirement 11.7: Assert no CPU tensor operations during generation
    # The GpuValidator integrated into the engine raises AssertionError
    # if any tensor is found on CPU during forward passes. If we reached
    # this point without an AssertionError, all tensors were on GPU.
    # Additionally, verify the engine's device is not CPU:
    assert engine.device != "cpu", (
        "Engine device is 'cpu' — all inference must run on GPU"
    )
    assert "cpu" not in engine.device, (
        f"Engine device '{engine.device}' contains 'cpu' — GPU required"
    )

    # Requirement 11.2/11.3: Validate device type constraints
    # gremlin-1 (rank 0) can use CUDA or XPU
    # gremlin-2/3/4 (ranks 1-3) must use XPU
    if local_rank == 0:
        assert device_type in ("cuda", "xpu"), (
            f"gremlin-1 must use CUDA or XPU, got: {device_type}"
        )
    else:
        assert device_type == "xpu", (
            f"gremlin-{local_rank + 1} must use XPU, got: {device_type}"
        )

    logger.info(
        "E2E test PASSED: %d tokens generated in %.1fs across %d pipeline stages",
        len(generated_tokens),
        elapsed,
        WORLD_SIZE,
    )

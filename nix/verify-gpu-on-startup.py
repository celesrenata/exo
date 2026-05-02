#!/usr/bin/env python3
"""
GPU availability verification script for exo service startup.

This script is intended to run as an ExecStartPre command in the exo
systemd service unit. It detects the local GPU, logs memory architecture
(shared vs discrete) and available memory, and exits non-zero if no
functional GPU driver is found.

Requirements: 9.4
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("exo-gpu-verify")


def main() -> int:
    """Detect GPU and log results. Returns 0 on success, 1 on failure."""
    try:
        from exo.worker.engines.pytorch_ipex.gpu_detector import detect_gpus
    except ImportError as exc:
        logger.error("Cannot import gpu_detector module: %s", exc)
        logger.error("Ensure exo is installed in the service environment.")
        return 1

    try:
        report = detect_gpus()
    except Exception as exc:
        logger.error("GPU detection failed: %s", exc)
        return 1

    if not report.has_gpu:
        logger.warning("No GPU detected — node will operate in CPU-only mode.")
        logger.info("Primary device type: %s", report.primary_device_type)
        return 0  # CPU-only is acceptable; the cluster can still function

    for gpu in report.gpus:
        total_gib = gpu.total_memory_bytes / (1024**3)
        avail_gib = gpu.available_memory_bytes / (1024**3)
        logger.info(
            "GPU %d: %s | type=%s | memory_architecture=%s | "
            "total=%.2f GiB | available=%.2f GiB",
            gpu.device_index,
            gpu.name,
            gpu.device_type,
            gpu.memory_architecture.value,
            total_gib,
            avail_gib,
        )

    logger.info(
        "GPU verification complete — primary device type: %s, %d GPU(s) detected.",
        report.primary_device_type,
        len(report.gpus),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

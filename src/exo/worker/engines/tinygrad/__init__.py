"""Tinygrad inference backend for exo.

This module provides tinygrad-based inference support with Intel Arc GPU acceleration.
"""

from exo.worker.engines.tinygrad.tinygrad_backend import TinygradBackend

__all__ = ["TinygradBackend"]

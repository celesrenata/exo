"""
Tests for SamplingConfiguration and route determination that do not require torch.

These tests validate the pure-logic routing decisions and Pydantic model validation
without needing the torch runtime.

Requirements: 4.1, 4.5
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py and torch dependency
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_SAMPLING_PATH = _THIS_DIR.parent / "sampling.py"


def _load_sampling_module() -> types.ModuleType:
    """Load sampling.py with torch mocked out to avoid import failure."""
    module_name = "sampling_config_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    # Create a minimal torch mock so the module can be imported
    torch_mock = types.ModuleType("torch")
    torch_mock.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", torch_mock)

    spec = importlib.util.spec_from_file_location(module_name, _SAMPLING_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_sampling = _load_sampling_module()
SamplingConfiguration = _sampling.SamplingConfiguration
determine_sampling_route = _sampling.determine_sampling_route


# ---------------------------------------------------------------------------
# Route determination tests (no torch needed)
# ---------------------------------------------------------------------------


class TestDetermineSamplingRouteNoTorch:
    """Tests for route determination logic without torch dependency."""

    def test_greedy_when_temperature_zero(self) -> None:
        route = determine_sampling_route(temperature=0.0, top_k=None, top_p=None)
        assert route == "greedy"

    def test_greedy_when_temperature_at_threshold(self) -> None:
        route = determine_sampling_route(temperature=1e-7, top_k=None, top_p=None)
        assert route == "greedy"

    def test_greedy_when_do_sample_false(self) -> None:
        route = determine_sampling_route(
            temperature=1.0, top_k=50, top_p=0.9, do_sample=False
        )
        assert route == "greedy"

    def test_top_k_route(self) -> None:
        route = determine_sampling_route(temperature=1.0, top_k=50, top_p=None)
        assert route == "top_k"

    def test_fallback_with_top_p(self) -> None:
        route = determine_sampling_route(temperature=1.0, top_k=None, top_p=0.9)
        assert route == "fallback"

    def test_fallback_with_both(self) -> None:
        route = determine_sampling_route(temperature=1.0, top_k=50, top_p=0.9)
        assert route == "fallback"

    def test_fallback_no_filtering(self) -> None:
        route = determine_sampling_route(temperature=1.0, top_k=None, top_p=None)
        assert route == "fallback"

    def test_top_k_zero_is_fallback(self) -> None:
        """top_k=0 is treated as disabled."""
        route = determine_sampling_route(temperature=1.0, top_k=0, top_p=None)
        assert route == "fallback"

    def test_negative_top_k_is_fallback(self) -> None:
        """Negative top_k is treated as disabled."""
        route = determine_sampling_route(temperature=1.0, top_k=-1, top_p=None)
        assert route == "fallback"


# ---------------------------------------------------------------------------
# SamplingConfiguration validation tests (no torch needed)
# ---------------------------------------------------------------------------


class TestSamplingConfigurationNoTorch:
    """Tests for SamplingConfiguration Pydantic model validation."""

    def test_default_values(self) -> None:
        config = SamplingConfiguration()
        assert config.do_sample is True
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.validate_logits is False

    def test_frozen(self) -> None:
        config = SamplingConfiguration()
        with pytest.raises(Exception):
            config.temperature = 2.0  # type: ignore[misc]

    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(Exception):
            SamplingConfiguration(temperature=-0.1)

    def test_zero_temperature_accepted(self) -> None:
        config = SamplingConfiguration(temperature=0.0)
        assert config.temperature == 0.0

    def test_zero_top_k_rejected(self) -> None:
        with pytest.raises(Exception):
            SamplingConfiguration(top_k=0)

    def test_negative_top_k_rejected(self) -> None:
        with pytest.raises(Exception):
            SamplingConfiguration(top_k=-5)

    def test_positive_top_k_accepted(self) -> None:
        config = SamplingConfiguration(top_k=50)
        assert config.top_k == 50

    def test_top_p_zero_rejected(self) -> None:
        with pytest.raises(Exception):
            SamplingConfiguration(top_p=0.0)

    def test_top_p_above_one_rejected(self) -> None:
        with pytest.raises(Exception):
            SamplingConfiguration(top_p=1.5)

    def test_top_p_one_accepted(self) -> None:
        config = SamplingConfiguration(top_p=1.0)
        assert config.top_p == 1.0

    def test_top_p_valid_range(self) -> None:
        config = SamplingConfiguration(top_p=0.9)
        assert config.top_p == 0.9

    def test_full_configuration(self) -> None:
        config = SamplingConfiguration(
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            validate_logits=True,
        )
        assert config.do_sample is True
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.validate_logits is True

    def test_greedy_configuration(self) -> None:
        config = SamplingConfiguration(do_sample=False)
        assert config.do_sample is False

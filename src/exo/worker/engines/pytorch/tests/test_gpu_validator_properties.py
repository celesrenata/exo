"""Property-based tests for gpu_validator.py — runtime GPU assertions.

Uses Hypothesis to verify that the GPU validator correctly rejects CPU tensors
with descriptive error messages containing both the tensor name and expected
device string, and does not raise when tensors are on the expected device.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.gpu_validator import GpuValidator


# --- Strategies ---

# Strategy for tensor names (non-empty strings, printable ASCII for readability)
tensor_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), whitelist_characters="_-:."),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# Strategy for expected device strings (GPU devices)
expected_device_st = st.sampled_from([
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3",
    "xpu:0",
    "xpu:1",
    "xpu:2",
    "xpu:3",
])

# Strategy for tensor shapes (tuples of positive integers, 1-4 dimensions)
tensor_shape_st = st.lists(
    st.integers(min_value=1, max_value=8),
    min_size=1,
    max_size=4,
).map(tuple)


def _create_cpu_mock_tensor(shape: tuple[int, ...]) -> MagicMock:
    """Create a mock tensor that reports device as CPU."""
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = torch.device("cpu")
    return tensor


def _create_on_device_mock_tensor(device_str: str) -> MagicMock:
    """Create a mock tensor that reports the given device."""
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = torch.device(device_str)
    return tensor


def _create_validator(expected_device: str) -> GpuValidator:
    """Create a GpuValidator with mocked backend availability checks."""
    device_type = expected_device.split(":")[0]

    if device_type == "cuda":
        with patch("torch.cuda.is_available", return_value=True):
            return GpuValidator(expected_device=expected_device, debug_mode=True)
    elif device_type == "xpu":
        with patch.object(torch, "xpu", create=True) as mock_xpu:
            mock_xpu.is_available = MagicMock(return_value=True)
            return GpuValidator(expected_device=expected_device, debug_mode=True)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


class TestGpuValidatorRejectsCpuTensors:
    """Property 3: GPU validator rejects CPU tensors with descriptive errors.

    *For any* tensor and any expected device string (e.g., "xpu:0", "cuda:0"),
    if the tensor resides on CPU, the GPU validator SHALL raise an AssertionError
    whose message contains both the tensor's name and the expected device string.
    If the tensor resides on the expected device, the validator SHALL not raise.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """

    @settings(max_examples=100)
    @given(
        tensor_name=tensor_name_st,
        expected_device=expected_device_st,
        shape=tensor_shape_st,
    )
    def test_cpu_tensor_raises_assertion_error(
        self,
        tensor_name: str,
        expected_device: str,
        shape: tuple[int, ...],
    ) -> None:
        """For any CPU tensor and any expected GPU device, assert_on_device() raises AssertionError.

        Requirement 4.4: IF any tensor is found on CPU during inference, THEN raise
        an assertion error.
        """
        validator = _create_validator(expected_device)
        cpu_tensor = _create_cpu_mock_tensor(shape)

        with pytest.raises(AssertionError):
            validator.assert_on_device(cpu_tensor, tensor_name)

    @settings(max_examples=100)
    @given(
        tensor_name=tensor_name_st,
        expected_device=expected_device_st,
        shape=tensor_shape_st,
    )
    def test_error_message_contains_tensor_name(
        self,
        tensor_name: str,
        expected_device: str,
        shape: tuple[int, ...],
    ) -> None:
        """The error message SHALL contain the tensor's name.

        Requirement 4.4: raise an assertion error with a message identifying
        the offending tensor.
        """
        validator = _create_validator(expected_device)
        cpu_tensor = _create_cpu_mock_tensor(shape)

        with pytest.raises(AssertionError) as exc_info:
            validator.assert_on_device(cpu_tensor, tensor_name)

        assert tensor_name in str(exc_info.value), (
            f"Error message should contain tensor name '{tensor_name}', "
            f"but got: {exc_info.value}"
        )

    @settings(max_examples=100)
    @given(
        tensor_name=tensor_name_st,
        expected_device=expected_device_st,
        shape=tensor_shape_st,
    )
    def test_error_message_contains_expected_device(
        self,
        tensor_name: str,
        expected_device: str,
        shape: tuple[int, ...],
    ) -> None:
        """The error message SHALL contain the expected device string.

        Requirement 4.4: raise an assertion error with a message identifying
        the expected device.
        """
        validator = _create_validator(expected_device)
        cpu_tensor = _create_cpu_mock_tensor(shape)

        with pytest.raises(AssertionError) as exc_info:
            validator.assert_on_device(cpu_tensor, tensor_name)

        assert expected_device in str(exc_info.value), (
            f"Error message should contain expected device '{expected_device}', "
            f"but got: {exc_info.value}"
        )

    @settings(max_examples=100)
    @given(
        tensor_name=tensor_name_st,
        expected_device=expected_device_st,
    )
    def test_tensor_on_expected_device_does_not_raise(
        self,
        tensor_name: str,
        expected_device: str,
    ) -> None:
        """For any tensor on the expected device, assert_on_device() SHALL not raise.

        This validates the positive case: tensors correctly placed on the expected
        GPU device pass validation without error.
        """
        validator = _create_validator(expected_device)
        on_device_tensor = _create_on_device_mock_tensor(expected_device)

        # Should not raise any exception
        validator.assert_on_device(on_device_tensor, tensor_name)

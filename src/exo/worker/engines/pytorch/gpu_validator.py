"""Runtime GPU assertions ensuring inference never falls back to CPU.

Validates that all tensors reside on the expected GPU device during model
loading, forward passes, and output generation. Supports debug mode (every
forward pass) and production mode (first forward pass only).

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

import logging
from typing import final

import torch

logger = logging.getLogger(__name__)


@final
class GpuValidator:
    """Validates that all tensors reside on the expected GPU device.

    Raises AssertionError with a descriptive message identifying the offending
    tensor and expected device if any tensor is found on CPU during inference.

    Parameters:
        expected_device: The device string tensors must reside on (e.g. "xpu:0", "cuda:0").
        debug_mode: When True, validates every forward pass. When False, validates
            only the first forward pass (production mode).
    """

    def __init__(self, expected_device: str, debug_mode: bool = False) -> None:
        self._expected_device: str = expected_device
        self._debug_mode: bool = debug_mode
        self._first_forward_validated: bool = False

        # Extract device type from the expected_device string (e.g. "xpu" from "xpu:0")
        self._device_type: str = expected_device.split(":")[0]

        # Verify the GPU backend is available before any inference (Requirements 4.6, 4.7)
        self._verify_backend_available()

        logger.info(
            "GpuValidator initialized: expected_device=%s, debug_mode=%s",
            self._expected_device,
            self._debug_mode,
        )

    def _verify_backend_available(self) -> None:
        """Verify the GPU backend is available.

        Requirement 4.6: WHEN running on XPU, verify torch.xpu.is_available() returns True.
        Requirement 4.7: WHEN running on CUDA, verify torch.cuda.is_available() returns True.
        """
        if self._device_type == "xpu":
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise AssertionError(
                    f"XPU backend is not available (torch.xpu.is_available() returned False) "
                    f"but expected device is '{self._expected_device}'"
                )
        elif self._device_type == "cuda":
            if not torch.cuda.is_available():
                raise AssertionError(
                    f"CUDA backend is not available (torch.cuda.is_available() returned False) "
                    f"but expected device is '{self._expected_device}'"
                )

    def assert_on_device(self, tensor: torch.Tensor, name: str) -> None:
        """Assert tensor is on expected device. Raises AssertionError if on CPU.

        Requirement 4.4: IF any tensor is found on CPU during inference, THEN raise
        an assertion error with a message identifying the offending tensor and its
        expected device.

        Args:
            tensor: The tensor to validate.
            name: A descriptive name for the tensor (used in error messages).

        Raises:
            AssertionError: If the tensor is not on the expected device.
        """
        tensor_device = str(tensor.device)
        if tensor_device != self._expected_device:
            raise AssertionError(
                f"Tensor '{name}' is on device '{tensor_device}' but expected "
                f"'{self._expected_device}'. GPU inference requires all tensors "
                f"to reside on the target device — CPU fallback is not acceptable."
            )

    def validate_model_parameters(self, model: torch.nn.Module) -> None:
        """Assert all model parameters are on expected device.

        Requirement 4.1: WHEN loading a model, assert all model parameter tensors
        reside on the selected GPU device.

        Args:
            model: The PyTorch model whose parameters to validate.

        Raises:
            AssertionError: If any parameter is not on the expected device.
        """
        for param_name, param in model.named_parameters():
            self.assert_on_device(param.data, f"parameter:{param_name}")

        logger.debug(
            "All model parameters validated on device '%s'",
            self._expected_device,
        )

    def validate_forward_pass(
        self, inputs: dict[str, torch.Tensor], outputs: torch.Tensor
    ) -> None:
        """Validate input and output tensors are on device.

        Requirement 4.2: Assert input tensors are on the selected GPU device.
        Requirement 4.3: Assert output tensors reside on the selected GPU device.
        Requirement 4.5: Perform device validation on every forward pass in debug
        mode, first forward pass only in production mode.

        Args:
            inputs: Dictionary of named input tensors.
            outputs: The output tensor from the forward pass.

        Raises:
            AssertionError: If any input or output tensor is not on the expected device.
        """
        # In production mode, skip validation after the first forward pass
        if not self._debug_mode and self._first_forward_validated:
            return

        # Validate all input tensors (Requirement 4.2)
        for input_name, input_tensor in inputs.items():
            self.assert_on_device(input_tensor, f"input:{input_name}")

        # Validate output tensor (Requirement 4.3)
        self.assert_on_device(outputs, "output")

        # Mark first forward pass as validated for production mode
        if not self._first_forward_validated:
            self._first_forward_validated = True
            logger.info(
                "First forward pass validated — all tensors on '%s'",
                self._expected_device,
            )

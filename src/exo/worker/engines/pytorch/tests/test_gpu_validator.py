"""Unit tests for gpu_validator.py — runtime GPU assertions.

Tests the GpuValidator class for correct device assertion behavior,
model parameter validation, forward pass validation, and debug/production
mode switching.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7**
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import torch

from exo.worker.engines.pytorch.gpu_validator import GpuValidator


# --- Fixtures ---


@pytest.fixture
def validator_cuda() -> GpuValidator:
    """Create a GpuValidator expecting cuda:0 in debug mode."""
    with patch("torch.cuda.is_available", return_value=True):
        return GpuValidator(expected_device="cuda:0", debug_mode=True)


@pytest.fixture
def validator_production() -> GpuValidator:
    """Create a GpuValidator expecting cuda:0 in production mode."""
    with patch("torch.cuda.is_available", return_value=True):
        return GpuValidator(expected_device="cuda:0", debug_mode=False)


# --- Tests for assert_on_device ---


class TestAssertOnDevice:
    """Tests for GpuValidator.assert_on_device()."""

    def test_cpu_tensor_raises_assertion_error(self, validator_cuda: GpuValidator) -> None:
        """Requirement 4.4: CPU tensor raises AssertionError with tensor name and expected device."""
        tensor = torch.zeros(2, 3, device="cpu")
        with pytest.raises(AssertionError, match="my_tensor"):
            validator_cuda.assert_on_device(tensor, "my_tensor")

    def test_cpu_tensor_error_includes_expected_device(self, validator_cuda: GpuValidator) -> None:
        """Requirement 4.4: Error message identifies expected device."""
        tensor = torch.zeros(2, 3, device="cpu")
        with pytest.raises(AssertionError, match="cuda:0"):
            validator_cuda.assert_on_device(tensor, "test_tensor")

    def test_cpu_tensor_error_includes_actual_device(self, validator_cuda: GpuValidator) -> None:
        """Requirement 4.4: Error message identifies the offending tensor's actual device."""
        tensor = torch.zeros(2, 3, device="cpu")
        with pytest.raises(AssertionError, match="cpu"):
            validator_cuda.assert_on_device(tensor, "offending_tensor")

    def test_correct_device_does_not_raise(self) -> None:
        """Tensor on expected device passes without error."""
        # Use CPU as the "expected" device for this test since we can't guarantee GPU
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cpu", debug_mode=True)
        tensor = torch.zeros(2, 3, device="cpu")
        # Should not raise
        validator.assert_on_device(tensor, "good_tensor")


# --- Tests for validate_model_parameters ---


class TestValidateModelParameters:
    """Tests for GpuValidator.validate_model_parameters()."""

    def test_cpu_model_raises_assertion_error(self, validator_cuda: GpuValidator) -> None:
        """Requirement 4.1: Model with CPU parameters raises AssertionError."""
        model = torch.nn.Linear(10, 5)  # Default is CPU
        with pytest.raises(AssertionError, match="parameter:"):
            validator_cuda.validate_model_parameters(model)

    def test_error_identifies_parameter_name(self, validator_cuda: GpuValidator) -> None:
        """Requirement 4.1: Error identifies which parameter is on wrong device."""
        model = torch.nn.Linear(10, 5)
        with pytest.raises(AssertionError, match="weight|bias"):
            validator_cuda.validate_model_parameters(model)

    def test_empty_model_passes(self, validator_cuda: GpuValidator) -> None:
        """Model with no parameters passes validation."""
        model = torch.nn.Module()
        # Should not raise
        validator_cuda.validate_model_parameters(model)


# --- Tests for validate_forward_pass ---


class TestValidateForwardPass:
    """Tests for GpuValidator.validate_forward_pass()."""

    def test_cpu_input_raises_assertion_error(self, validator_cuda: GpuValidator) -> None:
        """Requirement 4.2: CPU input tensor raises AssertionError."""
        inputs = {"input_ids": torch.zeros(1, 10, device="cpu")}
        # We need a valid output for the test — use CPU since it will fail on input first
        output = torch.zeros(1, 10, device="cpu")
        with pytest.raises(AssertionError, match="input:input_ids"):
            validator_cuda.validate_forward_pass(inputs, output)

    def test_cpu_output_raises_assertion_error(self) -> None:
        """Requirement 4.3: CPU output tensor raises AssertionError."""
        # Use a validator that expects "cpu" for inputs but we'll test output separately
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cuda:0", debug_mode=True)

        # Mock inputs that pass (by making them appear on cuda:0)
        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device = torch.device("cuda:0")
        inputs = {"input_ids": input_tensor}

        # Output on CPU should fail
        output = torch.zeros(1, 10, device="cpu")
        with pytest.raises(AssertionError, match="output"):
            validator.validate_forward_pass(inputs, output)

    def test_all_tensors_on_correct_device_passes(self) -> None:
        """All tensors on expected device passes without error."""
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cpu", debug_mode=True)

        inputs = {"input_ids": torch.zeros(1, 10, device="cpu")}
        output = torch.zeros(1, 10, device="cpu")
        # Should not raise
        validator.validate_forward_pass(inputs, output)


# --- Tests for debug vs production mode ---


class TestModeSwitch:
    """Tests for debug mode vs production mode behavior."""

    def test_debug_mode_validates_every_pass(self) -> None:
        """Requirement 4.5: Debug mode validates every forward pass."""
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cpu", debug_mode=True)

        inputs = {"x": torch.zeros(1, device="cpu")}
        output = torch.zeros(1, device="cpu")

        # First pass
        validator.validate_forward_pass(inputs, output)

        # Second pass — should still validate (and catch errors)
        bad_inputs = {"x": MagicMock(spec=torch.Tensor)}
        bad_inputs["x"].device = torch.device("cuda:0")

        with pytest.raises(AssertionError):
            validator.validate_forward_pass(bad_inputs, output)

    def test_production_mode_validates_first_pass_only(self) -> None:
        """Requirement 4.5: Production mode validates first forward pass only."""
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cpu", debug_mode=False)

        inputs = {"x": torch.zeros(1, device="cpu")}
        output = torch.zeros(1, device="cpu")

        # First pass — validates
        validator.validate_forward_pass(inputs, output)

        # Second pass — skips validation, so bad tensors don't raise
        bad_inputs = {"x": MagicMock(spec=torch.Tensor)}
        bad_inputs["x"].device = torch.device("cuda:0")

        # Should NOT raise because production mode skips after first pass
        validator.validate_forward_pass(bad_inputs, output)

    def test_production_mode_first_pass_still_validates(self) -> None:
        """Requirement 4.5: Production mode still validates the first forward pass."""
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cuda:0", debug_mode=False)

        inputs = {"x": torch.zeros(1, device="cpu")}
        output = torch.zeros(1, device="cpu")

        # First pass should still validate and catch CPU tensors
        with pytest.raises(AssertionError, match="input:x"):
            validator.validate_forward_pass(inputs, output)


# --- Tests for backend availability verification ---


class TestBackendAvailability:
    """Tests for GPU backend availability checks."""

    def test_xpu_unavailable_raises_assertion(self) -> None:
        """Requirement 4.6: XPU unavailable raises AssertionError."""
        with patch.object(torch, "xpu", create=True) as mock_xpu:
            mock_xpu.is_available = MagicMock(return_value=False)
            with pytest.raises(AssertionError, match="XPU backend is not available"):
                GpuValidator(expected_device="xpu:0", debug_mode=True)

    def test_cuda_unavailable_raises_assertion(self) -> None:
        """Requirement 4.7: CUDA unavailable raises AssertionError."""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(AssertionError, match="CUDA backend is not available"):
                GpuValidator(expected_device="cuda:0", debug_mode=True)

    def test_cuda_available_succeeds(self) -> None:
        """Requirement 4.7: CUDA available allows construction."""
        with patch("torch.cuda.is_available", return_value=True):
            validator = GpuValidator(expected_device="cuda:0", debug_mode=True)
        assert validator._expected_device == "cuda:0"

    def test_xpu_available_succeeds(self) -> None:
        """Requirement 4.6: XPU available allows construction."""
        with patch.object(torch, "xpu", create=True) as mock_xpu:
            mock_xpu.is_available = MagicMock(return_value=True)
            validator = GpuValidator(expected_device="xpu:0", debug_mode=True)
        assert validator._expected_device == "xpu:0"

"""Tests for UnifiedPyTorchEngine.

Validates that the engine correctly implements the InferenceBackend protocol,
routes tensor operations to the correct device, and integrates with the
GpuValidator, sampling, and model loading modules.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from exo.worker.engines.base import InferenceBackend
from exo.worker.engines.pytorch.engine import UnifiedPyTorchEngine


class TestEngineInitialization:
    """Test engine construction and device configuration."""

    def test_accepts_cuda_device_type(self) -> None:
        """Requirement 2.3: Accept device_type 'cuda'."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)
            assert engine.device_type == "cuda"
            assert engine.device == "cuda:0"

    def test_accepts_xpu_device_type(self) -> None:
        """Requirement 2.3: Accept device_type 'xpu'."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.xpu = MagicMock()
            mock_torch.xpu.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = False
            engine = UnifiedPyTorchEngine(device_type="xpu", device_index=0)
            assert engine.device_type == "xpu"
            assert engine.device == "xpu:0"

    def test_accepts_device_index(self) -> None:
        """Test non-zero device index."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=1)
            assert engine.device == "cuda:1"

    def test_accepts_pipeline_config(self) -> None:
        """Test pipeline config parameter."""
        from exo.worker.engines.pytorch.pipeline.coordinator import PipelineConfig

        config = PipelineConfig(
            world_size=4,
            rank=0,
            total_layers=32,
            master_addr="10.1.1.12",
            master_port=29500,
            transport="ethernet",
        )
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(
                device_type="cuda", device_index=0, pipeline_config=config
            )
            assert engine._pipeline_config is config

    def test_implements_inference_backend(self) -> None:
        """Requirement 2.1: Implements InferenceBackend ABC."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)
            assert isinstance(engine, InferenceBackend)


class TestEncodeAndDecode:
    """Test tokenization methods."""

    @pytest.fixture
    def engine_with_tokenizer(self) -> UnifiedPyTorchEngine:
        """Create an engine with a mock tokenizer."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = np.array([[1, 2, 3, 4]])
        mock_tokenizer.decode.return_value = "hello world"
        engine._tokenizer = mock_tokenizer
        return engine

    @pytest.mark.asyncio
    async def test_encode_returns_numpy_array(
        self, engine_with_tokenizer: UnifiedPyTorchEngine
    ) -> None:
        """Test encode returns token IDs as numpy array."""
        shard = MagicMock()
        result = await engine_with_tokenizer.encode(shard, "hello world")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    @pytest.mark.asyncio
    async def test_encode_raises_without_tokenizer(self) -> None:
        """Test encode raises RuntimeError if no tokenizer loaded."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        shard = MagicMock()
        with pytest.raises(RuntimeError, match="No tokenizer loaded"):
            await engine.encode(shard, "hello")

    @pytest.mark.asyncio
    async def test_decode_returns_string(
        self, engine_with_tokenizer: UnifiedPyTorchEngine
    ) -> None:
        """Test decode returns text string."""
        shard = MagicMock()
        tokens = np.array([1, 2, 3])
        result = await engine_with_tokenizer.decode(shard, tokens)
        assert isinstance(result, str)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_decode_raises_without_tokenizer(self) -> None:
        """Test decode raises RuntimeError if no tokenizer loaded."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        shard = MagicMock()
        with pytest.raises(RuntimeError, match="No tokenizer loaded"):
            await engine.decode(shard, np.array([1, 2, 3]))


class TestInferTensor:
    """Test tensor inference."""

    @pytest.mark.asyncio
    async def test_raises_without_model(self) -> None:
        """Test infer_tensor raises RuntimeError if no model loaded."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        shard = MagicMock()
        with pytest.raises(RuntimeError, match="No model loaded"):
            await engine.infer_tensor("req-1", shard, np.array([1, 2, 3]))

    @pytest.mark.asyncio
    async def test_infer_tensor_with_model(self) -> None:
        """Test infer_tensor runs forward pass and returns numpy output."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        # Override device to CPU for testing (no GPU in CI)
        engine._device_str = "cpu"

        # Create a simple model that returns input unchanged
        model = MagicMock()
        output_tensor = torch.randn(1, 10)
        model.return_value = output_tensor
        engine._model = model

        # Mock the validator to not check device (we're on CPU in tests)
        engine._validator = MagicMock()

        shard = MagicMock()
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result, state = await engine.infer_tensor("req-1", shard, input_data)

        assert isinstance(result, np.ndarray)
        assert state is not None
        assert state["request_id"] == "req-1"
        assert state["seq_position"] == 3  # input had 3 elements in last dim

    @pytest.mark.asyncio
    async def test_infer_tensor_preserves_inference_state(self) -> None:
        """Test that inference state seq_position accumulates."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        # Override device to CPU for testing
        engine._device_str = "cpu"

        model = MagicMock()
        model.return_value = torch.randn(1, 10)
        engine._model = model
        engine._validator = MagicMock()

        shard = MagicMock()
        input_data = np.array([[1.0, 2.0]], dtype=np.float32)

        # First call
        _, state1 = await engine.infer_tensor("req-1", shard, input_data)
        assert state1["seq_position"] == 2

        # Second call with previous state
        _, state2 = await engine.infer_tensor("req-1", shard, input_data, state1)
        assert state2["seq_position"] == 4


class TestSample:
    """Test token sampling."""

    @pytest.mark.asyncio
    async def test_sample_returns_numpy_array(self) -> None:
        """Requirement 2.6: Token sampling works on GPU tensors."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        # Create logits on CPU (since we don't have GPU in tests)
        logits = np.array([0.1, 0.2, 0.5, 0.1, 0.1], dtype=np.float32)

        # Mock torch.from_numpy().to() to return a CPU tensor
        with patch.object(engine, "_device_str", "cpu"):
            result = await engine.sample(logits)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] >= 1

    @pytest.mark.asyncio
    async def test_sample_greedy_with_zero_temperature(self) -> None:
        """Test greedy sampling selects argmax."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        engine._temperature = 0.0

        # Logits where index 2 is clearly the highest
        logits = np.array([-10.0, -10.0, 100.0, -10.0, -10.0], dtype=np.float32)

        with patch.object(engine, "_device_str", "cpu"):
            result = await engine.sample(logits)

        assert result[0] == 2


class TestLoadCheckpoint:
    """Test checkpoint loading."""

    @pytest.mark.asyncio
    async def test_raises_on_missing_path(self) -> None:
        """Test load_checkpoint raises on non-existent path."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        shard = MagicMock()
        with pytest.raises(RuntimeError, match="does not exist"):
            await engine.load_checkpoint(shard, "/nonexistent/path")

    @pytest.mark.asyncio
    async def test_loads_tokenizer_and_weights(self, tmp_path: Path) -> None:
        """Test load_checkpoint loads tokenizer and safetensors weights."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        # Create a fake safetensors file
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        fake_safetensors = model_dir / "model.safetensors"
        fake_safetensors.write_bytes(b"fake")

        # Mock the tokenizer loading (imported inside load_checkpoint)
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            # Mock the safetensors loader
            fake_state_dict = {"layer.weight": torch.randn(10, 10)}
            with patch(
                "exo.worker.engines.pytorch.engine.load_safetensors_to_device"
            ) as mock_loader:
                mock_loader.return_value = iter([fake_state_dict])

                shard = MagicMock()
                await engine.load_checkpoint(shard, str(model_dir))

        assert engine._tokenizer is mock_tokenizer
        assert engine._state_dict is not None
        assert "layer.weight" in engine._state_dict


class TestDeviceAgnosticOperations:
    """Test that all tensor operations use device-agnostic APIs.

    Requirement 2.8: Use PyTorch's device-agnostic APIs for all tensor allocation.
    """

    def test_forward_layers_uses_device_to(self) -> None:
        """Test forward_layers moves tensors to target device."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        # Override device to CPU for testing
        engine._device_str = "cpu"
        # Use a real-like mock that allows assert_on_device as a regular method
        validator_mock = MagicMock(spec=None)
        validator_mock.assert_on_device = MagicMock()
        engine._validator = validator_mock

        hidden = torch.randn(1, 256)
        result = engine.forward_layers(hidden, 0, 4)
        assert result.device == torch.device("cpu")

    def test_embed_tokens_allocates_on_device(self) -> None:
        """Test embed_tokens creates tensors on target device."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        engine._device_str = "cpu"
        engine._validator = MagicMock()

        tokens = torch.tensor([1, 2, 3], dtype=torch.long)
        result = engine.embed_tokens(tokens)
        assert result.device == torch.device("cpu")

    def test_lm_head_allocates_on_device(self) -> None:
        """Test lm_head creates tensors on target device."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        engine._device_str = "cpu"
        engine._validator = MagicMock()

        hidden = torch.randn(1, 256)
        result = engine.lm_head(hidden)
        assert result.device == torch.device("cpu")
        assert result.shape[-1] == 32000  # vocab size


class TestSampleTokenFromTensor:
    """Test the tensor-based sampling helper for pipeline integration."""

    def test_returns_numpy_array(self) -> None:
        """Test sample_token_from_tensor returns numpy array."""
        with patch("exo.worker.engines.pytorch.gpu_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            engine = UnifiedPyTorchEngine(device_type="cuda", device_index=0)

        engine._temperature = 0.0
        logits = torch.tensor([[-10.0, -10.0, 100.0, -10.0]])
        result = engine.sample_token_from_tensor(logits)
        assert isinstance(result, np.ndarray)
        assert result[0] == 2

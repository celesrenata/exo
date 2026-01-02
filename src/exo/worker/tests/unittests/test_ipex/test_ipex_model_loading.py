"""Test IPEX model loading and initialization."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch

from exo.worker.engines.ipex.utils_ipex import (
    initialize_ipex, 
    load_ipex_model, 
    get_model_info,
    detect_model_quantization,
    get_optimal_dtype_for_intel_gpu
)
from exo.worker.engines.ipex import (
    IPEXModel, 
    IPEXTokenizerWrapper, 
    IPEXModelLoadError, 
    IPEXMemoryError,
    IPEXInitializationError
)
from exo.shared.types.worker.instances import BoundInstance
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance
from exo.worker.tests.constants import MODEL_A_ID, RUNNER_1_ID, NODE_A


class TestIPEXModelLoading:
    """Test IPEX model loading functionality."""

    @pytest.fixture
    def mock_bound_instance(self):
        """Create a mock bound instance for testing."""
        return get_bound_mlx_ring_instance(
            instance_id="test_instance",
            model_id=MODEL_A_ID,
            runner_id=RUNNER_1_ID,
            node_id=NODE_A
        )

    @pytest.fixture
    def mock_model_path(self):
        """Create a mock model path."""
        return Path("/mock/model/path")

    def test_initialize_ipex_success(self, mock_bound_instance):
        """Test successful IPEX initialization."""
        with patch('exo.worker.engines.ipex.utils_ipex.setup_ipex_logging'), \
             patch('exo.worker.engines.ipex.utils_ipex.validate_intel_gpu_environment'), \
             patch('exo.worker.engines.ipex.utils_ipex.enable_intel_gpu_kernel_optimizations'), \
             patch('exo.worker.engines.ipex.utils_ipex.create_intel_gpu_optimized_sampler') as mock_sampler, \
             patch('exo.worker.engines.ipex.utils_ipex.initialize_single_device_ipex') as mock_init:
            
            # Mock return values
            mock_model = Mock(spec=IPEXModel)
            mock_tokenizer = Mock(spec=IPEXTokenizerWrapper)
            mock_sampler_func = Mock()
            
            mock_sampler.return_value = mock_sampler_func
            mock_init.return_value = (mock_model, mock_tokenizer, mock_sampler_func)
            
            # Test initialization
            model, tokenizer, sampler = initialize_ipex(mock_bound_instance)
            
            assert model == mock_model
            assert tokenizer == mock_tokenizer
            assert sampler == mock_sampler_func
            
            # Verify calls
            mock_init.assert_called_once_with(mock_bound_instance, mock_sampler_func)

    def test_initialize_ipex_distributed(self, mock_bound_instance):
        """Test IPEX initialization for distributed inference."""
        # Mock multiple nodes for distributed setup
        mock_bound_instance.instance.shard_assignments.node_to_runner = {
            "node1": "runner1",
            "node2": "runner2"
        }
        
        with patch('exo.worker.engines.ipex.utils_ipex.setup_ipex_logging'), \
             patch('exo.worker.engines.ipex.utils_ipex.validate_intel_gpu_environment'), \
             patch('exo.worker.engines.ipex.utils_ipex.enable_intel_gpu_kernel_optimizations'), \
             patch('exo.worker.engines.ipex.utils_ipex.create_intel_gpu_optimized_sampler') as mock_sampler, \
             patch('exo.worker.engines.ipex.utils_ipex.initialize_distributed_ipex') as mock_init_dist:
            
            # Mock return values
            mock_model = Mock(spec=IPEXModel)
            mock_tokenizer = Mock(spec=IPEXTokenizerWrapper)
            mock_sampler_func = Mock()
            
            mock_sampler.return_value = mock_sampler_func
            mock_init_dist.return_value = (mock_model, mock_tokenizer, mock_sampler_func)
            
            # Test initialization
            model, tokenizer, sampler = initialize_ipex(mock_bound_instance)
            
            assert model == mock_model
            assert tokenizer == mock_tokenizer
            assert sampler == mock_sampler_func
            
            # Verify distributed initialization was called
            mock_init_dist.assert_called_once_with(mock_bound_instance, mock_sampler_func)

    def test_initialize_ipex_validation_failure(self, mock_bound_instance):
        """Test IPEX initialization when environment validation fails."""
        with patch('exo.worker.engines.ipex.utils_ipex.setup_ipex_logging'), \
             patch('exo.worker.engines.ipex.utils_ipex.validate_intel_gpu_environment', 
                   side_effect=IPEXInitializationError("Validation failed")):
            
            with pytest.raises(IPEXInitializationError) as exc_info:
                initialize_ipex(mock_bound_instance)
            
            assert "Validation failed" in str(exc_info.value)

    def test_load_ipex_model_success(self, mock_model_path):
        """Test successful IPEX model loading."""
        with patch('exo.worker.engines.ipex.utils_ipex.validate_intel_gpu_environment'), \
             patch('intel_extension_for_pytorch') as mock_ipex, \
             patch('torch.xpu') as mock_xpu, \
             patch('exo.worker.engines.ipex.utils_ipex.detect_model_quantization') as mock_detect_quant, \
             patch('exo.worker.engines.ipex.utils_ipex.get_optimal_dtype_for_intel_gpu') as mock_dtype, \
             patch('exo.worker.engines.ipex.utils_ipex.optimize_model_for_intel_gpu') as mock_optimize, \
             patch('transformers.AutoModelForCausalLM') as mock_auto_model, \
             patch('transformers.AutoConfig') as mock_auto_config:
            
            # Mock Intel GPU setup
            mock_xpu.device_count.return_value = 1
            mock_xpu.is_available.return_value = True
            
            # Mock model loading
            mock_config = Mock()
            mock_auto_config.from_pretrained.return_value = mock_config
            
            mock_model = Mock()
            mock_auto_model.from_pretrained.return_value = mock_model
            
            # Mock optimization functions
            mock_detect_quant.return_value = {"is_quantized": False}
            mock_dtype.return_value = torch.float16
            mock_optimize.return_value = mock_model
            
            # Test model loading
            result = load_ipex_model(mock_model_path, mock_config)
            
            assert result == mock_model
            
            # Verify calls
            mock_auto_model.from_pretrained.assert_called_once()
            mock_optimize.assert_called_once()

    def test_load_ipex_model_memory_error(self, mock_model_path):
        """Test IPEX model loading with memory error."""
        with patch('exo.worker.engines.ipex.utils_ipex.validate_intel_gpu_environment'), \
             patch('intel_extension_for_pytorch') as mock_ipex, \
             patch('torch.xpu') as mock_xpu, \
             patch('exo.worker.engines.ipex.utils_ipex.detect_model_quantization'), \
             patch('exo.worker.engines.ipex.utils_ipex.get_optimal_dtype_for_intel_gpu'), \
             patch('transformers.AutoModelForCausalLM') as mock_auto_model, \
             patch('transformers.AutoConfig') as mock_auto_config:
            
            # Mock Intel GPU setup
            mock_xpu.device_count.return_value = 1
            mock_xpu.is_available.return_value = True
            
            # Mock config
            mock_config = Mock()
            mock_auto_config.from_pretrained.return_value = mock_config
            
            # Mock memory error during model loading
            mock_auto_model.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
            
            with pytest.raises(IPEXModelLoadError) as exc_info:
                load_ipex_model(mock_model_path, mock_config)
            
            assert "Model loading failed" in str(exc_info.value)

    def test_load_ipex_model_no_ipex(self, mock_model_path):
        """Test IPEX model loading when IPEX is not available."""
        with patch('intel_extension_for_pytorch', side_effect=ImportError("IPEX not found")):
            
            mock_config = Mock()
            
            with pytest.raises(IPEXDriverError) as exc_info:
                load_ipex_model(mock_model_path, mock_config)
            
            assert "Intel Extension for PyTorch not available" in str(exc_info.value)

    def test_get_model_info_success(self, mock_model_path):
        """Test successful model info extraction."""
        with patch('transformers.AutoConfig') as mock_auto_config:
            
            # Mock config with model information
            mock_config = Mock()
            mock_config.model_type = "llama"
            mock_config.vocab_size = 32000
            mock_config.hidden_size = 4096
            mock_config.num_hidden_layers = 32
            mock_config.num_attention_heads = 32
            
            mock_auto_config.from_pretrained.return_value = mock_config
            
            info = get_model_info(mock_model_path)
            
            assert info["model_type"] == "llama"
            assert info["vocab_size"] == 32000
            assert info["hidden_size"] == 4096
            assert info["num_layers"] == 32
            assert info["num_attention_heads"] == 32

    def test_get_model_info_failure(self, mock_model_path):
        """Test model info extraction failure."""
        with patch('transformers.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.side_effect = Exception("Config loading failed")
            
            info = get_model_info(mock_model_path)
            
            assert info == {}

    def test_detect_model_quantization_not_quantized(self, mock_model_path):
        """Test quantization detection for non-quantized model."""
        with patch('transformers.AutoConfig') as mock_auto_config:
            
            # Mock config without quantization
            mock_config = Mock()
            del mock_config.quantization_config  # No quantization config
            
            mock_auto_config.from_pretrained.return_value = mock_config
            
            quant_info = detect_model_quantization(mock_model_path)
            
            assert quant_info["is_quantized"] is False
            assert quant_info["quantization_method"] is None
            assert quant_info["bits"] is None

    def test_detect_model_quantization_gptq(self, mock_model_path):
        """Test quantization detection for GPTQ quantized model."""
        with patch('transformers.AutoConfig') as mock_auto_config:
            
            # Mock config with GPTQ quantization
            mock_config = Mock()
            mock_config.gptq = True
            
            mock_auto_config.from_pretrained.return_value = mock_config
            
            quant_info = detect_model_quantization(mock_model_path)
            
            assert quant_info["is_quantized"] is True
            assert quant_info["quantization_method"] == "gptq"

    def test_detect_model_quantization_with_config(self, mock_model_path):
        """Test quantization detection with quantization config."""
        with patch('transformers.AutoConfig') as mock_auto_config:
            
            # Mock config with quantization config
            mock_config = Mock()
            mock_quant_config = Mock()
            mock_quant_config.bits = 4
            mock_quant_config.quant_method = "awq"
            mock_config.quantization_config = mock_quant_config
            
            mock_auto_config.from_pretrained.return_value = mock_config
            
            quant_info = detect_model_quantization(mock_model_path)
            
            assert quant_info["is_quantized"] is True
            assert quant_info["quantization_method"] == "awq"
            assert quant_info["bits"] == 4

    def test_get_optimal_dtype_for_intel_gpu_bfloat16(self):
        """Test optimal dtype selection when bfloat16 is supported."""
        with patch('torch.xpu') as mock_xpu:
            
            mock_xpu.is_available.return_value = True
            
            # Mock device properties with bfloat16 support
            mock_props = Mock()
            mock_props.supports_bfloat16 = True
            mock_xpu.get_device_properties.return_value = mock_props
            
            dtype = get_optimal_dtype_for_intel_gpu()
            
            assert dtype == torch.bfloat16

    def test_get_optimal_dtype_for_intel_gpu_float16(self):
        """Test optimal dtype selection when only float16 is supported."""
        with patch('torch.xpu') as mock_xpu:
            
            mock_xpu.is_available.return_value = True
            
            # Mock device properties without bfloat16 support
            mock_props = Mock()
            mock_props.supports_bfloat16 = False
            mock_xpu.get_device_properties.return_value = mock_props
            
            dtype = get_optimal_dtype_for_intel_gpu()
            
            assert dtype == torch.float16

    def test_get_optimal_dtype_for_intel_gpu_no_xpu(self):
        """Test optimal dtype selection when XPU is not available."""
        with patch('torch.xpu') as mock_xpu:
            
            mock_xpu.is_available.return_value = False
            
            dtype = get_optimal_dtype_for_intel_gpu()
            
            assert dtype == torch.float16  # Default fallback

    def test_get_optimal_dtype_for_intel_gpu_error(self):
        """Test optimal dtype selection with error."""
        with patch('torch.xpu', side_effect=Exception("XPU error")):
            
            dtype = get_optimal_dtype_for_intel_gpu()
            
            assert dtype == torch.float16  # Default fallback


class TestIPEXTokenizerWrapper:
    """Test IPEX tokenizer wrapper functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock HuggingFace tokenizer."""
        tokenizer = Mock()
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token_id = 2
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "decoded text"
        tokenizer.apply_chat_template.return_value = "formatted chat"
        return tokenizer

    def test_tokenizer_wrapper_initialization(self, mock_tokenizer):
        """Test tokenizer wrapper initialization."""
        wrapper = IPEXTokenizerWrapper(mock_tokenizer)
        
        assert wrapper.tokenizer == mock_tokenizer
        assert wrapper.bos_token == "<s>"
        assert wrapper.eos_token_ids == [2]

    def test_tokenizer_wrapper_initialization_no_eos(self, mock_tokenizer):
        """Test tokenizer wrapper initialization when eos_token_id is None."""
        mock_tokenizer.eos_token_id = None
        
        wrapper = IPEXTokenizerWrapper(mock_tokenizer)
        
        assert wrapper.eos_token_ids == []

    def test_tokenizer_wrapper_encode(self, mock_tokenizer):
        """Test tokenizer wrapper encode method."""
        wrapper = IPEXTokenizerWrapper(mock_tokenizer)
        
        result = wrapper.encode("test text", add_special_tokens=True)
        
        assert result == [1, 2, 3, 4, 5]
        mock_tokenizer.encode.assert_called_once_with("test text", add_special_tokens=True)

    def test_tokenizer_wrapper_decode(self, mock_tokenizer):
        """Test tokenizer wrapper decode method."""
        wrapper = IPEXTokenizerWrapper(mock_tokenizer)
        
        result = wrapper.decode([1, 2, 3], skip_special_tokens=True)
        
        assert result == "decoded text"
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)

    def test_tokenizer_wrapper_apply_chat_template(self, mock_tokenizer):
        """Test tokenizer wrapper apply_chat_template method."""
        wrapper = IPEXTokenizerWrapper(mock_tokenizer)
        
        messages = [{"role": "user", "content": "Hello"}]
        result = wrapper.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        assert result == "formatted chat"
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )
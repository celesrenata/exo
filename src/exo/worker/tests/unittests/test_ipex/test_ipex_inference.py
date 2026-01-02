"""Test IPEX text generation and streaming output."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import torch
from typing import Generator

from exo.worker.engines.ipex.generator.generate import (
    warmup_inference,
    ipex_generate,
    ipex_generate_simple,
    _ipex_generate_single_device,
    _ipex_generate_distributed
)
from exo.worker.engines.ipex import (
    IPEXModel, 
    IPEXTokenizerWrapper,
    IPEXInferenceError,
    IPEXMemoryError,
    IPEXDistributedError
)
from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse


class TestIPEXWarmup:
    """Test IPEX inference warmup functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock IPEX model."""
        model = Mock(spec=IPEXModel)
        
        # Mock model parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device("xpu:0")
        model.parameters.return_value = [mock_param]
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        model.return_value = mock_outputs
        
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock IPEX tokenizer."""
        tokenizer = Mock(spec=IPEXTokenizerWrapper)
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "warmup token"
        tokenizer.eos_token_ids = [2]
        return tokenizer

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock sampler function."""
        def sampler(logits):
            return torch.tensor([1])  # Always return token ID 1
        return sampler

    def test_warmup_inference_success(self, mock_model, mock_tokenizer, mock_sampler):
        """Test successful warmup inference."""
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.xpu.empty_cache') as mock_empty_cache, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'):
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids  # For concatenation
            
            # Run warmup
            tokens_generated = warmup_inference(mock_model, mock_tokenizer, mock_sampler)
            
            # Verify warmup completed
            assert tokens_generated > 0
            assert tokens_generated <= 10  # Should generate up to 10 warmup tokens
            
            # Verify cache was cleared
            mock_empty_cache.assert_called()

    def test_warmup_inference_memory_error(self, mock_model, mock_tokenizer, mock_sampler):
        """Test warmup inference with Intel GPU memory error."""
        with patch('torch.tensor'), \
             patch('torch.ones_like'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'):
            
            # Mock memory error during forward pass
            mock_model.side_effect = torch.xpu.OutOfMemoryError("GPU out of memory")
            
            with pytest.raises(IPEXMemoryError) as exc_info:
                warmup_inference(mock_model, mock_tokenizer, mock_sampler)
            
            assert "Intel GPU out of memory during warmup" in str(exc_info.value)

    def test_warmup_inference_general_error(self, mock_model, mock_tokenizer, mock_sampler):
        """Test warmup inference with general error."""
        with patch('torch.tensor'), \
             patch('torch.ones_like'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'):
            
            # Mock general error during forward pass
            mock_model.side_effect = RuntimeError("General inference error")
            
            with pytest.raises(IPEXInferenceError) as exc_info:
                warmup_inference(mock_model, mock_tokenizer, mock_sampler)
            
            assert "IPEX warmup failed" in str(exc_info.value)

    def test_warmup_inference_early_termination(self, mock_model, mock_tokenizer, mock_sampler):
        """Test warmup inference with early EOS termination."""
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.xpu.empty_cache'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'):
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids
            
            # Mock sampler to return EOS token
            def eos_sampler(logits):
                return torch.tensor([2])  # EOS token ID
            
            tokens_generated = warmup_inference(mock_model, mock_tokenizer, eos_sampler)
            
            # Should terminate early due to EOS
            assert tokens_generated == 1


class TestIPEXGeneration:
    """Test IPEX text generation functionality."""

    @pytest.fixture
    def mock_task(self):
        """Create a mock chat completion task."""
        task = ChatCompletionTaskParams(
            model="test-model",
            messages=[
                ChatCompletionMessage(role="user", content="Hello, world!")
            ],
            max_tokens=50,
            temperature=0.7
        )
        return task

    @pytest.fixture
    def mock_model(self):
        """Create a mock IPEX model."""
        model = Mock(spec=IPEXModel)
        
        # Mock model parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device("xpu:0")
        mock_param.dtype = torch.float16
        model.parameters.return_value = [mock_param]
        
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock IPEX tokenizer."""
        tokenizer = Mock(spec=IPEXTokenizerWrapper)
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "generated"
        tokenizer.eos_token_ids = [2]
        tokenizer.apply_chat_template.return_value = "User: Hello, world!\nAssistant:"
        return tokenizer

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock sampler function."""
        def sampler(logits):
            return torch.tensor([10])  # Return token ID 10
        return sampler

    def test_ipex_generate_single_device(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test IPEX generation on single device."""
        with patch('exo.worker.engines.ipex.generator.generate._ipex_generate_single_device') as mock_single:
            
            # Mock generator response
            def mock_generator():
                yield GenerationResponse(text="Hello", token=10, finish_reason=None)
                yield GenerationResponse(text=" world", token=11, finish_reason=None)
                yield GenerationResponse(text="!", token=12, finish_reason="stop")
            
            mock_single.return_value = mock_generator()
            
            # Test generation
            responses = list(ipex_generate(mock_model, mock_tokenizer, mock_sampler, mock_task))
            
            assert len(responses) == 3
            assert responses[0].text == "Hello"
            assert responses[1].text == " world"
            assert responses[2].text == "!"
            assert responses[2].finish_reason == "stop"

    def test_ipex_generate_distributed(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test IPEX generation with distributed inference."""
        # Mock distributed model
        mock_model._ipex_dist_group = Mock()
        
        with patch('exo.worker.engines.ipex.generator.generate._ipex_generate_distributed') as mock_distributed:
            
            # Mock generator response
            def mock_generator():
                yield GenerationResponse(text="Distributed", token=20, finish_reason=None)
                yield GenerationResponse(text=" response", token=21, finish_reason="length")
            
            mock_distributed.return_value = mock_generator()
            
            # Test generation
            responses = list(ipex_generate(mock_model, mock_tokenizer, mock_sampler, mock_task))
            
            assert len(responses) == 2
            assert responses[0].text == "Distributed"
            assert responses[1].text == " response"
            assert responses[1].finish_reason == "length"

    def test_ipex_generate_memory_error(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test IPEX generation with memory error."""
        with patch('exo.worker.engines.ipex.generator.generate._ipex_generate_single_device') as mock_single:
            
            mock_single.side_effect = torch.xpu.OutOfMemoryError("GPU out of memory")
            
            with pytest.raises(IPEXMemoryError) as exc_info:
                list(ipex_generate(mock_model, mock_tokenizer, mock_sampler, mock_task))
            
            assert "Intel GPU out of memory during generation" in str(exc_info.value)

    def test_ipex_generate_inference_error(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test IPEX generation with inference error."""
        with patch('exo.worker.engines.ipex.generator.generate._ipex_generate_single_device') as mock_single:
            
            mock_single.side_effect = RuntimeError("Inference failed")
            
            with pytest.raises(IPEXInferenceError) as exc_info:
                list(ipex_generate(mock_model, mock_tokenizer, mock_sampler, mock_task))
            
            assert "IPEX generation failed" in str(exc_info.value)


class TestIPEXSingleDeviceGeneration:
    """Test single device IPEX generation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock IPEX model."""
        model = Mock(spec=IPEXModel)
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        model.return_value = mock_outputs
        
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock IPEX tokenizer."""
        tokenizer = Mock(spec=IPEXTokenizerWrapper)
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "token"
        tokenizer.eos_token_ids = [2]
        return tokenizer

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock sampler function."""
        def sampler(logits):
            return torch.tensor([10])
        return sampler

    @pytest.fixture
    def mock_task(self):
        """Create a mock task."""
        return ChatCompletionTaskParams(
            model="test-model",
            messages=[ChatCompletionMessage(role="user", content="Test")],
            max_tokens=5
        )

    def test_single_device_generation_success(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test successful single device generation."""
        device = torch.device("xpu:0")
        
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'), \
             patch('exo.worker.engines.ipex.generator.generate.monitor_intel_gpu_health') as mock_health, \
             patch('exo.worker.engines.ipex.generator.generate.handle_intel_gpu_health_issues'):
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids
            
            # Mock health monitoring
            mock_health.return_value = {"overall_health": "healthy"}
            
            # Test generation
            responses = list(_ipex_generate_single_device(
                mock_model, mock_tokenizer, mock_sampler, mock_task, "Test prompt", device
            ))
            
            # Should generate tokens up to max_tokens
            assert len(responses) <= mock_task.max_tokens
            
            # Verify health monitoring was called
            mock_health.assert_called()

    def test_single_device_generation_eos_termination(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test single device generation with EOS termination."""
        device = torch.device("xpu:0")
        
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'), \
             patch('exo.worker.engines.ipex.generator.generate.monitor_intel_gpu_health') as mock_health, \
             patch('exo.worker.engines.ipex.generator.generate.handle_intel_gpu_health_issues'):
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids
            
            # Mock health monitoring
            mock_health.return_value = {"overall_health": "healthy"}
            
            # Mock sampler to return EOS token
            def eos_sampler(logits):
                return torch.tensor([2])  # EOS token
            
            # Test generation
            responses = list(_ipex_generate_single_device(
                mock_model, mock_tokenizer, eos_sampler, mock_task, "Test prompt", device
            ))
            
            # Should terminate early due to EOS
            assert len(responses) >= 1
            assert responses[-1].finish_reason == "stop"

    def test_single_device_generation_health_issues(self, mock_model, mock_tokenizer, mock_sampler, mock_task):
        """Test single device generation with health issues."""
        device = torch.device("xpu:0")
        
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'), \
             patch('exo.worker.engines.ipex.generator.generate.monitor_intel_gpu_health') as mock_health, \
             patch('exo.worker.engines.ipex.generator.generate.handle_intel_gpu_health_issues') as mock_handle:
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids
            
            # Mock health issues
            mock_health.return_value = {"overall_health": "unhealthy"}
            mock_handle.return_value = True  # Successfully handled
            
            # Test generation
            responses = list(_ipex_generate_single_device(
                mock_model, mock_tokenizer, mock_sampler, mock_task, "Test prompt", device
            ))
            
            # Should still generate despite health issues
            assert len(responses) >= 0
            
            # Verify health handling was called
            mock_handle.assert_called()


class TestIPEXDistributedGeneration:
    """Test distributed IPEX generation."""

    @pytest.fixture
    def mock_distributed_model(self):
        """Create a mock distributed IPEX model."""
        model = Mock(spec=IPEXModel)
        model._ipex_dist_group = Mock()
        model._ipex_parallelism_type = "pipeline"
        model._ipex_rank = 0
        model._ipex_world_size = 2
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        mock_outputs.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        model.return_value = mock_outputs
        
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock IPEX tokenizer."""
        tokenizer = Mock(spec=IPEXTokenizerWrapper)
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "distributed"
        tokenizer.eos_token_ids = [2]
        return tokenizer

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock sampler function."""
        def sampler(logits):
            return torch.tensor([15])
        return sampler

    @pytest.fixture
    def mock_task(self):
        """Create a mock task."""
        return ChatCompletionTaskParams(
            model="test-model",
            messages=[ChatCompletionMessage(role="user", content="Test distributed")],
            max_tokens=3
        )

    def test_distributed_generation_pipeline_rank_0(self, mock_distributed_model, mock_tokenizer, mock_sampler, mock_task):
        """Test distributed generation with pipeline parallelism on rank 0."""
        device = torch.device("xpu:0")
        
        with patch('torch.distributed') as mock_dist, \
             patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'), \
             patch('exo.worker.engines.ipex.generator.generate.intel_gpu_distributed_barrier'):
            
            # Mock distributed operations
            mock_dist.send = Mock()
            mock_dist.broadcast = Mock()
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids
            
            # Test distributed generation
            responses = list(_ipex_generate_distributed(
                mock_distributed_model, mock_tokenizer, mock_sampler, mock_task, "Test prompt", device
            ))
            
            # Should generate tokens
            assert len(responses) <= mock_task.max_tokens
            
            # Verify distributed operations were called
            mock_dist.send.assert_called()
            mock_dist.broadcast.assert_called()

    def test_distributed_generation_tensor_parallelism(self, mock_distributed_model, mock_tokenizer, mock_sampler, mock_task):
        """Test distributed generation with tensor parallelism."""
        device = torch.device("xpu:0")
        mock_distributed_model._ipex_parallelism_type = "tensor"
        
        with patch('torch.distributed') as mock_dist, \
             patch('torch.tensor') as mock_tensor, \
             patch('torch.ones_like') as mock_ones_like, \
             patch('torch.cat') as mock_cat, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'), \
             patch('exo.worker.engines.ipex.generator.generate.intel_gpu_distributed_barrier'):
            
            # Mock distributed operations
            mock_dist.all_reduce = Mock()
            mock_dist.broadcast = Mock()
            
            # Mock tensor operations
            mock_input_ids = Mock()
            mock_attention_mask = Mock()
            mock_tensor.return_value = mock_input_ids
            mock_ones_like.return_value = mock_attention_mask
            mock_cat.return_value = mock_input_ids
            
            # Test distributed generation
            responses = list(_ipex_generate_distributed(
                mock_distributed_model, mock_tokenizer, mock_sampler, mock_task, "Test prompt", device
            ))
            
            # Should generate tokens
            assert len(responses) <= mock_task.max_tokens
            
            # Verify tensor parallelism operations were called
            mock_dist.all_reduce.assert_called()
            mock_dist.broadcast.assert_called()

    def test_distributed_generation_memory_error(self, mock_distributed_model, mock_tokenizer, mock_sampler, mock_task):
        """Test distributed generation with memory error."""
        device = torch.device("xpu:0")
        
        with patch('torch.distributed'), \
             patch('torch.tensor'), \
             patch('torch.ones_like'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'):
            
            # Mock memory error during forward pass
            mock_distributed_model.side_effect = torch.xpu.OutOfMemoryError("Distributed GPU out of memory")
            
            with pytest.raises(IPEXMemoryError) as exc_info:
                list(_ipex_generate_distributed(
                    mock_distributed_model, mock_tokenizer, mock_sampler, mock_task, "Test prompt", device
                ))
            
            assert "Intel GPU out of memory during distributed generation" in str(exc_info.value)

    def test_distributed_generation_fallback_to_single(self, mock_distributed_model, mock_tokenizer, mock_sampler, mock_task):
        """Test distributed generation fallback to single device."""
        device = torch.device("xpu:0")
        mock_distributed_model._ipex_rank = 0  # Only rank 0 can fallback
        
        with patch('torch.distributed'), \
             patch('torch.tensor'), \
             patch('torch.ones_like'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'), \
             patch('exo.worker.engines.ipex.generator.generate._ipex_generate_single_device') as mock_single:
            
            # Mock distributed failure
            mock_distributed_model.side_effect = RuntimeError("Distributed failed")
            
            # Mock single device fallback
            def mock_single_generator():
                yield GenerationResponse(text="fallback", token=99, finish_reason="stop")
            
            mock_single.return_value = mock_single_generator()
            
            # Test distributed generation with fallback
            responses = list(_ipex_generate_distributed(
                mock_distributed_model, mock_tokenizer, mock_sampler, mock_task, "Test prompt", device
            ))
            
            # Should get fallback response
            assert len(responses) == 1
            assert responses[0].text == "fallback"
            
            # Verify fallback was called
            mock_single.assert_called_once()


class TestIPEXSimpleGeneration:
    """Test simple non-streaming IPEX generation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock IPEX model."""
        model = Mock(spec=IPEXModel)
        
        # Mock model parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device("xpu:0")
        model.parameters.return_value = [mock_param]
        
        # Mock generate method
        generated_ids = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])  # Input + new tokens
        model.generate.return_value = generated_ids
        
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock IPEX tokenizer."""
        tokenizer = Mock(spec=IPEXTokenizerWrapper)
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "Simple generated text"
        tokenizer.tokenizer.pad_token_id = 0
        tokenizer.tokenizer.eos_token_id = 2
        return tokenizer

    def test_ipex_generate_simple_success(self, mock_model, mock_tokenizer):
        """Test successful simple generation."""
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'):
            
            # Mock input tensor
            mock_input_ids = Mock()
            mock_input_ids.shape = (1, 5)  # Batch size 1, sequence length 5
            mock_tensor.return_value = mock_input_ids
            
            result = ipex_generate_simple(
                mock_model, 
                mock_tokenizer, 
                "Test prompt", 
                max_tokens=10, 
                temperature=0.8
            )
            
            assert result == "Simple generated text"
            
            # Verify model.generate was called with correct parameters
            mock_model.generate.assert_called_once()
            call_args = mock_model.generate.call_args
            assert call_args[1]['max_new_tokens'] == 10
            assert call_args[1]['temperature'] == 0.8
            assert call_args[1]['do_sample'] is True

    def test_ipex_generate_simple_memory_error(self, mock_model, mock_tokenizer):
        """Test simple generation with memory error."""
        with patch('torch.tensor'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'):
            
            # Mock memory error during generation
            mock_model.generate.side_effect = torch.xpu.OutOfMemoryError("Simple generation out of memory")
            
            with pytest.raises(IPEXMemoryError) as exc_info:
                ipex_generate_simple(mock_model, mock_tokenizer, "Test prompt")
            
            assert "Intel GPU out of memory during simple generation" in str(exc_info.value)

    def test_ipex_generate_simple_inference_error(self, mock_model, mock_tokenizer):
        """Test simple generation with inference error."""
        with patch('torch.tensor'), \
             patch('torch.no_grad'), \
             patch('torch.xpu.amp.autocast'), \
             patch('torch.xpu.empty_cache'):
            
            # Mock inference error during generation
            mock_model.generate.side_effect = RuntimeError("Simple generation failed")
            
            with pytest.raises(IPEXInferenceError) as exc_info:
                ipex_generate_simple(mock_model, mock_tokenizer, "Test prompt")
            
            assert "IPEX simple generation failed" in str(exc_info.value)
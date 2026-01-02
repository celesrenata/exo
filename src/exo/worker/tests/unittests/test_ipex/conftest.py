"""
Pytest configuration and fixtures for IPEX engine tests.

This module provides common fixtures and configuration for all IPEX tests.
"""

import pytest
from unittest.mock import Mock, patch
import torch
from pathlib import Path

from exo.worker.engines.ipex import IPEXModel, IPEXTokenizerWrapper
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.api import ChatCompletionMessage


@pytest.fixture
def mock_intel_gpu_available():
    """Mock Intel GPU as available for testing."""
    with patch('torch.xpu') as mock_xpu:
        mock_xpu.is_available.return_value = True
        mock_xpu.device_count.return_value = 1
        mock_xpu.empty_cache = Mock()
        mock_xpu.memory_allocated.return_value = 1024**3  # 1GB
        mock_xpu.memory_reserved.return_value = 1.5 * 1024**3  # 1.5GB
        
        # Mock device properties
        mock_props = Mock()
        mock_props.name = "Intel Arc A770"
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_props.max_compute_units = 512
        mock_props.supports_bfloat16 = True
        mock_xpu.get_device_properties.return_value = mock_props
        
        yield mock_xpu


@pytest.fixture
def mock_intel_gpu_unavailable():
    """Mock Intel GPU as unavailable for testing."""
    with patch('torch.xpu') as mock_xpu:
        mock_xpu.is_available.return_value = False
        mock_xpu.device_count.return_value = 0
        yield mock_xpu


@pytest.fixture
def mock_ipex_available():
    """Mock IPEX as available for testing."""
    with patch('intel_extension_for_pytorch') as mock_ipex:
        mock_ipex.__version__ = "1.13.0"
        mock_ipex.optimize = Mock(side_effect=lambda model, **kwargs: model)
        mock_ipex.llm = Mock()
        mock_ipex.llm.optimize = Mock(side_effect=lambda model, **kwargs: model)
        
        # Mock quantization
        mock_ipex.quantization = Mock()
        mock_ipex.quantization.prepare = Mock(side_effect=lambda model, **kwargs: model)
        mock_ipex.quantization.convert = Mock(side_effect=lambda model, **kwargs: model)
        mock_ipex.quantization.default_dynamic_qconfig = Mock()
        
        yield mock_ipex


@pytest.fixture
def mock_ipex_unavailable():
    """Mock IPEX as unavailable for testing."""
    with patch('intel_extension_for_pytorch', side_effect=ImportError("IPEX not found")):
        yield


@pytest.fixture
def mock_transformers():
    """Mock transformers library components."""
    with patch('transformers.AutoModelForCausalLM') as mock_model, \
         patch('transformers.AutoTokenizer') as mock_tokenizer, \
         patch('transformers.AutoConfig') as mock_config:
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.parameters.return_value = [Mock(device=torch.device("xpu:0"))]
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.bos_token = "<s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.decode.return_value = "decoded text"
        mock_tokenizer_instance.apply_chat_template.return_value = "formatted chat"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock config
        mock_config_instance = Mock()
        mock_config_instance.model_type = "llama"
        mock_config_instance.vocab_size = 32000
        mock_config_instance.hidden_size = 4096
        mock_config_instance.num_hidden_layers = 32
        mock_config_instance.num_attention_heads = 32
        mock_config.from_pretrained.return_value = mock_config_instance
        
        yield {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'config': mock_config,
            'model_instance': mock_model_instance,
            'tokenizer_instance': mock_tokenizer_instance,
            'config_instance': mock_config_instance
        }


@pytest.fixture
def mock_ipex_model():
    """Create a mock IPEX model for testing."""
    model = Mock(spec=IPEXModel)
    
    # Mock model parameters
    mock_param = Mock()
    mock_param.device = torch.device("xpu:0")
    mock_param.dtype = torch.float16
    model.parameters.return_value = [mock_param]
    
    # Mock forward pass
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    mock_outputs.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    model.return_value = mock_outputs
    
    # Mock generate method
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])
    
    return model


@pytest.fixture
def mock_ipex_tokenizer():
    """Create a mock IPEX tokenizer for testing."""
    tokenizer = Mock(spec=IPEXTokenizerWrapper)
    
    # Mock tokenizer properties
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token_ids = [2]
    
    # Mock tokenizer methods
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "User: Hello\nAssistant:"
    
    # Mock underlying tokenizer
    tokenizer.tokenizer = Mock()
    tokenizer.tokenizer.pad_token_id = 0
    tokenizer.tokenizer.eos_token_id = 2
    
    return tokenizer


@pytest.fixture
def mock_sampler():
    """Create a mock sampler function for testing."""
    def sampler(logits):
        return torch.tensor([10])  # Always return token ID 10
    return sampler


@pytest.fixture
def sample_chat_task():
    """Create a sample chat completion task for testing."""
    return ChatCompletionTaskParams(
        model="test-model",
        messages=[
            ChatCompletionMessage(role="user", content="Hello, world!")
        ],
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )


@pytest.fixture
def sample_model_path():
    """Create a sample model path for testing."""
    return Path("/mock/model/path")


@pytest.fixture
def mock_bound_instance():
    """Create a mock bound instance for testing."""
    from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance
    from exo.worker.tests.constants import MODEL_A_ID, RUNNER_1_ID, NODE_A
    
    return get_bound_mlx_ring_instance(
        instance_id="test_instance",
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NODE_A
    )


@pytest.fixture
def mock_distributed_model():
    """Create a mock distributed IPEX model for testing."""
    model = Mock(spec=IPEXModel)
    
    # Mock distributed properties
    model._ipex_dist_group = Mock()
    model._ipex_parallelism_type = "pipeline"
    model._ipex_rank = 0
    model._ipex_world_size = 2
    
    # Mock model parameters
    mock_param = Mock()
    mock_param.device = torch.device("xpu:0")
    mock_param.dtype = torch.float16
    model.parameters.return_value = [mock_param]
    
    # Mock forward pass
    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    mock_outputs.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    model.return_value = mock_outputs
    
    return model


@pytest.fixture
def mock_torch_operations():
    """Mock common torch operations for testing."""
    with patch('torch.tensor') as mock_tensor, \
         patch('torch.ones_like') as mock_ones_like, \
         patch('torch.cat') as mock_cat, \
         patch('torch.no_grad'), \
         patch('torch.device') as mock_device:
        
        # Mock tensor creation
        mock_tensor_instance = Mock()
        mock_tensor_instance.device = torch.device("xpu:0")
        mock_tensor_instance.dtype = torch.float16
        mock_tensor_instance.shape = (1, 5)
        mock_tensor_instance.item.return_value = 10
        mock_tensor_instance.unsqueeze.return_value = mock_tensor_instance
        mock_tensor.return_value = mock_tensor_instance
        
        # Mock other operations
        mock_ones_like.return_value = mock_tensor_instance
        mock_cat.return_value = mock_tensor_instance
        mock_device.return_value = torch.device("xpu:0")
        
        yield {
            'tensor': mock_tensor,
            'ones_like': mock_ones_like,
            'cat': mock_cat,
            'device': mock_device,
            'tensor_instance': mock_tensor_instance
        }


@pytest.fixture
def mock_logging():
    """Mock logging for cleaner test output."""
    with patch('exo.worker.runner.bootstrap.logger') as mock_logger:
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        yield mock_logger


# Test markers for categorizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu_required = pytest.mark.gpu_required


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu_required: mark test as requiring GPU hardware")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in item.nodeid or "test_ipex_engine_detection" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid or "dashboard" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to inference tests
        if "inference" in item.nodeid or "generation" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add GPU required marker to tests that need actual GPU
        if "gpu_required" in item.nodeid:
            item.add_marker(pytest.mark.gpu_required)
"""
Tests for ModelLoader component.

These tests validate model loading, XPU optimization, sharding, and validation.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.pytorch_xpu.model_loader import ModelLoader, TransformerShard


@pytest.fixture
def mock_shard_metadata():
    """Create mock shard metadata for testing."""
    model_card = ModelCard(
        model_id=ModelId("test/model"),
        storage_size=Memory(bytes=1000000),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[],
    )

    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


@pytest.fixture
def model_loader():
    """Create ModelLoader instance for testing."""
    return ModelLoader()


def test_model_loader_initialization(model_loader):
    """Test that ModelLoader initializes correctly."""
    assert model_loader is not None
    assert hasattr(model_loader, "_model_cache")
    assert hasattr(model_loader, "_tokenizer_cache")
    assert len(model_loader._model_cache) == 0
    assert len(model_loader._tokenizer_cache) == 0


def test_model_loader_dependencies(model_loader):
    """Test that ModelLoader checks for required dependencies."""
    # Should have checked for torch, ipex, and transformers
    assert hasattr(model_loader, "_torch_available")
    assert hasattr(model_loader, "_ipex_available")
    assert hasattr(model_loader, "_transformers_available")


@pytest.mark.asyncio
async def test_load_model_without_dependencies():
    """Test that load_model fails gracefully without dependencies."""
    loader = ModelLoader()
    loader._torch_available = False
    loader._transformers_available = False

    model_card = ModelCard(
        model_id=ModelId("test/model"),
        storage_size=Memory(bytes=1000000),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[],
    )

    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )

    with pytest.raises(RuntimeError, match="PyTorch and Transformers are required"):
        await loader.load_model(shard_metadata, "cpu", 0)


def test_validate_model_with_invalid_layer_count(model_loader, mock_shard_metadata):
    """Test model validation with mismatched layer count."""
    # Create mock model with wrong number of layers
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.model = Mock()
    mock_model.model.layers = [Mock() for _ in range(24)]  # Wrong count

    with pytest.raises(ValueError, match="Model has 24 layers but metadata specifies 32"):
        model_loader._validate_model(mock_model, mock_shard_metadata)


def test_validate_model_with_invalid_shard_boundaries(model_loader):
    """Test model validation with invalid shard boundaries."""
    model_card = ModelCard(
        model_id=ModelId("test/model"),
        storage_size=Memory(bytes=1000000),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[],
    )

    # Test invalid start_layer
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=-1,
        end_layer=32,
        n_layers=32,
    )

    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.model = Mock()
    mock_model.model.layers = [Mock() for _ in range(32)]

    with pytest.raises(ValueError, match="Invalid start_layer"):
        model_loader._validate_model(mock_model, shard_metadata)

    # Test invalid end_layer
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=40,
        n_layers=32,
    )

    with pytest.raises(ValueError, match="Invalid end_layer"):
        model_loader._validate_model(mock_model, shard_metadata)

    # Test invalid range
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=20,
        end_layer=10,
        n_layers=32,
    )

    with pytest.raises(ValueError, match="Invalid shard range"):
        model_loader._validate_model(mock_model, shard_metadata)


def test_validate_model_success(model_loader, mock_shard_metadata):
    """Test successful model validation."""
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.model = Mock()
    mock_model.model.layers = [Mock() for _ in range(32)]

    # Should not raise
    model_loader._validate_model(mock_model, mock_shard_metadata)


def test_transformer_shard_initialization():
    """Test TransformerShard initialization."""
    mock_model = Mock()
    mock_model.model = Mock()
    mock_model.model.layers = [Mock() for _ in range(32)]
    mock_model.model.embed_tokens = Mock()
    mock_model.model.norm = Mock()
    mock_model.lm_head = Mock()

    shard = TransformerShard(
        model=mock_model,
        start_layer=0,
        end_layer=16,
        is_first_layer=True,
        is_last_layer=False,
    )

    assert shard.start_layer == 0
    assert shard.end_layer == 16
    assert shard.is_first_layer is True
    assert shard.is_last_layer is False
    assert len(shard.layers) == 16
    assert shard.embed_tokens is not None
    assert shard.norm is None
    assert shard.lm_head is None


def test_transformer_shard_middle_layers():
    """Test TransformerShard for middle layers."""
    mock_model = Mock()
    mock_model.model = Mock()
    mock_model.model.layers = [Mock() for _ in range(32)]
    mock_model.model.embed_tokens = Mock()
    mock_model.model.norm = Mock()
    mock_model.lm_head = Mock()

    shard = TransformerShard(
        model=mock_model,
        start_layer=8,
        end_layer=24,
        is_first_layer=False,
        is_last_layer=False,
    )

    assert len(shard.layers) == 16
    assert shard.embed_tokens is None
    assert shard.norm is None
    assert shard.lm_head is None


def test_transformer_shard_last_layers():
    """Test TransformerShard for last layers."""
    mock_model = Mock()
    mock_model.model = Mock()
    mock_model.model.layers = [Mock() for _ in range(32)]
    mock_model.model.embed_tokens = Mock()
    mock_model.model.norm = Mock()
    mock_model.lm_head = Mock()

    shard = TransformerShard(
        model=mock_model,
        start_layer=16,
        end_layer=32,
        is_first_layer=False,
        is_last_layer=True,
    )

    assert len(shard.layers) == 16
    assert shard.embed_tokens is None
    assert shard.norm is not None
    assert shard.lm_head is not None


def test_clear_cache(model_loader):
    """Test cache clearing."""
    # Add some items to cache
    model_loader._model_cache["test"] = Mock()
    model_loader._tokenizer_cache["test"] = Mock()

    assert len(model_loader._model_cache) == 1
    assert len(model_loader._tokenizer_cache) == 1

    # Clear cache
    model_loader.clear_cache()

    assert len(model_loader._model_cache) == 0
    assert len(model_loader._tokenizer_cache) == 0


@pytest.mark.asyncio
async def test_encode_without_tokenizer(model_loader):
    """Test encode fails without loaded tokenizer."""
    with pytest.raises(RuntimeError, match="Tokenizer for test/model not loaded"):
        await model_loader.encode("test/model", "Hello, world!")


@pytest.mark.asyncio
async def test_decode_without_tokenizer(model_loader):
    """Test decode fails without loaded tokenizer."""
    tokens = np.array([1, 2, 3])
    with pytest.raises(RuntimeError, match="Tokenizer for test/model not loaded"):
        await model_loader.decode("test/model", tokens)

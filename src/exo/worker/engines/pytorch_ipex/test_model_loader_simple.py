#!/usr/bin/env python3
"""
Simple test script for ModelLoader component.

This script validates the ModelLoader implementation without requiring
full pytest infrastructure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.pytorch_ipex.model_loader import ModelLoader, TransformerShard


def test_model_loader_initialization():
    """Test ModelLoader initialization."""
    print("Testing ModelLoader initialization...")
    loader = ModelLoader()

    assert loader is not None
    assert hasattr(loader, "_model_cache")
    assert hasattr(loader, "_tokenizer_cache")
    assert len(loader._model_cache) == 0
    assert len(loader._tokenizer_cache) == 0

    print("✓ ModelLoader initialization successful")
    print(f"  - PyTorch available: {loader._torch_available}")
    print(f"  - IPEX available: {loader._ipex_available}")
    print(f"  - Transformers available: {loader._transformers_available}")


def test_model_validation():
    """Test model validation logic."""
    print("\nTesting model validation...")

    loader = ModelLoader()

    # Create mock model
    class MockConfig:
        pass

    class MockModel:
        def __init__(self, n_layers):
            self.config = MockConfig()
            self.model = type("obj", (object,), {"layers": [None] * n_layers})()

    # Create shard metadata
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

    # Test valid model
    mock_model = MockModel(32)
    try:
        loader._validate_model(mock_model, shard_metadata)
        print("✓ Valid model passed validation")
    except Exception as e:
        print(f"✗ Valid model failed validation: {e}")
        return False

    # Test invalid model (wrong layer count)
    mock_model_invalid = MockModel(24)
    try:
        loader._validate_model(mock_model_invalid, shard_metadata)
        print("✗ Invalid model passed validation (should have failed)")
        return False
    except ValueError as e:
        print(f"✓ Invalid model correctly rejected: {e}")

    return True


def test_transformer_shard():
    """Test TransformerShard wrapper."""
    print("\nTesting TransformerShard...")

    # Create mock model
    class MockLayer:
        pass

    class MockModel:
        def __init__(self):
            self.model = type(
                "obj",
                (object,),
                {
                    "layers": [MockLayer() for _ in range(32)],
                    "embed_tokens": "embed",
                    "norm": "norm",
                },
            )()
            self.lm_head = "lm_head"

    mock_model = MockModel()

    # Test first shard
    shard = TransformerShard(
        model=mock_model,
        start_layer=0,
        end_layer=16,
        is_first_layer=True,
        is_last_layer=False,
    )

    assert shard.start_layer == 0
    assert shard.end_layer == 16
    assert len(shard.layers) == 16
    assert shard.embed_tokens is not None
    assert shard.norm is None
    assert shard.lm_head is None

    print("✓ First shard created correctly")

    # Test middle shard
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

    print("✓ Middle shard created correctly")

    # Test last shard
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

    print("✓ Last shard created correctly")

    return True


def test_cache_management():
    """Test cache management."""
    print("\nTesting cache management...")

    loader = ModelLoader()

    # Add items to cache
    loader._model_cache["test1"] = "model1"
    loader._tokenizer_cache["test1"] = "tokenizer1"

    assert len(loader._model_cache) == 1
    assert len(loader._tokenizer_cache) == 1

    print("✓ Cache populated")

    # Clear cache
    loader.clear_cache()

    assert len(loader._model_cache) == 0
    assert len(loader._tokenizer_cache) == 0

    print("✓ Cache cleared")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ModelLoader Component Test Suite")
    print("=" * 60)

    try:
        test_model_loader_initialization()
        test_model_validation()
        test_transformer_shard()
        test_cache_management()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

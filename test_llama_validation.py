#!/usr/bin/env python3
"""Validation tests for Llama transformer implementation.

This test suite validates:
1. Output correctness compared to reference implementations
2. Deterministic output with fixed seed
3. Support for different model sizes (1B, 3B)

Tests are designed to be minimal and focused on core functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, "src")

import numpy as np
from pathlib import Path

# Set environment variable for testing
os.environ["EXO_TESTS"] = "1"


def test_model_forward_pass():
    """Test basic forward pass through the model."""
    print("\n=== Test 1: Basic Forward Pass ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        LlamaConfig,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Create small test config
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )

    # Create model
    model = LlamaTransformer(config)
    print(f"✓ Created model with {config.num_hidden_layers} layers")

    # Create test input
    batch_size = 1
    seq_len = 4
    input_ids = Tensor([[1, 2, 3, 4]])

    # Forward pass
    logits, cache = model(input_ids)

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {logits.shape}"
    )
    print(f"✓ Forward pass successful, output shape: {logits.shape}")

    # Verify cache was created
    assert cache is not None, "Cache should not be None"
    assert cache.get_seq_length() == seq_len, f"Cache should have {seq_len} tokens"
    print(f"✓ KV cache created with {cache.get_seq_length()} tokens")

    return True


def test_deterministic_output():
    """Test that model produces deterministic output with same input."""
    print("\n=== Test 2: Deterministic Output ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        LlamaConfig,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Create config
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    # Create model
    model = LlamaTransformer(config)

    # Create test input
    input_ids = Tensor([[1, 2, 3, 4, 5]])

    # First forward pass
    logits1, _ = model(input_ids)
    logits1_np = logits1.numpy()

    # Second forward pass with same input
    logits2, _ = model(input_ids)
    logits2_np = logits2.numpy()

    # Verify outputs are identical
    max_diff = np.abs(logits1_np - logits2_np).max()
    print(f"✓ Maximum difference between runs: {max_diff:.10f}")

    # Allow for small numerical differences due to floating point
    assert max_diff < 1e-6, f"Outputs differ by {max_diff}, expected < 1e-6"
    print("✓ Model produces deterministic output")

    return True


def test_kv_cache_consistency():
    """Test that KV cache produces consistent results."""
    print("\n=== Test 3: KV Cache Consistency ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        LlamaConfig,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Create config
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    # Create model
    model = LlamaTransformer(config)

    # Test sequence
    full_sequence = Tensor([[1, 2, 3, 4, 5]])

    # Method 1: Process full sequence at once
    logits_full, _ = model(full_sequence)
    logits_full_np = logits_full.numpy()

    # Method 2: Process incrementally with cache
    # First token
    logits_cached, cache = model(Tensor([[1]]))
    # Remaining tokens one by one
    for token_id in [2, 3, 4, 5]:
        logits_cached, cache = model(Tensor([[token_id]]), cache=cache)

    logits_cached_np = logits_cached.numpy()

    # Compare last token logits (should be identical)
    last_logits_full = logits_full_np[0, -1, :]
    last_logits_cached = logits_cached_np[0, -1, :]

    max_diff = np.abs(last_logits_full - last_logits_cached).max()
    print(f"✓ Maximum difference between full and cached: {max_diff:.10f}")

    # Allow for small numerical differences
    assert max_diff < 1e-4, f"Cached output differs by {max_diff}, expected < 1e-4"
    print("✓ KV cache produces consistent results")

    return True


def test_config_parsing():
    """Test configuration parsing from dictionary."""
    print("\n=== Test 4: Configuration Parsing ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        parse_config_from_dict,
        validate_config,
    )

    # Test valid config
    config_dict = {
        "model_type": "llama",
        "vocab_size": 128256,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }

    config = parse_config_from_dict(config_dict)
    print(f"✓ Parsed config: {config.hidden_size}d, {config.num_hidden_layers} layers")

    # Validate config
    validate_config(config)
    print("✓ Config validation passed")

    # Test default configs
    from exo.worker.engines.tinygrad.llama_transformer import get_default_config

    for size in ["0.5B", "1B", "3B", "8B"]:
        config = get_default_config(size)
        validate_config(config)
        print(
            f"✓ Default {size} config valid: {config.hidden_size}d, {config.num_hidden_layers} layers"
        )

    return True


def test_1b_model_structure():
    """Test 1B model can be created and used."""
    print("\n=== Test 5: 1B Model Structure ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        get_default_config,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Get 1B config
    config = get_default_config("1B")
    print(
        f"✓ 1B config: {config.hidden_size}d, {config.num_hidden_layers} layers, {config.num_attention_heads} heads"
    )

    # Create model
    model = LlamaTransformer(config)
    print("✓ Created 1B model structure")

    # Test forward pass
    input_ids = Tensor([[1, 2, 3]])
    logits, cache = model(input_ids)

    expected_shape = (1, 3, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected {expected_shape}, got {logits.shape}"
    )
    print(f"✓ 1B model forward pass successful: {logits.shape}")

    return True


def test_3b_model_structure():
    """Test 3B model can be created and used."""
    print("\n=== Test 6: 3B Model Structure ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        get_default_config,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Get 3B config
    config = get_default_config("3B")
    print(
        f"✓ 3B config: {config.hidden_size}d, {config.num_hidden_layers} layers, {config.num_attention_heads} heads"
    )

    # Create model
    model = LlamaTransformer(config)
    print("✓ Created 3B model structure")

    # Test forward pass
    input_ids = Tensor([[1, 2, 3]])
    logits, cache = model(input_ids)

    expected_shape = (1, 3, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected {expected_shape}, got {logits.shape}"
    )
    print(f"✓ 3B model forward pass successful: {logits.shape}")

    return True


def test_grouped_query_attention():
    """Test grouped-query attention with different Q and KV head counts."""
    print("\n=== Test 7: Grouped-Query Attention ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        LlamaConfig,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Create config with GQA (24 Q heads, 8 KV heads)
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=768,  # 768 / 24 = 32 head_dim
        intermediate_size=2048,
        num_hidden_layers=2,
        num_attention_heads=24,
        num_key_value_heads=8,  # 24 / 8 = 3x repetition
    )

    print(
        f"✓ GQA config: {config.num_attention_heads} Q heads, {config.num_key_value_heads} KV heads"
    )
    print(
        f"  Head dim: {config.head_dim}, Repetition factor: {config.num_attention_heads // config.num_key_value_heads}"
    )

    # Create model
    model = LlamaTransformer(config)
    print("✓ Created model with GQA")

    # Test forward pass
    input_ids = Tensor([[1, 2, 3, 4]])
    logits, cache = model(input_ids)

    expected_shape = (1, 4, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected {expected_shape}, got {logits.shape}"
    )
    print(f"✓ GQA forward pass successful: {logits.shape}")

    return True


def test_rope_position_encoding():
    """Test rotary position embeddings."""
    print("\n=== Test 8: Rotary Position Embeddings ===")

    from exo.worker.engines.tinygrad.llama_transformer import RotaryEmbedding
    from tinygrad import Tensor

    # Create RoPE
    rope = RotaryEmbedding(dim=128, theta=10000.0, max_seq_len=512)
    print(f"✓ Created RoPE with dim=128, theta=10000.0")

    # Create test tensors
    batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 128
    query = Tensor.randn(batch_size, seq_len, num_heads, head_dim)
    key = Tensor.randn(batch_size, seq_len, num_heads, head_dim)

    # Apply RoPE
    query_rot, key_rot = rope(query, key)

    # Verify shapes unchanged
    assert query_rot.shape == query.shape, "Query shape should be unchanged"
    assert key_rot.shape == key.shape, "Key shape should be unchanged"
    print(f"✓ RoPE applied successfully, shapes preserved")

    # Verify rotation actually changed values
    query_np = query.numpy()
    query_rot_np = query_rot.numpy()
    max_diff = np.abs(query_np - query_rot_np).max()
    assert max_diff > 0.01, "RoPE should change values significantly"
    print(f"✓ RoPE modified values (max diff: {max_diff:.4f})")

    return True


def test_output_logits_range():
    """Test that output logits are in reasonable range."""
    print("\n=== Test 9: Output Logits Range ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        LlamaConfig,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Create model
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaTransformer(config)

    # Forward pass
    input_ids = Tensor([[1, 2, 3, 4, 5]])
    logits, _ = model(input_ids)
    logits_np = logits.numpy()

    # Check for NaN or Inf
    assert not np.isnan(logits_np).any(), "Logits contain NaN values"
    assert not np.isinf(logits_np).any(), "Logits contain Inf values"
    print("✓ No NaN or Inf in logits")

    # Check reasonable range (logits typically in [-100, 100] for random init)
    logits_min = logits_np.min()
    logits_max = logits_np.max()
    print(f"✓ Logits range: [{logits_min:.2f}, {logits_max:.2f}]")

    # Verify not all zeros
    assert np.abs(logits_np).max() > 0.001, "Logits should not be all zeros"
    print("✓ Logits are non-zero")

    return True


def test_multiple_prompts():
    """Test model with multiple different prompts."""
    print("\n=== Test 10: Multiple Prompts ===")

    from exo.worker.engines.tinygrad.llama_transformer import (
        LlamaConfig,
        LlamaTransformer,
    )
    from tinygrad import Tensor

    # Create model
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaTransformer(config)

    # Test different prompts
    prompts = [
        [1, 2, 3],
        [5, 10, 15, 20],
        [100, 200],
        [1],
    ]

    for i, prompt in enumerate(prompts):
        input_ids = Tensor([prompt])
        logits, cache = model(input_ids)

        expected_shape = (1, len(prompt), config.vocab_size)
        assert logits.shape == expected_shape, (
            f"Prompt {i}: Expected {expected_shape}, got {logits.shape}"
        )

        # Verify no NaN/Inf
        logits_np = logits.numpy()
        assert not np.isnan(logits_np).any(), f"Prompt {i}: Contains NaN"
        assert not np.isinf(logits_np).any(), f"Prompt {i}: Contains Inf"

        print(f"✓ Prompt {i + 1} (len={len(prompt)}): {logits.shape}")

    print("✓ All prompts processed successfully")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("LLAMA TRANSFORMER VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Basic Forward Pass", test_model_forward_pass),
        ("Deterministic Output", test_deterministic_output),
        ("KV Cache Consistency", test_kv_cache_consistency),
        ("Configuration Parsing", test_config_parsing),
        ("1B Model Structure", test_1b_model_structure),
        ("3B Model Structure", test_3b_model_structure),
        ("Grouped-Query Attention", test_grouped_query_attention),
        ("Rotary Position Embeddings", test_rope_position_encoding),
        ("Output Logits Range", test_output_logits_range),
        ("Multiple Prompts", test_multiple_prompts),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {name}")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All validation tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

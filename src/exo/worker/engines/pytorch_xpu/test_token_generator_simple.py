"""
Simple test for TokenGenerator

This script tests the TokenGenerator class with basic sampling operations.
"""

import sys


def test_token_generator() -> None:
    """Test TokenGenerator basic functionality."""
    print("Testing TokenGenerator...")

    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping test")
        sys.exit(0)

    from exo.worker.engines.pytorch_xpu.token_generator import TokenGenerator

    # Create token generator
    generator = TokenGenerator(eos_token_id=2, pad_token_id=0)
    print("✓ TokenGenerator created")

    # Create sample logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    print(f"✓ Created logits with shape {logits.shape}")

    # Test basic sampling
    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)
    print(f"✓ Basic sampling: token={result.token_id}, prob={result.probability:.4f}")
    assert 0 <= result.token_id < vocab_size, "Token ID out of range"
    assert 0.0 <= result.probability <= 1.0, "Probability out of range"

    # Test temperature scaling
    result = generator.sample(logits, temperature=0.5, top_p=1.0, top_k=0)
    print(f"✓ Temperature sampling: token={result.token_id}, prob={result.probability:.4f}")

    # Test top-k sampling
    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=50)
    print(f"✓ Top-k sampling: token={result.token_id}, prob={result.probability:.4f}")

    # Test top-p sampling
    result = generator.sample(logits, temperature=1.0, top_p=0.9, top_k=0)
    print(f"✓ Top-p sampling: token={result.token_id}, prob={result.probability:.4f}")

    # Test combined sampling
    result = generator.sample(logits, temperature=0.8, top_p=0.95, top_k=100)
    print(f"✓ Combined sampling: token={result.token_id}, prob={result.probability:.4f}")

    # Test EOS detection
    eos_logits = torch.zeros(vocab_size)
    eos_logits[2] = 100.0  # Make EOS token very likely
    result = generator.sample(eos_logits, temperature=1.0, top_p=1.0, top_k=0)
    print(f"✓ EOS detection: token={result.token_id}, is_eos={result.is_eos}")
    assert result.is_eos, "Failed to detect EOS token"

    # Test PAD detection
    pad_logits = torch.zeros(vocab_size)
    pad_logits[0] = 100.0  # Make PAD token very likely
    result = generator.sample(pad_logits, temperature=1.0, top_p=1.0, top_k=0)
    print(f"✓ PAD detection: token={result.token_id}, is_pad={result.is_pad}")
    assert result.is_pad, "Failed to detect PAD token"

    # Test special token update
    generator.set_special_tokens(eos_token_id=5, pad_token_id=1)
    print("✓ Special tokens updated")

    # Test 2D logits
    logits_2d = torch.randn(2, vocab_size)
    result = generator.sample(logits_2d, temperature=1.0, top_p=1.0, top_k=0)
    print(f"✓ 2D logits sampling: token={result.token_id}, prob={result.probability:.4f}")

    print("\n✅ All TokenGenerator tests passed!")


if __name__ == "__main__":
    test_token_generator()

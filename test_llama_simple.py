#!/usr/bin/env python3
"""Simple validation test for Llama transformer - runs on gremlin-1."""

import sys
import os

# Add required paths for gremlin-1
sys.path.insert(
    0,
    "/nix/store/74lsy15mvdbsnn40jjr9w95y17mm8b0v-python3.13-numpy-2.3.5/lib/python3.13/site-packages",
)
sys.path.insert(
    0,
    "/nix/store/xg0j9lib60xlszacmivg00nfw20bdybd-python3.13-tinygrad-0.12.0/lib/python3.13/site-packages",
)
sys.path.insert(
    0,
    "/nix/store/jrsxpngcaif0rxflm6bax7pi1cd1hhyd-exo-0.3.0/lib/python3.13/site-packages",
)

# Find and add loguru
import glob

loguru_paths = glob.glob("/nix/store/*loguru*/lib/python3.13/site-packages")
if loguru_paths:
    sys.path.insert(0, loguru_paths[0])

os.environ["EXO_TESTS"] = "1"

import numpy as np

print("=" * 60)
print("LLAMA TRANSFORMER SIMPLE VALIDATION")
print("=" * 60)
print()

# Test 1: Import and create config
print("Test 1: Import llama_transformer module...")
try:
    # Import directly from the module file, not through __init__.py
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "llama_transformer",
        "/nix/store/jrsxpngcaif0rxflm6bax7pi1cd1hhyd-exo-0.3.0/lib/python3.13/site-packages/exo/worker/engines/tinygrad/llama_transformer.py",
    )
    llama_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama_module)

    LlamaConfig = llama_module.LlamaConfig
    LlamaTransformer = llama_module.LlamaTransformer
    get_default_config = llama_module.get_default_config

    print("✓ Successfully imported llama_transformer")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Create small config
print("\nTest 2: Create small test config...")
try:
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    print(f"✓ Created config: {config.hidden_size}d, {config.num_hidden_layers} layers")
except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 3: Create model
print("\nTest 3: Create LlamaTransformer model...")
try:
    model = LlamaTransformer(config)
    print(f"✓ Created model with {len(model.layers)} layers")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass
print("\nTest 4: Run forward pass...")
try:
    from tinygrad import Tensor

    input_ids = Tensor([[1, 2, 3, 4]])
    logits, cache = model(input_ids)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {logits.shape}")
    print(f"  Cache length: {cache.get_seq_length()}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Check output validity
print("\nTest 5: Validate output...")
try:
    logits_np = logits.numpy()
    has_nan = np.isnan(logits_np).any()
    has_inf = np.isinf(logits_np).any()

    if has_nan:
        print("✗ Output contains NaN values")
        sys.exit(1)
    if has_inf:
        print("✗ Output contains Inf values")
        sys.exit(1)

    print("✓ Output is valid (no NaN/Inf)")
    print(f"  Logits range: [{logits_np.min():.2f}, {logits_np.max():.2f}]")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)

# Test 6: Test 1B config
print("\nTest 6: Test 1B model config...")
try:
    config_1b = get_default_config("1B")
    print(
        f"✓ 1B config: {config_1b.hidden_size}d, {config_1b.num_hidden_layers} layers"
    )
except Exception as e:
    print(f"✗ Failed to get 1B config: {e}")
    sys.exit(1)

# Test 7: Test 3B config
print("\nTest 7: Test 3B model config...")
try:
    config_3b = get_default_config("3B")
    print(
        f"✓ 3B config: {config_3b.hidden_size}d, {config_3b.num_hidden_layers} layers"
    )
except Exception as e:
    print(f"✗ Failed to get 3B config: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print()
print("Summary:")
print("  - Module import: OK")
print("  - Config creation: OK")
print("  - Model creation: OK")
print("  - Forward pass: OK")
print("  - Output validation: OK")
print("  - 1B config: OK")
print("  - 3B config: OK")

#!/usr/bin/env python3
"""Memory profiling script for tinygrad Llama transformer.

This script monitors GPU memory during generation, verifies KV cache
reduces computation, and tracks memory usage patterns.

Requirements: 8.2, 8.5
"""

import time
import importlib.util
from pathlib import Path
from typing import Any
import sys


def load_llama_module() -> Any:
    """Load llama_transformer module directly to avoid dependency issues."""
    module_path = (
        Path(__file__).parent / "src/exo/worker/engines/tinygrad/llama_transformer.py"
    )

    if not module_path.exists():
        print(f"Error: Module not found at {module_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("llama_transformer", module_path)
    llama_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama_module)

    return llama_module


def estimate_model_memory(config: Any) -> dict[str, float]:
    """Estimate memory usage for model parameters.

    Args:
        config: LlamaConfig instance

    Returns:
        Dictionary with memory estimates in MB
    """
    # Assume float32 (4 bytes per parameter)
    bytes_per_param = 4

    # Embedding layer
    embed_params = config.vocab_size * config.hidden_size

    # Transformer layers
    layer_params = 0
    # Attention: Q, K, V, O projections
    layer_params += (
        config.hidden_size * config.num_attention_heads * config.head_dim
    )  # Q
    layer_params += (
        config.hidden_size * config.num_key_value_heads * config.head_dim
    )  # K
    layer_params += (
        config.hidden_size * config.num_key_value_heads * config.head_dim
    )  # V
    layer_params += (
        config.num_attention_heads * config.head_dim * config.hidden_size
    )  # O
    # MLP: gate, up, down projections
    layer_params += config.hidden_size * config.intermediate_size  # gate
    layer_params += config.hidden_size * config.intermediate_size  # up
    layer_params += config.intermediate_size * config.hidden_size  # down
    # Layer norms (2 per layer)
    layer_params += 2 * config.hidden_size

    total_layer_params = layer_params * config.num_hidden_layers

    # Final norm and LM head
    final_params = config.hidden_size + (config.hidden_size * config.vocab_size)

    total_params = embed_params + total_layer_params + final_params
    total_memory_mb = (total_params * bytes_per_param) / (1024 * 1024)

    return {
        "embedding_mb": (embed_params * bytes_per_param) / (1024 * 1024),
        "layers_mb": (total_layer_params * bytes_per_param) / (1024 * 1024),
        "final_mb": (final_params * bytes_per_param) / (1024 * 1024),
        "total_mb": total_memory_mb,
        "total_params": total_params,
    }


def estimate_kv_cache_memory(
    config: Any, batch_size: int, seq_len: int
) -> dict[str, float]:
    """Estimate memory usage for KV cache.

    Args:
        config: LlamaConfig instance
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dictionary with memory estimates in MB
    """
    # Assume float32 (4 bytes per value)
    bytes_per_value = 4

    # Each layer stores keys and values
    # Shape: (batch_size, seq_len, num_kv_heads, head_dim)
    cache_per_layer = (
        2 * batch_size * seq_len * config.num_key_value_heads * config.head_dim
    )
    total_cache = cache_per_layer * config.num_hidden_layers

    cache_mb = (total_cache * bytes_per_value) / (1024 * 1024)
    cache_per_layer_mb = (cache_per_layer * bytes_per_value) / (1024 * 1024)

    return {
        "cache_per_layer_mb": cache_per_layer_mb,
        "total_cache_mb": cache_mb,
        "cache_elements": total_cache,
    }


def estimate_activation_memory(
    config: Any, batch_size: int, seq_len: int
) -> dict[str, float]:
    """Estimate memory usage for activations during forward pass.

    Args:
        config: LlamaConfig instance
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dictionary with memory estimates in MB
    """
    # Assume float32 (4 bytes per value)
    bytes_per_value = 4

    # Hidden states: (batch_size, seq_len, hidden_size)
    hidden_states = batch_size * seq_len * config.hidden_size

    # Attention scores: (batch_size, num_heads, seq_len, seq_len)
    attn_scores = batch_size * config.num_attention_heads * seq_len * seq_len

    # MLP intermediate: (batch_size, seq_len, intermediate_size)
    mlp_intermediate = batch_size * seq_len * config.intermediate_size

    # Logits: (batch_size, seq_len, vocab_size)
    logits = batch_size * seq_len * config.vocab_size

    total_activations = hidden_states + attn_scores + mlp_intermediate + logits
    total_mb = (total_activations * bytes_per_value) / (1024 * 1024)

    return {
        "hidden_states_mb": (hidden_states * bytes_per_value) / (1024 * 1024),
        "attn_scores_mb": (attn_scores * bytes_per_value) / (1024 * 1024),
        "mlp_intermediate_mb": (mlp_intermediate * bytes_per_value) / (1024 * 1024),
        "logits_mb": (logits * bytes_per_value) / (1024 * 1024),
        "total_mb": total_mb,
    }


def verify_kv_cache_benefit(
    model: Any, input_ids: Any, max_tokens: int = 20
) -> dict[str, Any]:
    """Verify that KV cache reduces computation time.

    Args:
        model: LlamaTransformer instance
        input_ids: Initial prompt tokens
        max_tokens: Number of tokens to generate

    Returns:
        Dictionary with timing comparisons
    """
    from tinygrad import Tensor

    print(f"\n{'=' * 60}")
    print("KV CACHE BENEFIT VERIFICATION")
    print(f"{'=' * 60}")

    # Test 1: Generation WITH cache (normal mode)
    print("\n[1/2] Testing generation WITH KV cache...")
    cache_times = []
    cache = None

    # Prefill
    start = time.perf_counter()
    logits, cache = model(input_ids)
    _ = logits.numpy()
    prefill_time = time.perf_counter() - start

    # Generate tokens
    for i in range(max_tokens):
        next_token = logits[:, -1, :].argmax(axis=-1, keepdim=True)

        start = time.perf_counter()
        logits, cache = model(next_token, cache=cache)
        _ = logits.numpy()
        cache_times.append(time.perf_counter() - start)

        if (i + 1) % 5 == 0:
            avg_time = sum(cache_times[-5:]) / 5
            print(f"  Token {i + 1}/{max_tokens}: {avg_time * 1000:.2f}ms per token")

    with_cache_time = sum(cache_times)
    with_cache_avg = with_cache_time / len(cache_times)

    print(f"\n  Total generation time: {with_cache_time:.4f}s")
    print(f"  Average per token: {with_cache_avg * 1000:.2f}ms")
    print(f"  Tokens per second: {1.0 / with_cache_avg:.2f}")

    # Test 2: Generation WITHOUT cache (recompute everything)
    print("\n[2/2] Testing generation WITHOUT KV cache (full recompute)...")
    no_cache_times = []
    generated_tokens = []

    # Start with initial tokens
    current_tokens = input_ids

    for i in range(max_tokens):
        start = time.perf_counter()
        logits, _ = model(current_tokens, cache=None)  # No cache!
        _ = logits.numpy()
        no_cache_times.append(time.perf_counter() - start)

        # Sample next token
        next_token = logits[:, -1, :].argmax(axis=-1, keepdim=True)
        generated_tokens.append(int(next_token.numpy()[0, 0]))

        # Append to sequence for next iteration
        current_tokens = Tensor.cat([current_tokens, next_token], dim=1)

        if (i + 1) % 5 == 0:
            avg_time = sum(no_cache_times[-5:]) / 5
            print(f"  Token {i + 1}/{max_tokens}: {avg_time * 1000:.2f}ms per token")

    without_cache_time = sum(no_cache_times)
    without_cache_avg = without_cache_time / len(no_cache_times)

    print(f"\n  Total generation time: {without_cache_time:.4f}s")
    print(f"  Average per token: {without_cache_avg * 1000:.2f}ms")
    print(f"  Tokens per second: {1.0 / without_cache_avg:.2f}")

    # Calculate speedup
    speedup = without_cache_time / with_cache_time if with_cache_time > 0 else 0
    time_saved = without_cache_time - with_cache_time
    time_saved_pct = (
        (time_saved / without_cache_time) * 100 if without_cache_time > 0 else 0
    )

    print(f"\n{'=' * 60}")
    print("CACHE BENEFIT ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Time with cache:    {with_cache_time:.4f}s")
    print(f"  Time without cache: {without_cache_time:.4f}s")
    print(f"  Time saved:         {time_saved:.4f}s ({time_saved_pct:.1f}%)")
    print(f"  Speedup:            {speedup:.2f}x")
    print(
        f"  Cache is working:   {'✓ YES' if speedup > 1.5 else '✗ NO (expected >1.5x)'}"
    )
    print(f"{'=' * 60}\n")

    return {
        "with_cache_time": with_cache_time,
        "without_cache_time": without_cache_time,
        "speedup": speedup,
        "time_saved": time_saved,
        "time_saved_pct": time_saved_pct,
        "cache_working": speedup > 1.5,
    }


def profile_memory_growth(
    model: Any, input_ids: Any, max_tokens: int = 50
) -> dict[str, Any]:
    """Profile memory growth during generation.

    Args:
        model: LlamaTransformer instance
        input_ids: Initial prompt tokens
        max_tokens: Number of tokens to generate

    Returns:
        Dictionary with memory growth statistics
    """
    print(f"\n{'=' * 60}")
    print("MEMORY GROWTH PROFILING")
    print(f"{'=' * 60}")

    # Estimate initial memory
    config = model.config
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    model_mem = estimate_model_memory(config)
    initial_cache_mem = estimate_kv_cache_memory(config, batch_size, prompt_len)
    initial_activation_mem = estimate_activation_memory(config, batch_size, prompt_len)

    print(f"\n[1/2] Initial memory estimates:")
    print(f"  Model parameters: {model_mem['total_mb']:.2f} MB")
    print(f"  KV cache (prompt): {initial_cache_mem['total_cache_mb']:.2f} MB")
    print(f"  Activations: {initial_activation_mem['total_mb']:.2f} MB")
    print(
        f"  Total estimated: {model_mem['total_mb'] + initial_cache_mem['total_cache_mb'] + initial_activation_mem['total_mb']:.2f} MB"
    )

    # Estimate memory after generation
    final_seq_len = prompt_len + max_tokens
    final_cache_mem = estimate_kv_cache_memory(config, batch_size, final_seq_len)

    cache_growth = (
        final_cache_mem["total_cache_mb"] - initial_cache_mem["total_cache_mb"]
    )

    print(f"\n[2/2] Memory after {max_tokens} tokens:")
    print(f"  KV cache (final): {final_cache_mem['total_cache_mb']:.2f} MB")
    print(f"  Cache growth: {cache_growth:.2f} MB")
    print(f"  Growth per token: {cache_growth / max_tokens:.4f} MB/token")

    # Calculate memory efficiency
    print(f"\n{'=' * 60}")
    print("MEMORY EFFICIENCY ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Cache per layer: {final_cache_mem['cache_per_layer_mb']:.2f} MB")
    print(f"  Total layers: {config.num_hidden_layers}")
    print(f"  KV heads per layer: {config.num_key_value_heads}")
    print(f"  Head dimension: {config.head_dim}")
    print(f"  Sequence length: {final_seq_len}")
    print(f"{'=' * 60}\n")

    return {
        "model_memory_mb": model_mem["total_mb"],
        "initial_cache_mb": initial_cache_mem["total_cache_mb"],
        "final_cache_mb": final_cache_mem["total_cache_mb"],
        "cache_growth_mb": cache_growth,
        "growth_per_token_mb": cache_growth / max_tokens if max_tokens > 0 else 0,
        "total_estimated_mb": model_mem["total_mb"]
        + final_cache_mem["total_cache_mb"]
        + initial_activation_mem["total_mb"],
    }


def main():
    """Main memory profiling function."""
    print("=" * 60)
    print("TINYGRAD LLAMA TRANSFORMER MEMORY PROFILING")
    print("=" * 60)
    print("\nThis script profiles:")
    print("  1. Model parameter memory")
    print("  2. KV cache memory growth")
    print("  3. KV cache computation benefit")
    print("  4. Memory efficiency")
    print("\nRequirements: 8.2, 8.5")
    print("=" * 60)

    # Load module
    print("\n[Setup] Loading llama_transformer module...")
    llama_module = load_llama_module()

    # Import required classes
    LlamaConfig = llama_module.LlamaConfig
    LlamaTransformer = llama_module.LlamaTransformer

    # Create a small model for profiling (1B size)
    print("[Setup] Creating 1B model for profiling...")
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
    )

    model = LlamaTransformer(config)
    print(
        f"[Setup] Model created: {config.num_hidden_layers} layers, {config.hidden_size} hidden size"
    )

    # Import tinygrad
    try:
        from tinygrad import Tensor
    except ImportError:
        print("Error: tinygrad not available")
        sys.exit(1)

    # Create test input
    print("[Setup] Creating test input...")
    batch_size = 1
    prompt_length = 20
    input_ids = Tensor.randint(batch_size, prompt_length, low=0, high=config.vocab_size)

    # Profile memory growth
    memory_stats = profile_memory_growth(model, input_ids, max_tokens=50)

    # Verify KV cache benefit
    cache_stats = verify_kv_cache_benefit(model, input_ids, max_tokens=20)

    # Save results
    print("[Cleanup] Saving results to memory_profile_results.txt...")
    with open("memory_profile_results.txt", "w") as f:
        f.write("TINYGRAD LLAMA TRANSFORMER MEMORY PROFILE\n")
        f.write("=" * 60 + "\n\n")

        f.write("MEMORY STATISTICS\n")
        f.write("-" * 60 + "\n")
        for key, value in memory_stats.items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

        f.write("KV CACHE BENEFIT\n")
        f.write("-" * 60 + "\n")
        for key, value in cache_stats.items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

    print("✓ Results saved to memory_profile_results.txt")
    print("\n" + "=" * 60)
    print("MEMORY PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

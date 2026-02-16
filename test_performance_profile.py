#!/usr/bin/env python3
"""Performance profiling script for tinygrad Llama transformer.

This script profiles generation speed, measures tokens per second,
and identifies bottlenecks in the inference pipeline.

Requirements: 8.1, 8.3
"""

import time
import importlib.util
from pathlib import Path
from typing import Any
import sys


def load_llama_module() -> Any:
    """Load llama_transformer module directly to avoid dependency issues."""
    # Find the module in the source tree
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


def profile_forward_pass(
    model: Any, input_ids: Any, cache: Any = None, num_iterations: int = 10
) -> dict[str, float]:
    """Profile a single forward pass through the model.

    Args:
        model: LlamaTransformer instance
        input_ids: Input token IDs tensor
        cache: Optional KV cache
        num_iterations: Number of iterations to average over

    Returns:
        Dictionary with timing statistics
    """
    from tinygrad import Tensor

    times = []

    # Warmup
    for _ in range(2):
        _, _ = model(input_ids, cache=cache)

    # Profile
    for _ in range(num_iterations):
        start = time.perf_counter()
        logits, updated_cache = model(input_ids, cache=cache)
        # Force computation by accessing the data
        _ = logits.numpy()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times),
    }


def profile_component(
    component: Any, input_tensor: Any, num_iterations: int = 10
) -> dict[str, float]:
    """Profile a single component (RMSNorm, Attention, MLP, etc.).

    Args:
        component: Component to profile
        input_tensor: Input tensor
        num_iterations: Number of iterations to average over

    Returns:
        Dictionary with timing statistics
    """
    times = []

    # Warmup
    for _ in range(2):
        _ = component(input_tensor)

    # Profile
    for _ in range(num_iterations):
        start = time.perf_counter()
        output = component(input_tensor)
        # Force computation
        if hasattr(output, "numpy"):
            _ = output.numpy()
        elif isinstance(output, tuple):
            for item in output:
                if hasattr(item, "numpy"):
                    _ = item.numpy()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def profile_generation_speed(
    model: Any, initial_tokens: Any, max_tokens: int = 50
) -> dict[str, Any]:
    """Profile end-to-end generation speed.

    Args:
        model: LlamaTransformer instance
        initial_tokens: Initial prompt tokens
        max_tokens: Number of tokens to generate

    Returns:
        Dictionary with generation statistics
    """
    from tinygrad import Tensor

    print(f"\n{'=' * 60}")
    print("GENERATION SPEED PROFILING")
    print(f"{'=' * 60}")

    # Prefill phase
    print("\n[1/3] Prefill phase...")
    prefill_start = time.perf_counter()
    logits, cache = model(initial_tokens)
    _ = logits.numpy()  # Force computation
    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    prompt_length = initial_tokens.shape[1]
    prefill_tps = prompt_length / prefill_time if prefill_time > 0 else 0

    print(f"  Prompt length: {prompt_length} tokens")
    print(f"  Prefill time: {prefill_time:.4f}s")
    print(f"  Prefill TPS: {prefill_tps:.2f} tokens/sec")

    # Generation phase
    print("\n[2/3] Generation phase...")
    generation_times = []
    generated_tokens = []

    for i in range(max_tokens):
        # Sample next token (greedy for consistency)
        next_token_id = logits[:, -1, :].argmax(axis=-1, keepdim=True)
        generated_tokens.append(int(next_token_id.numpy()[0, 0]))

        # Generate next token
        gen_start = time.perf_counter()
        logits, cache = model(next_token_id, cache=cache)
        _ = logits.numpy()  # Force computation
        gen_end = time.perf_counter()

        generation_times.append(gen_end - gen_start)

        if (i + 1) % 10 == 0:
            avg_time = sum(generation_times[-10:]) / 10
            tps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"  Token {i + 1}/{max_tokens}: {tps:.2f} tokens/sec (last 10 avg)")

    # Calculate statistics
    total_gen_time = sum(generation_times)
    mean_gen_time = total_gen_time / len(generation_times)
    generation_tps = 1.0 / mean_gen_time if mean_gen_time > 0 else 0

    print(f"\n[3/3] Generation statistics:")
    print(f"  Tokens generated: {len(generated_tokens)}")
    print(f"  Total generation time: {total_gen_time:.4f}s")
    print(f"  Mean time per token: {mean_gen_time:.4f}s")
    print(f"  Generation TPS: {generation_tps:.2f} tokens/sec")
    print(f"  Min time per token: {min(generation_times):.4f}s")
    print(f"  Max time per token: {max(generation_times):.4f}s")

    # Overall statistics
    total_time = prefill_time + total_gen_time
    total_tokens = prompt_length + len(generated_tokens)
    overall_tps = total_tokens / total_time if total_time > 0 else 0

    print(f"\n{'=' * 60}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Overall TPS: {overall_tps:.2f} tokens/sec")
    print(f"{'=' * 60}\n")

    return {
        "prefill_time": prefill_time,
        "prefill_tps": prefill_tps,
        "generation_time": total_gen_time,
        "generation_tps": generation_tps,
        "mean_time_per_token": mean_gen_time,
        "min_time_per_token": min(generation_times),
        "max_time_per_token": max(generation_times),
        "total_time": total_time,
        "total_tokens": total_tokens,
        "overall_tps": overall_tps,
        "generated_tokens": generated_tokens,
    }


def profile_components(
    model: Any, batch_size: int = 1, seq_len: int = 10
) -> dict[str, dict[str, float]]:
    """Profile individual components to identify bottlenecks.

    Args:
        model: LlamaTransformer instance
        batch_size: Batch size for profiling
        seq_len: Sequence length for profiling

    Returns:
        Dictionary mapping component names to timing statistics
    """
    from tinygrad import Tensor

    print(f"\n{'=' * 60}")
    print("COMPONENT PROFILING")
    print(f"{'=' * 60}")

    hidden_size = model.config.hidden_size
    results = {}

    # Profile RMSNorm
    print("\n[1/6] Profiling RMSNorm...")
    test_input = Tensor.randn(batch_size, seq_len, hidden_size)
    norm_stats = profile_component(model.norm, test_input)
    results["RMSNorm"] = norm_stats
    print(f"  Mean time: {norm_stats['mean_time'] * 1000:.4f}ms")

    # Profile Embedding
    print("\n[2/6] Profiling Embedding...")
    test_ids = Tensor.randint(batch_size, seq_len, low=0, high=model.config.vocab_size)
    embed_stats = profile_component(model.embed_tokens, test_ids)
    results["Embedding"] = embed_stats
    print(f"  Mean time: {embed_stats['mean_time'] * 1000:.4f}ms")

    # Profile a single transformer layer
    print("\n[3/6] Profiling TransformerLayer...")
    layer = model.layers[0]
    layer_stats = profile_component(layer, test_input)
    results["TransformerLayer"] = layer_stats
    print(f"  Mean time: {layer_stats['mean_time'] * 1000:.4f}ms")

    # Profile Attention
    print("\n[4/6] Profiling Attention...")
    attn_stats = profile_component(layer.self_attn, test_input)
    results["Attention"] = attn_stats
    print(f"  Mean time: {attn_stats['mean_time'] * 1000:.4f}ms")

    # Profile MLP
    print("\n[5/6] Profiling MLP...")
    mlp_stats = profile_component(layer.mlp, test_input)
    results["MLP"] = mlp_stats
    print(f"  Mean time: {mlp_stats['mean_time'] * 1000:.4f}ms")

    # Profile LM Head
    print("\n[6/6] Profiling LM Head...")
    lm_head_stats = profile_component(model.lm_head, test_input)
    results["LM_Head"] = lm_head_stats
    print(f"  Mean time: {lm_head_stats['mean_time'] * 1000:.4f}ms")

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPONENT TIMING SUMMARY")
    print(f"{'=' * 60}")
    for name, stats in results.items():
        print(f"  {name:20s}: {stats['mean_time'] * 1000:8.4f}ms")
    print(f"{'=' * 60}\n")

    return results


def identify_bottlenecks(
    component_stats: dict[str, dict[str, float]], generation_stats: dict[str, Any]
) -> None:
    """Identify performance bottlenecks based on profiling data.

    Args:
        component_stats: Component timing statistics
        generation_stats: Generation timing statistics
    """
    print(f"\n{'=' * 60}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'=' * 60}")

    # Find slowest component
    slowest_component = max(component_stats.items(), key=lambda x: x[1]["mean_time"])
    print(f"\n[1] Slowest component: {slowest_component[0]}")
    print(f"    Time: {slowest_component[1]['mean_time'] * 1000:.4f}ms")

    # Check if generation TPS meets requirement (>10 tokens/sec)
    gen_tps = generation_stats["generation_tps"]
    requirement_met = gen_tps >= 10.0

    print(f"\n[2] Generation speed: {gen_tps:.2f} tokens/sec")
    print(f"    Requirement (>10 tok/s): {'✓ PASS' if requirement_met else '✗ FAIL'}")

    if not requirement_met:
        print(f"    Gap: {10.0 - gen_tps:.2f} tokens/sec below requirement")

    # Analyze time distribution
    print(f"\n[3] Time distribution:")
    prefill_pct = (
        generation_stats["prefill_time"] / generation_stats["total_time"]
    ) * 100
    gen_pct = (
        generation_stats["generation_time"] / generation_stats["total_time"]
    ) * 100
    print(f"    Prefill: {prefill_pct:.1f}%")
    print(f"    Generation: {gen_pct:.1f}%")

    # Recommendations
    print(f"\n[4] Optimization recommendations:")
    if slowest_component[0] == "Attention":
        print("    - Consider implementing Flash Attention")
        print("    - Optimize KV cache memory layout")
    elif slowest_component[0] == "MLP":
        print("    - Consider kernel fusion for gate/up projections")
        print("    - Optimize SwiGLU activation")
    elif slowest_component[0] == "LM_Head":
        print("    - Consider vocabulary projection optimization")
        print("    - Check if weight tying is enabled")

    if gen_tps < 10.0:
        print("    - Profile GPU utilization (see task 13.3)")
        print("    - Check for CPU fallbacks")
        print("    - Verify tinygrad JIT compilation is working")

    print(f"{'=' * 60}\n")


def main():
    """Main profiling function."""
    print("=" * 60)
    print("TINYGRAD LLAMA TRANSFORMER PERFORMANCE PROFILING")
    print("=" * 60)
    print("\nThis script profiles:")
    print("  1. Generation speed (tokens per second)")
    print("  2. Component-level timing")
    print("  3. Bottleneck identification")
    print("\nRequirements: 8.1, 8.3")
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

    # Profile components
    component_stats = profile_components(model, batch_size=batch_size, seq_len=10)

    # Profile generation speed
    generation_stats = profile_generation_speed(model, input_ids, max_tokens=50)

    # Identify bottlenecks
    identify_bottlenecks(component_stats, generation_stats)

    # Save results
    print("[Cleanup] Saving results to performance_profile_results.txt...")
    with open("performance_profile_results.txt", "w") as f:
        f.write("TINYGRAD LLAMA TRANSFORMER PERFORMANCE PROFILE\n")
        f.write("=" * 60 + "\n\n")

        f.write("GENERATION STATISTICS\n")
        f.write("-" * 60 + "\n")
        for key, value in generation_stats.items():
            if key != "generated_tokens":
                f.write(f"{key:25s}: {value}\n")
        f.write("\n")

        f.write("COMPONENT STATISTICS\n")
        f.write("-" * 60 + "\n")
        for component, stats in component_stats.items():
            f.write(f"\n{component}:\n")
            for key, value in stats.items():
                f.write(f"  {key:15s}: {value * 1000:.4f}ms\n")
        f.write("\n")

    print("✓ Results saved to performance_profile_results.txt")
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

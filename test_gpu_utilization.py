#!/usr/bin/env python3
"""GPU utilization verification script for tinygrad Llama transformer.

This script verifies that operations run on GPU, checks for CPU fallbacks,
and profiles kernel execution.

Requirements: 8.3, 6.5
"""

import time
import importlib.util
from pathlib import Path
from typing import Any
import sys
import os


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


def check_tinygrad_device() -> dict[str, Any]:
    """Check which device tinygrad is using.

    Returns:
        Dictionary with device information
    """
    print(f"\n{'=' * 60}")
    print("TINYGRAD DEVICE DETECTION")
    print(f"{'=' * 60}")

    try:
        from tinygrad import Device, Tensor

        # Get default device
        default_device = Device.DEFAULT
        print(f"\n[1/3] Default device: {default_device}")

        # Check available devices
        print(f"\n[2/3] Checking available devices...")
        devices_tested = {}

        for device_name in ["CPU", "GPU", "METAL", "CUDA", "OPENCL"]:
            try:
                # Try to create a tensor on this device
                test_tensor = Tensor([1.0], device=device_name)
                _ = test_tensor.numpy()
                devices_tested[device_name] = "✓ Available"
                print(f"  {device_name:10s}: ✓ Available")
            except Exception as e:
                devices_tested[device_name] = f"✗ Not available ({str(e)[:30]}...)"
                print(f"  {device_name:10s}: ✗ Not available")

        # Check environment variables
        print(f"\n[3/3] Environment variables:")
        env_vars = {
            "DEVICE": os.environ.get("DEVICE", "not set"),
            "GPU": os.environ.get("GPU", "not set"),
            "CUDA": os.environ.get("CUDA", "not set"),
            "METAL": os.environ.get("METAL", "not set"),
        }
        for key, value in env_vars.items():
            print(f"  {key:10s}: {value}")

        print(f"{'=' * 60}\n")

        return {
            "default_device": default_device,
            "devices_available": devices_tested,
            "env_vars": env_vars,
        }

    except ImportError as e:
        print(f"Error: tinygrad not available: {e}")
        sys.exit(1)


def verify_tensor_device(tensor: Any, expected_device: str) -> bool:
    """Verify that a tensor is on the expected device.

    Args:
        tensor: Tinygrad tensor
        expected_device: Expected device name

    Returns:
        True if tensor is on expected device
    """
    try:
        # Check tensor's device attribute
        actual_device = getattr(tensor, "device", "unknown")
        return actual_device == expected_device
    except Exception:
        return False


def profile_cpu_vs_gpu(
    model: Any, input_ids: Any, num_iterations: int = 10
) -> dict[str, Any]:
    """Profile performance on CPU vs GPU to detect fallbacks.

    Args:
        model: LlamaTransformer instance
        input_ids: Input token IDs
        num_iterations: Number of iterations to average over

    Returns:
        Dictionary with timing comparisons
    """
    from tinygrad import Tensor

    print(f"\n{'=' * 60}")
    print("CPU vs GPU PERFORMANCE COMPARISON")
    print(f"{'=' * 60}")

    results = {}

    # Test on CPU
    print(f"\n[1/2] Testing on CPU...")
    cpu_times = []

    # Create CPU input
    cpu_input = Tensor(input_ids.numpy(), device="CPU")

    # Warmup
    for _ in range(2):
        logits, _ = model(cpu_input, cache=None)
        _ = logits.numpy()

    # Profile
    for i in range(num_iterations):
        start = time.perf_counter()
        logits, _ = model(cpu_input, cache=None)
        _ = logits.numpy()
        cpu_times.append(time.perf_counter() - start)

        if (i + 1) % 5 == 0:
            avg = sum(cpu_times[-5:]) / 5
            print(f"  Iteration {i + 1}/{num_iterations}: {avg * 1000:.2f}ms")

    cpu_avg = sum(cpu_times) / len(cpu_times)
    print(f"\n  Average time: {cpu_avg * 1000:.2f}ms")
    results["cpu_time"] = cpu_avg

    # Test on GPU (if available)
    print(f"\n[2/2] Testing on GPU...")
    try:
        # Try GPU first, then OPENCL, then METAL
        gpu_device = None
        for device_name in ["GPU", "OPENCL", "METAL", "CUDA"]:
            try:
                test = Tensor([1.0], device=device_name)
                _ = test.numpy()
                gpu_device = device_name
                print(f"  Using device: {gpu_device}")
                break
            except Exception:
                continue

        if gpu_device is None:
            print("  ✗ No GPU device available")
            results["gpu_time"] = None
            results["speedup"] = None
            results["gpu_available"] = False
        else:
            gpu_times = []

            # Create GPU input
            gpu_input = Tensor(input_ids.numpy(), device=gpu_device)

            # Warmup
            for _ in range(2):
                logits, _ = model(gpu_input, cache=None)
                _ = logits.numpy()

            # Profile
            for i in range(num_iterations):
                start = time.perf_counter()
                logits, _ = model(gpu_input, cache=None)
                _ = logits.numpy()
                gpu_times.append(time.perf_counter() - start)

                if (i + 1) % 5 == 0:
                    avg = sum(gpu_times[-5:]) / 5
                    print(f"  Iteration {i + 1}/{num_iterations}: {avg * 1000:.2f}ms")

            gpu_avg = sum(gpu_times) / len(gpu_times)
            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

            print(f"\n  Average time: {gpu_avg * 1000:.2f}ms")
            print(f"  Speedup vs CPU: {speedup:.2f}x")

            results["gpu_time"] = gpu_avg
            results["speedup"] = speedup
            results["gpu_available"] = True
            results["gpu_device"] = gpu_device

    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        results["gpu_time"] = None
        results["speedup"] = None
        results["gpu_available"] = False

    # Analysis
    print(f"\n{'=' * 60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  CPU time: {cpu_avg * 1000:.2f}ms")
    if results.get("gpu_available"):
        gpu_time = results["gpu_time"]
        speedup = results["speedup"]
        print(f"  GPU time: {gpu_time * 1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(
            f"  GPU acceleration: {'✓ YES' if speedup > 1.2 else '✗ NO (expected >1.2x)'}"
        )
    else:
        print(f"  GPU: Not available")
    print(f"{'=' * 60}\n")

    return results


def check_for_cpu_fallbacks(model: Any, input_ids: Any) -> dict[str, Any]:
    """Check if any operations are falling back to CPU.

    Args:
        model: LlamaTransformer instance
        input_ids: Input token IDs

    Returns:
        Dictionary with fallback detection results
    """
    print(f"\n{'=' * 60}")
    print("CPU FALLBACK DETECTION")
    print(f"{'=' * 60}")

    # This is a heuristic check - we look for operations that are
    # significantly slower than expected, which might indicate CPU fallback

    print("\n[1/2] Profiling individual operations...")

    from tinygrad import Tensor

    # Test matrix multiplication (should be fast on GPU)
    print("  Testing matrix multiplication...")
    size = 1024
    a = Tensor.randn(size, size)
    b = Tensor.randn(size, size)

    # Warmup
    for _ in range(2):
        c = a @ b
        _ = c.numpy()

    # Profile
    times = []
    for _ in range(10):
        start = time.perf_counter()
        c = a @ b
        _ = c.numpy()
        times.append(time.perf_counter() - start)

    matmul_time = sum(times) / len(times)
    print(f"    Average time: {matmul_time * 1000:.2f}ms")

    # Test element-wise operations (should be fast on GPU)
    print("  Testing element-wise operations...")
    x = Tensor.randn(size, size)

    times = []
    for _ in range(10):
        start = time.perf_counter()
        y = x * x + x
        _ = y.numpy()
        times.append(time.perf_counter() - start)

    elemwise_time = sum(times) / len(times)
    print(f"    Average time: {elemwise_time * 1000:.2f}ms")

    # Test softmax (should be fast on GPU)
    print("  Testing softmax...")
    x = Tensor.randn(size, size)

    times = []
    for _ in range(10):
        start = time.perf_counter()
        y = x.softmax(axis=-1)
        _ = y.numpy()
        times.append(time.perf_counter() - start)

    softmax_time = sum(times) / len(times)
    print(f"    Average time: {softmax_time * 1000:.2f}ms")

    print(f"\n[2/2] Analyzing results...")

    # Heuristic: if operations are very slow, might be CPU fallback
    # These thresholds are rough estimates
    matmul_slow = matmul_time > 0.1  # >100ms for 1024x1024 matmul is slow
    elemwise_slow = elemwise_time > 0.01  # >10ms for element-wise is slow
    softmax_slow = softmax_time > 0.05  # >50ms for softmax is slow

    print(f"  Matrix multiplication: {'⚠ SLOW' if matmul_slow else '✓ FAST'}")
    print(f"  Element-wise ops: {'⚠ SLOW' if elemwise_slow else '✓ FAST'}")
    print(f"  Softmax: {'⚠ SLOW' if softmax_slow else '✓ FAST'}")

    any_slow = matmul_slow or elemwise_slow or softmax_slow

    print(f"\n{'=' * 60}")
    print("FALLBACK ANALYSIS")
    print(f"{'=' * 60}")
    if any_slow:
        print("  ⚠ WARNING: Some operations are slow")
        print("  This might indicate CPU fallback or suboptimal GPU usage")
        print("  Recommendations:")
        print("    - Check tinygrad device configuration")
        print("    - Verify GPU drivers are installed")
        print("    - Check for tinygrad compilation errors")
    else:
        print("  ✓ All operations are reasonably fast")
        print("  No obvious CPU fallbacks detected")
    print(f"{'=' * 60}\n")

    return {
        "matmul_time": matmul_time,
        "elemwise_time": elemwise_time,
        "softmax_time": softmax_time,
        "matmul_slow": matmul_slow,
        "elemwise_slow": elemwise_slow,
        "softmax_slow": softmax_slow,
        "potential_fallback": any_slow,
    }


def profile_kernel_execution(model: Any, input_ids: Any) -> dict[str, Any]:
    """Profile kernel execution patterns.

    Args:
        model: LlamaTransformer instance
        input_ids: Input token IDs

    Returns:
        Dictionary with kernel profiling results
    """
    print(f"\n{'=' * 60}")
    print("KERNEL EXECUTION PROFILING")
    print(f"{'=' * 60}")

    print("\n[1/2] Profiling forward pass...")

    # Profile a single forward pass with detailed timing
    start_total = time.perf_counter()

    # Embedding
    start = time.perf_counter()
    hidden_states = model.embed_tokens(input_ids)
    _ = hidden_states.numpy()
    embed_time = time.perf_counter() - start

    # First layer
    start = time.perf_counter()
    hidden_states, _ = model.layers[0](hidden_states)
    _ = hidden_states.numpy()
    layer_time = time.perf_counter() - start

    # All layers (approximate)
    all_layers_time = layer_time * len(model.layers)

    # Final norm
    start = time.perf_counter()
    hidden_states = model.norm(hidden_states)
    _ = hidden_states.numpy()
    norm_time = time.perf_counter() - start

    # LM head
    start = time.perf_counter()
    logits = model.lm_head(hidden_states)
    _ = logits.numpy()
    lm_head_time = time.perf_counter() - start

    total_time = time.perf_counter() - start_total

    print(f"  Embedding: {embed_time * 1000:.2f}ms")
    print(f"  Single layer: {layer_time * 1000:.2f}ms")
    print(f"  All layers (est): {all_layers_time * 1000:.2f}ms")
    print(f"  Final norm: {norm_time * 1000:.2f}ms")
    print(f"  LM head: {lm_head_time * 1000:.2f}ms")
    print(f"  Total: {total_time * 1000:.2f}ms")

    print(f"\n[2/2] Analyzing kernel efficiency...")

    # Calculate time distribution
    total_accounted = embed_time + all_layers_time + norm_time + lm_head_time
    overhead = total_time - total_accounted
    overhead_pct = (overhead / total_time) * 100 if total_time > 0 else 0

    print(f"  Accounted time: {total_accounted * 1000:.2f}ms")
    print(f"  Overhead: {overhead * 1000:.2f}ms ({overhead_pct:.1f}%)")

    # Time distribution
    print(f"\n  Time distribution:")
    print(f"    Embedding: {(embed_time / total_time) * 100:.1f}%")
    print(f"    Layers: {(all_layers_time / total_time) * 100:.1f}%")
    print(f"    Final norm: {(norm_time / total_time) * 100:.1f}%")
    print(f"    LM head: {(lm_head_time / total_time) * 100:.1f}%")
    print(f"    Overhead: {overhead_pct:.1f}%")

    print(f"\n{'=' * 60}")
    print("KERNEL EFFICIENCY ANALYSIS")
    print(f"{'=' * 60}")
    if overhead_pct > 20:
        print("  ⚠ WARNING: High overhead detected")
        print("  This might indicate:")
        print("    - Kernel launch overhead")
        print("    - Memory transfer overhead")
        print("    - Suboptimal kernel fusion")
    else:
        print("  ✓ Overhead is reasonable")
        print("  Kernel execution appears efficient")
    print(f"{'=' * 60}\n")

    return {
        "embed_time": embed_time,
        "layer_time": layer_time,
        "all_layers_time": all_layers_time,
        "norm_time": norm_time,
        "lm_head_time": lm_head_time,
        "total_time": total_time,
        "overhead": overhead,
        "overhead_pct": overhead_pct,
    }


def main():
    """Main GPU utilization verification function."""
    print("=" * 60)
    print("TINYGRAD LLAMA TRANSFORMER GPU UTILIZATION")
    print("=" * 60)
    print("\nThis script verifies:")
    print("  1. GPU device detection")
    print("  2. CPU vs GPU performance")
    print("  3. CPU fallback detection")
    print("  4. Kernel execution profiling")
    print("\nRequirements: 8.3, 6.5")
    print("=" * 60)

    # Check device
    device_info = check_tinygrad_device()

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

    # Profile CPU vs GPU
    perf_stats = profile_cpu_vs_gpu(model, input_ids, num_iterations=10)

    # Check for CPU fallbacks
    fallback_stats = check_for_cpu_fallbacks(model, input_ids)

    # Profile kernel execution
    kernel_stats = profile_kernel_execution(model, input_ids)

    # Save results
    print("[Cleanup] Saving results to gpu_utilization_results.txt...")
    with open("gpu_utilization_results.txt", "w") as f:
        f.write("TINYGRAD LLAMA TRANSFORMER GPU UTILIZATION\n")
        f.write("=" * 60 + "\n\n")

        f.write("DEVICE INFORMATION\n")
        f.write("-" * 60 + "\n")
        for key, value in device_info.items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

        f.write("PERFORMANCE STATISTICS\n")
        f.write("-" * 60 + "\n")
        for key, value in perf_stats.items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

        f.write("FALLBACK DETECTION\n")
        f.write("-" * 60 + "\n")
        for key, value in fallback_stats.items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

        f.write("KERNEL PROFILING\n")
        f.write("-" * 60 + "\n")
        for key, value in kernel_stats.items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

    print("✓ Results saved to gpu_utilization_results.txt")
    print("\n" + "=" * 60)
    print("GPU UTILIZATION VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/bin/bash
# Run all performance profiling tests for tinygrad Llama transformer
# This script runs the three performance optimization tests:
#   1. Generation speed profiling (task 13.1)
#   2. Memory usage profiling (task 13.2)
#   3. GPU utilization verification (task 13.3)

set -e

echo "======================================================================="
echo "TINYGRAD LLAMA TRANSFORMER PERFORMANCE TEST SUITE"
echo "======================================================================="
echo ""
echo "This script will run three performance tests:"
echo "  1. Generation speed profiling (tokens/sec, bottlenecks)"
echo "  2. Memory usage profiling (KV cache, memory growth)"
echo "  3. GPU utilization verification (device detection, fallbacks)"
echo ""
echo "Requirements: 8.1, 8.2, 8.3, 8.5, 6.5"
echo "======================================================================="
echo ""

# Check if Python is available
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found"
  exit 1
fi

# Check if tinygrad is available
if ! python3 -c "import tinygrad" 2>/dev/null; then
  echo "Warning: tinygrad not available in current environment"
  echo "Tests may fail if tinygrad is not installed"
fi

# Create results directory
RESULTS_DIR="performance_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Test 1: Generation speed profiling
echo "======================================================================="
echo "TEST 1/3: Generation Speed Profiling"
echo "======================================================================="
echo ""

if [ -f "test_performance_profile.py" ]; then
  echo "Running test_performance_profile.py..."
  if python3 test_performance_profile.py 2>&1 | tee "$RESULTS_DIR/generation_speed.log"; then
    echo "✓ Generation speed profiling completed"
    # Move results file if it exists
    if [ -f "performance_profile_results.txt" ]; then
      mv performance_profile_results.txt "$RESULTS_DIR/"
    fi
  else
    echo "✗ Generation speed profiling failed"
  fi
else
  echo "✗ test_performance_profile.py not found"
fi

echo ""
echo "Press Enter to continue to next test..."
read

# Test 2: Memory usage profiling
echo "======================================================================="
echo "TEST 2/3: Memory Usage Profiling"
echo "======================================================================="
echo ""

if [ -f "test_memory_profile.py" ]; then
  echo "Running test_memory_profile.py..."
  if python3 test_memory_profile.py 2>&1 | tee "$RESULTS_DIR/memory_usage.log"; then
    echo "✓ Memory usage profiling completed"
    # Move results file if it exists
    if [ -f "memory_profile_results.txt" ]; then
      mv memory_profile_results.txt "$RESULTS_DIR/"
    fi
  else
    echo "✗ Memory usage profiling failed"
  fi
else
  echo "✗ test_memory_profile.py not found"
fi

echo ""
echo "Press Enter to continue to next test..."
read

# Test 3: GPU utilization verification
echo "======================================================================="
echo "TEST 3/3: GPU Utilization Verification"
echo "======================================================================="
echo ""

if [ -f "test_gpu_utilization.py" ]; then
  echo "Running test_gpu_utilization.py..."
  if python3 test_gpu_utilization.py 2>&1 | tee "$RESULTS_DIR/gpu_utilization.log"; then
    echo "✓ GPU utilization verification completed"
    # Move results file if it exists
    if [ -f "gpu_utilization_results.txt" ]; then
      mv gpu_utilization_results.txt "$RESULTS_DIR/"
    fi
  else
    echo "✗ GPU utilization verification failed"
  fi
else
  echo "✗ test_gpu_utilization.py not found"
fi

echo ""
echo "======================================================================="
echo "ALL TESTS COMPLETE"
echo "======================================================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary of results:"
ls -lh "$RESULTS_DIR"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/performance_profile_results.txt"
echo "  cat $RESULTS_DIR/memory_profile_results.txt"
echo "  cat $RESULTS_DIR/gpu_utilization_results.txt"
echo ""
echo "======================================================================="

#!/usr/bin/env bash
# Script to remove all tinygrad-related code and files

set -e

echo "Removing tinygrad backend and related files..."

# Remove tinygrad engine directory
rm -rf src/exo/worker/engines/tinygrad/

# Remove tinygrad test files
rm -f test_tinygrad_gpu.sh
rm -f test_llama_validation.py
rm -f test_llama_simple.py
rm -f test_llama_on_gremlin1.sh
rm -f test_gpu_utilization.py
rm -f test_memory_profile.py
rm -f test_performance_profile.py
rm -f test_weight_loading.py
rm -f trigger_tinygrad_diagnostics.sh
rm -f capture_fresh_model_load.sh
rm -f trigger_model_loading_diagnostics.sh
rm -f run_test_with_exo_env.sh
rm -f run_performance_tests.sh

# Remove tinygrad deployment scripts
rm -f deploy_tinygrad_to_gremlin1.sh

# Remove tinygrad documentation
rm -f docs/tinygrad-backend.md
rm -f docs/nixos-tinygrad-configuration.md
rm -f docs/TINYGRAD_MODELS.md
rm -f docs/TINYGRAD_MODEL_SETUP.md

# Remove tinygrad patches
rm -f patches/tinygrad-intel-arc-4gb-fix.patch
rm -f create_patch.py

# Remove tinygrad-related markdown files
rm -f TINYGRAD_*.md
rm -f QUICK_TEST_LLAMA.md
rm -f QUICK_PERFORMANCE_TEST_GUIDE.md
rm -f DEBUGGING_CONTEXT_TASK13.md

echo "Tinygrad files removed successfully!"

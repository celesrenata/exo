#!/usr/bin/env bash
# Run simple validation test

echo "Running simple Llama transformer test..."
echo ""

# Set environment
export PYTHONNOUSERSITE='true'
export EXO_TINYGRAD_ENABLED='true'

# Build PYTHONPATH
export PYTHONPATH="/nix/store/jrsxpngcaif0rxflm6bax7pi1cd1hhyd-exo-0.3.0/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/74lsy15mvdbsnn40jjr9w95y17mm8b0v-python3.13-numpy-2.3.5/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/xg0j9lib60xlszacmivg00nfw20bdybd-python3.13-tinygrad-0.12.0/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/gj1nk0ff7qh9m1cwkv2sma0v5px1z7vj-python3.13-loguru-0.7.3/lib/python3.13/site-packages"

# Use python
PYTHON=/nix/store/qzc04a3npl70cyyy6flnnrb2ig3kayxm-python3-3.13.11/bin/python3.13

# Run simple test
exec $PYTHON /tmp/test_llama_simple.py

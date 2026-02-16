#!/bin/bash
# Run simple validation test on gremlin-1

export PYTHONNOUSERSITE='true'
export EXO_TINYGRAD_ENABLED='true'

# Use the python from exo
PYTHON=/nix/store/qzc04a3npl70cyyy6flnnrb2ig3kayxm-python3-3.13.11/bin/python3.13

echo "Running Llama transformer validation on gremlin-1..."
echo ""

$PYTHON /tmp/test_llama_simple.py

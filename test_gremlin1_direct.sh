#!/usr/bin/env bash
# Direct test of exo on gremlin-1 without nix package build
# This bypasses the MLX dependency issues by running from git checkout

set -euo pipefail

GREMLIN1="root@10.1.1.12"

echo "=== Testing exo on gremlin-1 directly ==="
echo ""

# Step 1: Clone repo
echo "Step 1: Cloning repo..."
ssh $GREMLIN1 "rm -rf /tmp/exo-test && git clone -b ipex https://github.com/celesrenata/exo.git /tmp/exo-test"

# Step 2: Install dependencies with nix-shell
echo "Step 2: Testing tinygrad import..."
ssh $GREMLIN1 "cd /tmp/exo-test && nix-shell -p python313 python313Packages.tinygrad python313Packages.numpy --run 'python3 -c \"import tinygrad; print(f\\\"tinygrad {tinygrad.__version__}\\\")\"'"

# Step 3: Check if we can import exo modules
echo "Step 3: Testing exo imports..."
ssh $GREMLIN1 "cd /tmp/exo-test && nix-shell -p python313 python313Packages.tinygrad python313Packages.numpy python313Packages.pydantic python313Packages.aiohttp --run 'python3 -c \"import sys; sys.path.insert(0, \\\"src\\\"); from exo.shared.constants import EXO_TINYGRAD_ENABLED; print(f\\\"EXO_TINYGRAD_ENABLED: {EXO_TINYGRAD_ENABLED}\\\")\"'"

echo ""
echo "✓ Basic imports work!"
echo ""
echo "To run exo on gremlin-1:"
echo "  ssh $GREMLIN1"
echo "  cd /tmp/exo-test"
echo "  nix-shell -p python313 python313Packages.tinygrad python313Packages.numpy python313Packages.pydantic python313Packages.aiohttp python313Packages.fastapi python313Packages.huggingface-hub python313Packages.psutil python313Packages.loguru python313Packages.filelock python313Packages.rustworkx python313Packages.tiktoken python313Packages.hypercorn python313Packages.httpx python313Packages.toml python313Packages.pillow python313Packages.safetensors python313Packages.transformers"
echo '  export PYTHONPATH=$PWD/src:$PYTHONPATH'
echo "  export EXO_TINYGRAD_ENABLED=true"
echo "  python3 -m exo.main"

#!/usr/bin/env bash
set -euo pipefail

echo "=== Deploying PyTorch + IPEX build to gremlin-1 ==="

# Build the exo package
echo "Building exo with PyTorch + IPEX..."
nix build .#exo --print-build-logs

# Copy to gremlin-1
echo "Copying build to gremlin-1..."
nix copy --to ssh-ng://gremlin-1 .#exo

echo "=== Deployment complete ==="
echo ""
echo "To test on gremlin-1, run:"
echo "  ssh gremlin-1"
echo "  nix shell /home/celes/sources/celesrenata/exo#exo -c python -c 'import torch; import intel_extension_for_pytorch as ipex; print(f\"PyTorch: {torch.__version__}\"); print(f\"IPEX: {ipex.__version__}\"); print(f\"XPU available: {torch.xpu.is_available()}\")'"

#!/usr/bin/env bash
set -euo pipefail

echo "=== Testing PyTorch + IPEX on gremlin-1 ==="

# Test PyTorch and IPEX availability
echo "1. Testing PyTorch and IPEX installation..."
ssh gremlin-1 "cd /home/celes/sources/celesrenata/exo && nix develop .#default -c python -c '
import torch
import intel_extension_for_pytorch as ipex
print(f\"✓ PyTorch version: {torch.__version__}\")
print(f\"✓ IPEX version: {ipex.__version__}\")
print(f\"✓ XPU available: {torch.xpu.is_available()}\")
if torch.xpu.is_available():
    print(f\"✓ XPU device count: {torch.xpu.device_count()}\")
    for i in range(torch.xpu.device_count()):
        print(f\"  Device {i}: {torch.xpu.get_device_name(i)}\")
'"

echo ""
echo "2. Testing Intel Arc GPU detection..."
ssh gremlin-1 "cd /home/celes/sources/celesrenata/exo && nix develop .#default -c python src/exo/worker/engines/pytorch_ipex/detect_intel_arc.py"

echo ""
echo "3. Testing device manager..."
ssh gremlin-1 "cd /home/celes/sources/celesrenata/exo && nix develop .#default -c python src/exo/worker/engines/pytorch_ipex/test_device_manager_simple.py"

echo ""
echo "=== All tests complete ==="

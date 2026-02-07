# Single-Node Validation Guide

This guide covers the validation process for Intel hardware support on a single node (gremlin-1).

## Overview

The validation process consists of 7 subtasks that verify:
1. Build system and dependencies
2. Service startup and initialization
3. API endpoints and health checks
4. Intel Arc iGPU detection and configuration
5. Intel NPU detection (optional)
6. Model download and loading
7. Inference execution and performance

## Prerequisites

### On Development Machine (esnixi)

- SSH access to gremlin-1 (10.1.1.12)
- Git repository with Intel hardware support
- Network connectivity to gremlin cluster

### On Target Node (gremlin-1)

- NixOS with Intel hardware support configured
- Intel Core Ultra 9 185H processor
- Intel Arc iGPU drivers installed
- Network connectivity

## Quick Start

### Option 1: Automated Validation

Run the comprehensive validation script:

```bash
# From esnixi, test gremlin-1
./tests/validate_gremlin_single_node.sh gremlin-1

# Or test localhost
./tests/validate_gremlin_single_node.sh localhost
```

This script will:
- Check connectivity
- Verify build and dependencies
- Start exo service (if local)
- Test all endpoints
- Validate GPU and NPU detection
- Download and test a model
- Run inference and measure performance

### Option 2: Manual Validation

Follow the step-by-step process below.

## Step-by-Step Validation

### Task 9.1: Build exo with Intel Hardware Support

**Objective**: Verify that exo builds successfully with tinygrad backend.

#### On gremlin-1:

```bash
# SSH to gremlin-1
ssh root@10.1.1.12

# Navigate to exo repository
cd /path/to/exo

# Build exo
nix build .#exo

# Verify build
./result/bin/exo --version

# Check tinygrad
python3 -c "import tinygrad; print(f'tinygrad {tinygrad.__version__}')"

# Check dependencies
python3 -c "import numpy, pyopencl"
```

**Expected Results**:
- ✓ Build completes without errors
- ✓ exo binary is available
- ✓ tinygrad package is importable
- ✓ All dependencies are available

**Troubleshooting**:
- If build fails, check `nix flake show` for module structure
- If tinygrad missing, verify `python/parts.nix` configuration
- If pyopencl missing, check OpenCL headers are installed

### Task 9.2: Start exo Service

**Objective**: Start exo with tinygrad backend enabled.

#### On gremlin-1:

```bash
# Start exo with tinygrad backend
EXO_TINYGRAD_ENABLED=true exo -vv

# Or with more verbose logging
EXO_TINYGRAD_ENABLED=true EXO_INTEL_ARC_ENABLED=true exo -vv
```

#### From esnixi:

```bash
# Check if service is running
curl http://10.1.1.12:52415/health

# Check logs (if using systemd)
ssh root@10.1.1.12 "journalctl -u exo -f"
```

**Expected Results**:
- ✓ Service starts without errors
- ✓ Logs show tinygrad backend initialization
- ✓ No crash or immediate exit

**Troubleshooting**:
- Check logs for "tinygrad" initialization messages
- Verify GPU device access: `ls -la /dev/dri/`
- Check for permission issues: `groups` should include `video`

### Task 9.3: Verify Web Service Endpoint

**Objective**: Confirm API is accessible and responding.

```bash
# Test health endpoint
curl http://10.1.1.12:52415/health

# Test OpenAI-compatible API
curl http://10.1.1.12:52415/v1/models

# Test metrics endpoint (if available)
curl http://10.1.1.12:52415/metrics
```

**Expected Results**:
- ✓ Health endpoint returns `{"status": "ok"}` or similar
- ✓ Models endpoint returns JSON with `{"object": "list", "data": [...]}`
- ✓ Metrics endpoint returns metrics data (optional)

**Troubleshooting**:
- If connection refused, check if exo is running: `ps aux | grep exo`
- If timeout, check firewall: `iptables -L`
- If 404, verify API routes are registered

### Task 9.4: Validate Intel GPU Detection

**Objective**: Verify Intel Arc iGPU is detected and configured.

#### Hardware Detection:

```bash
# Check for DRI devices
ls -la /dev/dri/

# Check for Intel GPU
lspci | grep -i "VGA.*Intel"

# Check GPU info
intel_gpu_top  # If available
```

#### Runtime Detection:

```bash
# Check Level Zero
ls -la /run/opengl-driver/lib/libze_loader.so*

# Test Level Zero with tinygrad
python3 << EOF
import os
os.environ['GPU'] = '1'
os.environ['LEVEL_ZERO'] = '1'
from tinygrad import Device
Device.DEFAULT = 'GPU'
print("Level Zero OK")
EOF

# Check OpenCL (fallback)
clinfo | grep -i intel
```

#### Metrics Check:

```bash
# Check if GPU appears in metrics
curl http://10.1.1.12:52415/metrics | grep -i "gpu\|intel"
```

**Expected Results**:
- ✓ DRI render devices found (`/dev/dri/renderD128`, etc.)
- ✓ Intel GPU detected via lspci
- ✓ Level Zero runtime available OR OpenCL available
- ✓ GPU information appears in metrics

**Troubleshooting**:
- If no DRI devices, check kernel modules: `lsmod | grep i915`
- If Level Zero missing, install: `nix-env -iA nixpkgs.intel-compute-runtime`
- If OpenCL missing, install: `nix-env -iA nixpkgs.ocl-icd`

### Task 9.5: Validate Intel NPU Detection

**Objective**: Verify Intel NPU is detected (optional, Core Ultra only).

#### Hardware Detection:

```bash
# Check for NPU device
ls -la /dev/accel/

# Check kernel modules
lsmod | grep -E "intel_vpu|ivpu"

# Load module if needed
sudo modprobe intel_vpu
```

#### OpenVINO Check:

```bash
# Check OpenVINO NPU access
python3 << EOF
import openvino as ov
core = ov.Core()
devices = core.available_devices()
print(f"Available devices: {devices}")
assert any('NPU' in d for d in devices), "NPU not found"
print("NPU OK")
EOF
```

#### Capability Report:

```bash
# Run NPU capability report
python3 -m exo.worker.engines.npu.capability_report
```

**Expected Results**:
- ✓ NPU device node found (`/dev/accel/accel0`)
- ✓ Kernel module loaded (`intel_vpu` or `ivpu`)
- ✓ OpenVINO can access NPU
- ✓ Capability report shows NPU available

**Note**: NPU support is optional and experimental. Failures here do not block deployment.

**Troubleshooting**:
- If no device node, check kernel version: `uname -r` (need 6.2+)
- If module not loading, check dmesg: `dmesg | grep -i vpu`
- If OpenVINO fails, check installation: `python3 -c "import openvino"`

### Task 9.6: Download and Load Tiny Model

**Objective**: Download and load a small model for testing.

```bash
# Send inference request (triggers download)
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 1,
    "stream": false
  }'

# Check if model is loaded
curl http://10.1.1.12:52415/v1/models | jq .

# Monitor download progress (if local)
tail -f /tmp/exo.log | grep -i "download\|loading"
```

**Expected Results**:
- ✓ Model download initiates
- ✓ Model appears in models list after download
- ✓ No errors during download or loading

**Troubleshooting**:
- If download fails, check internet connectivity
- If loading fails, check available memory: `free -h`
- If timeout, increase wait time (large models take longer)

### Task 9.7: Run Inference on Tiny Model

**Objective**: Execute inference and verify GPU usage.

```bash
# Run inference
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10,
    "stream": false
  }' | jq .

# Check GPU usage during inference
intel_gpu_top  # In another terminal

# Check metrics
curl http://10.1.1.12:52415/metrics | grep -i "tokens\|gpu"
```

**Expected Results**:
- ✓ Inference completes successfully
- ✓ Tokens are generated
- ✓ Performance metrics available
- ✓ GPU is being used (not CPU fallback)

**Troubleshooting**:
- If inference fails, check logs for errors
- If slow, verify GPU is being used (not CPU)
- If GPU not used, check Level Zero/OpenCL configuration

## Performance Benchmarks

### Expected Performance (TinyLlama-1.1B)

| Backend | Device | Tokens/sec | Notes |
|---------|--------|------------|-------|
| tinygrad | Intel Arc iGPU (Level Zero) | 20-40 | Target performance |
| tinygrad | Intel Arc iGPU (OpenCL) | 15-30 | Fallback performance |
| tinygrad | CPU | 5-10 | Baseline |

### Measuring Performance

```bash
# Run benchmark
time curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Write a short poem"}],
    "max_tokens": 100,
    "stream": false
  }' | jq '.usage'
```

## Validation Checklist

Use this checklist to track validation progress:

- [ ] **9.1 Build**
  - [ ] exo builds successfully
  - [ ] tinygrad is available
  - [ ] All dependencies present

- [ ] **9.2 Service**
  - [ ] Service starts without errors
  - [ ] Logs show proper initialization
  - [ ] No immediate crashes

- [ ] **9.3 Endpoints**
  - [ ] Health endpoint responds
  - [ ] OpenAI API available
  - [ ] Metrics endpoint accessible

- [ ] **9.4 GPU**
  - [ ] DRI devices present
  - [ ] Intel GPU detected
  - [ ] Level Zero OR OpenCL available
  - [ ] GPU appears in metrics

- [ ] **9.5 NPU** (Optional)
  - [ ] NPU device node present
  - [ ] Kernel module loaded
  - [ ] OpenVINO can access NPU
  - [ ] Capability report shows available

- [ ] **9.6 Model**
  - [ ] Model downloads successfully
  - [ ] Model loads without errors
  - [ ] Model appears in models list

- [ ] **9.7 Inference**
  - [ ] Inference completes successfully
  - [ ] Tokens generated correctly
  - [ ] Performance metrics available
  - [ ] GPU is being used

## Common Issues

### Issue: Build Fails

**Symptoms**: `nix build` fails with errors

**Solutions**:
1. Check flake structure: `nix flake show`
2. Update flake lock: `nix flake update`
3. Check for syntax errors in `flake.nix`
4. Verify nixpkgs version is compatible

### Issue: Service Won't Start

**Symptoms**: exo exits immediately or crashes

**Solutions**:
1. Check logs: `journalctl -u exo -n 100`
2. Verify dependencies: `python3 -c "import tinygrad"`
3. Check device permissions: `ls -la /dev/dri/`
4. Run with verbose logging: `exo -vv`

### Issue: GPU Not Detected

**Symptoms**: Falls back to CPU, slow performance

**Solutions**:
1. Check hardware: `lspci | grep Intel`
2. Load kernel module: `modprobe i915`
3. Install drivers: `nix-env -iA nixpkgs.intel-compute-runtime`
4. Check device access: `ls -la /dev/dri/`

### Issue: Inference Fails

**Symptoms**: Errors during inference, no tokens generated

**Solutions**:
1. Check model is loaded: `curl http://10.1.1.12:52415/v1/models`
2. Verify memory available: `free -h`
3. Check logs for OOM errors
4. Try smaller model or reduce max_tokens

## Next Steps

After successful validation:

1. **Document Results**: Record performance metrics and any issues
2. **Update Configuration**: Apply any necessary configuration changes
3. **Deploy to Cluster**: Proceed to multi-node deployment (task 10)
4. **Monitor Stability**: Run extended tests (24+ hours)
5. **Benchmark Performance**: Compare against baseline

## References

- [Intel Hardware Setup Guide](intel-hardware-setup.md)
- [Gremlin Cluster Deployment](gremlin-cluster-deployment.md)
- [Validation Script](../tests/validate_gremlin_single_node.sh)
- [Hardware Config Test](../tests/test_intel_hardware_config.sh)

## Support

If validation fails:

1. Review this guide and troubleshooting sections
2. Check logs for specific error messages
3. Verify hardware compatibility
4. Consult Intel hardware setup documentation
5. Report issues with full logs and system information

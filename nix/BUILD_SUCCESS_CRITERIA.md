# Build Success Criteria for PyTorch + IPEX with XPU Support

This document defines the success criteria for verifying that PyTorch and Intel Extension for PyTorch (IPEX) have been built correctly with Intel XPU (Arc GPU) support.

## Overview

The build verification process tests all requirements from task 10.4:
1. Test `torch.xpu.is_available()` after build
2. Verify `torch.xpu.device_count()` returns devices
3. Test basic tensor operations on XPU
4. Verify IPEX import and optimization
5. Document build success criteria

## Critical Requirements (Must Pass)

These requirements must pass for the build to be considered successful:

### 1. PyTorch Import
**Test**: `import torch`

**Success Criteria**:
- PyTorch imports without errors
- Version is 2.5.0 or higher
- No missing dependencies

**Failure Indicators**:
- ImportError when importing torch
- Missing shared libraries
- Version mismatch

### 2. XPU Module Compiled
**Test**: `hasattr(torch, 'xpu')`

**Success Criteria**:
- `torch.xpu` module exists
- This proves PyTorch was built with `USE_XPU=ON`

**Failure Indicators**:
- `torch.xpu` module not found
- AttributeError when accessing torch.xpu
- Build was done without XPU support

### 3. IPEX Import
**Test**: `import intel_extension_for_pytorch as ipex`

**Success Criteria**:
- IPEX imports without errors
- Version is 2.5.0+xpu or higher
- No missing dependencies

**Failure Indicators**:
- ImportError when importing IPEX
- Missing shared libraries
- Version mismatch with PyTorch

### 4. IPEX CPU Optimization
**Test**: Create and optimize a simple model on CPU

**Success Criteria**:
- `ipex.optimize()` succeeds on CPU
- Model inference produces correct output shapes
- No runtime errors

**Failure Indicators**:
- Optimization fails
- Inference produces wrong shapes
- Runtime errors during forward pass

## Optional Requirements (Hardware-Dependent)

These requirements are only tested if Intel Arc GPU hardware is present:

### 5. XPU Hardware Detection
**Test**: `torch.xpu.is_available()`

**Success Criteria**:
- Returns `True` when Intel Arc GPU is present
- Returns `False` when no Intel Arc GPU (expected on build machines)

**Notes**:
- This test is skipped in non-strict mode if no GPU present
- In strict mode, this test must pass

### 6. XPU Device Enumeration
**Test**: `torch.xpu.device_count()`

**Success Criteria**:
- Returns count > 0 when Intel Arc GPU present
- Device properties can be queried
- Device name, memory, compute units are reported

**Failure Indicators**:
- Returns 0 when GPU should be present
- Cannot query device properties
- Driver issues

### 7. XPU Tensor Operations
**Test**: Create tensors and perform operations on XPU

**Success Criteria**:
- Tensors can be created on `xpu:0` device
- Matrix multiplication works
- Element-wise operations work
- Reduction operations work
- Data can be transferred between XPU and CPU

**Failure Indicators**:
- Cannot create tensors on XPU
- Operations fail or produce incorrect results
- Memory errors
- Driver crashes

### 8. IPEX XPU Optimization
**Test**: Create and optimize a model for XPU with IPEX

**Success Criteria**:
- Model can be moved to XPU device
- `ipex.optimize()` succeeds with `dtype=torch.bfloat16`
- `ipex.optimize()` succeeds with `dtype=torch.float32`
- Inference on XPU produces correct output shapes
- Output tensors are on XPU device

**Failure Indicators**:
- Cannot move model to XPU
- Optimization fails
- Inference produces wrong shapes or errors
- Output tensors not on XPU

## Verification Modes

### Non-Strict Mode (Default)
```bash
python nix/verify-build.py
# or
./test_build_verification.sh
```

**Behavior**:
- Critical requirements (1-4) must pass
- Optional requirements (5-8) are skipped if no GPU present
- Exit code 0 if critical requirements pass
- Exit code 1 if critical requirements fail

**Use Case**:
- Building on machines without Intel Arc GPU
- CI/CD pipelines
- Development environments

### Strict Mode
```bash
python nix/verify-build.py --strict
# or
./test_build_verification.sh --strict
```

**Behavior**:
- All requirements (1-8) must pass
- Fails if no Intel Arc GPU detected
- Exit code 0 if all requirements pass
- Exit code 1 if any requirement fails
- Exit code 2 if no XPU hardware available

**Use Case**:
- Testing on machines with Intel Arc GPU
- Production deployment verification
- Hardware validation

## Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | All critical tests passed |
| 1 | Failure | Critical tests failed |
| 2 | No Hardware | XPU hardware not available (strict mode only) |

## Expected Results by Environment

### Build Machine (No Intel Arc GPU)
**Non-Strict Mode**:
```
✓ PASS    PyTorch Import
✓ PASS    XPU Available (torch.xpu.is_available)
⊘ SKIP    XPU Device Count (torch.xpu.device_count)
⊘ SKIP    XPU Tensor Operations
✓ PASS    IPEX Import
✓ PASS    IPEX Optimization (CPU)
⊘ SKIP    IPEX Optimization (XPU)

Result: BUILD VERIFICATION PASSED (without XPU hardware)
Exit Code: 0
```

**Strict Mode**:
```
✓ PASS    PyTorch Import
✗ FAIL    XPU Available (torch.xpu.is_available)
⊘ SKIP    XPU Device Count (torch.xpu.device_count)
⊘ SKIP    XPU Tensor Operations
✓ PASS    IPEX Import
✓ PASS    IPEX Optimization (CPU)
⊘ SKIP    IPEX Optimization (XPU)

Result: XPU Hardware Not Available
Exit Code: 2
```

### Deployment Machine (With Intel Arc GPU)
**Both Modes**:
```
✓ PASS    PyTorch Import
✓ PASS    XPU Available (torch.xpu.is_available)
✓ PASS    XPU Device Count (torch.xpu.device_count)
✓ PASS    XPU Tensor Operations
✓ PASS    IPEX Import
✓ PASS    IPEX Optimization (CPU)
✓ PASS    IPEX Optimization (XPU)

Result: BUILD VERIFICATION PASSED (with XPU support)
Exit Code: 0
```

## Troubleshooting

### PyTorch Import Fails
**Symptoms**:
- `ImportError: cannot import name 'torch'`
- Missing shared library errors

**Solutions**:
1. Check PyTorch was built: `nix build .#pytorch-xpu`
2. Verify dependencies in `nix/pytorch-xpu.nix`
3. Check build logs for errors
4. Ensure oneAPI dependencies are available

### XPU Module Not Found
**Symptoms**:
- `AttributeError: module 'torch' has no attribute 'xpu'`
- XPU support not compiled

**Solutions**:
1. Verify `USE_XPU=ON` in `nix/pytorch-xpu.nix`
2. Check CMake configuration in build logs
3. Rebuild PyTorch: `nix build .#pytorch-xpu --rebuild`
4. Verify oneAPI compiler is available

### IPEX Import Fails
**Symptoms**:
- `ImportError: cannot import name 'intel_extension_for_pytorch'`
- Version mismatch errors

**Solutions**:
1. Check IPEX was built: `nix build .#ipex-xpu`
2. Verify IPEX links against correct PyTorch version
3. Check `nix/ipex-xpu.nix` configuration
4. Ensure PyTorch XPU is built first

### XPU Not Available (Hardware Present)
**Symptoms**:
- `torch.xpu.is_available()` returns `False`
- GPU should be present but not detected

**Solutions**:
1. Check Intel GPU drivers: `clinfo | grep -i intel`
2. Verify compute-runtime installed: `nix-store -q --references $(which python3) | grep compute-runtime`
3. Check Level Zero: `ls /usr/lib/libze_loader.so*`
4. Verify GPU visible: `lspci | grep -i vga`
5. Check dmesg for driver errors: `dmesg | grep -i i915`

### XPU Operations Fail
**Symptoms**:
- Tensor creation fails on XPU
- Operations produce errors
- Driver crashes

**Solutions**:
1. Update Intel GPU drivers
2. Check GPU memory: `intel_gpu_top`
3. Verify no other processes using GPU
4. Check system logs: `journalctl -xe | grep -i gpu`
5. Try with smaller tensors to rule out memory issues

### IPEX Optimization Fails
**Symptoms**:
- `ipex.optimize()` raises errors
- Inference fails after optimization

**Solutions**:
1. Check IPEX version matches PyTorch version
2. Try different dtype (float32 vs bfloat16)
3. Verify model is in eval mode
4. Check for unsupported operations in model
5. Review IPEX documentation for model compatibility

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Verify Build
  run: |
    nix build .#pytorch-xpu
    nix build .#ipex-xpu
    ./test_build_verification.sh
```

### NixOS Deployment
```bash
# Build on CI server (no GPU)
./test_build_verification.sh

# Deploy to production (with GPU)
ssh production-server "cd /path/to/exo && ./test_build_verification.sh --strict"
```

## References

- Task 10.4: Create build verification script
- Requirement 11.4: Validate on Intel Arc hardware
- `nix/pytorch-xpu.nix`: PyTorch XPU build configuration
- `nix/ipex-xpu.nix`: IPEX XPU build configuration
- `nix/verify-build.py`: Verification script implementation
- `test_build_verification.sh`: Shell wrapper for verification

## Version History

- 2026-02-17: Initial version for task 10.4
- Covers PyTorch 2.5.0 and IPEX 2.5.0+xpu
- Tested on NixOS with Intel Arc GPU

# Build Success Criteria for PyTorch + IPEX with Intel XPU

## Overview

This document defines the success criteria for PyTorch and IPEX installation with Intel Arc GPU (XPU) support. Since we're using pip-installed pre-built wheels instead of building from source, "build success" means "installation success".

See `PYTORCH_IPEX_INSTALLATION.md` for installation instructions and `TASK_10_PIVOT.md` for the rationale behind using pip instead of Nix builds.

## Critical Requirements (Must Pass)

These requirements must be met on any system, regardless of GPU presence:

### CR-1: PyTorch Import
**Requirement**: PyTorch must be importable
**Test**: `python -c "import torch"`
**Success**: No ImportError
**Failure**: Installation incomplete or corrupted

### CR-2: PyTorch Version
**Requirement**: PyTorch version must include '+xpu' suffix
**Test**: `python -c "import torch; print(torch.__version__)"`
**Success**: Output contains '+xpu' (e.g., "2.5.1+xpu")
**Failure**: Wrong PyTorch version installed (standard instead of XPU)

### CR-3: IPEX Import
**Requirement**: IPEX must be importable
**Test**: `python -c "import intel_extension_for_pytorch as ipex"`
**Success**: No ImportError
**Failure**: IPEX not installed or incompatible version

### CR-4: IPEX Version
**Requirement**: IPEX version must include '+xpu' suffix
**Test**: `python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"`
**Success**: Output contains '+xpu' (e.g., "2.5.10+xpu")
**Failure**: Wrong IPEX version installed

### CR-5: XPU Module Available
**Requirement**: torch.xpu module must be accessible
**Test**: `python -c "import torch; torch.xpu.is_available()"`
**Success**: No AttributeError (returns True or False)
**Failure**: XPU module not available (wrong PyTorch build)

### CR-6: IPEX CPU Optimization
**Requirement**: IPEX optimization must work on CPU
**Test**: `python -c "import torch, intel_extension_for_pytorch as ipex; m = torch.nn.Linear(10,10); ipex.optimize(m)"`
**Success**: No exceptions
**Failure**: IPEX optimization broken

## Optional Requirements (Require Intel Arc GPU)

These requirements can only be tested on systems with Intel Arc GPU hardware:

### OR-1: XPU Available
**Requirement**: XPU must be detected when Intel Arc GPU is present
**Test**: `python -c "import torch; print(torch.xpu.is_available())"`
**Success**: Returns True
**Failure**: Returns False despite GPU being present
**Note**: Returns False on systems without Intel Arc GPU (expected)

### OR-2: XPU Device Count
**Requirement**: At least one XPU device must be detected
**Test**: `python -c "import torch; print(torch.xpu.device_count())"`
**Success**: Returns >= 1
**Failure**: Returns 0 despite GPU being present

### OR-3: XPU Device Name
**Requirement**: XPU device name must be retrievable
**Test**: `python -c "import torch; print(torch.xpu.get_device_name(0))"`
**Success**: Returns device name (e.g., "Intel(R) Arc(TM) A770 Graphics")
**Failure**: Exception or empty string

### OR-4: Tensor Creation on XPU
**Requirement**: Tensors must be creatable on XPU device
**Test**: `python -c "import torch; x = torch.randn(3,3).to('xpu'); print(x.device)"`
**Success**: Device type is 'xpu'
**Failure**: Exception or wrong device type

### OR-5: Tensor Operations on XPU
**Requirement**: Basic tensor operations must work on XPU
**Test**: `python -c "import torch; x = torch.randn(3,3).to('xpu'); y = torch.randn(3,3).to('xpu'); z = torch.matmul(x,y); print(z.shape)"`
**Success**: Returns torch.Size([3, 3])
**Failure**: Exception or wrong shape

### OR-6: IPEX XPU Optimization
**Requirement**: IPEX optimization must work on XPU
**Test**: `python -c "import torch, intel_extension_for_pytorch as ipex; m = torch.nn.Linear(10,10).to('xpu'); m = ipex.optimize(m); x = torch.randn(1,10).to('xpu'); y = m(x); print(y.device)"`
**Success**: Output device is 'xpu'
**Failure**: Exception or wrong device

## Verification Script

The `nix/verify-pytorch-ipex-xpu.py` script automates all these checks:

```bash
# Run all checks (allows missing GPU)
python nix/verify-pytorch-ipex-xpu.py

# Run all checks (requires GPU)
python nix/verify-pytorch-ipex-xpu.py --strict
```

Or use the wrapper script:

```bash
# Run all checks (allows missing GPU)
./test_pytorch_ipex_verification.sh

# Run all checks (requires GPU)
./test_pytorch_ipex_verification.sh --strict
```

## Exit Codes

The verification script uses the following exit codes:

- **0**: All checks passed (or GPU checks skipped due to no GPU)
- **1**: Critical failure (PyTorch or IPEX import failed)
- **2**: GPU not available (only in strict mode)
- **3**: GPU tests failed (GPU present but operations failed)

## Interpretation Guide

### Exit Code 0 (Success)

**Without GPU**:
```
Checks passed: 6/9
⚠️  WARNING: No Intel Arc GPU detected
   PyTorch and IPEX are installed correctly
   XPU features will not be available without Intel Arc GPU
```
**Interpretation**: Installation is correct. GPU tests were skipped because no GPU is present. This is expected on build machines or systems without Intel Arc GPUs.

**With GPU**:
```
Checks passed: 9/9
✅ SUCCESS: All checks passed
   PyTorch and IPEX are correctly installed with XPU support
   Intel Arc GPU is available and working
```
**Interpretation**: Installation is correct and GPU is working. Ready for production use.

### Exit Code 1 (Critical Failure)

```
❌ CRITICAL FAILURE: PyTorch or IPEX import failed
   Please install PyTorch+IPEX following nix/PYTORCH_IPEX_INSTALLATION.md
```
**Interpretation**: PyTorch or IPEX is not installed or is corrupted. Follow installation guide.

**Common causes**:
- PyTorch not installed
- Wrong PyTorch version (standard instead of XPU)
- IPEX not installed
- Version mismatch between PyTorch and IPEX
- Python environment issues

**Resolution**:
1. Check Python version (must be 3.10-3.12)
2. Reinstall PyTorch with XPU support
3. Reinstall IPEX with XPU support
4. Verify versions match

### Exit Code 2 (No GPU in Strict Mode)

```
❌ FAILURE: No Intel Arc GPU detected (strict mode)
   XPU is not available on this system
```
**Interpretation**: Strict mode requires GPU but none was found.

**Common causes**:
- No Intel Arc GPU in system
- GPU drivers not installed
- GPU not enabled in BIOS
- Permissions issues

**Resolution**:
1. Check hardware: `lspci | grep -i vga`
2. Check drivers: `clinfo` and `sycl-ls`
3. Check permissions: `groups` (should include 'video' or 'render')
4. Install drivers if missing

### Exit Code 3 (GPU Tests Failed)

```
❌ FAILURE: GPU tests failed
   Intel Arc GPU is detected but some operations failed
```
**Interpretation**: GPU is present but not working correctly.

**Common causes**:
- Driver version mismatch
- Incomplete driver installation
- GPU firmware issues
- Memory allocation failures

**Resolution**:
1. Update GPU drivers
2. Check `dmesg | grep -i gpu` for errors
3. Verify `/dev/dri/renderD*` exists
4. Check GPU memory: `intel_gpu_top`
5. Reinstall compute runtime

## Troubleshooting Matrix

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| ImportError: torch | PyTorch not installed | Install PyTorch with XPU |
| ImportError: intel_extension_for_pytorch | IPEX not installed | Install IPEX with XPU |
| Version without '+xpu' | Wrong package installed | Reinstall with correct index URL |
| AttributeError: 'module' object has no attribute 'xpu' | Standard PyTorch installed | Reinstall PyTorch with XPU |
| xpu.is_available() returns False | No GPU or drivers | Check hardware and drivers |
| Tensor creation fails | GPU memory issue | Check GPU memory availability |
| IPEX optimization fails | Version mismatch | Ensure PyTorch and IPEX versions match |

## Continuous Integration

For CI/CD pipelines:

```bash
# In CI without GPU (build verification)
python nix/verify-pytorch-ipex-xpu.py
# Should exit 0 with warning about no GPU

# In CI with GPU (full verification)
python nix/verify-pytorch-ipex-xpu.py --strict
# Should exit 0 with all checks passed
```

## References

- Installation Guide: `nix/PYTORCH_IPEX_INSTALLATION.md`
- Pivot Document: `.kiro/specs/pytorch-ipex-intel-arc/TASK_10_PIVOT.md`
- Verification Script: `nix/verify-pytorch-ipex-xpu.py`
- Test Wrapper: `test_pytorch_ipex_verification.sh`

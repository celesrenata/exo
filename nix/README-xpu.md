# PyTorch XPU Support - Nix Derivation

## Overview

This directory contains a Nix derivation for PyTorch with Intel XPU (Arc GPU) support. This is part of the PyTorch XPU Intel Arc GPU integration project.

> **Note:** IPEX (Intel Extension for PyTorch) is discontinued. PyTorch 2.11+ includes native XPU support via `torch.xpu`.

## What This Provides

- **PyTorch XPU**: Native Intel Arc GPU acceleration via `torch.xpu` (PyTorch 2.11+)
- **XPU Support**: Enables Intel Arc GPU acceleration for PyTorch models
- **Nix Integration**: Reproducible builds with all dependencies managed by Nix

## Requirements

### Prerequisites

1. **PyTorch with XPU support**
   - PyTorch 2.11+ includes native XPU support
   - No separate IPEX package needed

2. **Intel GPU Runtime** (included in derivation)
   - intel-compute-runtime
   - level-zero
   - Intel MKL

3. **Build Resources**
   - Disk space: ~10GB for build artifacts
   - RAM: ~16GB recommended
   - Time: 20-40 minutes on modern hardware

### Hardware (Optional for Build)

- Intel Arc GPU is NOT required for building
- XPU support is compiled in, but runtime detection requires actual hardware
- Build can succeed on any Linux system
- Deploy to Intel Arc GPU hardware for full functionality

## Files

### Core Files

- **`xpu.nix`**: Main Nix derivation for PyTorch XPU support (placeholder)
- **`verify-xpu.py`**: Verification script to test XPU functionality
- **`verify-pytorch-xpu-setup.py`**: Comprehensive PyTorch XPU verification
- **`README-xpu.md`**: This documentation file

### Test Scripts

- **`../test_ipex_xpu_build.sh`**: Automated build and test script

## Building

### Quick Start

```bash
# Build PyTorch XPU derivation
nix build .#pytorch-xpu

# Run verification tests
python nix/verify-xpu.py

# Or run comprehensive verification
python nix/verify-pytorch-xpu-setup.py
```

### Step-by-Step Build

1. **Build PyTorch XPU**:
   ```bash
   nix build .#pytorch-xpu --print-build-logs
   ```

2. **Verify the build**:
   ```bash
   python nix/verify-xpu.py
   ```

## Verification

The verification script tests:

1. **PyTorch XPU Support**: Verifies PyTorch with XPU is available
2. **XPU Device Detection**: Tests that Intel Arc GPUs are detected
3. **Tensor Operations**: Verifies basic tensor operations on XPU
4. **Model Optimization**: Tests model optimization on XPU (requires Intel Arc GPU)

### Running Verification

```bash
# Quick verification
python nix/verify-xpu.py

# Comprehensive verification (PyTorch XPU)
python nix/verify-pytorch-xpu-setup.py

# Or use the shell wrapper
./test_build_verification.sh
```

See `nix/BUILD_SUCCESS_CRITERIA.md` for detailed success criteria.

### Expected Results

#### Without Intel Arc GPU

```
✓ PASS    PyTorch XPU
✓ PASS    XPU Import Check
⊘ SKIP    XPU Device Operations

✓ All required checks passed!

Note: XPU-specific tests were skipped (no Intel Arc GPU detected).
This is expected if building without Intel Arc GPU hardware.
```

#### With Intel Arc GPU

```
✓ PASS    PyTorch XPU
✓ PASS    XPU Import Check
✓ PASS    XPU Device Operations

✓ All required checks passed!
✓ XPU-specific tests also passed!
PyTorch XPU is fully functional with Intel Arc GPU.
```

## Integration with exo

### Flake Integration

Add to `flake.nix`:

```nix
packages = {
  # ... existing packages ...
  
  pytorch-xpu = pkgs.python313.pkgs.callPackage ./nix/pytorch-xpu.nix {};
};
```

### Using in exo

```python
import torch

# Detect Intel Arc GPU
device = torch.device("xpu:0") if torch.xpu.is_available() else torch.device("cpu")

# Load model
model = load_model()
model = model.to(device)

# Run inference (native PyTorch XPU, no IPEX needed)
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

## Build Configuration

### Enabled Features

- **XPU Support**: Intel Arc GPU via Level Zero
- **PyTorch Integration**: Native `torch.xpu` support
- **Shared Libraries**: Dynamic linking

### Disabled Features

- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- Tests (require GPU and are very slow)

### Environment Variables

Runtime:
```bash
LD_LIBRARY_PATH=<intel-libs>
PYTORCH_ENABLE_XPU=1
EXO_PYTORCH_XPU_ENABLED=true
```

## Dependencies

### Build Dependencies

- cmake, ninja, git, which, pkg-config
- makeWrapper, addDriverRunpath

### Intel GPU Dependencies

- intel-compute-runtime (OpenCL and Level Zero runtime)
- level-zero (Low-level GPU API)
- mkl (Intel Math Kernel Library) - **unfree license**

### Python Dependencies

- numpy, pyyaml, typing-extensions
- setuptools, pybind11, psutil

## Troubleshooting

### XPU Not Available

**Expected behavior**: `torch.xpu.is_available()` returns False without Intel Arc GPU.

**Note**: This is expected. The build can succeed without a GPU. XPU support is compiled in, but runtime detection requires actual hardware.

### Import Errors

**Error**: `ImportError: cannot import name 'torch'` or similar

**Solution**:
1. Verify PyTorch built successfully
2. Check PYTHONPATH includes PyTorch
3. Ensure LD_LIBRARY_PATH includes Intel libraries

### Version Mismatch

**Solution**:
1. Ensure PyTorch 2.11+ is installed for native XPU support
2. Rebuild if necessary

## Testing on Hardware

### Deploy to gremlin-1 (Intel Arc GPU Test Machine)

```bash
# Update flake.nix to include pytorch-xpu
# Then deploy
bash force_update_gremlin1.sh

# SSH to gremlin-1
ssh root@10.1.1.12

# Run verification
python nix/verify-xpu.py

# Test with exo
systemctl status exo
journalctl -u exo -f
```

### Expected Performance

With Intel Arc GPU:
- Model loading: <10 seconds for 3B models
- Inference: >15 tokens/sec for Llama-3.2-3B
- Memory: <6GB VRAM for 3B models

## References

- **Design Document**: `.kiro/specs/pytorch-model-cards/design.md`
- **Requirements**: `.kiro/specs/pytorch-model-cards/requirements.md`
- **PyTorch XPU Build**: `nix/README-pytorch-xpu.md`
- [PyTorch Intel GPU Support](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Intel GPU Drivers](https://dgpu-docs.intel.com/)

## Notes

- IPEX is discontinued — PyTorch 2.11+ includes native XPU support
- No separate IPEX package needed
- All dependencies are declaratively specified via Nix
- Build is reproducible via Nix flake.lock
- MKL unfree license is acceptable for this use case
- Build time is significant but expected for PyTorch
- XPU support is compiled in even without hardware present

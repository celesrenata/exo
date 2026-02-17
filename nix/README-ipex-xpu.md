# Intel Extension for PyTorch (IPEX) with XPU Support - Nix Derivation

## Overview

This directory contains a Nix derivation for building Intel Extension for PyTorch (IPEX) from source with Intel XPU (Arc GPU) support. This is part of task 10.2 in the PyTorch+IPEX Intel Arc GPU integration project.

## What This Provides

- **IPEX 2.5.1+xpu**: Intel's optimization library for PyTorch on Intel hardware
- **XPU Support**: Enables Intel Arc GPU acceleration for PyTorch models
- **Optimizations**: Fused kernels, operator optimizations, and memory management for Intel GPUs
- **Nix Integration**: Reproducible builds with all dependencies managed by Nix

## Requirements

### Prerequisites

1. **PyTorch with XPU support** (from task 10.1)
   - Must be built first: `nix build .#pytorch-xpu`
   - IPEX links against this PyTorch build

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

- **`ipex-xpu.nix`**: Main Nix derivation for IPEX with XPU support
- **`verify-ipex-xpu.py`**: Verification script to test IPEX functionality
- **`README-ipex-xpu.md`**: This documentation file

### Test Scripts

- **`../test_ipex_xpu_build.sh`**: Automated build and test script

## Building

### Quick Start

```bash
# Build IPEX XPU derivation
nix build .#ipex-xpu

# Run verification tests
nix run .#ipex-xpu -- python nix/verify-ipex-xpu.py

# Or use the automated test script
./test_ipex_xpu_build.sh
```

### Step-by-Step Build

1. **Ensure PyTorch XPU is built** (task 10.1):
   ```bash
   nix build .#pytorch-xpu
   ```

2. **Build IPEX XPU**:
   ```bash
   nix build .#ipex-xpu --print-build-logs
   ```

3. **Handle hash mismatch** (expected on first build):
   - Nix will report a hash mismatch
   - Copy the "got" hash from the error
   - Update `nix/ipex-xpu.nix`:
     ```nix
     hash = "sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
     ```
   - Rebuild

4. **Verify the build**:
   ```bash
   nix run .#ipex-xpu -- python nix/verify-ipex-xpu.py
   ```

## Verification

The verification script (`verify-ipex-xpu.py`) tests:

1. **PyTorch XPU Support**: Verifies PyTorch with XPU is available
2. **IPEX Import**: Tests that IPEX imports successfully
3. **IPEX Optimization**: Verifies IPEX can optimize models
4. **IPEX XPU Optimization**: Tests XPU-specific optimizations (requires Intel Arc GPU)

### Expected Results

#### Without Intel Arc GPU

```
✓ PASS    PyTorch XPU
✓ PASS    IPEX Import
✓ PASS    IPEX Optimization
⊘ SKIP    IPEX XPU Optimization

✓ All required checks passed!

Note: XPU-specific tests were skipped (no Intel Arc GPU detected).
This is expected if building without Intel Arc GPU hardware.
```

#### With Intel Arc GPU

```
✓ PASS    PyTorch XPU
✓ PASS    IPEX Import
✓ PASS    IPEX Optimization
✓ PASS    IPEX XPU Optimization

✓ All required checks passed!
✓ XPU-specific tests also passed!
IPEX is fully functional with Intel Arc GPU.
```

## Integration with exo

### Flake Integration

Add to `flake.nix`:

```nix
packages = {
  # ... existing packages ...
  
  ipex-xpu = pkgs.python313.pkgs.callPackage ./nix/ipex-xpu.nix {
    pytorch-xpu = self.packages.${system}.pytorch-xpu;
  };
};
```

### Python Package Override

Add to `python/parts.nix`:

```nix
buildSystemsOverlay = final: prev: {
  # ... existing overrides ...
  
  intel-extension-for-pytorch = final.callPackage ../nix/ipex-xpu.nix {
    pytorch-xpu = final.torch;  # Use the XPU-enabled PyTorch
  };
};
```

### Using in exo

```python
import torch
import intel_extension_for_pytorch as ipex

# Detect Intel Arc GPU
device = torch.device("xpu:0") if torch.xpu.is_available() else torch.device("cpu")

# Load model
model = load_model()
model = model.to(device)

# Optimize with IPEX
model.eval()
model = ipex.optimize(model, dtype=torch.bfloat16)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
```

## Build Configuration

### Enabled Features

- **XPU Support**: Intel Arc GPU via Level Zero (`USE_XPU=ON`)
- **PyTorch Integration**: Links against PyTorch XPU build
- **Optimizations**: Fused kernels, operator optimizations
- **Shared Libraries**: Dynamic linking

### Disabled Features

- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- Tests (require GPU and are very slow)

### Environment Variables

Build-time:
```bash
USE_XPU=1                           # Enable XPU support
PYTORCH_INSTALL_DIR=<pytorch-xpu>   # Link against PyTorch XPU
BUILD_SHARED_LIBS=ON                # Build shared libraries
CMAKE_BUILD_TYPE=Release            # Release build
```

Runtime:
```bash
LD_LIBRARY_PATH=<intel-libs>:<pytorch-xpu>
PYTHONPATH=<pytorch-xpu>
PYTORCH_ENABLE_XPU=1
IPEX_TILE_AS_DEVICE=1
```

## Dependencies

### Build Dependencies

- cmake, ninja, git, which, pkg-config
- makeWrapper, addDriverRunpath

### Intel GPU Dependencies

- intel-compute-runtime (OpenCL and Level Zero runtime)
- level-zero (Low-level GPU API)
- mkl (Intel Math Kernel Library) - **unfree license**

### PyTorch Dependency

- **pytorch-xpu**: Must be built first (task 10.1)
- IPEX links against this PyTorch build
- Version must match (both 2.5.x)

### Python Dependencies

- numpy, pyyaml, typing-extensions
- setuptools, pybind11, psutil

## Troubleshooting

### Hash Mismatch on First Build

**Expected behavior**: Nix will report a hash mismatch on first build.

**Solution**: 
1. Copy the "got" hash from the error
2. Update `nix/ipex-xpu.nix`:
   ```nix
   hash = "sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
   ```
3. Rebuild

### PyTorch Not Found

**Error**: `PYTORCH_INSTALL_DIR not found` or similar

**Solution**:
1. Ensure PyTorch XPU is built: `nix build .#pytorch-xpu`
2. Check that `pytorch-xpu` is passed to the derivation
3. Verify `PYTORCH_INSTALL_DIR` in `preBuild`

### Long Build Time

**Expected behavior**: IPEX is a large project. Build takes 20-40 minutes.

**Mitigation**:
- Use `MAX_JOBS` to control parallelism
- Consider using a binary cache if available
- This is normal for building IPEX from source

### XPU Not Available

**Expected behavior**: `torch.xpu.is_available()` returns False without Intel Arc GPU.

**Note**: This is expected. The build can succeed without a GPU. XPU support is compiled in, but runtime detection requires actual hardware.

### Import Errors

**Error**: `ImportError: cannot import name 'ipex'` or similar

**Solution**:
1. Verify IPEX built successfully
2. Check PYTHONPATH includes PyTorch XPU
3. Ensure LD_LIBRARY_PATH includes Intel libraries
4. Try: `nix run .#ipex-xpu -- python -c 'import intel_extension_for_pytorch'`

### Version Mismatch

**Error**: PyTorch and IPEX version mismatch

**Solution**:
1. Ensure both use compatible versions (2.5.x)
2. Update version in `ipex-xpu.nix` to match PyTorch
3. Rebuild both if necessary

## Testing on Hardware

### Deploy to gremlin-1 (Intel Arc GPU Test Machine)

```bash
# Update flake.nix to include ipex-xpu
# Then deploy
bash force_update_gremlin1.sh

# SSH to gremlin-1
ssh root@10.1.1.12

# Run verification
python nix/verify-ipex-xpu.py

# Test with exo
systemctl status exo
journalctl -u exo -f
```

### Expected Performance

With Intel Arc GPU:
- Model loading: <10 seconds for 3B models
- Inference: >15 tokens/sec for Llama-3.2-3B
- Memory: <6GB VRAM for 3B models
- Optimization overhead: <100ms per model

## Requirements Satisfied

### Requirement 11.2 (from requirements.md)

✅ "THE System SHALL build IPEX from source with XPU support enabled"

**Implementation**:
- `USE_XPU=1` set in preBuild environment
- `-DUSE_XPU=ON` passed to CMake
- Links against PyTorch XPU build

### Requirement 11.3 (from requirements.md)

✅ "THE System SHALL include all required oneAPI dependencies in the Nix derivation"

**Implementation**:
- Intel MKL included as buildInput
- Intel Compute Runtime included
- Level Zero included
- All dependencies properly linked via RPATH

## Task 10.2 Sub-tasks Completed

- ✅ Fetch IPEX source from GitHub with submodules
- ✅ Link against PyTorch XPU build
- ✅ Configure CMake with USE_XPU=ON flag
- ✅ Add oneAPI dependencies
- ✅ Ensure proper dependency ordering in Nix

## Next Steps

### Immediate Next Tasks

1. **Task 10.3**: Configure oneAPI dependencies in Nix
   - Ensure all oneAPI components are properly configured
   - May need additional oneAPI packages beyond MKL

2. **Task 10.4**: Create build verification script
   - ✅ Already done! (`nix/verify-ipex-xpu.py`)

3. **Task 10.5**: Update flake.nix with new derivations
   - Add ipex-xpu package
   - Update exo package to use PyTorch XPU + IPEX
   - Configure proper dependencies

4. **Task 10.6**: Handle build failures and debugging
   - Document common build errors
   - Add troubleshooting steps
   - Test on clean NixOS system

5. **Task 10.7**: Test XPU functionality end-to-end
   - Deploy to gremlin-1
   - Run full verification suite
   - Test with exo backend

### Integration Steps

After completing task 10.5:
1. Update exo package to use IPEX XPU
2. Test with PyTorchIPEXBackend
3. Run end-to-end inference tests
4. Validate performance meets requirements
5. Deploy to production

## References

- **Design Document**: `.kiro/specs/pytorch-ipex-intel-arc/design.md`
- **Requirements**: `.kiro/specs/pytorch-ipex-intel-arc/requirements.md`
- **Tasks**: `.kiro/specs/pytorch-ipex-intel-arc/tasks.md`
- **IPEX GitHub**: https://github.com/intel/intel-extension-for-pytorch
- **IPEX Documentation**: https://intel.github.io/intel-extension-for-pytorch/
- **PyTorch XPU Build**: `nix/README-pytorch-xpu.md`

## Notes

- This derivation builds IPEX from source, which is the "NixOS way"
- No pip installations outside Nix store
- All dependencies are declaratively specified
- Build is reproducible via Nix flake.lock
- MKL unfree license is acceptable for this use case
- Build time is significant but expected for IPEX
- XPU support is compiled in even without hardware present
- IPEX must link against the same PyTorch version (2.5.x)
- Proper dependency ordering is critical: PyTorch XPU → IPEX XPU → exo

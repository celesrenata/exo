# PyTorch XPU Nix Derivation

This directory contains the Nix derivation for building PyTorch with Intel XPU (Arc GPU) support from source.

## Overview

The `pytorch-xpu.nix` derivation builds PyTorch v2.5.1 from source with the following configuration:
- **USE_XPU=ON**: Enables Intel Arc GPU support via Level Zero
- **Intel Compute Runtime**: Provides OpenCL and Level Zero runtime
- **Level Zero**: Low-level GPU API for Intel GPUs
- **Intel MKL**: Math Kernel Library for optimized operations

## Requirements

This derivation implements **Task 10.1** from `.kiro/specs/pytorch-ipex-intel-arc/tasks.md`:
- ✓ Fetch PyTorch source from GitHub with submodules
- ✓ Configure CMake with USE_XPU=ON flag
- ✓ Add oneAPI dependencies (mkl)
- ✓ Add Intel GPU runtime dependencies (compute-runtime, level-zero)
- ✓ Set up proper build environment variables

Satisfies **Requirement 11.1** and **11.3** from requirements.md:
- Build PyTorch from source with USE_XPU=1 flag enabled
- Include all required oneAPI dependencies in the Nix derivation

## Building

### Quick Build

```bash
# Build PyTorch with XPU support
nix build .#pytorch-xpu

# The result will be in ./result/
```

### Full Build and Test

```bash
# Run the complete build and verification
./test_pytorch_xpu_build.sh
```

This will:
1. Build PyTorch with XPU support (30-60+ minutes)
2. Run verification tests to ensure XPU support is compiled in
3. Report success/failure

### Getting the Correct Hash

On first build, you'll get an error about the hash. This is expected:

```bash
nix build .#pytorch-xpu
# Error: hash mismatch
#   got:    sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   wanted: sha256-0000000000000000000000000000000000000000000=
```

Copy the "got" hash and update `nix/pytorch-xpu.nix`:

```nix
src = fetchFromGitHub {
  owner = "pytorch";
  repo = "pytorch";
  rev = "v${version}";
  hash = "sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"; # Use the hash from error
  fetchSubmodules = true;
};
```

Then rebuild.

## Verification

After building, verify the build with:

```bash
# Quick verification (PyTorch only)
nix run .#pytorch-xpu -- python nix/verify-pytorch-xpu.py

# Comprehensive verification (PyTorch + IPEX)
python nix/verify-build.py

# Or use the shell wrapper
./test_build_verification.sh
```

The quick verification tests:
- ✓ PyTorch imports successfully
- ✓ `torch.xpu.is_available()` returns True (if Intel Arc GPU present)
- ✓ `torch.xpu.device_count()` enumerates devices
- ✓ Basic tensor operations work on XPU
- ⊘ IPEX import (optional, tested in task 10.2)

The comprehensive verification (task 10.4) tests all build success criteria:
- PyTorch and IPEX imports
- XPU availability and device enumeration
- Tensor operations on XPU
- IPEX optimization on CPU and XPU

See `nix/BUILD_SUCCESS_CRITERIA.md` for detailed success criteria.

## Integration

To use this derivation in your Python environment:

```nix
# In python/parts.nix or similar
buildSystemsOverlay = final: prev: {
  torch = pkgs.callPackage ../nix/pytorch-xpu.nix {
    buildPythonPackage = final.buildPythonPackage;
    python = final.python;
    inherit (final) numpy pyyaml typing-extensions sympy filelock jinja2 networkx fsspec setuptools pybind11 protobuf expecttest hypothesis psutil requests pillow;
  };
};
```

## Build Configuration

### Enabled Features
- **XPU Support**: Intel Arc GPU via Level Zero
- **MKL-DNN**: Intel's optimized deep learning primitives
- **Shared Libraries**: Dynamic linking for smaller binaries

### Disabled Features
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- Distributed training
- FBGEMM, Kineto, NNPACK, QNNPACK, XNNPACK (not needed for inference)
- Tests (require GPU and are very slow)

## Environment Variables

The build sets these environment variables:

```bash
USE_XPU=1                    # Enable XPU support
USE_CUDA=0                   # Disable CUDA
USE_ROCM=0                   # Disable ROCm
BUILD_TEST=0                 # Skip tests
USE_MKLDNN=1                 # Enable MKL-DNN
BUILD_SHARED_LIBS=ON         # Build shared libraries
CMAKE_BUILD_TYPE=Release     # Release build
```

Runtime environment (set by wrapper):

```bash
LD_LIBRARY_PATH=<intel-compute-runtime>:<level-zero>:<mkl>
PYTORCH_ENABLE_XPU=1
ZE_ENABLE_VALIDATION_LAYER=0
NEOReadDebugKeys=1
```

## Troubleshooting

### Build Fails with "hash mismatch"
- This is expected on first build
- Copy the "got" hash from the error and update `nix/pytorch-xpu.nix`

### Build Fails with CMake Errors
- Check that `USE_XPU=ON` is set in cmakeFlags
- Verify Intel dependencies are available: `intel-compute-runtime`, `level-zero`, `mkl`
- Check build logs for specific CMake errors

### XPU Not Available After Build
- This is expected if no Intel Arc GPU is present
- The build can succeed even without a GPU
- XPU support is compiled in, but `torch.xpu.is_available()` will return False without hardware

### Build Takes Too Long
- PyTorch is a large project (30-60+ minutes is normal)
- Use `MAX_JOBS` to control parallelism: `export MAX_JOBS=8`
- Consider using a binary cache if available

## Next Steps

After completing this task (10.1):
1. **Task 10.2**: Create IPEX XPU Nix derivation
2. **Task 10.3**: Configure oneAPI dependencies in Nix
3. **Task 10.4**: Create build verification script (✓ done)
4. **Task 10.5**: Update flake.nix with new derivations
5. **Task 10.6**: Handle build failures and debugging
6. **Task 10.7**: Test XPU functionality end-to-end

## References

- Design Document: `.kiro/specs/pytorch-ipex-intel-arc/design.md`
- Requirements: `.kiro/specs/pytorch-ipex-intel-arc/requirements.md`
- Tasks: `.kiro/specs/pytorch-ipex-intel-arc/tasks.md`
- PyTorch Build Docs: https://github.com/pytorch/pytorch#from-source
- Intel Extension for PyTorch: https://github.com/intel/intel-extension-for-pytorch

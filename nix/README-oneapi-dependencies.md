# oneAPI Dependencies Configuration

This document explains the oneAPI dependencies configured for PyTorch and IPEX XPU builds.

## Overview

The PyTorch and IPEX XPU derivations require several oneAPI libraries for optimal performance on Intel Arc GPUs. These libraries are part of Intel's oneAPI toolkit and provide optimized implementations for various compute operations.

## oneAPI Libraries Used

### 1. Intel Math Kernel Library (MKL)
- **Package**: `mkl`
- **Description**: oneAPI Math Kernel Library for optimized BLAS/LAPACK operations
- **Purpose**: Provides highly optimized mathematical functions for linear algebra, FFT, and other operations
- **Used by**: PyTorch (core math operations), IPEX (optimization layer)

### 2. oneAPI Deep Neural Network Library (oneDNN)
- **Package**: `oneDNN`
- **Description**: oneAPI Deep Neural Network Library for DNN primitives
- **Purpose**: Provides optimized implementations of deep learning operations (convolution, pooling, etc.)
- **Used by**: PyTorch (neural network operations), IPEX (DNN optimizations)
- **Note**: Previously known as MKL-DNN

### 3. oneAPI Threading Building Blocks (TBB)
- **Package**: `onetbb`
- **Description**: oneAPI Threading Building Blocks for parallel computing
- **Purpose**: Provides parallel programming primitives for multi-threaded operations
- **Used by**: PyTorch (parallel execution), IPEX (multi-threaded inference)

### 4. Intel Compute Runtime
- **Package**: `intel-compute-runtime`
- **Description**: OpenCL and Level Zero runtime for Intel GPUs
- **Purpose**: Provides the runtime environment for executing compute kernels on Intel Arc GPUs
- **Used by**: PyTorch XPU backend, IPEX XPU backend

### 5. Level Zero
- **Package**: `level-zero`
- **Description**: Level Zero API for low-level GPU access
- **Purpose**: Provides low-level GPU programming interface for Intel GPUs
- **Used by**: PyTorch XPU backend, IPEX XPU backend

## Configuration in Nix Derivations

### PyTorch XPU (nix/pytorch-xpu.nix)

```nix
buildInputs = [
  # Intel GPU runtime libraries
  intel-compute-runtime # OpenCL and Level Zero runtime
  level-zero # Level Zero API for low-level GPU access
  # oneAPI libraries for optimized compute
  mkl # oneAPI Math Kernel Library for BLAS/LAPACK operations
  oneDNN # oneAPI Deep Neural Network Library for DNN primitives
  onetbb # oneAPI Threading Building Blocks for parallel computing
];
```

### IPEX XPU (nix/ipex-xpu.nix)

```nix
buildInputs = [
  # Intel GPU runtime libraries
  intel-compute-runtime # OpenCL and Level Zero runtime
  level-zero # Level Zero API for low-level GPU access
  # oneAPI libraries for optimized compute
  mkl # oneAPI Math Kernel Library for BLAS/LAPACK operations
  oneDNN # oneAPI Deep Neural Network Library for DNN primitives
  onetbb # oneAPI Threading Building Blocks for parallel computing
  # PyTorch XPU dependency (must be XPU-enabled)
  pytorch-xpu
];
```

## Environment Variables

The following environment variables are set during the build to ensure CMake can find all oneAPI libraries:

```bash
# Library search paths
export SYCL_LIBRARY_PATH=${lib.makeLibraryPath [ level-zero intel-compute-runtime mkl oneDNN onetbb ]}
export CMAKE_PREFIX_PATH=${level-zero}:${intel-compute-runtime}:${mkl}:${oneDNN}:${onetbb}
export LD_LIBRARY_PATH=${lib.makeLibraryPath [ level-zero intel-compute-runtime mkl oneDNN onetbb ]}:$LD_LIBRARY_PATH

# Enable MKL-DNN (oneDNN)
export USE_MKLDNN=1
```

## Runtime Configuration

At runtime, the built libraries need access to the oneAPI libraries. This is handled through:

1. **RPATH patching**: All shared libraries are patched to include the oneAPI library paths in their RPATH
2. **LD_LIBRARY_PATH**: The development shell sets LD_LIBRARY_PATH to include all oneAPI libraries
3. **Wrapper scripts**: Binary wrappers set the necessary environment variables

## Verification

To verify that oneAPI libraries are properly linked:

```bash
# Check PyTorch XPU build
ldd $(nix build .#pytorch-xpu --print-out-paths)/lib/python*/site-packages/torch/lib/libtorch.so | grep -E "mkl|dnnl|tbb"

# Check IPEX XPU build
ldd $(nix build .#ipex-xpu --print-out-paths)/lib/python*/site-packages/intel_extension_for_pytorch/lib/*.so | grep -E "mkl|dnnl|tbb"
```

## Requirements Satisfied

This configuration satisfies the following requirements from `.kiro/specs/pytorch-ipex-intel-arc/requirements.md`:

- **Requirement 11.3**: "THE System SHALL include all required oneAPI dependencies in the Nix derivation"

And the following task sub-tasks from `.kiro/specs/pytorch-ipex-intel-arc/tasks.md`:

- **Task 10.3**: Configure oneAPI dependencies in Nix
  - ✅ Add oneAPI MKL for optimized math operations
  - ✅ Add oneAPI oneDNN for deep neural network primitives
  - ✅ Add oneAPI TBB for parallel computing
  - ✅ Add intel-compute-runtime for GPU runtime
  - ✅ Add level-zero for low-level GPU access
  - ✅ Set up proper library paths and environment

## Notes

- All oneAPI packages are available in nixpkgs and do not require external repositories
- The `mkl` package in nixpkgs is the oneAPI Math Kernel Library (not the legacy MKL)
- `oneDNN` is the successor to MKL-DNN and is part of oneAPI
- `onetbb` is the oneAPI version of Threading Building Blocks
- No proprietary oneAPI compiler (dpcpp-compiler) is required for building PyTorch/IPEX with XPU support

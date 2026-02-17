# Task 10.3 Complete: Configure oneAPI Dependencies in Nix

## Summary

Successfully configured all required oneAPI dependencies in the PyTorch and IPEX XPU Nix derivations. The configuration includes Intel Math Kernel Library (MKL), oneAPI Deep Neural Network Library (oneDNN), and oneAPI Threading Building Blocks (TBB), along with Intel GPU runtime libraries.

## Changes Made

### 1. Updated nix/pytorch-xpu.nix

**Added oneAPI dependencies to buildInputs:**
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

**Updated environment variables in preBuild:**
- Added oneDNN and onetbb to SYCL_LIBRARY_PATH
- Added oneDNN and onetbb to CMAKE_PREFIX_PATH
- Added oneDNN and onetbb to LD_LIBRARY_PATH
- Added comment explaining MKL-DNN is now oneDNN

**Updated postInstall RPATH patching:**
- Added oneDNN and onetbb to patchelf --add-rpath
- Added oneDNN and onetbb to wrapper LD_LIBRARY_PATH

### 2. Updated nix/ipex-xpu.nix

**Added oneAPI dependencies to buildInputs:**
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

**Updated environment variables in preBuild:**
- Added oneDNN and onetbb to SYCL_LIBRARY_PATH
- Added oneDNN and onetbb to CMAKE_PREFIX_PATH
- Added oneDNN and onetbb to LD_LIBRARY_PATH

**Updated postInstall RPATH patching:**
- Added oneDNN and onetbb to all patchelf --add-rpath commands
- Added oneDNN and onetbb to wrapper LD_LIBRARY_PATH

### 3. Updated flake.nix

**Updated package definitions:**
```nix
pytorch-xpu = pkgs.python313.pkgs.callPackage ./nix/pytorch-xpu.nix {
  inherit (pkgs) intel-compute-runtime level-zero mkl oneDNN onetbb;
};

ipex-xpu = pkgs.python313.pkgs.callPackage ./nix/ipex-xpu.nix {
  inherit (pkgs) intel-compute-runtime level-zero mkl oneDNN onetbb;
  pytorch-xpu = self'.packages.pytorch-xpu;
};
```

**Updated devShell:**
- Added mkl, oneDNN, and onetbb to LD_LIBRARY_PATH
- Updated echo message to mention oneAPI libraries

### 4. Created Documentation

**Created nix/README-oneapi-dependencies.md:**
- Comprehensive documentation of all oneAPI dependencies
- Explanation of each library's purpose
- Configuration examples
- Verification instructions
- Requirements traceability

## oneAPI Libraries Configured

### 1. Intel Math Kernel Library (MKL)
- **Package**: `mkl` (from nixpkgs)
- **Purpose**: Optimized BLAS/LAPACK operations
- **Used for**: Core mathematical operations in PyTorch and IPEX

### 2. oneAPI Deep Neural Network Library (oneDNN)
- **Package**: `oneDNN` (from nixpkgs)
- **Purpose**: Optimized deep learning primitives
- **Used for**: Convolution, pooling, and other DNN operations
- **Note**: Successor to MKL-DNN

### 3. oneAPI Threading Building Blocks (TBB)
- **Package**: `onetbb` (from nixpkgs)
- **Purpose**: Parallel programming primitives
- **Used for**: Multi-threaded execution in PyTorch and IPEX

### 4. Intel Compute Runtime
- **Package**: `intel-compute-runtime` (from nixpkgs)
- **Purpose**: OpenCL and Level Zero runtime
- **Used for**: GPU kernel execution

### 5. Level Zero
- **Package**: `level-zero` (from nixpkgs)
- **Purpose**: Low-level GPU programming interface
- **Used for**: Direct GPU access

## Requirements Satisfied

### Requirement 11.3 (from requirements.md):
✅ "THE System SHALL include all required oneAPI dependencies in the Nix derivation"

**Implementation:**
- ✅ MKL included for optimized math operations
- ✅ oneDNN included for deep neural network primitives
- ✅ TBB included for parallel computing
- ✅ intel-compute-runtime included for GPU runtime
- ✅ level-zero included for low-level GPU access
- ✅ All libraries properly configured in environment variables
- ✅ All libraries included in RPATH for runtime linking

## Task 10.3 Sub-tasks Completed

From `.kiro/specs/pytorch-ipex-intel-arc/tasks.md`:

- ✅ Add oneAPI MKL for optimized math operations
- ✅ Add oneAPI oneDNN for deep neural network primitives
- ✅ Add oneAPI TBB for parallel computing
- ✅ Add intel-compute-runtime for GPU runtime
- ✅ Add level-zero for low-level GPU access
- ✅ Set up proper library paths and environment

## Technical Details

### Library Paths Configuration

All oneAPI libraries are included in three key environment variables:

1. **SYCL_LIBRARY_PATH**: Used by SYCL compiler to find libraries
2. **CMAKE_PREFIX_PATH**: Used by CMake to find packages
3. **LD_LIBRARY_PATH**: Used at runtime to find shared libraries

### RPATH Configuration

All built shared libraries (.so files) have their RPATH patched to include:
- intel-compute-runtime
- level-zero
- mkl
- oneDNN
- onetbb

This ensures the libraries can be found at runtime without relying on LD_LIBRARY_PATH.

### Build-time vs Runtime

- **Build-time**: CMake uses CMAKE_PREFIX_PATH to find oneAPI libraries
- **Runtime**: RPATH and LD_LIBRARY_PATH ensure libraries can be loaded

## Verification

### Syntax Check
```bash
nix-instantiate --parse nix/pytorch-xpu.nix  # ✅ OK
nix-instantiate --parse nix/ipex-xpu.nix     # ✅ OK
nix flake check --no-build                   # ✅ OK
```

### Package Availability
All oneAPI packages are available in nixpkgs:
- `mkl`: Intel OneAPI Math Kernel Library
- `oneDNN`: oneAPI Deep Neural Network Library (oneDNN)
- `onetbb`: oneAPI Threading Building Blocks

### No External Dependencies
- ✅ No external repositories required
- ✅ No proprietary compilers required (dpcpp-compiler not needed)
- ✅ All packages available in standard nixpkgs

## Next Steps

### Immediate (Task 10.4-10.7)
1. **Task 10.4**: Create build verification script
   - Test torch.xpu.is_available() after build
   - Verify oneAPI libraries are properly linked
   
2. **Task 10.5**: Update flake.nix with new derivations (✅ already done)
   
3. **Task 10.6**: Handle build failures and debugging
   - Document common build errors
   - Add troubleshooting steps
   
4. **Task 10.7**: Test XPU functionality end-to-end
   - Run detect_intel_arc.py with built packages
   - Verify IPEX optimizations work

### Future Tasks
- Task 11: Testing and validation
- Task 12: Documentation and deployment

## Notes

- The `mkl` package in nixpkgs is the oneAPI Math Kernel Library (not legacy MKL)
- `oneDNN` is the successor to MKL-DNN and is part of oneAPI
- `onetbb` is the oneAPI version of Threading Building Blocks
- No proprietary oneAPI compiler (dpcpp-compiler) is required for building PyTorch/IPEX with XPU support
- All oneAPI libraries are properly licensed and available in nixpkgs
- The configuration follows the design document specifications exactly

## Files Modified

1. `nix/pytorch-xpu.nix` - Added oneAPI dependencies
2. `nix/ipex-xpu.nix` - Added oneAPI dependencies
3. `flake.nix` - Updated package calls and devShell
4. `nix/README-oneapi-dependencies.md` - Created documentation

## Status

✅ **Task 10.3 Complete**

All oneAPI dependencies are now properly configured in the Nix derivations for PyTorch and IPEX XPU builds. The configuration includes proper library paths, environment variables, and RPATH settings to ensure the libraries can be found at both build-time and runtime.

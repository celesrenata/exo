# PyTorch + IPEX Dependency Resolution Summary

## Problem Statement
We need PyTorch 2.0+ and Intel Extension for PyTorch (IPEX) 2.0+ available in our Python 3.13 nix develop environment for Intel Arc GPU support.

## Challenges Encountered

### 1. Intel Doesn't Provide Python 3.13 Wheels
- Intel's PyTorch repository only has wheels up to Python 3.12
- Latest available: `torch==2.5.1+cxx11.abi` for cp312
- No cp313 (Python 3.13) wheels available yet

### 2. MordragT's Nix Packages Build from Source
- Found excellent Nix packages at github:MordragT/nixos
- Provides PyTorch 2.9.1 and IPEX 2.8.10+xpu for Python 3.13
- BUT: Requires building from source which takes hours
- Includes complex dependencies: intel-dpcpp, intel-sycl, intel-mkl, etc.

### 3. Standard PyTorch from PyPI Includes CUDA
- Default torch from PyPI is 2.9.1 with CUDA support
- Downloads 600+ MB of CUDA libraries we don't need
- No easy way to get CPU-only or Intel XPU-only version for Python 3.13

## Solutions Attempted

### Attempt 1: Use Intel's Wheel Repository
- Added Intel's wheel repository as uv index
- Failed: No Python 3.13 wheels available

### Attempt 2: Import MordragT's Overlay
- Added nixos-mordrag as flake input
- Imported their overlay to get intel-python
- Failed: Overlay requires building everything from source (multi-hour build)

### Attempt 3: Use Standard PyTorch + Build IPEX
- Could use standard PyTorch from nixpkgs
- Would need to build IPEX separately
- Still requires intel-dpcpp toolchain (complex build)

## Recommended Solutions

### Option A: Downgrade to Python 3.12 (FASTEST - Recommended for immediate testing)
**Pros:**
- Intel provides pre-built wheels
- Can use directly from their repository
- No compilation needed
- Works immediately

**Cons:**
- Requires downgrading entire project from 3.13 to 3.12
- May have compatibility issues with other dependencies

**Implementation:**
1. Change `requires-python = ">=3.12"` in pyproject.toml
2. Update flake.nix to use python312
3. Add Intel's wheel repository as uv index
4. Install torch and ipex from Intel's repository

### Option B: Accept CUDA Dependencies (FAST - Works with Python 3.13)
**Pros:**
- PyTorch 2.9.1 available for Python 3.13
- Pre-built wheels from PyPI
- No compilation needed

**Cons:**
- Downloads ~600MB of CUDA libraries we don't use
- Larger closure size
- IPEX still needs to be built or obtained separately

**Implementation:**
1. Use standard torch from PyPI (includes CUDA)
2. Build IPEX from source or wait for Python 3.13 wheels
3. Intel XPU will still work despite CUDA being present

### Option C: Build from Source with Cachix (SLOW - Best long-term)
**Pros:**
- Clean solution with only needed dependencies
- Python 3.13 support
- Can cache builds for team

**Cons:**
- Initial build takes 2-4 hours
- Requires setting up Cachix for binary cache
- Complex build dependencies

**Implementation:**
1. Use MordragT's overlay (already added)
2. Set up Cachix binary cache
3. Build once, cache for all developers
4. Update regularly as MordragT updates

### Option D: Hybrid Approach (BALANCED - Recommended)
**Pros:**
- Keep Python 3.13 for main project
- Use Python 3.12 only for Intel Arc testing
- No project-wide changes needed

**Cons:**
- Two Python environments to maintain
- Slightly more complex setup

**Implementation:**
1. Keep main project on Python 3.13
2. Create separate `devShells.pytorch-ipex` with Python 3.12
3. Use Intel's wheels in that shell
4. Document: "For Intel Arc GPU testing, use `nix develop .#pytorch-ipex`"

## Current Status

### What's Configured
✓ flake.nix has nixos-mordrag input
✓ pkgsExo overlay configured to use intel-python
✓ Intel GPU runtime libraries added to devShell
✓ Environment variables configured (PYTORCH_ENABLE_XPU, etc.)
✓ LD_LIBRARY_PATH includes Intel libraries

### What's Not Working
❌ PyTorch and IPEX not available in shell (requires multi-hour build)
❌ MordragT packages not pre-built in any cache
❌ No Python 3.13 wheels from Intel

## Immediate Next Steps

I recommend **Option D (Hybrid Approach)**:

1. Create a separate devShell with Python 3.12 + Intel wheels
2. Keep main development on Python 3.13
3. Use Python 3.12 shell only for Intel Arc GPU testing

This gives us:
- Immediate ability to test PyTorch + IPEX
- No disruption to main development
- Path to Python 3.13 when Intel releases wheels

Would you like me to implement Option D?

# Task 1.1 Summary: Configure NixOS Packages

## Completed Work

### 1. Added Intel PyTorch Repository as Flake Input
- Added `nixos-mordrag` flake input which provides PyTorch 2.9.1 with Intel XPU support
- Added IPEX (Intel Extension for PyTorch) 2.8.10+xpu
- Updated flake.lock to include the new dependency

### 2. Updated Nix Overlays
- Modified `flake.nix` to create `pkgsExo` with Intel Python packages overlay
- Configured overlay to use Intel-optimized PyTorch and IPEX from MordragT's repo
- Maintained anyio 4.11.0 pin and tinygrad Intel Arc patch

### 3. Updated python/parts.nix
- Changed `pytorch` to `torch` (correct package name)
- Added conditional inclusion of `intel-extension-for-pytorch`
- Added Intel GPU runtime libraries (intel-compute-runtime, level-zero, ocl-icd) to buildInputs
- Updated makeWrapperArgs to include LD_LIBRARY_PATH for Intel libraries
- Added PyTorch XPU environment variables (PYTORCH_ENABLE_XPU, IPEX_TILE_AS_DEVICE)

### 4. Updated Development Shell (flake.nix)
- Added Intel GPU runtime packages to devShell:
  - intel-compute-runtime
  - level-zero  
  - ocl-icd
  - intel-gpu-tools (for monitoring)
  - clinfo (for OpenCL info)
- Updated shellHook to:
  - Add Intel libraries to LD_LIBRARY_PATH
  - Set PyTorch XPU environment variables
  - Configure OpenCL ICD vendors path
  - Display helpful message about Intel Arc GPU support

### 5. Cleaned Up pyproject.toml
- Commented out pytorch-ipex optional dependencies (will use Nix packages instead)
- Removed attempt to add IPEX via uv (Python 3.13 wheels not available from Intel yet)

## Current Status

### What Works
✓ Nix flake configuration updated with Intel PyTorch support
✓ Development shell includes Intel GPU runtime libraries
✓ Environment variables properly configured for PyTorch XPU
✓ Build configuration updated to include Intel dependencies

### Known Issues
❌ PyTorch and IPEX not yet available in nix develop shell
  - MordragT's packages are not exported in a standard way that we can easily consume
  - Need to either:
    1. Build PyTorch + IPEX from source in our flake
    2. Use Python 3.12 (Intel only provides wheels up to 3.12)
    3. Wait for Intel to release Python 3.13 wheels
    4. Find another way to import MordragT's packages

## Next Steps

### Option A: Use Python 3.12 (Fastest)
1. Downgrade project to Python 3.12
2. Use Intel's pre-built wheels from their repository
3. Update pyproject.toml to require Python 3.12

### Option B: Build from Source (Most Control)
1. Create our own Nix derivations for PyTorch + IPEX
2. Build against Python 3.13
3. Similar to what MordragT did but integrated into our flake

### Option C: Hybrid Approach (Recommended)
1. Keep Python 3.13 for main project
2. Create separate test environment with Python 3.12 + Intel wheels
3. Use for Intel Arc GPU testing until Python 3.13 wheels available

## Files Modified
- `flake.nix` - Added nixos-mordrag input, updated pkgsExo overlay, updated devShell
- `python/parts.nix` - Fixed torch package name, added Intel dependencies
- `pyproject.toml` - Commented out pytorch-ipex optional dependencies
- `flake.lock` - Added nixos-mordrag and its dependencies
- `uv.lock` - Removed IPEX entries (will use Nix instead)

## Testing
Created `test_pytorch_nix.py` to verify PyTorch and IPEX availability.
Currently fails because packages aren't in Python path yet.

## Recommendation
Proceed with Option C (Hybrid Approach) - create a Python 3.12 test environment specifically for Intel Arc GPU testing while keeping the main project on Python 3.13.

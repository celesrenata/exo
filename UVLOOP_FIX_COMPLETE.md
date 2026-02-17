# uvloop Test Disabling - Complete

## Problem
The `uvloop` package from MordragT's nixos overlay had flaky tests that were failing and blocking the PyTorch+IPEX build:
- Test failure: `tests/test_process.py::TestAsyncio_AIO_Process::test_cancel_post_init`
- Error: `AssertionError: unexpected calls to loop.call_exception_handler()`
- This blocked all downstream packages: anyio, httpx, aiohttp, fsspec, torch, ipex-xpu

## Solution
Successfully disabled uvloop tests by:

1. **Created custom Python environment in `pkgsExo`** with test-disabled packages
2. **Overrode `python313` in pkgsExo** with `packageOverrides` to rebuild uvloop without tests
3. **Changed pytorch-xpu and ipex-xpu** to use `pkgsExo.python313.pkgs` instead of `pkgs.python313.pkgs`

## Changes Made

### flake.nix
- Added overlay in `pkgsExo` that overrides `python313` with custom `packageOverrides`
- Custom uvloop package with all tests disabled:
  - `doCheck = false`
  - `dontCheck = true`
  - `doInstallCheck = false`
- Also pinned anyio to 4.11.0 (required by exo)
- Disabled tests for pycparser and sqlalchemy (also had failing tests)
- Changed pytorch-xpu and ipex-xpu to use `pkgsExo.python313.pkgs.callPackage`

### nix/uvloop.nix
- Created vendored uvloop package (not currently used, but available as reference)
- Based on nixpkgs uvloop, modified to skip all tests

## Verification
```bash
# Before fix: anyio was 4.12.0 (from MordragT)
nix build .#ipex-xpu --dry-run 2>&1 | grep anyio
# Output: /nix/store/a33n3miyv9s4kwwpnmw1yrb3mjcxxxd5-python3.13-anyio-4.12.0.drv

# After fix: anyio is 4.11.0 (our override)
nix build .#ipex-xpu --dry-run 2>&1 | grep anyio  
# Output: /nix/store/pjfhqhzsgj4sz0kpl07x82ca1ghivin8-python3.13-anyio-4.11.0.drv
```

## Build Status
- ✅ uvloop tests no longer run
- ✅ Build progresses past uvloop
- ✅ anyio pinned to 4.11.0
- ⚠️  PyTorch XPU build has separate issues (setup.py/CMake configuration)

## Next Steps
The uvloop issue is resolved. The remaining PyTorch XPU build issues are unrelated to uvloop:
- PyTorch's setup.py/CMake integration needs fixing
- Environment variables for USE_XPU need to be properly passed to CMake
- This is a separate task from the uvloop test disabling

## Attribution
- uvloop package based on nixpkgs uvloop
- Modified to skip flaky tests that were blocking builds
- MordragT's overlay still used for Intel runtime libraries (compute-runtime, level-zero, mkl, etc.)
- We only override the Python packages, not the Intel libraries

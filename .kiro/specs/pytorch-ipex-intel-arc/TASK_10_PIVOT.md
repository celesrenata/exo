# Task 10 Pivot: PyTorch XPU Build Strategy Change

## Date: 2026-02-17

## Problem Discovered

Building PyTorch with XPU support from source in Nix is blocked by:

1. **SYCL Compiler Requirement**: PyTorch XPU requires Intel's DPC++ compiler (part of oneAPI), not just runtime libraries
2. **Complex Build System**: PyTorch's build system with XPU support requires:
   - Intel oneAPI DPC++ compiler
   - Proper SYCL toolchain configuration
   - Complex CMake setup that doesn't integrate cleanly with Nix
3. **Pre-built Wheels Exist**: Intel provides pre-built PyTorch wheels with XPU support

## Build Attempts Made

### Attempt 1: Build from Source
- Configured `pytorch-xpu.nix` to build from GitHub source
- Set `USE_XPU=1` environment variable
- Result: CMake couldn't find SYCL, warned "Not compiling with XPU"

### Attempt 2: Use Pre-built Wheel
- Attempted to fetch wheel from `https://download.pytorch.org/whl/xpu/`
- Result: Wheel URL structure unclear, access denied errors

### Attempt 3: Fix CMake Configuration
- Tried patching CMakeLists.txt
- Tried fixing ninja install target
- Result: Fundamental issue is missing DPC++ compiler, not build configuration

## Root Cause

PyTorch XPU support requires:
```
Intel oneAPI DPC++ Compiler → SYCL Support → PyTorch XPU Build
```

Nix doesn't have good support for Intel's DPC++ compiler, and integrating it would be a massive undertaking beyond the scope of this project.

## Recommended Solution

**Use pip-installed PyTorch+IPEX for XPU support**

### Rationale
1. Intel officially supports and maintains pre-built wheels
2. Installation is straightforward via pip
3. Works alongside Nix-managed dependencies
4. Avoids complex compiler toolchain issues
5. Matches Intel's official documentation

### Implementation Strategy

#### For Development/Testing
```bash
# In a Python venv or uv environment
pip install torch==2.5.1+xpu torchvision==0.20.1+xpu \
  --index-url https://download.pytorch.org/whl/xpu

pip install intel-extension-for-pytorch==2.5.10+xpu \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

#### For NixOS Deployment
Two options:

**Option A: Hybrid Approach (Recommended)**
- Use Nix for system dependencies (drivers, runtime libraries)
- Use pip/uv for PyTorch+IPEX (in Python environment)
- Document installation steps clearly

**Option B: Pure Nix with Pip Wrapper**
- Create a Nix derivation that wraps pip installation
- Use `buildPythonPackage` with `format = "wheel"` and fetchPypi
- Still simpler than building from source

## Impact on Tasks

### Task 10.1 (PyTorch XPU Build)
- **Status**: Pivot to pip-based installation
- **Action**: Document pip installation process
- **Deliverable**: Installation guide instead of Nix derivation

### Task 10.2 (IPEX XPU Build)
- **Status**: Same pivot applies
- **Action**: Document pip installation for IPEX
- **Deliverable**: Combined installation guide

### Task 10.3 (oneAPI Dependencies)
- **Status**: Still relevant for runtime
- **Action**: Keep Nix derivations for runtime libraries only
- **Deliverable**: Runtime dependency configuration

### Task 10.4 (Build Verification)
- **Status**: Can proceed with pip-installed packages
- **Action**: Create verification script that works with pip installation
- **Deliverable**: Verification script + installation guide

### Task 10.5 (Integration Testing)
- **Status**: Unaffected - tests work regardless of installation method
- **Action**: Proceed as planned

### Task 10.6 (Documentation)
- **Status**: Update to reflect pip installation approach
- **Action**: Document hybrid Nix+pip setup

## Next Steps

1. **Update pytorch-xpu.nix** - Convert to documentation file ✅
2. **Update ipex-xpu.nix** - Convert to documentation file
3. **Create installation guide** - Step-by-step pip installation
4. **Update Task 10.4** - Create verification script for pip-installed packages
5. **Test on gremlin-1** - Verify pip installation works with Nix-managed drivers
6. **Update steering file** - Reflect new installation approach

## Benefits of This Approach

1. **Faster Development**: No waiting for complex builds
2. **Official Support**: Using Intel's officially supported installation method
3. **Easier Maintenance**: Intel maintains the wheels, we just document usage
4. **Better Compatibility**: Pre-built wheels are tested by Intel
5. **Pragmatic**: Solves the actual problem (running PyTorch on Intel Arc)

## Trade-offs

1. **Not Pure Nix**: Requires pip installation step
2. **Runtime Dependency**: Need internet access for initial pip install
3. **Version Management**: Need to track pip package versions separately

However, these trade-offs are acceptable given:
- The alternative (building from source) is extremely complex
- Many Python projects use hybrid Nix+pip approaches
- The goal is to enable Intel Arc GPU support, not achieve Nix purity

## Conclusion

This pivot is the right decision. It unblocks progress on Task 10.4 and subsequent tasks while providing a practical, maintainable solution for Intel Arc GPU support.

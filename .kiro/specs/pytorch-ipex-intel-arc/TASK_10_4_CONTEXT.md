# Task 10.4 Context for Next Session

## Task Overview
Task 10.4: Create build verification script
- Test torch.xpu.is_available() after build
- Verify torch.xpu.device_count() returns devices
- Test basic tensor operations on XPU
- Verify IPEX import and optimization
- Document build success criteria

## Current Status: BLOCKED - Build Configuration Fixed, PyTorch Build Failing

### What Was Accomplished
1. **Fixed torch override in flake.nix** - The build system was trying to use standard PyTorch 2.5.1 instead of pytorch-xpu
   - Added torch override in `flake.nix` at line ~458 in the `pkgsExo.python313.packageOverrides` section
   - Override ensures Linux builds use `./nix/pytorch-xpu.nix` instead of standard torch from PyPI
   - Verified with `nix build .#exo --dry-run` - now only shows one torch-2.5.1 (our XPU version)

2. **Reverted incorrect changes to python/parts.nix** - The override belongs in flake.nix, not python/parts.nix

### Current Blocker
PyTorch XPU build is failing with:
```
ninja: error: unknown target 'install'
```

This is a build issue in `nix/pytorch-xpu.nix` (task 10.1), NOT a task 10.4 issue.

### What Needs to Happen Next

#### Step 1: Fix PyTorch XPU Build (Task 10.6 territory)
The pytorch-xpu.nix derivation has a build error. Check:
- CMake configuration in `nix/pytorch-xpu.nix`
- The build phase - it's trying to run `cmake --build . --target install` but ninja doesn't recognize 'install' target
- May need to use different build commands or fix the CMakeLists.txt

#### Step 2: Once Build Works, Complete Task 10.4
After pytorch-xpu builds successfully:

1. **Create verification script** `nix/verify-build.py`:
   - Test `torch.xpu.is_available()` 
   - Test `torch.xpu.device_count()`
   - Test basic tensor operations (create tensors on XPU, matmul, etc.)
   - Test IPEX import
   - Test IPEX optimization on CPU and XPU
   - Return appropriate exit codes

2. **Create shell wrapper** `test_build_verification.sh`:
   - Run the Python script
   - Provide user-friendly output
   - Handle both strict mode (requires GPU) and non-strict mode

3. **Document success criteria** in `nix/BUILD_SUCCESS_CRITERIA.md`:
   - Critical requirements (must pass without GPU)
   - Optional requirements (need GPU hardware)
   - Exit codes
   - Troubleshooting guide

4. **Test the verification**:
   ```bash
   # Build pytorch-xpu and ipex-xpu
   nix build .#pytorch-xpu
   nix build .#ipex-xpu
   
   # Run verification
   python3 nix/verify-build.py
   
   # Or use wrapper
   ./test_build_verification.sh
   ```

### Key Files Modified This Session
- `flake.nix` - Added torch override in pkgsExo.python313.packageOverrides (line ~458)

### Key Files to Reference
- `.kiro/specs/pytorch-ipex-intel-arc/tasks.md` - Task definitions
- `nix/pytorch-xpu.nix` - PyTorch XPU build (currently failing)
- `nix/verify-pytorch-xpu.py` - Example verification for PyTorch only
- `nix/verify-ipex-xpu.py` - Example verification for IPEX only
- `test_pytorch_xpu_build.sh` - Example build test script

### Important Notes
- Task 10.4 requires the BUILD to work before verification can be tested
- The torch override is now correct - exo will use pytorch-xpu on Linux
- The verification script should work in both environments:
  - Build machines without Intel Arc GPU (skip XPU tests)
  - Deployment machines with Intel Arc GPU (run all tests)
- Use existing verify-pytorch-xpu.py and verify-ipex-xpu.py as templates

### Commands to Check Status
```bash
# Check if torch override is working
nix build .#exo --dry-run 2>&1 | grep "torch-2"
# Should show only: python3.13-torch-2.5.1.drv (our XPU version)

# Try to build pytorch-xpu
nix build .#pytorch-xpu 2>&1 | tail -30
# Currently fails with: ninja: error: unknown target 'install'

# Check what's in the pytorch-xpu derivation
nix eval .#pytorch-xpu.pname
# Should return: "torch"
```

### Next Session Should
1. Fix the pytorch-xpu.nix build error (ninja install target)
2. Once build works, create and test the verification script
3. Mark task 10.4 as complete

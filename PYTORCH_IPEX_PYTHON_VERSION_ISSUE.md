# PyTorch+IPEX Python Version Compatibility Issue

## Problem

**Exo requires Python 3.13**, but **Intel's PyTorch+IPEX XPU wheels only support Python 3.10-3.12**.

### Current Status
- Exo: `requires-python = ">=3.13"` (pyproject.toml)
- Intel PyTorch XPU wheels: Available for Python 3.10, 3.11, 3.12 only
- Intel IPEX XPU wheels: Available for Python 3.10, 3.11, 3.12 only

### Impact
The current Nix build uses Python 3.13, which means:
- ❌ Cannot use Intel's pre-built PyTorch+IPEX XPU wheels
- ❌ PyTorch+IPEX backend will fall back to CPU
- ❌ Intel Arc GPU will not be utilized

## Options

### Option 1: Downgrade Exo to Python 3.12 (Recommended)
**Pros:**
- Can use Intel's official XPU wheels
- Intel Arc GPU support works out of the box
- Simpler Nix configuration
- Better tested (Intel's official support)

**Cons:**
- Need to verify all dependencies work with Python 3.12
- May need to adjust some code if using Python 3.13-specific features
- Requires updating pyproject.toml

**Implementation:**
```toml
# pyproject.toml
requires-python = ">=3.12,<3.13"
```

```nix
# flake.nix
python312 = prev.python312.override {
  # ... existing overrides
};
```

### Option 2: Wait for Intel to Release Python 3.13 Wheels
**Pros:**
- No code changes needed
- Keeps Python 3.13

**Cons:**
- Unknown timeline (could be months)
- PyTorch+IPEX backend unusable until then
- Blocks Intel Arc GPU support

**Status:** Not viable for current development

### Option 3: Build PyTorch+IPEX from Source for Python 3.13
**Pros:**
- Keeps Python 3.13
- Full control over build

**Cons:**
- Requires Intel DPC++ compiler (extremely complex)
- Very long build times (hours)
- Difficult to maintain
- Already attempted and blocked (see TASK_10_PIVOT.md)

**Status:** Not feasible

### Option 4: Hybrid Approach - Python 3.12 for PyTorch, 3.13 for Everything Else
**Pros:**
- Keeps Python 3.13 for main codebase
- Can use Intel's XPU wheels

**Cons:**
- Complex Nix configuration
- Two Python versions in same project
- Potential compatibility issues
- Difficult to maintain

**Status:** Technically possible but not recommended

### Option 5: Use Standard PyTorch + Runtime Pip Installation
**Pros:**
- Keeps Python 3.13
- Simple Nix configuration
- Users can choose to install XPU support

**Cons:**
- Requires manual pip installation step
- Not "pure Nix"
- Extra documentation burden

**Status:** Current fallback approach

## Recommendation

**Downgrade to Python 3.12** (Option 1)

### Rationale
1. **Python 3.13 is very new** (released October 2024)
2. **Most Python packages still target 3.12** as the latest stable
3. **Intel's ecosystem is on 3.12** and will likely stay there for months
4. **No critical Python 3.13 features** are being used in exo
5. **Better ecosystem compatibility** overall

### Verification Needed
Before downgrading, verify these dependencies work with Python 3.12:
- ✓ aiofiles
- ✓ fastapi
- ✓ uvicorn
- ✓ tinygrad
- ✓ safetensors
- ✓ transformers
- ✓ pillow
- ✓ mflux (check this one specifically)
- ✓ All other dependencies

### Implementation Plan

1. **Update pyproject.toml**:
   ```toml
   requires-python = ">=3.12,<3.13"
   ```

2. **Update flake.nix**:
   - Change `python313` to `python312` throughout
   - Update Python package overrides
   - Update dev shell

3. **Update basedpyright config**:
   ```toml
   pythonVersion = "3.12"
   ```

4. **Test all functionality**:
   - Run test suite
   - Verify all imports work
   - Check type checking passes
   - Test on both macOS and Linux

5. **Update pytorch-xpu.nix**:
   - Fetch Python 3.12 wheel from Intel
   - Get correct hash
   - Test installation

6. **Update ipex-xpu.nix**:
   - Fetch Python 3.12 wheel from Intel
   - Get correct hash
   - Test installation

7. **Deploy and test on gremlin-1**:
   - Verify Intel Arc GPU is detected
   - Run verification script
   - Test inference on GPU

## Current Workaround

Until a decision is made, users can manually install PyTorch+IPEX with XPU support:

```bash
# Create Python 3.12 environment
python3.12 -m venv .venv-pytorch
source .venv-pytorch/bin/activate

# Install PyTorch+IPEX with XPU support
pip install torch==2.5.1+xpu torchvision==0.20.1+xpu \
  --index-url https://download.pytorch.org/whl/xpu

pip install intel-extension-for-pytorch==2.5.10+xpu \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Run exo
python -m exo.main --backend pytorch_ipex
```

## Decision Required

**Action needed:** Decide whether to downgrade to Python 3.12 or wait for Intel's Python 3.13 wheels.

**Recommendation:** Downgrade to Python 3.12 now to unblock Intel Arc GPU support.

## References

- PyTorch XPU wheel index: https://download.pytorch.org/whl/xpu/torch/
- IPEX documentation: https://intel.github.io/intel-extension-for-pytorch/
- Task 10 Pivot: `.kiro/specs/pytorch-ipex-intel-arc/TASK_10_PIVOT.md`
- Python 3.13 release: https://www.python.org/downloads/release/python-3130/

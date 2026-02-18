# Python 3.12 + PyTorch 2.6.0 Migration Summary

## Changes Made

### 1. Python Version Downgrade: 3.13 → 3.12

**Reason**: Intel's PyTorch+IPEX XPU wheels only support Python 3.10-3.12

**Files Modified**:
- `pyproject.toml`: Changed `requires-python = ">=3.12,<3.13"` and `pythonVersion = "3.12"`
- `flake.nix`: Changed all `python313` references to `python312`

### 2. PyTorch Version Update: 2.5.1 → 2.6.0

**Reason**: PyTorch 2.5.x+xpu wheels are not available for Python 3.12 (cp312)

**Available versions for Python 3.12**:
- ❌ PyTorch 2.5.0+xpu - Not available for cp312
- ❌ PyTorch 2.5.1+xpu - Not available for cp312
- ✅ PyTorch 2.6.0+xpu - Available for cp312
- ✅ PyTorch 2.7.0+xpu - Available for cp312
- ✅ PyTorch 2.10.0+xpu - Available for cp312

**Chosen**: PyTorch 2.6.0+xpu (most stable of available options)

### 3. IPEX Version Update: 2.5.10 → 2.6.10

**Reason**: IPEX version must match PyTorch major.minor version

**Version Compatibility**:
- PyTorch 2.6.0+xpu ↔ IPEX 2.6.10+xpu ✅
- PyTorch 2.7.0+xpu ↔ IPEX 2.7.10+xpu ✅

**Chosen**: IPEX 2.6.10+xpu (matches PyTorch 2.6.0)

## Package Details

### PyTorch 2.6.0+xpu
- **Wheel**: `torch-2.6.0+xpu-cp312-cp312-linux_x86_64.whl`
- **URL**: https://download.pytorch.org/whl/xpu/torch-2.6.0%2Bxpu-cp312-cp312-linux_x86_64.whl
- **Hash**: `sha256-xMXGdiXK88NXZcK5Th/hZuPjP0pUUhshJaWtG+sLD8I=`
- **Index**: https://download.pytorch.org/whl/xpu

### IPEX 2.6.10+xpu
- **Wheel**: `intel_extension_for_pytorch-2.6.10+xpu-cp312-cp312-linux_x86_64.whl`
- **URL**: https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-2.6.10%2Bxpu-cp312-cp312-linux_x86_64.whl
- **Hash**: To be computed on first build
- **Index**: https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

## Files Created/Modified

### Modified
1. `pyproject.toml` - Python version requirement
2. `flake.nix` - Python 3.12 throughout
3. `nix/pytorch-xpu.nix` - Fetch PyTorch 2.6.0+xpu wheel
4. `nix/ipex-xpu.nix` - Fetch IPEX 2.6.10+xpu wheel

### Created
- `PYTHON_312_PYTORCH_26_MIGRATION.md` - This file
- `PYTORCH_IPEX_PYTHON_VERSION_ISSUE.md` - Problem analysis

## Testing Plan

### 1. Build Verification
```bash
# Build PyTorch XPU package
nix build .#pytorch-xpu

# Build IPEX XPU package
nix build .#ipex-xpu

# Build full exo package
nix build .#exo
```

### 2. Import Verification
```bash
# Enter dev shell
nix develop

# Test PyTorch import
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

# Test IPEX import
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

### 3. GPU Verification (on gremlin-1)
```bash
# Run verification script
python nix/verify-pytorch-ipex-xpu.py

# Should show:
# - PyTorch 2.6.0+xpu
# - IPEX 2.6.10+xpu
# - XPU available: True
# - All GPU tests passing
```

### 4. Exo Integration Test
```bash
# Run exo with PyTorch+IPEX backend
uv run exo --backend pytorch_ipex

# Verify GPU is being used
# Check logs for XPU device detection
```

## Compatibility Notes

### PyTorch API Changes 2.5 → 2.6
PyTorch 2.6.0 is a minor version bump from 2.5.x. Key changes:
- API is backward compatible
- No breaking changes expected for our use case
- XPU support is stable in both versions

### Code Changes Required
Minimal to none. The PyTorch+IPEX backend code should work without modification because:
- We use standard PyTorch APIs
- XPU device API is stable
- IPEX optimization API is unchanged

### Potential Issues
1. **New PyTorch features**: Code using PyTorch 2.6+ features won't work on older versions
2. **Performance differences**: 2.6.0 may have different performance characteristics
3. **Bug fixes**: Some bugs fixed in 2.6.0 that existed in 2.5.x

## Rollback Plan

If issues arise, we can:

1. **Try PyTorch 2.7.0+xpu + IPEX 2.7.10+xpu**:
   - Update hashes in nix files
   - Rebuild

2. **Use Python 3.11 with PyTorch 2.5.x** (if available):
   - Check if 2.5.x+xpu wheels exist for cp311
   - Downgrade Python to 3.11

3. **Fall back to pip installation**:
   - Use standard PyTorch in Nix
   - Document pip installation for XPU support
   - Already documented in `nix/PYTORCH_IPEX_INSTALLATION.md`

## Success Criteria

✅ Python 3.12 environment builds successfully
✅ PyTorch 2.6.0+xpu imports without errors
✅ IPEX 2.6.10+xpu imports without errors
✅ `torch.xpu.is_available()` returns True on gremlin-1
✅ Basic tensor operations work on XPU
✅ IPEX optimization works
✅ Exo runs with pytorch_ipex backend
✅ Inference works on Intel Arc GPU

## Next Steps

1. ✅ Update pyproject.toml (DONE)
2. ✅ Update flake.nix (DONE)
3. ✅ Create pytorch-xpu.nix (DONE)
4. ✅ Create ipex-xpu.nix (DONE)
5. ⏳ Build and test locally
6. ⏳ Deploy to gremlin-1
7. ⏳ Run verification script
8. ⏳ Test inference on GPU
9. ⏳ Update documentation

## References

- PyTorch XPU wheels: https://download.pytorch.org/whl/xpu/torch/
- IPEX XPU wheels: https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
- PyTorch 2.6 release notes: https://github.com/pytorch/pytorch/releases/tag/v2.6.0
- IPEX documentation: https://intel.github.io/intel-extension-for-pytorch/

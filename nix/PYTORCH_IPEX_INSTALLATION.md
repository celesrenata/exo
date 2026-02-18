# PyTorch + IPEX Installation Guide for Intel Arc GPUs

## Overview

This guide explains how to install PyTorch and Intel Extension for PyTorch (IPEX) with Intel Arc GPU (XPU) support.

## Why Pip Instead of Nix?

Building PyTorch with XPU support from source requires:
- Intel oneAPI DPC++ compiler
- Complex SYCL toolchain configuration
- Extensive build time (hours)

Intel provides pre-built wheels that:
- Are officially supported and tested
- Install in minutes instead of hours
- Work reliably with Intel Arc GPUs
- Are updated regularly by Intel

See `.kiro/specs/pytorch-ipex-intel-arc/TASK_10_PIVOT.md` for full rationale.

## Prerequisites

### System Requirements (Managed by Nix)

The following are installed via NixOS configuration:

```nix
{
  # Intel GPU drivers and runtime
  hardware.graphics.extraPackages = [
    intel-compute-runtime  # OpenCL and Level Zero runtime
    level-zero            # Low-level GPU API
  ];

  # oneAPI libraries (optional, for performance)
  environment.systemPackages = [
    mkl      # Math Kernel Library
    oneDNN   # Deep Neural Network Library
    onetbb   # Threading Building Blocks
  ];
}
```

### Python Environment

Use Python 3.10, 3.11, or 3.12. Python 3.13 support may be limited.

## Installation Steps

### Step 1: Create Python Environment

Using uv (recommended):
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Using venv:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install PyTorch with XPU Support

```bash
pip install torch==2.5.1+xpu torchvision==0.20.1+xpu torchaudio==2.5.1+xpu \
  --index-url https://download.pytorch.org/whl/xpu
```

This installs:
- `torch` 2.5.1 with XPU support
- `torchvision` 0.20.1 with XPU support
- `torchaudio` 2.5.1 with XPU support

### Step 3: Install IPEX with XPU Support

```bash
pip install intel-extension-for-pytorch==2.5.10+xpu \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

This installs:
- `intel_extension_for_pytorch` 2.5.10 with XPU support
- Additional Intel-optimized kernels and operations

### Step 4: Verify Installation

Run the verification script:
```bash
python nix/verify-pytorch-ipex-xpu.py
```

Or manually verify:
```python
import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch version: {torch.__version__}")
print(f"IPEX version: {ipex.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")

if torch.xpu.is_available():
    print(f"XPU device count: {torch.xpu.device_count()}")
    print(f"XPU device name: {torch.xpu.get_device_name(0)}")
    
    # Test basic tensor operations
    x = torch.randn(3, 3).to('xpu')
    y = torch.randn(3, 3).to('xpu')
    z = torch.matmul(x, y)
    print(f"Tensor operation successful: {z.shape}")
```

Expected output (with Intel Arc GPU):
```
PyTorch version: 2.5.1+xpu
IPEX version: 2.5.10+xpu
XPU available: True
XPU device count: 1
XPU device name: Intel(R) Arc(TM) A770 Graphics
Tensor operation successful: torch.Size([3, 3])
```

Expected output (without Intel Arc GPU):
```
PyTorch version: 2.5.1+xpu
IPEX version: 2.5.10+xpu
XPU available: False
```

## Version Compatibility

| PyTorch | IPEX | Python | Status |
|---------|------|--------|--------|
| 2.5.1+xpu | 2.5.10+xpu | 3.10-3.12 | ✅ Recommended |
| 2.5.0+xpu | 2.5.0+xpu | 3.10-3.12 | ✅ Supported |
| 2.4.x+xpu | 2.4.x+xpu | 3.10-3.11 | ⚠️ Older |

Always match PyTorch and IPEX major.minor versions.

## Troubleshooting

### XPU Not Available

If `torch.xpu.is_available()` returns `False`:

1. **Check GPU drivers**:
   ```bash
   clinfo  # Should show Intel GPU
   sycl-ls  # Should show Level Zero devices
   ```

2. **Check Level Zero**:
   ```bash
   ls /dev/dri/  # Should show renderD128 or similar
   ```

3. **Check permissions**:
   ```bash
   groups  # Should include 'video' or 'render'
   ```

4. **Reinstall with correct index**:
   ```bash
   pip uninstall torch torchvision torchaudio intel-extension-for-pytorch
   # Then reinstall following steps above
   ```

### Import Errors

If you get import errors:

1. **Check installation**:
   ```bash
   pip list | grep torch
   pip list | grep intel-extension
   ```

2. **Verify Python version**:
   ```bash
   python --version  # Should be 3.10-3.12
   ```

3. **Check for conflicts**:
   ```bash
   pip check
   ```

### Performance Issues

If inference is slow:

1. **Enable IPEX optimizations**:
   ```python
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model)
   ```

2. **Use XPU device explicitly**:
   ```python
   model = model.to('xpu')
   ```

3. **Check GPU utilization**:
   ```bash
   intel_gpu_top  # Monitor GPU usage
   ```

## Integration with Exo

To use PyTorch+IPEX in exo:

1. **Install in exo's Python environment**:
   ```bash
   cd /path/to/exo
   source .venv/bin/activate  # or uv venv
   pip install torch==2.5.1+xpu torchvision==0.20.1+xpu \
     --index-url https://download.pytorch.org/whl/xpu
   pip install intel-extension-for-pytorch==2.5.10+xpu \
     --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
   ```

2. **Verify backend is available**:
   ```bash
   uv run python -c "from exo.worker.engines.factory import get_available_backends; print(get_available_backends())"
   ```

3. **Run exo with PyTorch+IPEX backend**:
   ```bash
   uv run exo --backend pytorch_ipex
   ```

## NixOS Integration

For NixOS deployments, you can create a wrapper script:

```nix
# In your NixOS configuration
environment.systemPackages = [
  (pkgs.writeShellScriptBin "exo-with-ipex" ''
    # Ensure Intel drivers are loaded
    export LD_LIBRARY_PATH=${pkgs.intel-compute-runtime}/lib:${pkgs.level-zero}/lib:$LD_LIBRARY_PATH
    
    # Run exo with pip-installed PyTorch+IPEX
    cd /path/to/exo
    ${pkgs.uv}/bin/uv run exo --backend pytorch_ipex "$@"
  '')
];
```

## References

- [PyTorch Intel GPU Support](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel GPU Drivers](https://dgpu-docs.intel.com/)
- [Task 10 Pivot Document](.kiro/specs/pytorch-ipex-intel-arc/TASK_10_PIVOT.md)

## Support

For issues:
1. Check this guide's troubleshooting section
2. Review Intel's official documentation
3. Check exo's GitHub issues
4. Verify GPU drivers are up to date

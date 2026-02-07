# NixOS Configuration for Tinygrad Backend with Intel Arc Support

This document describes the NixOS module configuration for enabling the tinygrad backend with Intel Arc GPU acceleration in exo.

## Overview

The `exo-intel` NixOS module provides declarative configuration for:
- Tinygrad backend with GPU/CPU execution
- Intel Arc iGPU support with Level Zero and OpenCL runtimes
- Intel NPU support (experimental)
- Automatic device detection and runtime selection
- Proper kernel modules, drivers, and environment variables

## Module Options

### `services.exo.intel.enable`

**Type**: `boolean`  
**Default**: `false`

Enable Intel hardware acceleration support for exo. This is the main switch that enables all Intel-specific features.

### `services.exo.intel.tinygrad.enable`

**Type**: `boolean`  
**Default**: `true` (when `services.exo.intel.enable = true`)

Enable the tinygrad backend for exo. This installs tinygrad and configures the necessary environment variables.

### `services.exo.intel.tinygrad.backend`

**Type**: `enum ["GPU", "CPU"]`  
**Default**: `"GPU"`

Select the tinygrad execution backend:
- `"GPU"`: Use GPU acceleration (requires Intel Arc iGPU)
- `"CPU"`: Use CPU-only execution

### `services.exo.intel.arc.enable`

**Type**: `boolean`  
**Default**: `true` (when `services.exo.intel.enable = true`)

Enable Intel Arc iGPU support. This installs GPU drivers, runtimes, and configures device permissions.

### `services.exo.intel.arc.runtime`

**Type**: `enum ["level-zero", "opencl", "auto"]`  
**Default**: `"auto"`

Select the GPU runtime for Intel Arc:
- `"level-zero"`: Use Level Zero API (recommended for best performance)
- `"opencl"`: Use OpenCL API (fallback option)
- `"auto"`: Automatically detect and use the best available runtime

### `services.exo.intel.npu.enable`

**Type**: `boolean`  
**Default**: `false`

Enable Intel NPU support (experimental). This is only available on Intel Core Ultra processors with NPU hardware.

### `services.exo.intel.npu.servicePort`

**Type**: `port`  
**Default**: `52416`

Port number for the NPU inference service.

## What the Module Configures

### Packages

When enabled, the module installs:
- `python313Packages.tinygrad` - Tinygrad deep learning framework
- `exo` - The exo distributed inference system
- `intel-gpu-tools` - GPU monitoring tools (intel_gpu_top)
- `clinfo` - OpenCL device information tool

### Graphics Drivers (Intel Arc)

When `arc.enable = true`, the module configures:
- `intel-compute-runtime` - Intel OpenCL runtime
- `level-zero` - Level Zero runtime and loader
- `intel-media-driver` - VA-API driver for Intel GPUs
- `ocl-icd` - OpenCL ICD loader

### Environment Variables

The module sets the following environment variables:

#### Tinygrad Backend
- `EXO_TINYGRAD_ENABLED="true"` - Enable tinygrad backend in exo
- `TINYGRAD_BACKEND="GPU"` or `"CPU"` - Tinygrad execution backend
- `TINYGRAD_OPTIMIZE="2"` - Enable GPU optimizations (when backend=GPU)
- `TINYGRAD_DISABLE_CACHE="1"` - Disable JIT cache to avoid permission issues

#### Intel Arc Runtime
- `TINYGRAD_INTEL_RUNTIME` - Set to "LEVEL_ZERO", "OPENCL", or "AUTO"
- `ZE_ENABLE_VALIDATION_LAYER="0"` - Disable Level Zero validation for performance
- `ZE_AFFINITY_MASK="0"` - Use first GPU device
- `OCL_ICD_VENDORS="/etc/OpenCL/vendors"` - OpenCL ICD vendor path
- `NEOReadDebugKeys="1"` - Enable Intel compute runtime debug keys

### Kernel Configuration

#### Kernel Modules
- `i915` - Intel GPU driver (for Arc iGPU)
- `intel_vpu` - Intel NPU driver (when NPU enabled)

#### Kernel Parameters
- `i915.force_probe=*` - Force probe all Intel GPUs
- `i915.enable_guc=3` - Enable GuC and HuC firmware loading

### Device Permissions

The module configures udev rules for device access:

#### Intel Arc GPU
```
SUBSYSTEM=="drm", KERNEL=="renderD*", ATTRS{vendor}=="0x8086", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", ATTRS{vendor}=="0x8086", MODE="0666"
```

#### Intel NPU
```
SUBSYSTEM=="accel", KERNEL=="accel[0-9]*", GROUP="exo", MODE="0660"
SUBSYSTEM=="drm", KERNEL=="renderD*", ATTRS{vendor}=="0x8086", GROUP="exo", MODE="0660"
```

### OpenCL ICD Configuration

The module creates `/etc/OpenCL/vendors/intel.icd` pointing to the Intel OpenCL runtime library.

## Example Configurations

### Minimal Configuration (Auto-detect everything)

```nix
{
  services.exo.intel = {
    enable = true;
    # All other options use defaults:
    # - tinygrad.enable = true
    # - tinygrad.backend = "GPU"
    # - arc.enable = true
    # - arc.runtime = "auto"
  };
}
```

### Explicit Level Zero Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    
    tinygrad = {
      enable = true;
      backend = "GPU";
    };
    
    arc = {
      enable = true;
      runtime = "level-zero";
    };
  };
}
```

### OpenCL Fallback Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    
    arc = {
      enable = true;
      runtime = "opencl";
    };
  };
}
```

### CPU-Only Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    
    tinygrad = {
      enable = true;
      backend = "CPU";
    };
    
    arc = {
      enable = false;  # No GPU support needed
    };
  };
}
```

### Full Configuration with NPU

```nix
{
  services.exo.intel = {
    enable = true;
    
    tinygrad = {
      enable = true;
      backend = "GPU";
    };
    
    arc = {
      enable = true;
      runtime = "auto";
    };
    
    npu = {
      enable = true;
      servicePort = 52416;
    };
  };
}
```

## Complete NixOS Configuration Example

See [docs/examples/nixos-intel-config.nix](examples/nixos-intel-config.nix) for a complete flake-based NixOS configuration example.

## Verification

After applying the configuration, verify the setup:

### Check GPU Detection

```bash
# Check if Intel GPU is detected
clinfo | grep -A 5 "Platform Name"

# Check Level Zero devices
ls -la /dev/dri/

# Monitor GPU usage
intel_gpu_top
```

### Check Environment Variables

```bash
# Check tinygrad configuration
env | grep TINYGRAD

# Check Intel runtime configuration
env | grep -E "(ZE_|OCL_|NEO)"
```

### Test Tinygrad

```python
import os
print(f"TINYGRAD_BACKEND: {os.environ.get('TINYGRAD_BACKEND')}")
print(f"EXO_TINYGRAD_ENABLED: {os.environ.get('EXO_TINYGRAD_ENABLED')}")

from tinygrad import Device
print(f"Tinygrad device: {Device.DEFAULT}")
```

## Troubleshooting

### GPU Not Detected

1. Check if the i915 kernel module is loaded:
   ```bash
   lsmod | grep i915
   ```

2. Check dmesg for GPU initialization errors:
   ```bash
   dmesg | grep -i "i915\|gpu"
   ```

3. Verify GPU device nodes exist:
   ```bash
   ls -la /dev/dri/
   ```

### Level Zero Not Working

1. Check if Level Zero loader is available:
   ```bash
   ldconfig -p | grep libze_loader
   ```

2. Try using OpenCL instead:
   ```nix
   services.exo.intel.arc.runtime = "opencl";
   ```

### OpenCL Not Working

1. Check OpenCL platforms:
   ```bash
   clinfo
   ```

2. Verify ICD configuration:
   ```bash
   cat /etc/OpenCL/vendors/intel.icd
   ```

### Permission Denied Errors

1. Check device permissions:
   ```bash
   ls -la /dev/dri/
   ```

2. Verify udev rules are applied:
   ```bash
   udevadm control --reload-rules
   udevadm trigger
   ```

## Performance Tuning

### GPU Frequency Scaling

For better performance, you can disable GPU frequency scaling:

```nix
{
  boot.kernelParams = [
    "i915.enable_guc=3"
    "i915.enable_fbc=1"
    "i915.enable_psr=0"
  ];
}
```

### Memory Allocation

For large models, increase GPU memory allocation:

```nix
{
  environment.variables = {
    # Increase GPU memory allocation (in MB)
    NEO_SHARED_FORCE_DEVICE_ALLOC = "1";
  };
}
```

## Related Documentation

- [Intel Hardware Setup Guide](intel-hardware-setup.md)
- [Tinygrad Backend Documentation](tinygrad-backend.md)
- [Example NixOS Configuration](examples/nixos-intel-config.nix)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)

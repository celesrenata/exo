# PyTorch+IPEX Backend NixOS Configuration

This guide explains how to configure the PyTorch+IPEX backend for Intel Arc GPU support on NixOS.

## Overview

The PyTorch+IPEX backend provides an alternative to the tinygrad backend for Intel Arc GPU inference. It uses PyTorch with Intel Extension for PyTorch (IPEX) for optimized performance on Intel hardware.

## Prerequisites

- NixOS system with Intel Arc GPU
- Flakes enabled in your NixOS configuration
- Access to the exo repository

## Basic Configuration

### Enable PyTorch+IPEX Backend

Add the PyTorch+IPEX backend to your NixOS configuration:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:exo-explore/exo";
  };
  
  outputs = { self, nixpkgs, exo }: {
    nixosConfigurations.your-hostname = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        exo.nixosModules.exo-intel
        {
          services.exo.intel = {
            enable = true;
            
            # Enable PyTorch+IPEX backend
            pytorch_ipex = {
              enable = true;
              preferredBackend = true;  # Use PyTorch+IPEX as primary backend
            };
            
            # Intel Arc GPU configuration
            arc = {
              enable = true;
              runtime = "auto";  # Auto-detect Level Zero or OpenCL
            };
          };
        }
      ];
    };
  };
}
```

### Dual Backend Configuration

You can enable both tinygrad and PyTorch+IPEX backends:

```nix
{
  services.exo.intel = {
    enable = true;
    
    # Enable tinygrad backend (default)
    tinygrad = {
      enable = true;
      backend = "GPU";
    };
    
    # Enable PyTorch+IPEX backend
    pytorch_ipex = {
      enable = true;
      preferredBackend = false;  # Use tinygrad as primary, PyTorch+IPEX as fallback
    };
    
    arc = {
      enable = true;
      runtime = "level-zero";  # Prefer Level Zero for best performance
    };
  };
}
```

## Configuration Options

### `services.exo.intel.pytorch_ipex`

- **`enable`** (boolean, default: `false`)
  - Enable PyTorch+IPEX backend support
  - When enabled, PyTorch and IPEX will be available for inference

- **`preferredBackend`** (boolean, default: `false`)
  - Use PyTorch+IPEX as the preferred backend over tinygrad
  - When `true`, PyTorch+IPEX will be tried first, with tinygrad as fallback
  - When `false`, tinygrad will be tried first, with PyTorch+IPEX as fallback

## Environment Variables

When PyTorch+IPEX is enabled, the following environment variables are automatically set:

- `EXO_PYTORCH_IPEX_ENABLED=true` - Enables PyTorch+IPEX backend
- `PYTORCH_ENABLE_XPU=1` - Enables Intel XPU device support in PyTorch
- `IPEX_TILE_AS_DEVICE=1` - Treats each GPU tile as a separate device

## Backend Selection

The backend selection follows this priority:

1. **With `preferredBackend = true`:**
   - PyTorch+IPEX (if available)
   - Tinygrad (fallback)
   - MLX (final fallback on macOS)

2. **With `preferredBackend = false`:**
   - Tinygrad (if available)
   - PyTorch+IPEX (fallback)
   - MLX (final fallback on macOS)

## Deployment

### 1. Deploy Configuration

```bash
# Rebuild your NixOS system
sudo nixos-rebuild switch --flake .#your-hostname
```

### 2. Verify PyTorch+IPEX Installation

Check if PyTorch and IPEX are available:

```bash
# Test PyTorch XPU availability
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"

# Test IPEX import
python -c "import intel_extension_for_pytorch as ipex; print('IPEX imported successfully')"
```

### 3. Start exo with PyTorch+IPEX

```bash
# Start exo with verbose logging
exo -vv

# Look for log messages indicating:
# - Backend initialization (pytorch_ipex)
# - Device detection (Intel XPU)
# - Device properties (memory, compute units)
```

Expected log output:
```
INFO Device selection: Intel Arc GPU with PyTorch+IPEX backend_type=pytorch_ipex device_type=XPU device_count=1 device_name="Intel Arc Graphics" memory_gb=16.00
INFO Backend initialized backend_type=pytorch_ipex device=xpu:0
```

## Troubleshooting

### PyTorch XPU Not Available

If PyTorch XPU is not detected:

1. Verify Intel compute runtime is installed:
   ```bash
   nix-store -q --references /run/current-system | grep intel-compute-runtime
   ```

2. Check Level Zero devices:
   ```bash
   ls -la /sys/class/drm/
   ```

3. Verify graphics drivers are loaded:
   ```bash
   lspci -k | grep -A 3 VGA
   ```

### IPEX Import Fails

If IPEX fails to import:

1. Check that PyTorch+IPEX is enabled in configuration
2. Verify the exo package includes PyTorch and IPEX dependencies
3. Check for version compatibility issues in logs

### Backend Falls Back to CPU

If the backend falls back to CPU:

1. Check XPU availability:
   ```bash
   python -c "import torch; print(torch.xpu.is_available())"
   ```

2. Verify environment variables are set:
   ```bash
   echo $PYTORCH_ENABLE_XPU
   echo $IPEX_TILE_AS_DEVICE
   ```

3. Check for error messages in exo logs:
   ```bash
   journalctl -u exo -n 100
   ```

## Performance Comparison

### Benchmarking PyTorch+IPEX vs Tinygrad

To compare performance between backends:

```bash
# Test with PyTorch+IPEX
EXO_PYTORCH_IPEX_ENABLED=true exo -vv

# Test with Tinygrad
EXO_TINYGRAD_ENABLED=true exo -vv
```

Monitor performance metrics in the dashboard at `http://localhost:52415`

### Expected Performance

- **PyTorch+IPEX**: Better for larger models (7B+), optimized kernels
- **Tinygrad**: Better for smaller models (1B-3B), lower overhead
- **CPU Fallback**: Significantly slower, use only when GPU unavailable

## Example Configurations

### Minimal PyTorch+IPEX Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    pytorch_ipex.enable = true;
    arc.enable = true;
  };
}
```

### Production Configuration with Monitoring

```nix
{
  services.exo.intel = {
    enable = true;
    
    pytorch_ipex = {
      enable = true;
      preferredBackend = true;
    };
    
    arc = {
      enable = true;
      runtime = "level-zero";
    };
  };
  
  # Add monitoring tools
  environment.systemPackages = with pkgs; [
    intel-gpu-tools  # intel_gpu_top
    clinfo           # OpenCL info
  ];
}
```

### Multi-Backend Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    
    # Enable both backends for maximum compatibility
    tinygrad = {
      enable = true;
      backend = "GPU";
    };
    
    pytorch_ipex = {
      enable = true;
      preferredBackend = false;  # Use as fallback
    };
    
    arc = {
      enable = true;
      runtime = "auto";
    };
  };
}
```

## Runtime Backend Selection

You can override the backend selection at runtime:

```bash
# Force PyTorch+IPEX backend
EXO_PYTORCH_IPEX_ENABLED=true exo

# Force Tinygrad backend
EXO_TINYGRAD_ENABLED=true exo

# Let exo auto-select based on configuration
exo
```

## Integration with Distributed Inference

PyTorch+IPEX backend integrates seamlessly with exo's distributed inference:

- Supports multi-node inference via PyTorchIPEXRingInstance
- Compatible with exo's Master/Worker coordination
- Works with model sharding across nodes
- Maintains OpenAI API compatibility

## Next Steps

After successful deployment:

1. Monitor performance in the dashboard
2. Compare inference speeds with tinygrad backend
3. Test with different model sizes
4. Report performance data and issues

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel Compute Runtime](https://github.com/intel/compute-runtime)
- [exo PyTorch+IPEX Backend Design](../specs/pytorch-ipex-intel-arc/design.md)

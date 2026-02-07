# Intel Hardware Support Setup Guide

This guide explains how to enable Intel hardware acceleration (Arc iGPU and NPU) for exo on NixOS.

## Prerequisites

- NixOS system with Intel hardware (Core Ultra 9 185H or similar)
- Flakes enabled in your NixOS configuration
- Access to the exo repository

## Configuration

### Basic Intel Arc iGPU Support

Add the exo-intel module to your NixOS configuration:

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
            
            arc = {
              enable = true;
              runtime = "auto";  # Options: "level-zero", "opencl", "auto"
            };
          };
        }
      ];
    };
  };
}
```

### With NPU Support (Experimental)

To enable experimental NPU support:

```nix
{
  services.exo.intel = {
    enable = true;
    
    arc = {
      enable = true;
      runtime = "level-zero";
    };
    
    npu = {
      enable = true;
      servicePort = 52416;  # Default port
    };
  };
}
```

## Deployment and Testing

### 1. Deploy Configuration

```bash
# Rebuild your NixOS system
sudo nixos-rebuild switch --flake .#your-hostname
```

### 2. Verify Level Zero Runtime

Check if Level Zero is available:

```bash
# Check for Level Zero loader library
ls -la /run/opengl-driver/lib/libze_loader.so*

# List Level Zero devices
clinfo | grep -A 10 "Platform Name.*Intel"
```

### 3. Verify OpenCL Runtime

Check if OpenCL is available:

```bash
# Check for OpenCL library
ls -la /run/opengl-driver/lib/libOpenCL.so*

# List OpenCL platforms and devices
clinfo
```

### 4. Test exo Startup

Start exo and check that it detects Intel hardware:

```bash
# Start exo with verbose logging
exo -vv

# Look for log messages indicating:
# - Backend initialization (tinygrad)
# - Device detection (Intel Arc Graphics)
# - Runtime selection (Level Zero or OpenCL)
```

Expected log output:
```
INFO Backend initialized backend_type=tinygrad device=GPU runtime=LEVEL_ZERO device_name="Intel Arc Graphics" memory_gb=16.0
```

### 5. Verify NPU Service (if enabled)

Check NPU service status:

```bash
# Check systemd service
sudo systemctl status exo-npu

# Check for NPU device node
ls -la /dev/accel/accel0

# Check kernel modules
lsmod | grep intel_vpu
```

## Troubleshooting

### Level Zero Not Available

If Level Zero is not detected:

1. Check that `intel-compute-runtime` is installed:
   ```bash
   nix-store -q --references /run/current-system | grep intel-compute-runtime
   ```

2. Verify graphics drivers are loaded:
   ```bash
   lspci -k | grep -A 3 VGA
   ```

3. Check for Level Zero devices:
   ```bash
   ls -la /sys/class/drm/
   ```

### OpenCL Fallback

If exo falls back to OpenCL:

1. Check Level Zero availability (see above)
2. Look for error messages in exo logs
3. Verify OpenCL is working:
   ```bash
   clinfo | grep "Platform Name"
   ```

### GPU Not Detected

If GPU is not detected at all:

1. Verify Intel Arc iGPU is present:
   ```bash
   lspci | grep VGA
   ```

2. Check that `hardware.graphics.enable` is true:
   ```bash
   nixos-option hardware.graphics.enable
   ```

3. Ensure graphics drivers are in extraPackages:
   ```bash
   nixos-option hardware.graphics.extraPackages
   ```

### NPU Service Fails to Start

If the NPU service fails:

1. Check service logs:
   ```bash
   sudo journalctl -u exo-npu -n 50
   ```

2. Verify NPU device exists:
   ```bash
   ls -la /dev/accel/
   ```

3. Check kernel module:
   ```bash
   sudo modprobe intel_vpu
   dmesg | grep vpu
   ```

### Dashboard Not Showing Hardware Info

If the dashboard doesn't display hardware information:

1. Verify backend events are being emitted:
   ```bash
   # Check logs for BackendInitialized events
   exo -vv 2>&1 | grep "Backend initialized"
   ```

2. Check that the dashboard is connected to the API:
   ```bash
   # Verify API is running
   curl http://localhost:52415/health
   ```

3. Clear browser cache and reload the dashboard

### Metrics Not Updating

If performance metrics aren't updating:

1. Verify metrics collector is initialized:
   ```bash
   # Look for "Metrics collector initialized" in logs
   exo -vv 2>&1 | grep "Metrics collector"
   ```

2. Check that inferences are actually running:
   ```bash
   # Send a test inference request
   curl -X POST http://localhost:52415/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "your-model", "messages": [{"role": "user", "content": "test"}]}'
   ```

3. Verify GPU metrics collection is working:
   ```bash
   # Check for GPU metrics in logs
   exo -vv 2>&1 | grep "GPU metrics"
   ```

### Logs Not Showing Structured Fields

If structured logging fields aren't visible:

1. Ensure you're using verbose mode:
   ```bash
   exo -vv  # Double verbose shows structured fields
   ```

2. Check loguru configuration in your environment

3. Verify log format includes structured data:
   ```bash
   # Logs should show key=value pairs
   exo -vv 2>&1 | grep "backend_type="
   ```

## Performance Validation

### Benchmark GPU vs CPU

Run inference on both GPU and CPU to compare performance:

```bash
# GPU inference (default with Intel Arc enabled)
exo -vv

# CPU-only inference (disable GPU in config)
# Modify configuration to set arc.enable = false
```

### Monitor GPU Utilization

Use `intel_gpu_top` to monitor GPU usage during inference:

```bash
# Install intel-gpu-tools if not available
nix-shell -p intel-gpu-tools

# Monitor GPU
sudo intel_gpu_top
```

### View Performance Metrics in Dashboard

The exo dashboard displays real-time hardware and performance metrics:

1. Open the dashboard at `http://localhost:52415` (or your configured port)
2. Navigate to the cluster view
3. Each node shows:
   - Active backend (tinygrad, mlx, etc.)
   - Device type (GPU, CPU, METAL)
   - Runtime (LEVEL_ZERO, OPENCL, etc.)
   - Memory usage
   - GPU utilization (if available)
   - Inference throughput (tokens/sec)

### Interpreting Metrics

**Backend Status Colors:**
- 🟢 Green: Optimal configuration (GPU with Level Zero, or Metal on macOS)
- 🟡 Yellow: Good but not optimal (GPU with OpenCL fallback)
- ⚪ Gray: CPU fallback (no GPU available)
- 🔵 Blue: Experimental (NPU)

**Performance Indicators:**
- **Avg Throughput**: Overall tokens per second across all inferences
- **Recent Throughput**: Tokens per second for the last 10 inferences
- **GPU Utilization**: Percentage of GPU compute being used (0-100%)
- **Memory Usage**: Current memory consumption vs total available

### Structured Logging

exo uses structured logging with loguru. Key log fields to monitor:

```
backend_type: "tinygrad" | "mlx" | "npu"
device_type: "CPU" | "GPU" | "METAL" | "NPU"
runtime: "LEVEL_ZERO" | "OPENCL" | "CUDA" | "METAL"
device_name: Human-readable device name
memory_gb: Available memory in gigabytes
```

Example log messages:

```
# Backend initialization
INFO Backend initialized backend_type=tinygrad device=GPU runtime=LEVEL_ZERO device_name="Intel Arc Graphics" memory_gb=16.0

# Runtime selection
INFO Runtime selection: Level Zero (optimal) backend_type=tinygrad runtime=LEVEL_ZERO gpu_vendor=Intel reason="Level Zero provides best performance for Intel Arc"

# Fallback event
WARNING Fallback to OpenCL runtime (Level Zero unavailable) backend_type=tinygrad runtime=OPENCL requested_runtime=LEVEL_ZERO fallback_runtime=OPENCL

# GPU to CPU fallback
WARNING GPU fallback to CPU (no runtime available) backend_type=tinygrad requested_device=GPU fallback_device=CPU reason="Intel Arc GPU detected but no runtime available"
```

### Collecting Metrics Programmatically

If you need to collect metrics programmatically, the backend exposes metrics via the `BackendMetricsCollector`:

```python
# Get current metrics statistics
stats = backend.get_metrics_stats()

print(f"Inferences: {stats['inference_count']}")
print(f"Avg throughput: {stats['avg_tokens_per_second']:.2f} tok/s")
print(f"Recent throughput: {stats['recent_tokens_per_second']:.2f} tok/s")
print(f"Avg memory: {stats['avg_memory_used_mb']:.1f} MB")
if stats['avg_gpu_utilization']:
    print(f"Avg GPU usage: {stats['avg_gpu_utilization']:.1f}%")
```

## Example Configuration for gremlin-1

Complete example for a node with Intel Core Ultra 9 185H:

```nix
# /etc/nixos/flake.nix
{
  description = "gremlin-1 NixOS configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:exo-explore/exo";
  };

  outputs = { self, nixpkgs, exo }: {
    nixosConfigurations.gremlin-1 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./hardware-configuration.nix
        exo.nixosModules.exo-intel
        {
          # Enable Intel hardware acceleration
          services.exo.intel = {
            enable = true;
            
            arc = {
              enable = true;
              runtime = "level-zero";  # Prefer Level Zero for best performance
            };
            
            npu = {
              enable = true;  # Experimental NPU support
              servicePort = 52416;
            };
          };

          # Additional system configuration
          networking.hostName = "gremlin-1";
          
          # Enable OpenGL/Vulkan support
          hardware.graphics = {
            enable = true;
            enable32Bit = true;
          };
        }
      ];
    };
  };
}
```

## Runtime Configuration

### Environment Variables

Control backend selection at runtime:

```bash
# Force tinygrad backend
export EXO_BACKEND=tinygrad

# Force GPU device
export EXO_DEVICE=GPU

# Force specific runtime
export EXO_RUNTIME=LEVEL_ZERO  # or OPENCL

# Enable tinygrad
export EXO_TINYGRAD_ENABLED=1

# Start exo
exo
```

### Backend Selection Priority

The system will automatically select backends in this order:

1. Intel Arc iGPU (Level Zero) - if available and enabled
2. Intel Arc iGPU (OpenCL) - if Level Zero unavailable
3. CPU (tinygrad) - if GPU unavailable
4. MLX - fallback on macOS

## Next Steps

After successful deployment:

1. Monitor cluster performance in the dashboard
2. Compare inference speeds between GPU and CPU
3. Test model sharding across mixed-backend nodes
4. Report any issues or performance data

## References

- [Intel Compute Runtime](https://github.com/intel/compute-runtime)
- [Level Zero Specification](https://spec.oneapi.io/level-zero/latest/index.html)
- [tinygrad Documentation](https://github.com/tinygrad/tinygrad)
- [NixOS Hardware Configuration](https://nixos.wiki/wiki/Intel_Graphics)

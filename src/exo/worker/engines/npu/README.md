# Intel NPU Support for exo

This module provides Intel NPU (Neural Processing Unit) discovery and integration for exo.

## Overview

Intel Core Ultra processors (Meteor Lake and newer) include an integrated NPU designed for efficient AI inference. This module enables:

1. **Hardware Discovery**: Detect NPU hardware and verify driver availability
2. **Capability Assessment**: Determine which model types are supported
3. **Smoke Testing**: Verify NPU functionality with OpenVINO

## Components

### `discovery.py`

Core NPU discovery module that detects:
- NPU device nodes (`/dev/accel/accel*` or `/dev/dri/renderD*`)
- Kernel modules (`intel_vpu`, `ivpu`)
- OpenVINO availability and NPU device support
- Supported model types

Usage:
```python
from exo.worker.engines.npu.discovery import discover_npu

capabilities = discover_npu()
if capabilities.available:
    print(f"NPU available at {capabilities.device_path}")
    print(f"Supported models: {capabilities.supported_model_types}")
else:
    print(f"NPU not available: {capabilities.error_message}")
```

### `capability_report.py`

Generates comprehensive capability reports including:
- Hardware details (CPU model, Core Ultra detection)
- Device nodes and permissions
- Kernel version and configuration
- Driver details and versions
- OpenVINO status

Run as:
```bash
uv run python -m exo.worker.engines.npu.capability_report
```

Outputs:
- Human-readable report to stdout
- JSON report to `npu_capability_report.json`

### `smoke_test.py`

Verifies NPU functionality with three tests:

1. **Simple Inference**: Creates and runs a minimal model on NPU
2. **NPU Verification**: Confirms NPU is actually used (not CPU fallback)
3. **Latency Comparison**: Compares NPU vs CPU performance

Run as:
```bash
uv run python -m exo.worker.engines.npu.smoke_test
```

Or use the convenience script:
```bash
./tests/test_npu_smoke.sh
```

## Requirements

### Hardware
- Intel Core Ultra processor (Meteor Lake or newer)
- NPU-enabled BIOS/firmware

### Software
- Linux kernel 6.6+ (6.8+ recommended)
- `intel_vpu` or `ivpu` kernel module
- OpenVINO 2023.0+ with NPU plugin

### Python Dependencies
- `openvino` (optional, required for NPU execution)
- `numpy` (optional, required for smoke tests)

Install with:
```bash
uv pip install openvino numpy
```

## Supported Model Types

The Intel NPU is optimized for:

✅ **Well-Supported**:
- Vision models (image classification, object detection)
- Audio models (speech recognition, VAD)
- Embedding models (text embeddings, sentence transformers)
- Small transformers (<1B parameters)

❌ **Not Recommended**:
- Large LLM decode (>1B parameters)
- Training workloads
- Custom operations

## Integration Status

This module is part of **Phase 3** (Optional) of the Intel hardware support implementation. It provides:

- ✅ NPU hardware discovery
- ✅ Capability assessment and reporting
- ✅ Smoke testing with OpenVINO
- ⏳ NPU sidecar service (future work)
- ⏳ Workload routing to NPU (future work)

## Troubleshooting

### NPU Not Detected

1. Verify CPU model:
   ```bash
   cat /proc/cpuinfo | grep "model name"
   ```

2. Check kernel modules:
   ```bash
   lsmod | grep -E "intel_vpu|ivpu"
   sudo modprobe intel_vpu
   ```

3. Check device nodes:
   ```bash
   ls -la /dev/accel/
   ls -la /dev/dri/
   ```

### OpenVINO Not Finding NPU

1. Verify OpenVINO installation:
   ```python
   import openvino as ov
   print(ov.__version__)
   ```

2. Check available devices:
   ```python
   core = ov.Core()
   print(core.available_devices())
   ```

3. Ensure NPU plugin is installed:
   ```bash
   find /nix/store -name "*openvino*npu*"
   ```

## Documentation

See also:
- [NPU Capability Report Documentation](../../../../../docs/npu-capability-report.md)
- [Intel Hardware Setup Guide](../../../../../docs/intel-hardware-setup.md)
- [Design Document](../../../../../.kiro/specs/intel-hardware-support/design.md)

## References

- [Intel NPU Documentation](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)
- [OpenVINO NPU Plugin](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_NPU.html)
- [Linux Kernel DRM Accel](https://www.kernel.org/doc/html/latest/accel/index.html)


## NPU Sidecar Service

### Overview

The NPU service runs as a separate process and handles inference tasks suitable for NPU execution. It provides an HTTP API for communication with exo workers.

### Architecture

```
┌─────────────────────────────────────────┐
│           exo Worker Node               │
│                                         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │   Runner     │    │ NPU Service  │  │
│  │   (MLX/TG)   │◄──►│  (OpenVINO)  │  │
│  └──────────────┘    └──────────────┘  │
│         │                    │          │
│         │                    │          │
│         ▼                    ▼          │
│      GPU/CPU               NPU          │
└─────────────────────────────────────────┘
```

### Components

- **`service.py`**: Main NPU inference service with OpenVINO integration
- **`protocol.py`**: HTTP API client and server protocol definitions
- **`routing.py`**: Workload routing logic (NPU vs GPU/CPU)
- **`exo-npu.service`**: Systemd service template

### Installation

#### NixOS (Recommended)

Add to your NixOS configuration:

```nix
{
  inputs.exo.url = "github:exo-explore/exo";

  outputs = { nixpkgs, exo, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        exo.nixosModules.exo-intel
        {
          services.exo.intel = {
            enable = true;
            npu = {
              enable = true;
              servicePort = 52416;
            };
          };
        }
      ];
    };
  };
}
```

#### Manual Installation

1. Install dependencies:
```bash
pip install openvino fastapi uvicorn aiohttp
```

2. Create service user:
```bash
sudo useradd -r -s /bin/false exo
sudo mkdir -p /var/cache/exo
sudo chown exo:exo /var/cache/exo
```

3. Install systemd service:
```bash
sudo cp exo-npu.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable exo-npu
sudo systemctl start exo-npu
```

### Usage

#### Starting the Service

```bash
# Direct execution
python -m exo.worker.engines.npu.service --port 52416

# With systemd
sudo systemctl start exo-npu

# Check status
sudo systemctl status exo-npu
```

#### API Endpoints

- `POST /infer` - Execute inference
- `GET /health` - Health check
- `GET /models` - List loaded models
- `POST /models/{model_id}/load` - Preload a model
- `DELETE /models/{model_id}` - Unload a model

#### Client Usage

```python
from exo.worker.engines.npu.protocol import NPUServiceClient, InferenceRequest

async with NPUServiceClient(host="localhost", port=52416) as client:
    # Check health
    health = await client.health()
    print(f"NPU available: {health.npu_available}")
    
    # Execute inference
    request = InferenceRequest(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        input_data={"input_ids": [[101, 2023, 2003, 1037, 3231, 102]]},
    )
    response = await client.infer(request)
    print(f"Inference time: {response.inference_time_ms}ms")
```

### Workload Routing

The routing module determines which tasks should use NPU:

```python
from exo.worker.engines.npu.routing import should_use_npu, NPURouter
from exo.worker.engines.npu.discovery import discover_npu

# Discover NPU capabilities
npu_caps = discover_npu()

# Create router
router = NPURouter(npu_caps, npu_service_available=True)

# Route tasks
device = router.route_task(task)  # Returns "NPU", "GPU", or "CPU"
```

**NPU-Suitable Workloads:**
- ✅ Embedding generation
- ✅ Vision models
- ✅ Audio processing
- ✅ Small transformers

**GPU/CPU-Suitable Workloads:**
- ❌ Large LLM decode
- ❌ Image generation (FLUX, SD)
- ❌ High-throughput streaming

### Security

The NPU service runs with strict security isolation:
- Separate user account (`exo`)
- Limited file system access
- No network access except localhost
- Resource limits (8GB memory, 2 CPU cores)
- Device access restricted to NPU only

### Performance

Typical performance characteristics:
- **Embedding models**: 2-5x faster than CPU
- **Vision models**: 1.5-3x faster than CPU
- **Audio models**: 2-4x faster than CPU
- **Power efficiency**: 3-5x better than CPU

Note: Performance varies by model size and complexity.

### Service Troubleshooting

#### Service Won't Start

```bash
# Check service logs
sudo journalctl -u exo-npu -f

# Check permissions
sudo ls -la /dev/accel/accel0

# Test manual start
sudo -u exo python -m exo.worker.engines.npu.service --port 52416
```

#### API Not Responding

```bash
# Check if service is listening
sudo netstat -tlnp | grep 52416

# Test health endpoint
curl http://localhost:52416/health
```

#### Performance Issues

- Ensure NPU is actually being used (check logs)
- Verify model is compatible with NPU
- Check resource limits in systemd service
- Monitor with `intel_gpu_top`

# PyTorch + IPEX Backend for Intel Arc GPU Support

This module provides Intel Arc GPU support for exo using PyTorch and Intel Extension for PyTorch (IPEX).

## Overview

The PyTorch + IPEX backend enables exo to leverage Intel Arc GPUs for distributed AI inference. This implementation replaces the previous tinygrad-based approach which encountered device selection issues.

## Components

### Detection Script (`detect_intel_arc.py`)

Detects Intel Arc GPUs using `torch.xpu` and logs device capabilities including:
- Device name, type, and platform
- Total, allocated, and free memory
- Compute units and work group sizes
- Driver version and other properties

**Usage:**
```bash
uv run python src/exo/worker/engines/pytorch_ipex/detect_intel_arc.py
```

### Validation Script (`validate_ipex.py`)

Validates IPEX functionality through comprehensive tests:
1. Basic tensor operations (creation, movement, arithmetic)
2. Matrix multiplication (matmul) performance benchmarking
3. Softmax operations
4. bfloat16 precision support
5. IPEX model optimization on a dummy neural network

**Usage:**
```bash
uv run python src/exo/worker/engines/pytorch_ipex/validate_ipex.py
```

### Test Harness (`test_pytorch_ipex_setup.sh`)

Automated test harness that runs both detection and validation scripts to verify the complete setup.

**Usage:**
```bash
./test_pytorch_ipex_setup.sh
```

## Requirements

### Software Dependencies

- Python 3.13+
- PyTorch 2.0+
- Intel Extension for PyTorch (IPEX) 2.0+
- HuggingFace transformers
- safetensors

### System Dependencies (Linux only)

- Intel compute-runtime
- Level Zero runtime and loader
- Intel GPU drivers (i915 kernel module)

## Installation

### NixOS

The dependencies are configured in `flake.nix` and `python/parts.nix`. On NixOS systems with the exo-intel module enabled, all dependencies are automatically installed.

### Manual Installation

For non-NixOS systems:

```bash
# Install PyTorch with Intel GPU support
pip install torch>=2.0.0

# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch>=2.0.0

# Install other dependencies
pip install transformers safetensors
```

## Configuration

### NixOS Module

The Intel Arc support is configured through the NixOS module in `flake.nix`:

```nix
services.exo.intel = {
  enable = true;
  
  arc = {
    enable = true;
    runtime = "auto";  # or "level-zero" or "opencl"
  };
};
```

### Environment Variables

Key environment variables for PyTorch + IPEX:

- `PYTORCH_ENABLE_XPU=1` - Enable XPU (Intel GPU) support
- `IPEX_TILE_AS_DEVICE=1` - Treat each GPU tile as a separate device

## Architecture

The PyTorch + IPEX backend follows the exo inference engine protocol and integrates with:

- **Device Manager**: Detects and selects Intel Arc GPUs
- **Model Loader**: Loads HuggingFace models and applies IPEX optimizations
- **KV Cache Manager**: Manages key-value cache for transformer inference
- **Token Generator**: Implements sampling strategies (temperature, top-p, top-k)
- **Distributed Coordinator**: Coordinates multi-node inference in ring topology

## Development Status

### Completed (Task 1)

- ✅ NixOS package configuration (PyTorch, IPEX, drivers)
- ✅ Intel Arc GPU detection script
- ✅ IPEX functionality validation script
- ✅ Test harness for automated validation

### Planned

- Device Manager component
- Model Loader with IPEX optimization
- KV Cache Manager
- Inference Engine implementation
- Token Generator
- Distributed Coordinator
- Integration with exo architecture
- Monitoring and logging
- Testing and validation
- Documentation

## Testing

Run the test harness to verify your setup:

```bash
./test_pytorch_ipex_setup.sh
```

This will:
1. Detect Intel Arc GPUs
2. Validate IPEX functionality
3. Report any issues

## Troubleshooting

### GPU Not Detected

If Intel Arc GPU is not detected:

1. Verify GPU is present: `lspci | grep VGA`
2. Check driver is loaded: `lsmod | grep i915`
3. Verify compute runtime: `clinfo`
4. Check Level Zero: `ls /dev/dri/renderD*`

### IPEX Import Errors

If IPEX fails to import:

1. Verify PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Check IPEX installation: `pip list | grep intel-extension-for-pytorch`
3. Ensure compatible versions (PyTorch 2.0+ with IPEX 2.0+)

### Performance Issues

For performance problems:

1. Check GPU utilization: `intel_gpu_top`
2. Monitor memory usage: `intel_gpu_top` or validation script
3. Verify bfloat16 support is working
4. Check IPEX optimizations are applied

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel Arc GPU Documentation](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/overview.html)
- [exo Architecture](../../../../../../docs/architecture.md)

## License

This module is part of the exo project and follows the same license.


## Components (Continued)

### DeviceManager (`device_manager.py`)

Handles device detection, selection, and monitoring for Intel Arc GPUs, NVIDIA GPUs, and CPU fallback.

**Status**: ✅ Complete (Task 2)

Features:
- Automatic device detection and enumeration
- Priority-based device selection (Intel Arc > NVIDIA > CPU)
- Memory monitoring and health checks
- Device statistics and diagnostics

**Usage:**
```python
from exo.worker.engines.pytorch_ipex import DeviceManager

# Initialize device manager
device_manager = DeviceManager()

# Detect all available devices
devices = device_manager.detect_devices()

# Select optimal device
device_type, device_id = device_manager.select_device()

# Get device memory info
total, free = device_manager.get_device_memory(device_type, device_id)

# Check device health
is_healthy = device_manager.is_device_available(device_type, device_id)

# Get device statistics
stats = device_manager.get_device_stats(device_type, device_id)
```

### ModelLoader (`model_loader.py`)

Handles model loading from HuggingFace, IPEX optimization, and model sharding for distributed inference.

**Status**: ✅ Complete (Task 3)

Features:
- Async model loading from HuggingFace hub or local cache
- IPEX optimization with bfloat16 precision
- Model sharding support via TransformerShard wrapper
- Model validation and compatibility checking
- Tokenizer management for encoding/decoding

**Usage:**
```python
from exo.worker.engines.pytorch_ipex import ModelLoader

# Initialize model loader
model_loader = ModelLoader()

# Load model
model, tokenizer = await model_loader.load_model(
    shard_metadata=shard_metadata,
    device_type="xpu",
    device_id=0,
)

# Encode prompt
tokens = await model_loader.encode(model_id, "Hello, world!")

# Decode tokens
text = await model_loader.decode(model_id, tokens)
```

#### TransformerShard

Wrapper class that enables pipeline parallelism by executing only a specific range of transformer layers. Supports:
- First shard: includes embedding layer
- Middle shards: only transformer layers
- Last shard: includes normalization and language model head

## Development Status (Updated)

### Completed

- ✅ Task 1: Development environment setup
  - NixOS package configuration
  - Intel Arc GPU detection
  - IPEX functionality validation
- ✅ Task 2: Device Manager component
  - Device detection and enumeration
  - Device selection with priority logic
  - Memory monitoring and health checks
- ✅ Task 3: Model Loader component
  - ModelLoader class with async loading
  - IPEX optimization
  - Model sharding support
  - Model validation

### In Progress

- [ ] Task 4: KV Cache Manager
- [ ] Task 5: PyTorchInferenceEngine
- [ ] Task 6: Token Generator
- [ ] Task 7: Distributed Coordinator
- [ ] Task 8: Integration with exo architecture
- [ ] Task 9: Monitoring and logging
- [ ] Task 10: Testing and validation
- [ ] Task 11: Documentation

## Design Decisions

1. **Device Priority**: Intel Arc > NVIDIA > CPU (configurable via preference parameter)
2. **Memory-Based Selection**: When multiple devices of same type exist, select one with most free memory
3. **Graceful Degradation**: Falls back to CPU if no GPU available
4. **Async Model Loading**: Model loading runs in executor to avoid blocking the event loop
5. **IPEX Optimization**: Applied automatically when loading models on Intel Arc GPUs
6. **Model Sharding**: TransformerShard wrapper enables pipeline parallelism without modifying model code
7. **Caching**: Models and tokenizers are cached to avoid redundant loading
8. **Validation**: Comprehensive validation ensures model compatibility before inference

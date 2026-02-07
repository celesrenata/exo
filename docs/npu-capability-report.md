# Intel NPU Capability Report

This document describes the NPU capability assessment for Intel Core Ultra processors.

## Overview

Intel Core Ultra processors (formerly Meteor Lake) include an integrated Neural Processing Unit (NPU), also known as iVPU (integrated Vision Processing Unit). This NPU is designed for efficient AI inference workloads, particularly for vision, audio, and embedding tasks.

## Running the Capability Report

To generate a capability report on your system:

```bash
# Run the capability report script
uv run python -m exo.worker.engines.npu.capability_report

# This will:
# 1. Detect NPU hardware
# 2. Check kernel modules and drivers
# 3. Verify OpenVINO availability
# 4. Generate a detailed report
# 5. Save JSON output to npu_capability_report.json
```

## Expected Results on Core Ultra 9 185H

### Hardware Detection

On a system with Intel Core Ultra 9 185H, you should see:

- **CPU Model**: Intel(R) Core(TM) Ultra 9 185H
- **Core Ultra Processor**: Yes
- **Device Path**: `/dev/accel/accel0` or `/dev/dri/renderD128`

### Kernel Modules

The following kernel modules should be loaded:

- `intel_vpu` (newer kernels, 6.8+)
- `ivpu` (older kernels)

To manually check:
```bash
lsmod | grep -E "intel_vpu|ivpu"
```

To manually load the module:
```bash
sudo modprobe intel_vpu
```

### Device Nodes

Expected device nodes:

- `/dev/accel/accel0` - Dedicated NPU device (kernel 6.8+)
- `/dev/dri/renderD*` - Render device (may be shared with GPU)

To check device nodes:
```bash
ls -la /dev/accel/
ls -la /dev/dri/
```

### Driver Information

The driver should report:

- **Version**: Varies by kernel version
- **Description**: Intel VPU driver
- **Firmware**: intel/vpu/mtl_vpu.bin (or similar)

### OpenVINO Support

For NPU to be usable, OpenVINO must be installed and detect the NPU device:

```python
import openvino as ov
core = ov.Core()
print(core.available_devices())
# Should include 'NPU' or 'NPU.0'
```

## Supported Model Types

The Intel NPU is optimized for:

### ✅ Well-Supported Workloads

1. **Vision Models**
   - Image classification (ResNet, MobileNet, EfficientNet)
   - Object detection (YOLO, SSD)
   - Image segmentation
   - Face detection/recognition

2. **Audio Models**
   - Speech recognition (Whisper small models)
   - Audio classification
   - Voice activity detection (VAD)
   - Keyword spotting

3. **Embedding Models**
   - Text embeddings (sentence-transformers)
   - Small BERT models (<110M params)
   - Feature extraction

4. **Small Transformers**
   - Models under 1B parameters
   - Encoder-only models (BERT, RoBERTa)
   - Quantized models (INT8, FP16)

### ❌ Not Recommended for NPU

1. **Large LLM Decode**
   - Models >1B parameters
   - Autoregressive generation
   - High-throughput streaming
   - Use GPU/CPU instead

2. **Training**
   - NPU is inference-only
   - No gradient computation

3. **Custom Operations**
   - Limited operator support
   - Stick to standard model architectures

## Troubleshooting

### NPU Not Detected

If the capability report shows NPU as unavailable:

1. **Check CPU Model**
   ```bash
   cat /proc/cpuinfo | grep "model name"
   ```
   Ensure it's a Core Ultra processor (Meteor Lake or newer)

2. **Check Kernel Version**
   ```bash
   uname -r
   ```
   NPU support requires kernel 6.6+ (6.8+ recommended)

3. **Check Kernel Modules**
   ```bash
   lsmod | grep -E "intel_vpu|ivpu"
   ```
   If not loaded, try:
   ```bash
   sudo modprobe intel_vpu
   ```

4. **Check Device Permissions**
   ```bash
   ls -la /dev/accel/accel0
   ```
   Ensure your user has access (may need to add to `render` or `video` group)

### OpenVINO Not Available

If OpenVINO is not installed on NixOS:

```nix
# Add to your NixOS configuration
environment.systemPackages = with pkgs; [
  openvino
];
```

Or install in Python environment:
```bash
uv pip install openvino
```

### NPU Device Not Showing in OpenVINO

If OpenVINO is installed but doesn't detect NPU:

1. **Check OpenVINO Version**
   ```python
   import openvino as ov
   print(ov.__version__)
   ```
   Ensure version 2023.0 or newer

2. **Check NPU Plugin**
   ```bash
   # OpenVINO NPU plugin should be present
   find /nix/store -name "*openvino*npu*"
   ```

3. **Check Environment Variables**
   ```bash
   # May need to set
   export OPENVINO_NPU_ENABLED=1
   ```

## Performance Expectations

Based on Intel's specifications for Core Ultra NPU:

- **Compute**: ~10 TOPS (INT8)
- **Power**: 1-2W typical
- **Latency**: Lower than CPU for supported models
- **Throughput**: Best for batch size 1-4

### Example Performance (Estimated)

| Model Type | NPU | CPU (185H) | GPU (Arc) |
|------------|-----|------------|-----------|
| MobileNetV2 | ~50ms | ~80ms | ~30ms |
| Whisper-tiny | ~100ms | ~150ms | ~60ms |
| BERT-base | ~20ms | ~40ms | ~15ms |
| Llama-3B | ❌ | ~500ms | ~200ms |

*Note: Actual performance depends on model, quantization, and workload*

## Integration with exo

The NPU capability report informs whether NPU support should be enabled in exo:

```python
from exo.worker.engines.npu.discovery import discover_npu

capabilities = discover_npu()

if capabilities.available:
    print("NPU is available and can be used for:")
    for model_type in capabilities.supported_model_types:
        print(f"  - {model_type}")
else:
    print(f"NPU not available: {capabilities.error_message}")
```

## Running the Smoke Test

After confirming NPU is available, run the smoke test to verify functionality:

```bash
# Run the complete smoke test suite
./tests/test_npu_smoke.sh

# Or run just the smoke test
uv run python -m exo.worker.engines.npu.smoke_test
```

The smoke test will:

1. **Test 1: Simple Model Inference**
   - Creates a minimal model (input → ReLU → output)
   - Compiles for NPU
   - Executes inference
   - Verifies output shape

2. **Test 2: Verify NPU Usage**
   - Confirms model is executing on NPU (not CPU fallback)
   - Queries execution devices
   - Measures NPU latency

3. **Test 3: Latency Comparison**
   - Compiles same model for NPU and CPU
   - Runs 20 iterations on each
   - Compares average latency
   - Reports speedup/slowdown

### Expected Results

On a working NPU setup, you should see:

```
=== Test 1: Simple Model Inference ===
✅ Inference successful! Output shape: (1, 3, 224, 224)

=== Test 2: Verify NPU Usage ===
Execution devices: ['NPU.0']
✅ Verified: Model is executing on NPU
Average NPU latency: 2.34ms

=== Test 3: Latency Comparison ===
NPU average: 2.34ms
CPU average: 3.56ms
✅ NPU is 1.52x faster than CPU

✅ All smoke tests passed!
```

**Note**: For simple models, NPU may not show significant speedup. Real benefits appear with larger models (MobileNet, BERT, etc.).

## Next Steps

After running the capability report and smoke test:

1. **If NPU is available and tests pass**: NPU is ready for integration
2. **If NPU is available but tests fail**: Check OpenVINO configuration
3. **If NPU is not available**: Document the blockers and decide whether to:
   - Update kernel/drivers
   - Install OpenVINO
   - Skip NPU support for now

## References

- [Intel NPU Documentation](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)
- [OpenVINO NPU Plugin](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_NPU.html)
- [Linux Kernel DRM Accel](https://www.kernel.org/doc/html/latest/accel/index.html)

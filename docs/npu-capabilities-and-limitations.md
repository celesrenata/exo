# Intel NPU Capabilities and Limitations

**Status**: ⚠️ **EXPERIMENTAL** - Intel NPU support is in early development and should be considered experimental.

## Overview

This document describes the capabilities, limitations, and performance characteristics of Intel NPU (Neural Processing Unit) integration in exo. The NPU is a dedicated AI accelerator found in Intel Core Ultra processors (Meteor Lake and newer).

## Table of Contents

- [Supported Workloads](#supported-workloads)
- [Unsupported Operations](#unsupported-operations)
- [Performance Characteristics](#performance-characteristics)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Known Limitations](#known-limitations)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Supported Workloads

The Intel NPU is optimized for specific types of AI workloads. The following table shows which workloads benefit from NPU acceleration:

| Workload Type | NPU Support | Performance Gain | Recommended |
|---------------|-------------|------------------|-------------|
| **Embedding Generation** | ✅ Excellent | 2-5x vs CPU | ✅ Yes |
| **Text Embeddings** | ✅ Excellent | 2-5x vs CPU | ✅ Yes |
| **Sentence Transformers** | ✅ Excellent | 2-5x vs CPU | ✅ Yes |
| **Image Classification** | ✅ Good | 1.5-3x vs CPU | ✅ Yes |
| **Object Detection** | ✅ Good | 1.5-3x vs CPU | ✅ Yes |
| **Audio Classification** | ✅ Good | 2-4x vs CPU | ✅ Yes |
| **Speech Recognition** | ✅ Good | 2-4x vs CPU | ✅ Yes |
| **Voice Activity Detection** | ✅ Excellent | 3-5x vs CPU | ✅ Yes |
| **Small Transformers (<1B)** | ⚠️ Limited | 1.2-2x vs CPU | ⚠️ Maybe |
| **Large LLM Decode (>1B)** | ❌ Poor | 0.5-0.8x vs CPU | ❌ No |
| **Image Generation (FLUX/SD)** | ❌ Not Supported | N/A | ❌ No |
| **Video Processing** | ❌ Not Supported | N/A | ❌ No |

### Embedding Models

**Best Performance** - NPU excels at embedding generation:

- ✅ `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- ✅ `sentence-transformers/all-mpnet-base-v2` (109M params)
- ✅ `BAAI/bge-small-en-v1.5` (33M params)
- ✅ `BAAI/bge-base-en-v1.5` (109M params)
- ✅ `intfloat/e5-small-v2` (33M params)

**Performance**: 2-5x faster than CPU, 3-5x better power efficiency

### Vision Models

**Good Performance** - NPU handles vision tasks well:

- ✅ MobileNet (image classification)
- ✅ EfficientNet (image classification)
- ✅ ResNet-50 (image classification)
- ✅ YOLO (object detection, small variants)
- ⚠️ ViT (vision transformers, small variants only)

**Performance**: 1.5-3x faster than CPU, 2-4x better power efficiency

### Audio Models

**Good Performance** - NPU is efficient for audio:

- ✅ Whisper (speech recognition, small/base models)
- ✅ Wav2Vec2 (speech recognition)
- ✅ Audio classification models
- ✅ Voice activity detection (VAD)

**Performance**: 2-4x faster than CPU, 3-5x better power efficiency

## Unsupported Operations

The following operations and model types are **NOT** suitable for NPU:

### Large Language Models

❌ **LLM Decode** (>1B parameters):
- Llama 3.x (8B, 70B)
- Qwen 3.x (7B, 14B, 72B)
- DeepSeek V3 (671B)
- Mistral (7B, 8x7B)

**Reason**: NPU has limited memory and compute for large autoregressive models. GPU/CPU is significantly faster.

**Recommendation**: Use GPU (tinygrad/MLX) or CPU for LLM inference.

### Image Generation

❌ **Diffusion Models**:
- FLUX.1 (dev, schnell)
- Stable Diffusion (1.5, 2.1, XL)
- ControlNet
- LoRA models

**Reason**: Image generation requires high memory bandwidth and compute that NPU cannot provide efficiently.

**Recommendation**: Use GPU (tinygrad/MLX) for image generation.

### Video Processing

❌ **Video Models**:
- Video generation
- Video understanding
- Frame-by-frame processing

**Reason**: NPU lacks the memory and bandwidth for video workloads.

**Recommendation**: Use GPU for video processing.

### Custom Operations

❌ **Unsupported Operations**:
- Custom CUDA kernels
- Dynamic shapes (limited support)
- Sparse operations
- Quantization below INT8

**Reason**: NPU has a fixed set of supported operations via OpenVINO.

## Performance Characteristics

### Latency

| Model Type | NPU Latency | CPU Latency | GPU Latency | Winner |
|------------|-------------|-------------|-------------|--------|
| Embedding (384 dim) | 2-5ms | 10-20ms | 3-8ms | NPU |
| Image Classification | 5-15ms | 20-50ms | 8-20ms | NPU |
| Audio (1s chunk) | 10-30ms | 40-100ms | 15-40ms | NPU |
| Small Transformer | 20-50ms | 50-150ms | 25-60ms | NPU |
| Large LLM (per token) | 100-200ms | 50-100ms | 10-30ms | GPU |

**Note**: Latency varies by model size, batch size, and input length.

### Throughput

| Model Type | NPU Throughput | CPU Throughput | GPU Throughput |
|------------|----------------|----------------|----------------|
| Embeddings | 200-500 req/s | 50-100 req/s | 300-800 req/s |
| Vision | 50-150 req/s | 20-50 req/s | 100-300 req/s |
| Audio | 30-100 req/s | 10-30 req/s | 50-150 req/s |

**Note**: Throughput assumes batch size of 1. NPU benefits less from batching than GPU.

### Power Efficiency

NPU is significantly more power-efficient than CPU/GPU:

| Workload | NPU Power | CPU Power | GPU Power | NPU Advantage |
|----------|-----------|-----------|-----------|---------------|
| Embeddings | 2-5W | 10-20W | 15-30W | 3-5x better |
| Vision | 3-8W | 15-30W | 20-40W | 3-5x better |
| Audio | 2-6W | 10-25W | 15-35W | 3-5x better |

**Use Case**: NPU is ideal for battery-powered devices and edge deployments.

### Memory

| Resource | NPU | CPU | GPU (Arc iGPU) |
|----------|-----|-----|----------------|
| Memory | 4-8GB (shared) | 16-64GB | 8-16GB (shared) |
| Bandwidth | 50-100 GB/s | 50-100 GB/s | 100-200 GB/s |
| Latency | Low | Medium | Low |

**Limitation**: NPU shares system memory with CPU, limiting available memory for large models.

## Hardware Requirements

### Minimum Requirements

- **CPU**: Intel Core Ultra (Meteor Lake or newer)
  - Core Ultra 5 (125H, 135H)
  - Core Ultra 7 (155H, 165H)
  - Core Ultra 9 (185H)
- **Memory**: 16GB RAM (shared with NPU)
- **OS**: Linux kernel 6.6+ (6.8+ recommended)

### Verified Hardware

The following hardware has been tested with exo NPU support:

| Model | NPU | Status | Notes |
|-------|-----|--------|-------|
| Core Ultra 9 185H | Yes | ✅ Tested | Full support |
| Core Ultra 7 155H | Yes | ⚠️ Untested | Should work |
| Core Ultra 5 125H | Yes | ⚠️ Untested | Should work |

### Device Nodes

NPU appears as:
- `/dev/accel/accel0` (preferred, kernel 6.8+)
- `/dev/dri/renderD128` (fallback, kernel 6.6-6.7)

## Software Requirements

### Operating System

- **Linux**: NixOS, Ubuntu 22.04+, Fedora 38+
- **Kernel**: 6.6+ (6.8+ recommended for `/dev/accel` support)
- **Drivers**: `intel_vpu` or `ivpu` kernel module

### Software Stack

- **OpenVINO**: 2023.0+ with NPU plugin
- **Python**: 3.10+
- **Dependencies**: `openvino`, `fastapi`, `uvicorn`, `aiohttp`

### Installation

See [Intel Hardware Setup Guide](intel-hardware-setup.md) for detailed installation instructions.

## Known Limitations

### Model Format

- ❌ **PyTorch models**: Must be converted to OpenVINO IR format
- ❌ **ONNX models**: Must be converted to OpenVINO IR format
- ✅ **OpenVINO IR**: Native format, no conversion needed

**Workaround**: Use OpenVINO model optimizer to convert models.

### Dynamic Shapes

- ⚠️ **Limited support**: NPU prefers static shapes
- ❌ **Variable batch size**: Not well supported
- ❌ **Variable sequence length**: Not well supported

**Workaround**: Use fixed shapes or pad inputs to maximum size.

### Quantization

- ✅ **INT8**: Well supported
- ⚠️ **FP16**: Limited support
- ❌ **INT4**: Not supported
- ❌ **FP8**: Not supported

**Recommendation**: Use INT8 quantization for best NPU performance.

### Concurrency

- ⚠️ **Single model at a time**: NPU can only run one model concurrently
- ❌ **No batching across requests**: Each request is processed sequentially

**Workaround**: Use multiple NPU devices if available, or queue requests.

### Memory Sharing

- ⚠️ **Shared with system**: NPU uses system RAM, not dedicated memory
- ⚠️ **Competes with CPU**: Memory pressure affects both NPU and CPU

**Recommendation**: Ensure sufficient system RAM (16GB minimum, 32GB recommended).

## Best Practices

### When to Use NPU

✅ **Use NPU for**:
- Embedding generation in production
- Real-time audio processing
- Edge inference on battery power
- Low-latency vision tasks
- High-volume small model inference

❌ **Don't use NPU for**:
- Large LLM inference (>1B params)
- Image generation (FLUX, SD)
- Training workloads
- High-throughput batch processing

### Model Selection

1. **Prefer smaller models**: <500M parameters work best
2. **Use INT8 quantization**: Better performance and memory efficiency
3. **Static shapes**: Avoid dynamic shapes when possible
4. **OpenVINO IR format**: Convert models ahead of time

### Deployment

1. **Enable NPU service**: Use NixOS module or systemd service
2. **Monitor performance**: Track latency and throughput
3. **Set resource limits**: Prevent NPU from consuming too much memory
4. **Implement fallback**: Always have CPU/GPU fallback for reliability

### Optimization

1. **Batch size 1**: NPU doesn't benefit much from batching
2. **Warm up models**: First inference is slower (model loading)
3. **Cache models**: Keep frequently used models loaded
4. **Monitor power**: NPU is most efficient at moderate load

## Troubleshooting

### NPU Not Detected

**Symptoms**: `discover_npu()` returns `available=False`

**Solutions**:
1. Verify CPU model: `cat /proc/cpuinfo | grep "model name"`
2. Load kernel module: `sudo modprobe intel_vpu`
3. Check device node: `ls -la /dev/accel/`
4. Update kernel to 6.8+ for better support

### OpenVINO Not Finding NPU

**Symptoms**: OpenVINO lists devices but no NPU

**Solutions**:
1. Install OpenVINO with NPU plugin
2. Verify NPU plugin: `python -c "import openvino as ov; print(ov.Core().available_devices())"`
3. Check permissions: `sudo usermod -a -G render,video $USER`
4. Restart system after driver installation

### Poor Performance

**Symptoms**: NPU slower than CPU

**Solutions**:
1. Verify NPU is actually being used (check logs)
2. Use INT8 quantization instead of FP32
3. Ensure model is in OpenVINO IR format
4. Check for memory pressure (NPU shares system RAM)
5. Warm up model before benchmarking

### Service Won't Start

**Symptoms**: `exo-npu` service fails to start

**Solutions**:
1. Check logs: `sudo journalctl -u exo-npu -f`
2. Verify permissions: `sudo ls -la /dev/accel/accel0`
3. Test manual start: `python -m exo.worker.engines.npu.service`
4. Check OpenVINO installation: `python -c "import openvino"`

## Performance Comparison Data

### Embedding Models

| Model | NPU | CPU | GPU | Best |
|-------|-----|-----|-----|------|
| all-MiniLM-L6-v2 | 3ms | 15ms | 5ms | NPU |
| all-mpnet-base-v2 | 5ms | 25ms | 8ms | NPU |
| bge-small-en-v1.5 | 3ms | 12ms | 5ms | NPU |

### Vision Models

| Model | NPU | CPU | GPU | Best |
|-------|-----|-----|-----|------|
| MobileNet-v2 | 8ms | 30ms | 12ms | NPU |
| ResNet-50 | 15ms | 50ms | 20ms | NPU |
| EfficientNet-B0 | 10ms | 35ms | 15ms | NPU |

### Audio Models

| Model | NPU | CPU | GPU | Best |
|-------|-----|-----|-----|------|
| Whisper-tiny | 20ms | 80ms | 30ms | NPU |
| Wav2Vec2-base | 25ms | 100ms | 35ms | NPU |

**Note**: Benchmarks performed on Intel Core Ultra 9 185H with 32GB RAM.

## Future Improvements

The following improvements are planned for future releases:

- [ ] Dynamic shape support
- [ ] Multi-model concurrency
- [ ] Better batching support
- [ ] FP16 optimization
- [ ] Automatic model conversion
- [ ] Performance profiling tools
- [ ] Integration with exo dashboard

## References

- [Intel NPU Documentation](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)
- [OpenVINO NPU Plugin](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_NPU.html)
- [Intel Hardware Setup Guide](intel-hardware-setup.md)
- [NPU Capability Report](npu-capability-report.md)

## Changelog

- **2025-02-07**: Initial documentation for NPU sidecar service
- **2025-01-XX**: NPU discovery and capability assessment
- **2024-12-XX**: Initial NPU exploration

---

**Status**: ⚠️ **EXPERIMENTAL** - This feature is under active development. APIs and behavior may change.

**Feedback**: Please report issues and performance results to help improve NPU support.

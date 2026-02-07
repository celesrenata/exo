# Tinygrad Backend Documentation

This document describes the tinygrad inference backend for exo, which enables cross-platform AI inference with support for Intel Arc GPUs, NVIDIA GPUs, and CPU execution.

## Overview

The tinygrad backend is an alternative to MLX that provides:

- **Cross-platform support**: Works on Linux, macOS, and Windows
- **Multiple hardware targets**: CPU, GPU (Intel Arc, NVIDIA, AMD), Metal
- **Intel Arc optimization**: Native support for Intel Arc iGPUs via Level Zero
- **Automatic fallback**: Gracefully falls back from GPU to CPU when needed
- **Performance monitoring**: Built-in metrics collection and observability

## Architecture

### Backend Components

```
src/exo/worker/engines/tinygrad/
├── tinygrad_backend.py    # Main backend implementation
├── device_config.py        # Hardware detection and configuration
├── intel_arc.py           # Intel Arc GPU detection and runtime selection
├── model_loader.py        # Model weight loading and conversion
├── generator.py           # Text generation logic
└── metrics.py             # Performance metrics collection (shared)
```

### Backend Lifecycle

1. **Initialization**: Backend is created with a shard downloader
2. **Device Configuration**: Hardware capabilities are detected
3. **Runtime Selection**: Best available runtime is chosen (Level Zero, OpenCL, etc.)
4. **Model Loading**: Weights are loaded and converted to tinygrad format
5. **Inference**: Text generation with metrics collection
6. **Cleanup**: Resources are released

## Hardware Support

### Intel Arc iGPU

**Supported Runtimes:**
- Level Zero (preferred) - Best performance
- OpenCL (fallback) - Broader compatibility

**Detection:**
- Checks `lspci` for Intel Arc devices
- Verifies `/sys/class/drm` for Intel GPU
- Tests runtime availability

**Configuration:**
```python
# Automatic configuration
backend = TinygradBackend(shard_downloader)
# Backend will detect Intel Arc and configure Level Zero

# Manual configuration via environment
os.environ["GPU"] = "1"
os.environ["LEVEL_ZERO"] = "1"
```

### NVIDIA GPU

**Supported Runtime:**
- CUDA

**Detection:**
- Checks for `nvidia-smi` command
- Verifies CUDA runtime availability

### Apple Metal

**Supported Runtime:**
- Metal (on macOS)

**Detection:**
- Automatically detected on macOS systems

### CPU Fallback

**When Used:**
- No GPU hardware available
- GPU detected but no runtime available
- GPU initialization fails

**Configuration:**
- Automatically configured with optimal CPU settings
- Uses all available CPU cores

## Device Configuration

### Automatic Detection

The backend automatically detects and configures the best available hardware:

```python
from exo.worker.engines.tinygrad import TinygradBackend

backend = TinygradBackend(shard_downloader)
# Device is automatically detected and configured
```

### Detection Priority

1. macOS: Metal GPU
2. Linux/Windows with Intel Arc: Level Zero → OpenCL → CPU
3. Linux/Windows with NVIDIA: CUDA → CPU
4. Linux/Windows with AMD: OpenCL → CPU
5. Fallback: CPU

### Manual Override

Override automatic detection with environment variables:

```bash
# Force GPU device
export GPU=1

# Force specific runtime
export LEVEL_ZERO=1  # Intel Level Zero
export OPENCL=1      # OpenCL
export CUDA=1        # NVIDIA CUDA
export METAL=1       # Apple Metal

# Force CPU (disable GPU)
unset GPU
```

## Runtime Selection

### Level Zero (Intel Arc)

**Advantages:**
- Optimal performance for Intel Arc GPUs
- Lower latency than OpenCL
- Better memory management

**Requirements:**
- `intel-compute-runtime` package
- `libze_loader.so` library
- Intel Arc or Xe graphics hardware

**Verification:**
```bash
# Check for Level Zero library
ldconfig -p | grep libze_loader

# Test Level Zero devices
clinfo | grep -A 10 "Level-Zero"
```

### OpenCL (Fallback)

**Advantages:**
- Broader hardware compatibility
- Works with Intel, NVIDIA, AMD GPUs
- Fallback when Level Zero unavailable

**Requirements:**
- OpenCL runtime (vendor-specific)
- `libOpenCL.so` library
- `pyopencl` Python package

**Verification:**
```bash
# Check for OpenCL library
ldconfig -p | grep libOpenCL

# List OpenCL platforms
clinfo
```

## Model Loading

### Weight Format

Tinygrad uses its own tensor format. The backend handles conversion:

1. Download weights from HuggingFace (safetensors format)
2. Load weights into memory
3. Convert to tinygrad tensor format
4. Transfer to configured device (GPU/CPU)

### Sharding Support

The backend supports model sharding for distributed inference:

```python
# Load a specific shard
await backend.load_checkpoint(shard_metadata, checkpoint_path)

# Shard metadata specifies which layers to load
# Backend handles partial model loading
```

## Performance Monitoring

### Metrics Collection

The backend includes built-in metrics collection:

```python
# Initialize metrics collector
backend.initialize_metrics_collector(runner_id)

# Metrics are automatically recorded during inference
# Access current statistics
stats = backend.get_metrics_stats()

print(f"Inferences: {stats['inference_count']}")
print(f"Avg throughput: {stats['avg_tokens_per_second']:.2f} tok/s")
print(f"Recent throughput: {stats['recent_tokens_per_second']:.2f} tok/s")
```

### Collected Metrics

- **Inference count**: Total number of inferences
- **Total tokens**: Total tokens generated
- **Total time**: Total inference time
- **Tokens per second**: Average and recent throughput
- **Memory usage**: Average memory consumption
- **GPU utilization**: GPU usage percentage (if available)

### GPU Metrics

For GPU backends, additional metrics are collected:

```python
# Collect current GPU metrics
metrics = backend.collect_metrics()

print(f"Device: {metrics['device_name']}")
print(f"Runtime: {metrics['runtime']}")
print(f"Memory: {metrics['memory_used_mb']}/{metrics['memory_total_mb']} MB")
print(f"Utilization: {metrics['utilization_percent']}%")
```

## Logging and Observability

### Structured Logging

The backend uses structured logging with loguru:

```python
# All log messages include structured fields
logger.info(
    "Backend initialized",
    backend_type="tinygrad",
    device="GPU",
    runtime="LEVEL_ZERO",
    device_name="Intel Arc Graphics",
    memory_gb=16.0
)
```

### Key Log Events

**Backend Initialization:**
```
INFO Backend initialized backend_type=tinygrad device=GPU runtime=LEVEL_ZERO
```

**Device Detection:**
```
INFO Device capabilities detected device_name="Intel Arc Graphics" device_type=GPU memory_gb=16.0
```

**Runtime Selection:**
```
INFO Runtime selection: Level Zero (optimal) runtime=LEVEL_ZERO reason="Best performance for Intel Arc"
```

**Fallback Events:**
```
WARNING Fallback to OpenCL runtime requested_runtime=LEVEL_ZERO fallback_runtime=OPENCL
WARNING GPU fallback to CPU requested_device=GPU fallback_device=CPU
```

### Event Emission

The backend emits events to the cluster state:

- `BackendInitialized`: When backend successfully initializes
- `BackendFailed`: When backend initialization fails
- `GPUMetricsCollected`: Periodic GPU performance metrics

These events are visible in the dashboard for real-time monitoring.

## Error Handling

### Initialization Errors

```python
try:
    backend = TinygradBackend(shard_downloader)
    backend._configure_device()
except Exception as e:
    logger.error(f"Backend initialization failed: {e}")
    # Fallback to CPU or alternative backend
```

### Runtime Errors

```python
try:
    await backend.infer_tensor(request_id, shard_metadata, input_data)
except RuntimeError as e:
    logger.error(f"Inference failed: {e}")
    # Handle error, possibly retry or fallback
```

### Graceful Degradation

The backend implements graceful degradation:

1. Try Level Zero (Intel Arc)
2. Fall back to OpenCL
3. Fall back to CPU
4. Report error if all fail

## Configuration Examples

### NixOS Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    
    arc = {
      enable = true;
      runtime = "level-zero";  # or "opencl" or "auto"
    };
  };
}
```

### Environment Variables

```bash
# Enable tinygrad backend
export EXO_TINYGRAD_ENABLED=1

# Force Intel Arc with Level Zero
export GPU=1
export LEVEL_ZERO=1

# Start exo
exo -vv
```

### Python Configuration

```python
from exo.worker.engines.tinygrad import TinygradBackend
from exo.download.shard_download import ShardDownloader

# Create backend
downloader = ShardDownloader()
backend = TinygradBackend(downloader)

# Configure device (automatic)
backend._configure_device()

# Or force specific device
backend.device = "GPU"
backend.runtime = "LEVEL_ZERO"
```

## Performance Tuning

### Memory Management

```python
# Monitor memory usage
metrics = backend.collect_metrics()
if metrics['memory_used_mb'] > threshold:
    # Reduce batch size or clear cache
    pass
```

### Batch Size Optimization

```python
# Adjust batch size based on available memory
available_memory = backend.device_capabilities.memory_gb
optimal_batch_size = calculate_batch_size(available_memory)
```

### GPU Utilization

```python
# Monitor GPU utilization
metrics = backend.collect_metrics()
if metrics['utilization_percent'] < 50:
    # Increase batch size or concurrent requests
    pass
```

## Troubleshooting

### Backend Not Initializing

**Symptoms:**
- Backend initialization fails
- No device detected

**Solutions:**
1. Check hardware availability: `lspci | grep VGA`
2. Verify runtime installation: `ldconfig -p | grep libze_loader`
3. Check logs for error messages: `exo -vv`

### Poor Performance

**Symptoms:**
- Low tokens per second
- High latency

**Solutions:**
1. Verify GPU is being used (not CPU fallback)
2. Check GPU utilization: `intel_gpu_top`
3. Monitor memory usage
4. Adjust batch size

### Memory Errors

**Symptoms:**
- Out of memory errors
- Crashes during inference

**Solutions:**
1. Reduce model size or use quantization
2. Decrease batch size
3. Monitor memory usage: `backend.collect_metrics()`
4. Clear cache between inferences

### Runtime Not Available

**Symptoms:**
- Falls back to OpenCL or CPU
- Level Zero not detected

**Solutions:**
1. Install `intel-compute-runtime` package
2. Verify library: `ldconfig -p | grep libze_loader`
3. Check device permissions: `ls -la /dev/dri/`
4. Reload graphics drivers

## API Reference

### TinygradBackend

```python
class TinygradBackend(InferenceBackend):
    """Tinygrad-based inference backend."""
    
    def __init__(self, shard_downloader: Any) -> None:
        """Initialize backend with shard downloader."""
    
    async def load_checkpoint(
        self, 
        shard_metadata: ShardMetadata, 
        path: str
    ) -> None:
        """Load model weights from checkpoint."""
    
    async def infer_tensor(
        self,
        request_id: str,
        shard_metadata: ShardMetadata,
        input_data: np.ndarray,
        inference_state: dict | None = None,
    ) -> tuple[np.ndarray, dict | None]:
        """Execute tensor inference."""
    
    def collect_metrics(self) -> dict[str, Any]:
        """Collect current GPU metrics."""
    
    def initialize_metrics_collector(self, runner_id: RunnerId) -> None:
        """Initialize metrics collection."""
    
    def get_metrics_stats(self) -> dict[str, Any]:
        """Get aggregated metrics statistics."""
```

### Device Configuration

```python
def detect_capabilities() -> DeviceCapabilities:
    """Detect available hardware capabilities."""

@dataclass
class DeviceCapabilities:
    device_type: Literal["CPU", "GPU", "METAL"]
    runtime: str | None
    memory_gb: float
    compute_units: int | None
    device_name: str
```

### Intel Arc Detection

```python
def detect_intel_arc() -> bool:
    """Check if Intel Arc iGPU is available."""

def check_level_zero_available() -> bool:
    """Verify Level Zero runtime is available."""

def check_opencl_available() -> bool:
    """Verify OpenCL runtime is available."""

def select_runtime() -> Literal["LEVEL_ZERO", "OPENCL"] | None:
    """Select best available runtime for Intel Arc."""
```

## Best Practices

1. **Always use verbose logging** during initial setup: `exo -vv`
2. **Monitor metrics** to ensure optimal performance
3. **Verify runtime selection** matches your hardware
4. **Test fallback behavior** by disabling runtimes
5. **Keep drivers updated** for best compatibility
6. **Use Level Zero** for Intel Arc when available
7. **Monitor GPU utilization** to optimize batch sizes
8. **Collect metrics** for performance analysis

## References

- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Intel Compute Runtime](https://github.com/intel/compute-runtime)
- [Level Zero Specification](https://spec.oneapi.io/level-zero/latest/)
- [OpenCL Documentation](https://www.khronos.org/opencl/)
- [Intel Arc Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/a-series/overview.html)

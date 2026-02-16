# PyTorch + IPEX Backend Usage Guide

## Overview

The PyTorchIPEXBackend provides Intel Arc GPU support for exo using PyTorch and Intel Extension for PyTorch (IPEX).

## Basic Usage

### Initialization

```python
from exo.worker.engines.pytorch_ipex import PyTorchIPEXBackend

# Create backend instance
backend = PyTorchIPEXBackend()

# Backend automatically:
# - Detects available devices (Intel Arc, NVIDIA, CPU)
# - Selects optimal device
# - Initializes KV cache manager
# - Sets up logging
```

### Encoding Text

```python
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.shared.models.model_cards import ModelCard

# Create shard metadata
shard_metadata = PipelineShardMetadata(
    model_card=ModelCard(model_id="meta-llama/Llama-3.2-3B-Instruct"),
    device_rank=0,
    world_size=1,
    start_layer=0,
    end_layer=28,
    n_layers=28,
)

# Encode prompt
prompt = "Hello, how are you?"
tokens = await backend.encode(shard_metadata, prompt)
print(f"Encoded to {len(tokens)} tokens")
```

### Running Inference

```python
import numpy as np

# Prepare input
input_data = tokens.reshape(1, -1)  # [batch_size, seq_len]

# Run inference
output, state = await backend.infer_tensor(
    request_id="request-123",
    shard_metadata=shard_metadata,
    input_data=input_data,
    inference_state=None,  # First inference
)

print(f"Output shape: {output.shape}")
print(f"State: {state}")
```

### Sampling Tokens

```python
# Sample next token from logits
sampled_token = await backend.sample(output)
print(f"Sampled token: {sampled_token}")
```

### Decoding Text

```python
# Decode tokens back to text
text = await backend.decode(shard_metadata, sampled_token)
print(f"Generated text: {text}")
```

### Getting Statistics

```python
# Get backend statistics
stats = backend.get_stats()
print(f"Device: {stats['device_type']}:{stats['device_id']}")
print(f"Loaded models: {stats['loaded_models']}")
print(f"Cache stats: {stats['cache']}")
```

### Cleanup

```python
# Clean up resources
backend.cleanup()
```

## Advanced Usage

### Device Selection

```python
from exo.worker.engines.pytorch_ipex import DeviceManager, DeviceType

# Create device manager
manager = DeviceManager()

# Detect all devices
devices = manager.detect_devices()
for device in devices:
    print(f"{device.device_type}: {device.name}")

# Select specific device type
device_type, device_id = manager.select_device(preference=DeviceType.INTEL_ARC)
```

### Model Loading

```python
from exo.worker.engines.pytorch_ipex import ModelLoader

# Create model loader
loader = ModelLoader()

# Load model
model, tokenizer = await loader.load_model(
    shard_metadata=shard_metadata,
    device_type="xpu",
    device_id=0,
)

# Model is automatically optimized with IPEX
```

### KV Cache Management

```python
from exo.worker.engines.pytorch_ipex import KVCacheManager

# Create cache manager
cache_manager = KVCacheManager(
    device_type="xpu",
    device_id=0,
    memory_threshold_percent=80.0,
)

# Create cache for request
cache = cache_manager.create_cache(
    request_id="request-123",
    num_layers=28,
    max_length=8192,
)

# Get cache statistics
stats = cache_manager.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"Active caches: {stats['active_caches']}")
```

## Error Handling

### Custom Exceptions

```python
from exo.worker.engines.pytorch_ipex import (
    DeviceError,
    ModelError,
    InferenceError,
    CacheError,
)

try:
    output, state = await backend.infer_tensor(...)
except DeviceError as e:
    print(f"Device error: {e}")
    print(f"Device: {e.device_type}:{e.device_id}")
except ModelError as e:
    print(f"Model error: {e}")
    print(f"Model: {e.model_id}")
except InferenceError as e:
    print(f"Inference error: {e}")
    print(f"Request: {e.request_id}")
except CacheError as e:
    print(f"Cache error: {e}")
    print(f"Request: {e.request_id}")
```

### Error Context

All custom exceptions include:
- Descriptive error message
- Context information (device, model, request)
- Original exception (if applicable)

## Complete Example

```python
import asyncio
import numpy as np
from exo.worker.engines.pytorch_ipex import PyTorchIPEXBackend
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.shared.models.model_cards import ModelCard


async def generate_text(prompt: str, max_tokens: int = 50) -> str:
    """Generate text using PyTorchIPEXBackend."""
    
    # Initialize backend
    backend = PyTorchIPEXBackend()
    
    try:
        # Create shard metadata
        shard_metadata = PipelineShardMetadata(
            model_card=ModelCard(model_id="meta-llama/Llama-3.2-3B-Instruct"),
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=28,
            n_layers=28,
        )
        
        # Encode prompt
        tokens = await backend.encode(shard_metadata, prompt)
        input_data = tokens.reshape(1, -1)
        
        # Generate tokens
        generated_tokens = []
        state = None
        
        for i in range(max_tokens):
            # Run inference
            output, state = await backend.infer_tensor(
                request_id=f"request-{i}",
                shard_metadata=shard_metadata,
                input_data=input_data,
                inference_state=state,
            )
            
            # Sample next token
            next_token = await backend.sample(output)
            generated_tokens.append(next_token)
            
            # Use next token as input for next iteration
            input_data = next_token
            
            # Check for EOS token (assuming token_id 2)
            if next_token[0, 0] == 2:
                break
        
        # Decode generated tokens
        generated_text = await backend.decode(
            shard_metadata,
            np.concatenate(generated_tokens, axis=1),
        )
        
        return generated_text
        
    finally:
        # Always cleanup
        backend.cleanup()


# Run generation
async def main():
    text = await generate_text("Hello, how are you?")
    print(f"Generated: {text}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Enable XPU support
export PYTORCH_ENABLE_XPU=1

# Enable IPEX tile-as-device
export IPEX_TILE_AS_DEVICE=1

# Set log level
export LOG_LEVEL=INFO
```

### Memory Management

The KV cache manager automatically evicts old caches when memory usage exceeds 80% of available GPU memory. You can adjust this threshold:

```python
cache_manager = KVCacheManager(
    device_type="xpu",
    device_id=0,
    memory_threshold_percent=70.0,  # More aggressive eviction
)
```

## Performance Tips

1. **Reuse Backend Instance**: Create one backend instance and reuse it for multiple requests
2. **Batch Requests**: Process multiple requests in parallel when possible
3. **Monitor Memory**: Use `get_stats()` to monitor memory usage
4. **Cache Cleanup**: Call `cleanup()` when done to free GPU memory
5. **Use bfloat16**: IPEX automatically uses bfloat16 for better performance

## Troubleshooting

### PyTorch Not Found

```
RuntimeError: PyTorch is required for PyTorchIPEXBackend
```

Install PyTorch:
```bash
pip install torch
```

### IPEX Not Found

IPEX is optional but recommended for Intel Arc GPUs:
```bash
pip install intel-extension-for-pytorch
```

### Device Not Available

```
DeviceError: Device is no longer available
```

Check device availability:
```python
manager = DeviceManager()
devices = manager.detect_devices()
print(devices)
```

### Out of Memory

```
DeviceError: Failed to move input tensor to device
```

Reduce batch size or enable more aggressive cache eviction:
```python
cache_manager = KVCacheManager(
    device_type="xpu",
    device_id=0,
    memory_threshold_percent=60.0,  # More aggressive
)
```

## See Also

- [Design Document](../../.kiro/specs/pytorch-ipex-intel-arc/design.md)
- [Requirements Document](../../.kiro/specs/pytorch-ipex-intel-arc/requirements.md)
- [Task List](../../.kiro/specs/pytorch-ipex-intel-arc/tasks.md)

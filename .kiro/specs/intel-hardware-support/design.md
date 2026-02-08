# Design Document

## Overview

This design implements Intel Arc GPU acceleration for exo by restoring tinygrad backend integration using https://github.com/Scottcjn/exo-cuda as the reference implementation. The exo-cuda repository demonstrates how to integrate tinygrad with exo for CUDA GPUs; we will adapt this pattern for Intel Arc GPUs using Level Zero or OpenCL runtimes.

**Key Principle**: Use exo-cuda as the blueprint for tinygrad backend wiring (backend selection, device detection, multi-node validation), then swap CUDA-specific runtime calls with Intel GPU equivalents.

**Execution Engine**: Tinygrad is the primary execution engine. Do NOT use IPEX or PyTorch unless pivoting to a completely different backend strategy.

## Architecture

### Reference: exo-cuda Implementation Pattern

The exo-cuda repository shows the complete pattern for tinygrad integration:

1. **Runner Integration**: Lazy-load tinygrad modules in runner.py when TinygradRingInstance is detected
2. **Device Detection**: Use tinygrad's device enumeration to discover GPUs
3. **Model Loading**: Load weights using tinygrad operations
4. **Inference Loop**: Execute forward passes with tinygrad, emit TokenChunk events
5. **Multi-Node**: Support TinygradRingInstance for distributed inference across nodes

### Adaptation for Intel Arc

**CUDA → Intel GPU Mapping**:
- CUDA device → Intel Arc iGPU (via Level Zero or OpenCL)
- `TINYGRAD_BACKEND=CUDA` → `TINYGRAD_BACKEND=GPU`
- CUDA runtime calls → Level Zero or OpenCL runtime calls
- CUDA device properties → Intel GPU properties (compute units, memory, etc.)

### Component Architecture

```
src/exo/worker/
├── runner/
│   ├── bootstrap.py           # Detect TinygradRingInstance, set env vars
│   └── runner.py              # Lazy-load tinygrad, implement inference loop
├── engines/
│   ├── tinygrad/
│   │   ├── tinygrad_backend.py    # Tinygrad inference engine
│   │   ├── model_loader.py        # Load models with tinygrad
│   │   ├── generator.py           # Text generation with tinygrad
│   │   ├── device_config.py       # Intel GPU device detection
│   │   └── intel_arc.py           # Intel Arc specific configuration
│   └── backend_selector.py    # Select backend based on instance type
```

## Design Components

### 1. Runner Integration (from exo-cuda)

**File**: `src/exo/worker/runner/runner.py`

**Pattern from exo-cuda**:
```python
def main(bound_instance, event_sender, task_receiver):
    instance = bound_instance.instance
    
    # Detect backend from instance type
    if isinstance(instance, TinygradRingInstance):
        backend_type = "tinygrad"
        
        # Lazy-load tinygrad modules
        from exo.worker.engines.tinygrad.tinygrad_backend import TinygradBackend
        from exo.worker.engines.tinygrad.model_loader import load_tinygrad_model
        from exo.worker.engines.tinygrad.generator import tinygrad_generate
        
        # Initialize backend
        backend = TinygradBackend(device="GPU")  # or "CPU"
        
        # Run inference loop
        for task in tasks:
            match task:
                case LoadModel():
                    model, tokenizer = load_tinygrad_model(...)
                case TextGeneration():
                    for chunk in tinygrad_generate(model, tokenizer, ...):
                        event_sender.send(ChunkGenerated(...))
    else:
        # MLX backend (existing code)
        ...
```

**Key Changes**:
- Check `isinstance(instance, TinygradRingInstance)` early
- Lazy-load tinygrad modules only when needed
- Implement same task handling loop as MLX (LoadModel, TextGeneration, etc.)
- Emit same events (ChunkGenerated, RunnerStatusUpdated, etc.)

### 2. Device Detection (adapted from exo-cuda)

**File**: `src/exo/worker/engines/tinygrad/device_config.py`

**Pattern from exo-cuda** (CUDA version):
```python
def detect_cuda_devices():
    """Detect CUDA GPUs using tinygrad."""
    import tinygrad
    devices = tinygrad.Device.enumerate("CUDA")
    return [{"name": d.name, "memory": d.memory} for d in devices]
```

**Adapted for Intel Arc**:
```python
def detect_intel_gpu():
    """Detect Intel Arc GPU using tinygrad."""
    import os
    
    # Set tinygrad to use GPU backend
    os.environ["TINYGRAD_BACKEND"] = "GPU"
    
    try:
        import tinygrad
        from tinygrad import Device
        
        # Try Level Zero first
        if check_level_zero_available():
            devices = Device.enumerate("GPU")  # tinygrad will use Level Zero
            return {
                "available": True,
                "runtime": "LEVEL_ZERO",
                "devices": [{"name": d.name, "memory": d.memory} for d in devices]
            }
        
        # Fall back to OpenCL
        elif check_opencl_available():
            devices = Device.enumerate("GPU")  # tinygrad will use OpenCL
            return {
                "available": True,
                "runtime": "OPENCL",
                "devices": [{"name": d.name, "memory": d.memory} for d in devices]
            }
        
        return {"available": False, "runtime": None, "devices": []}
    
    except Exception as e:
        logger.error(f"GPU detection failed: {e}")
        return {"available": False, "runtime": None, "devices": []}

def check_level_zero_available() -> bool:
    """Check if Level Zero runtime is available."""
    try:
        # Check for Level Zero library
        import ctypes
        ctypes.CDLL("libze_loader.so.1")
        return True
    except:
        return False

def check_opencl_available() -> bool:
    """Check if OpenCL runtime is available."""
    try:
        import pyopencl
        platforms = pyopencl.get_platforms()
        return len(platforms) > 0
    except:
        return False
```

### 3. Model Loading (from exo-cuda pattern)

**File**: `src/exo/worker/engines/tinygrad/model_loader.py`

**Pattern from exo-cuda**:
```python
def load_tinygrad_model(model_id: str, shard_metadata: ShardMetadata):
    """Load model weights using tinygrad."""
    from tinygrad import Tensor, nn
    
    # Download model files
    model_path = download_model(model_id)
    
    # Load weights with tinygrad
    weights = load_weights(model_path)
    
    # Create model architecture
    model = create_model_architecture(model_id, weights)
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_id)
    
    return model, tokenizer
```

**Key Points**:
- Use tinygrad's `Tensor` and `nn` modules
- Load weights from HuggingFace format
- Support pipeline sharding (load specific layers based on shard_metadata)
- Return model and tokenizer compatible with generation loop

### 4. Text Generation (from exo-cuda pattern)

**File**: `src/exo/worker/engines/tinygrad/generator.py`

**Pattern from exo-cuda**:
```python
def tinygrad_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float
) -> Generator[TokenChunk]:
    """Generate text using tinygrad model."""
    from tinygrad import Tensor
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt)
    
    # Generate tokens
    for i in range(max_tokens):
        # Forward pass with tinygrad
        logits = model(Tensor(input_ids))
        
        # Sample next token
        next_token = sample_token(logits, temperature)
        
        # Decode and yield
        text = tokenizer.decode([next_token])
        yield TokenChunk(text=text, token_id=next_token)
        
        input_ids.append(next_token)
```

**Key Points**:
- Use tinygrad `Tensor` for forward passes
- Implement sampling (temperature, top-p, etc.)
- Yield `TokenChunk` events compatible with exo's streaming API
- Handle KV cache for efficient generation

### 5. Multi-Node Support (from exo-cuda)

**Pattern from exo-cuda**:
- TinygradRingInstance supports ring communication between nodes
- Each node runs a tinygrad runner with a shard of the model
- Activations flow between nodes using the same ring protocol as MLX

**Adaptation**:
- No changes needed to ring communication protocol
- Intel Arc GPU will execute its shard locally
- Network communication remains the same

### 6. Bootstrap Configuration

**File**: `src/exo/worker/runner/bootstrap.py`

**Current Implementation** (already done):
```python
if isinstance(bound_instance.instance, TinygradRingInstance):
    # Tinygrad backend configuration
    os.environ["EXO_TINYGRAD_ENABLED"] = "true"
    if not os.environ.get("TINYGRAD_BACKEND"):
        os.environ["TINYGRAD_BACKEND"] = "GPU"  # Default to GPU for Intel Arc
    logger.info(f"Tinygrad backend: {os.environ.get('TINYGRAD_BACKEND')}")
```

**Enhancement Needed**:
- Detect Intel Arc GPU availability
- Set Level Zero or OpenCL specific environment variables
- Fall back to CPU if GPU unavailable

## Data Flow

### Inference Request Flow

```
1. User sends chat completion request to API
2. Master creates TinygradRingInstance via placement
3. Worker spawns runner process with TinygradRingInstance
4. Runner bootstrap detects Tinygrad, sets TINYGRAD_BACKEND=GPU
5. Runner main() lazy-loads tinygrad modules
6. Runner receives LoadModel task
7. Tinygrad backend loads model weights to Intel Arc GPU
8. Runner receives TextGeneration task
9. Tinygrad generator executes forward passes on GPU
10. Runner emits TokenChunk events back to API
11. API streams tokens to user
```

### Device Selection Flow

```
1. Bootstrap checks instance type
2. If TinygradRingInstance:
   a. Check for Intel Arc GPU
   b. Try Level Zero runtime
   c. Fall back to OpenCL if Level Zero unavailable
   d. Fall back to CPU if no GPU runtime available
   e. Set TINYGRAD_BACKEND environment variable
3. Runner imports tinygrad with configured backend
4. Tinygrad uses selected runtime for execution
```

## Error Handling

### GPU Initialization Failure

**Strategy**: Fail fast with clear error message (no silent CPU fallback)

```python
try:
    backend = TinygradBackend(device="GPU")
    backend.initialize()
except GPUInitializationError as e:
    logger.error(f"Intel Arc GPU initialization failed: {e}")
    event_sender.send(BackendFailed(
        runner_id=runner_id,
        backend_type="tinygrad",
        error_message=str(e),
        fallback_backend=None  # No automatic fallback
    ))
    raise  # Fail the task
```

**Rationale**: Silent fallbacks hide problems. Better to fail explicitly so users know GPU isn't working.

### Model Loading Failure

```python
try:
    model, tokenizer = load_tinygrad_model(model_id, shard_metadata)
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    event_sender.send(RunnerStatusUpdated(
        runner_id=runner_id,
        runner_status=RunnerFailed(error_message=str(e))
    ))
    raise
```

### Runtime Errors During Generation

```python
try:
    for chunk in tinygrad_generate(model, tokenizer, params):
        event_sender.send(ChunkGenerated(chunk=chunk))
except Exception as e:
    logger.error(f"Generation failed: {e}")
    event_sender.send(ChunkGenerated(
        chunk=ErrorChunk(error=str(e))
    ))
```

## Testing Strategy

### Unit Tests

1. **Device Detection Tests**
   - Mock Level Zero availability → verify detection
   - Mock OpenCL availability → verify fallback
   - No GPU available → verify CPU fallback

2. **Model Loading Tests**
   - Load small model on CPU → verify weights loaded
   - Test shard metadata handling → verify correct layers loaded

3. **Generation Tests**
   - Generate tokens on CPU → verify output format
   - Test sampling parameters → verify temperature, top-p work

### Integration Tests

1. **Single-Node GPU Test**
   - Launch TinygradRingInstance on node with Intel Arc
   - Load model to GPU
   - Generate text
   - Verify GPU was actually used (not CPU fallback)

2. **Multi-Node Test**
   - Launch TinygradRingInstance across 2+ nodes
   - Distribute model shards
   - Generate text
   - Verify activations flow correctly between nodes

3. **Fallback Test**
   - Simulate GPU initialization failure
   - Verify clear error message
   - Verify task fails (no silent fallback)

## Implementation Phases

### Phase 1: Runner Integration (Current)
- ✅ Lazy-load tinygrad modules in runner.py
- ✅ Detect TinygradRingInstance
- ⏳ Implement tinygrad inference loop (LoadModel, TextGeneration)

### Phase 2: Device Detection
- Implement Intel Arc GPU detection
- Add Level Zero runtime check
- Add OpenCL fallback check
- Set TINYGRAD_BACKEND appropriately

### Phase 3: Model Loading
- Implement tinygrad model loader
- Support HuggingFace model format
- Handle pipeline sharding

### Phase 4: Text Generation
- Implement tinygrad generation loop
- Add sampling (temperature, top-p)
- Emit TokenChunk events

### Phase 5: Multi-Node Validation
- Test distributed inference
- Verify ring communication
- Validate activation flow

### Phase 6: NixOS Configuration
- Add Level Zero packages
- Add OpenCL packages
- Configure environment variables

## Reference Implementation

**Primary Reference**: https://github.com/Scottcjn/exo-cuda

**Key Files to Study**:
- Runner integration pattern
- Device detection approach
- Model loading with tinygrad
- Generation loop implementation
- Multi-node setup

**Adaptation Strategy**:
1. Copy the structure from exo-cuda
2. Replace CUDA-specific calls with Intel GPU equivalents
3. Keep the same event flow and task handling
4. Maintain compatibility with exo's existing architecture

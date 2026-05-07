# exo-cuda Reference Implementation Study

## Overview

This document summarizes the key patterns and implementation details from the [exo-cuda](https://github.com/Scottcjn/exo-cuda) repository, which demonstrates how to integrate tinygrad backend with exo for NVIDIA CUDA GPUs. This serves as the blueprint for implementing Intel Arc GPU support.

**Repository**: https://github.com/Scottcjn/exo-cuda  
**Commit Studied**: 18517e2 (December 2024 - January 2025)  
**Purpose**: Restore tinygrad backend integration for NVIDIA CUDA distributed inference

## Key Architecture Patterns

### 1. Backend Selection and Initialization

**File**: `exo/main.py`

The inference engine is selected at startup based on command-line arguments or system detection:

```python
# Command-line argument
parser.add_argument("--inference-engine", type=str, default=None, 
                    help="Inference engine to use (mlx, tinygrad, or dummy)")

# Auto-detection fallback
inference_engine_name = args.inference_engine or \
    ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")

# Get the engine instance
inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
```

**Key Insight**: The system defaults to MLX on Apple Silicon, tinygrad on other systems. Users can override with `--inference-engine tinygrad`.

### 2. Inference Engine Factory Pattern

**File**: `exo/inference/inference_engine.py`

The factory pattern provides clean separation between backends:

```python
inference_engine_classes = {
  "mlx": "MLXDynamicShardInferenceEngine",
  "tinygrad": "TinygradDynamicShardInferenceEngine",
  "dummy": "DummyInferenceEngine",
}

def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader):
    if inference_engine_name == "tinygrad":
        from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
        import tinygrad.helpers
        tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
        return TinygradDynamicShardInferenceEngine(shard_downloader)
    # ... other backends
```

**Key Insight**: Lazy imports prevent loading tinygrad unless needed. Debug level is configurable via environment variable.

### 3. TinygradDynamicShardInferenceEngine Implementation

**File**: `exo/inference/tinygrad/inference.py`

This is the core inference engine that implements the `InferenceEngine` abstract base class:

```python
class TinygradDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.states = OrderedDict()  # request_id -> state mapping
        self.executor = _executor  # ThreadPoolExecutor singleton
```

**Key Methods**:

#### a. Model Loading (`ensure_shard`)

```python
async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
        return  # Already loaded
    
    # Download model if needed
    model_path = await self.shard_downloader.ensure_shard(
        shard, self.__class__.__name__
    )
    
    # Determine model size from model_id
    parameters = "1B" if "1b" in shard.model_id.lower() else \
                 "3B" if "3b" in shard.model_id.lower() else \
                 "8B" if "8b" in shard.model_id.lower() else "70B"
    
    # Build transformer in executor (tinygrad must run on same thread)
    loop = asyncio.get_running_loop()
    model_shard = await loop.run_in_executor(
        self.executor, 
        build_transformer, 
        model_path, 
        shard, 
        parameters
    )
    
    # Load tokenizer
    tokenizer_path = str(model_path if model_path.is_dir() else model_path.parent)
    self.tokenizer = await resolve_tokenizer(tokenizer_path)
    
    self.shard = shard
    self.model = model_shard
```

**Key Insights**:
- Model loading is async but runs in a dedicated thread executor
- Model size is inferred from the model_id string
- Tokenizer is loaded from the same directory as model weights
- State is cached to avoid reloading the same shard

#### b. Tensor Inference (`infer_tensor`)

```python
async def infer_tensor(
    self, 
    request_id: str, 
    shard: Shard, 
    input_data: np.ndarray, 
    inference_state: Optional[dict] = None
) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    
    def wrap_infer():
        # Convert numpy to tinygrad Tensor
        x = Tensor(input_data)
        
        # Embed tokens
        h = self.model.embed(x)
        
        # Get or create state for this request
        state = self.poll_state(h, request_id)
        
        # Forward pass with KV cache
        out = self.model.forward(h, **state)
        
        # Update position in cache
        self.states[request_id].start += x.shape[1]
        
        return out.numpy()
    
    # Run in executor
    output_data = await asyncio.get_running_loop().run_in_executor(
        self.executor, 
        wrap_infer
    )
    
    return output_data, inference_state
```

**Key Insights**:
- All tinygrad operations run in a dedicated thread executor
- State management uses an OrderedDict with LRU eviction
- KV cache is maintained per request_id
- Input/output conversion between numpy and tinygrad Tensor

#### c. Token Sampling (`sample`)

```python
async def sample(
    self, 
    x: np.ndarray, 
    temp=TEMPERATURE, 
    top_p: float = 0.0
) -> np.ndarray:
    def sample_wrapper():
        logits = x[:, -1, :]  # Get last position
        return sample_logits(
            Tensor(logits).flatten(), 
            temp, 
            0,      # top_k (disabled)
            0.8,    # nucleus_p
            top_p,  # top_p
            0.0     # alpha_f
        ).realize().numpy().astype(int)
    
    return await asyncio.get_running_loop().run_in_executor(
        self.executor, 
        sample_wrapper
    )
```

**Key Insights**:
- Sampling uses tinygrad's built-in `sample_logits` function
- Supports temperature and nucleus sampling
- Always runs in executor for thread safety

### 4. Model Architecture and Weight Loading

**File**: `exo/inference/tinygrad/inference.py`

```python
MODEL_PARAMS = {
    "1B": {
        "args": {
            "dim": 2048, "n_heads": 32, "n_kv_heads": 8, 
            "n_layers": 16, "vocab_size": 128256, ...
        },
        "files": 1
    },
    "3B": {...},
    "8B": {...},
    "70B": {...}
}

def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
    # Create model architecture
    linear = nn.Linear
    model = Transformer(
        **MODEL_PARAMS[model_size]["args"], 
        linear=linear, 
        max_context=8192, 
        jit=True, 
        shard=shard
    )
    
    # Load weights from disk
    if model_path.is_dir():
        if (model_path/"model.safetensors.index.json").exists():
            weights = load(str(model_path/"model.safetensors.index.json"), shard)
        elif (model_path/"model.safetensors").exists():
            weights = load(str(model_path/"model.safetensors"), shard)
    
    # Convert from HuggingFace format
    weights = convert_from_huggingface(
        weights, 
        model, 
        MODEL_PARAMS[model_size]["args"]["n_heads"],
        MODEL_PARAMS[model_size]["args"]["n_kv_heads"]
    )
    
    # Fix bf16 precision issues
    weights = fix_bf16(weights)
    
    # Load weights into model
    with Context(BEAM=0):
        load_state_dict(model, weights, strict=False, consume=False)
        model = TransformerShard(shard, model)
    
    return model
```

**Key Insights**:
- Model parameters are hardcoded for common Llama variants
- Supports both single-file and sharded safetensors format
- Weight conversion handles HuggingFace → tinygrad format differences
- Pipeline sharding is handled by `TransformerShard` wrapper

### 5. Device Detection and Configuration

**File**: `exo/interweave/backends/tinygrad_cuda.py`

The exo-cuda implementation shows device detection patterns:

```python
def _check_tinygrad(self) -> bool:
    """Check if tinygrad with CUDA is available"""
    try:
        from tinygrad import Tensor, Device
        
        # Try to set CUDA device
        if self.device == 'CUDA':
            if 'CUDA' in Device._devices or os.getenv('GPU', '1') == '1':
                logger.info("TinyGrad CUDA backend available")
                return True
            else:
                logger.warning("CUDA not available, falling back to GPU/CPU")
                self.device = 'GPU'
                return True
        return True
    except ImportError as e:
        logger.error(f"TinyGrad not available: {e}")
        return False
```

**Key Insights**:
- Device detection uses tinygrad's `Device._devices` registry
- Fallback chain: CUDA → GPU (OpenCL) → CPU
- Environment variable `GPU=1` can force GPU mode

### 6. Multi-Node Communication

**Pattern**: Ring topology with shard distribution

The exo-cuda implementation doesn't modify the ring communication protocol. Key points:

- Each node runs the same `TinygradDynamicShardInferenceEngine`
- Model shards are distributed based on `Shard` metadata (start_layer, end_layer)
- Activations flow between nodes using existing exo networking (GRPC)
- No backend-specific changes needed for multi-node support

### 7. Model Registry

**File**: `exo/models.py`

Models are registered with backend-specific HuggingFace repos:

```python
model_cards = {
    "llama-3.2-3b": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
        },
    },
    # ... more models
}
```

**Key Insight**: Different backends can use different model repos (e.g., MLX uses quantized, tinygrad uses full precision).

## Environment Variables

The exo-cuda implementation uses these environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `TINYGRAD_DEBUG` | Debug verbosity (0-6) | `0` |
| `TINYGRAD_BACKEND` | Force backend (CUDA/GPU/CPU) | Auto-detect |
| `CUDA_VISIBLE_DEVICES` | Limit GPU visibility | All GPUs |
| `GPU` | Enable GPU mode | `1` |
| `TEMPERATURE` | Sampling temperature | `0.85` |

## Key Differences: CUDA vs Intel Arc

To adapt exo-cuda for Intel Arc, these changes are needed:

| Aspect | CUDA (exo-cuda) | Intel Arc (our target) |
|--------|-----------------|------------------------|
| Backend env var | `TINYGRAD_BACKEND=CUDA` | `TINYGRAD_BACKEND=GPU` |
| Device detection | Check `'CUDA' in Device._devices` | Check Level Zero or OpenCL |
| Runtime library | CUDA Toolkit | Level Zero or OpenCL |
| Device query | `nvidia-smi` | Level Zero API or OpenCL |
| Memory query | CUDA API | Level Zero API or OpenCL |

## Implementation Checklist for Intel Arc

Based on the exo-cuda reference, here's what we need to implement:

### ✅ Already Done (in current exo)
- [ ] Inference engine factory pattern (`get_inference_engine`)
- [ ] Abstract `InferenceEngine` base class
- [ ] Model registry with backend-specific repos
- [ ] Shard download infrastructure
- [ ] Ring topology for multi-node

### ❌ Need to Implement
- [ ] `TinygradDynamicShardInferenceEngine` class
- [ ] `build_transformer` function for model loading
- [ ] Device detection for Intel Arc (Level Zero/OpenCL)
- [ ] State management with KV cache
- [ ] Token sampling with tinygrad
- [ ] Weight loading from HuggingFace format
- [ ] Integration with existing runner/worker architecture

## Code Reuse Strategy

We can directly reuse these components from exo-cuda:

1. **Inference Engine Structure**: The `TinygradDynamicShardInferenceEngine` class structure
2. **Model Loading Pattern**: The `build_transformer` and weight loading logic
3. **State Management**: The `poll_state` and OrderedDict caching pattern
4. **Async Executor Pattern**: Using ThreadPoolExecutor for tinygrad operations
5. **Sampling Logic**: The `sample_logits` integration

We need to adapt these components:

1. **Device Detection**: Replace CUDA checks with Level Zero/OpenCL checks
2. **Environment Variables**: Use `TINYGRAD_BACKEND=GPU` instead of `CUDA`
3. **Memory Queries**: Use Level Zero/OpenCL APIs instead of nvidia-smi
4. **Model Repos**: Use appropriate HuggingFace repos for Intel Arc

## Testing Strategy from exo-cuda

The exo-cuda repository demonstrates this testing approach:

1. **Single-Node Testing**: Start with `--inference-engine tinygrad` on one node
2. **Model Loading**: Test with small models first (llama-3.2-1b)
3. **Multi-Node Testing**: Add second node, verify shard distribution
4. **API Testing**: Use ChatGPT-compatible API for end-to-end validation

## References

- **exo-cuda Repository**: https://github.com/Scottcjn/exo-cuda
- **Key Files**:
  - `exo/inference/tinygrad/inference.py` - Main inference engine
  - `exo/inference/inference_engine.py` - Factory pattern
  - `exo/main.py` - Entry point and backend selection
  - `exo/models.py` - Model registry
  - `exo/interweave/backends/tinygrad_cuda.py` - Backend wrapper

## Next Steps

1. Copy the `TinygradDynamicShardInferenceEngine` structure
2. Implement Intel Arc device detection
3. Adapt CUDA-specific calls to Level Zero/OpenCL
4. Test with small model on single node
5. Validate multi-node distribution
6. Add NixOS configuration for Intel Arc support

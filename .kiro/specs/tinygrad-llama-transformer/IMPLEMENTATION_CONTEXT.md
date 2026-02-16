# Llama Transformer Implementation Context

This document provides essential context for implementing remaining tasks in the tinygrad Llama transformer project.

## Project Status

### Completed Tasks
- ✅ Task 9: Implement core transformer components (RMSNorm, Attention, MLP, RoPE)
- ✅ Task 10: Implement complete LlamaTransformer model
- ✅ Task 11: Implement weight loading from HuggingFace checkpoints
- ✅ Task 12.3: Validate output correctness
- ✅ Task 12.4: Test different model sizes

### Remaining Tasks
- [ ] Task 12.1: Write unit tests for components (optional)
- [ ] Task 12.2: Write integration tests (optional)
- [ ] Task 13: Integration with tinygrad backend
- [ ] Task 14: End-to-end generation testing

## Key Implementation Files

### Core Implementation
- **`src/exo/worker/engines/tinygrad/llama_transformer.py`** (2922 lines)
  - Complete Llama transformer architecture
  - Configuration parsing and validation
  - Weight loading utilities
  - KV cache system
  - All core components (RMSNorm, Attention, MLP, RoPE, etc.)

### Supporting Files
- **`src/exo/worker/engines/tinygrad/model_loader.py`**
  - Async model loading
  - Tokenizer integration
  - Weight filtering for sharding
  
- **`src/exo/worker/engines/tinygrad/generator.py`**
  - Text generation logic
  - Sampling strategies
  
- **`src/exo/worker/engines/tinygrad/tinygrad_backend.py`**
  - Backend integration with exo
  - Runner process management

## Testing Infrastructure

### Test Files Created
1. **`test_llama_simple.py`** - Simple validation test (works on gremlin-1)
2. **`test_llama_validation.py`** - Comprehensive test suite (10 tests)
3. **`test_weight_loading.py`** - Weight loading validation
4. **`run_simple_test.sh`** - Script to run tests on gremlin-1

### Testing on gremlin-1 (NixOS)

**Key Learning**: Testing on NixOS requires careful environment setup.

#### Environment Setup
```bash
# Set environment variables
export PYTHONNOUSERSITE='true'
export EXO_TINYGRAD_ENABLED='true'

# Build PYTHONPATH with all dependencies
export PYTHONPATH="/nix/store/jrsxpngcaif0rxflm6bax7pi1cd1hhyd-exo-0.3.0/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/74lsy15mvdbsnn40jjr9w95y17mm8b0v-python3.13-numpy-2.3.5/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/xg0j9lib60xlszacmivg00nfw20bdybd-python3.13-tinygrad-0.12.0/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/gj1nk0ff7qh9m1cwkv2sma0v5px1z7vj-python3.13-loguru-0.7.3/lib/python3.13/site-packages"

# Use the python from exo
PYTHON=/nix/store/qzc04a3npl70cyyy6flnnrb2ig3kayxm-python3-3.13.11/bin/python3.13
```

#### Import Strategy
**Problem**: Importing from `exo.worker.engines.tinygrad.llama_transformer` triggers `__init__.py` which imports other modules requiring many dependencies (pydantic, aiofiles, etc.).

**Solution**: Import directly from the module file:
```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "llama_transformer",
    "/nix/store/.../llama_transformer.py"
)
llama_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama_module)
```

### Deployment Workflow

1. **Commit and push changes**
   ```bash
   git add <files>
   git commit -m "message"
   git push
   ```

2. **Deploy to gremlin-1**
   ```bash
   bash force_update_gremlin1.sh
   ```
   This script:
   - Updates flake to latest commit
   - Rebuilds NixOS system
   - Restarts exo service
   - Checks service status

3. **Run tests on gremlin-1**
   ```bash
   scp test_llama_simple.py run_simple_test.sh root@10.1.1.12:/tmp/
   ssh root@10.1.1.12 "bash /tmp/run_simple_test.sh"
   ```

## Architecture Overview

### Model Structure
```
LlamaTransformer
├── embed_tokens (Embedding)
├── layers (List[TransformerLayer])
│   ├── input_layernorm (RMSNorm)
│   ├── self_attn (Attention)
│   │   ├── q_proj, k_proj, v_proj, o_proj (Linear)
│   │   └── rope (RotaryEmbedding)
│   ├── post_attention_layernorm (RMSNorm)
│   └── mlp (MLP)
│       ├── gate_proj, up_proj, down_proj (Linear)
│       └── SwiGLU activation
├── norm (RMSNorm)
└── lm_head (Linear)
```

### KV Cache System
```
KVCacheManager
└── KVCache (per request)
    └── LayerCache (per layer)
        ├── key_cache
        └── value_cache
```

### Configuration System
- **LlamaConfig**: Dataclass with all hyperparameters
- **Default configs**: 0.5B, 1B, 3B, 8B, 70B
- **Parsing**: From HuggingFace config.json
- **Validation**: Comprehensive parameter validation

## Key Design Decisions

### 1. Pure Tinygrad Implementation
- No PyTorch dependencies
- All operations use tinygrad Tensor
- Compatible with Intel Arc GPU via tinygrad

### 2. Grouped-Query Attention (GQA)
- Supports different Q and KV head counts
- KV heads are repeated to match Q heads
- Example: 24 Q heads, 8 KV heads = 3x repetition

### 3. Rotary Position Embeddings (RoPE)
- Pre-computed cos/sin cache for efficiency
- Applied to queries and keys
- Supports position extrapolation

### 4. Weight Loading
- Supports single safetensors files
- Supports sharded models (model.safetensors.index.json)
- Handles bfloat16 conversion
- Weight name mapping from HuggingFace format

### 5. Logging
- Uses loguru for structured logging
- DEBUG level for detailed tracing
- INFO level for key operations
- Logs shapes, dimensions, and validation results

## Common Patterns

### Creating a Model
```python
from exo.worker.engines.tinygrad.llama_transformer import (
    LlamaConfig,
    LlamaTransformer,
    get_default_config,
)

# Option 1: Use default config
config = get_default_config("3B")

# Option 2: Parse from file
config = parse_config_from_file(Path("config.json"))

# Option 3: Create custom config
config = LlamaConfig(
    vocab_size=128256,
    hidden_size=3072,
    num_hidden_layers=28,
    num_attention_heads=24,
    num_key_value_heads=8,
)

# Create model
model = LlamaTransformer(config)
```

### Forward Pass
```python
from tinygrad import Tensor

# Create input
input_ids = Tensor([[1, 2, 3, 4]])

# Forward pass (prefill)
logits, cache = model(input_ids)

# Forward pass with cache (generation)
next_token = Tensor([[5]])
logits, cache = model(next_token, cache=cache)
```

### Loading Weights
```python
from exo.worker.engines.tinygrad.llama_transformer import (
    assign_weights_to_model,
)
from exo.worker.engines.tinygrad.model_loader import (
    _load_safetensors_sync,
)

# Load weights
weights = _load_safetensors_sync(Path("model.safetensors"))

# Assign to model
num_loaded, num_expected = assign_weights_to_model(
    model, weights, device="GPU"
)
```

## Integration Points

### With tinygrad Backend
- **TinygradBackend** (`tinygrad_backend.py`)
  - Implements `InferenceBackend` interface
  - Manages model lifecycle
  - Handles inference requests
  - Integrates with exo's runner system

### With Model Loader
- **load_tinygrad_model** (`model_loader.py`)
  - Async model loading
  - Tokenizer integration
  - Weight filtering for sharding
  - Thread-safe execution

### With Generator
- **TinygradGenerator** (`generator.py`)
  - Text generation logic
  - Sampling strategies (greedy, top-p, temperature)
  - Stop token handling
  - Streaming support

## Performance Considerations

### Memory Management
- KV cache grows with sequence length
- Each layer stores keys and values
- Memory usage: `2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)`

### Optimization Opportunities
1. **Flash Attention**: Not yet implemented
2. **Quantization**: Not yet implemented
3. **Kernel Fusion**: Relies on tinygrad's optimizer
4. **Pipeline Parallelism**: Supported via sharding

## Known Limitations

### Current Implementation
1. **No actual weights loaded in tests**: Tests use random initialization
2. **No generation quality validation**: Requires real weights
3. **No comparison with HuggingFace**: Requires real weights
4. **No performance benchmarks**: Requires real workloads

### Testing Limitations
1. **NixOS environment complexity**: Requires careful PYTHONPATH setup
2. **Import dependencies**: `__init__.py` triggers many imports
3. **No GPU in local environment**: Must test on gremlin-1

## Next Steps for Remaining Tasks

### Task 13: Integration with tinygrad backend
**Focus**: Ensure TinygradBackend properly uses LlamaTransformer
- Verify model loading in backend
- Test inference through backend API
- Validate sharding support
- Check error handling

**Files to modify**:
- `src/exo/worker/engines/tinygrad/tinygrad_backend.py`
- `src/exo/worker/engines/tinygrad/generator.py`

### Task 14: End-to-end generation testing
**Focus**: Test complete generation pipeline with real models
- Download Llama-3.2-1B or 3B model
- Load weights into transformer
- Generate text from prompts
- Validate output quality
- Benchmark performance

**Requirements**:
- Real model weights (HuggingFace)
- Test prompts
- Expected outputs (from HuggingFace)
- Performance metrics

## Useful Commands

### Check exo service on gremlin-1
```bash
ssh root@10.1.1.12 "systemctl status exo"
ssh root@10.1.1.12 "journalctl -u exo -n 100"
```

### Check GPU
```bash
ssh root@10.1.1.12 "intel_gpu_top"
```

### Test API
```bash
curl -s 'http://10.1.1.12:52415/state' | python3 -m json.tool
```

### Run generation test
```bash
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## References

### Documentation
- Requirements: `.kiro/specs/tinygrad-llama-transformer/requirements.md`
- Design: `.kiro/specs/tinygrad-llama-transformer/design.md`
- Tasks: `.kiro/specs/tinygrad-llama-transformer/tasks.md`

### Completion Reports
- Task 9: `.kiro/specs/tinygrad-llama-transformer/TASK_9_COMPLETE.md`
- Task 11: `.kiro/specs/tinygrad-llama-transformer/TASK_11_COMPLETE.md`
- Tasks 12.3 & 12.4: `.kiro/specs/tinygrad-llama-transformer/TASKS_12_3_12_4_COMPLETE.md`

### Test Guides
- Validation Guide: `TEST_VALIDATION_GUIDE.md`
- Quick Reference: `QUICK_TEST_LLAMA.md`

### Papers
- Llama 2: https://arxiv.org/abs/2307.09288
- RoPE: https://arxiv.org/abs/2104.09864
- GLU Variants: https://arxiv.org/abs/2002.05202
- RMSNorm: https://arxiv.org/abs/1910.07467

## Tips for Success

1. **Always test on gremlin-1**: Local environment doesn't have GPU
2. **Use simple tests first**: Avoid complex dependencies
3. **Import directly when needed**: Bypass `__init__.py` if necessary
4. **Check logs**: `journalctl -u exo` is your friend
5. **Commit frequently**: Easy to rollback if needed
6. **Document learnings**: Update this file with new insights

## Contact Points

- **Hardware**: gremlin-1 (root@10.1.1.12)
- **Service**: exo.service (systemd)
- **API**: http://10.1.1.12:52415
- **Dashboard**: http://10.1.1.12:52415 (if enabled)

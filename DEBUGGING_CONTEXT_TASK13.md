# Debugging Context: Task 13 Performance Optimization

## Overview

This document provides comprehensive context for debugging and testing the Task 13 performance optimization implementation with real model weights (meta-llama/Llama-3.2-3B-Instruct) on gremlin-1.

**Date**: Task 13 completed
**Target Hardware**: gremlin-1 (Intel Arc A770, NixOS)
**Model**: meta-llama/Llama-3.2-3B-Instruct
**Backend**: tinygrad

## Quick Reference

### Deployment Command
```bash
bash force_update_gremlin1.sh
```

### Test Commands on gremlin-1
```bash
# Copy test files
scp test_performance_profile.py test_memory_profile.py test_gpu_utilization.py run_performance_tests.sh root@10.1.1.12:/tmp/

# SSH and run
ssh root@10.1.1.12
cd /tmp
bash run_performance_tests.sh
```

### Check Service Status
```bash
ssh root@10.1.1.12 "systemctl status exo"
ssh root@10.1.1.12 "journalctl -u exo -n 100 --no-pager"
```

## Completed Tasks Summary

### Task 9: Core Transformer Components ✓
**Files**: `src/exo/worker/engines/tinygrad/llama_transformer.py`
- RMSNorm layer normalization
- Rotary Position Embeddings (RoPE)
- Multi-head Attention with GQA
- MLP with SwiGLU activation
- Complete transformer layer

### Task 10: Complete LlamaTransformer ✓
**Files**: `src/exo/worker/engines/tinygrad/llama_transformer.py`
- Full model architecture
- KV cache system (LayerCache, KVCache, KVCacheManager)
- Forward pass with caching
- Configuration system (LlamaConfig)

### Task 11: Weight Loading ✓
**Files**: `src/exo/worker/engines/tinygrad/model_loader.py`, `llama_transformer.py`
- HuggingFace checkpoint loading
- Safetensors support (single and sharded)
- Weight name mapping
- bfloat16 conversion
- Tokenizer integration

### Task 12: Testing and Validation ✓
**Files**: `test_llama_simple.py`, `test_llama_validation.py`, `test_weight_loading.py`
- Component unit tests
- Integration tests
- Output correctness validation
- Different model sizes tested

### Task 13: Performance Optimization ✓ (NEW)
**Files**: `test_performance_profile.py`, `test_memory_profile.py`, `test_gpu_utilization.py`
- Generation speed profiling
- Memory usage optimization
- GPU utilization verification

## Task 13 Implementation Details

### 13.1 Generation Speed Profiling

**File**: `test_performance_profile.py` (400+ lines)

**What it does**:
- Measures tokens per second (TPS) for prefill and generation
- Profiles individual components (RMSNorm, Attention, MLP, Embedding, LM Head)
- Identifies performance bottlenecks
- Checks if generation meets >10 tokens/sec requirement

**Key functions**:
- `profile_generation_speed()`: End-to-end generation profiling
- `profile_components()`: Component-level timing
- `identify_bottlenecks()`: Analysis and recommendations

**Expected output**:
```
GENERATION STATISTICS
  Prefill TPS: XX.XX tokens/sec
  Generation TPS: XX.XX tokens/sec
  Overall TPS: XX.XX tokens/sec
  
COMPONENT TIMING SUMMARY
  RMSNorm:        X.XXms
  Embedding:      X.XXms
  TransformerLayer: X.XXms
  Attention:      X.XXms
  MLP:            X.XXms
  LM_Head:        X.XXms
```

### 13.2 Memory Usage Profiling

**File**: `test_memory_profile.py` (450+ lines)

**What it does**:
- Estimates model parameter memory
- Tracks KV cache memory growth
- Verifies KV cache reduces computation (>1.5x speedup)
- Calculates memory efficiency metrics

**Key functions**:
- `estimate_model_memory()`: Parameter memory calculation
- `estimate_kv_cache_memory()`: Cache memory calculation
- `verify_kv_cache_benefit()`: With/without cache comparison
- `profile_memory_growth()`: Track growth during generation

**Expected output**:
```
MEMORY STATISTICS
  Model parameters: XXX.XX MB
  KV cache (initial): XX.XX MB
  KV cache (final): XX.XX MB
  Cache growth: XX.XX MB
  Growth per token: X.XXXX MB/token

KV CACHE BENEFIT
  Time with cache: X.XXXXs
  Time without cache: X.XXXXs
  Speedup: X.XXx
  Cache is working: ✓ YES
```

### 13.3 GPU Utilization Verification

**File**: `test_gpu_utilization.py` (550+ lines)

**What it does**:
- Detects available tinygrad devices
- Compares CPU vs GPU performance
- Checks for CPU fallbacks
- Profiles kernel execution patterns

**Key functions**:
- `check_tinygrad_device()`: Device detection
- `profile_cpu_vs_gpu()`: Performance comparison
- `check_for_cpu_fallbacks()`: Heuristic fallback detection
- `profile_kernel_execution()`: Kernel timing breakdown

**Expected output**:
```
DEVICE INFORMATION
  Default device: GPU
  GPU: ✓ Available
  
PERFORMANCE ANALYSIS
  CPU time: XXX.XXms
  GPU time: XX.XXms
  Speedup: X.XXx
  GPU acceleration: ✓ YES

FALLBACK ANALYSIS
  Matrix multiplication: ✓ FAST
  Element-wise ops: ✓ FAST
  Softmax: ✓ FAST
```

## Testing with Real Model Weights

### Model Download Location

The model should be downloaded to:
```
~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/
```

Or specify custom path in exo configuration.

### Model Configuration

**Llama-3.2-3B-Instruct specs**:
- vocab_size: 128256
- hidden_size: 3072
- intermediate_size: 8192
- num_hidden_layers: 28
- num_attention_heads: 24
- num_key_value_heads: 8
- max_position_embeddings: 8192

### Modified Test Script for Real Weights

Create `test_performance_real_model.py`:

```python
#!/usr/bin/env python3
"""Performance test with real Llama-3.2-3B-Instruct model."""

import sys
from pathlib import Path

# Add exo to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from exo.worker.engines.tinygrad.model_loader import load_tinygrad_model
from exo.shared.types.worker.shards import ShardMetadata
from exo.shared.types.model_card import ModelCard

# Create shard metadata for full model
model_card = ModelCard(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    config={
        "vocab_size": 128256,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
    }
)

shard_metadata = ShardMetadata(
    model_card=model_card,
    start_layer=0,
    end_layer=28,
    n_layers=28,
    is_first_layer=True,
    is_last_layer=True,
)

# Load model
checkpoint_path = "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/latest"
model, tokenizer = await load_tinygrad_model(
    shard_metadata,
    checkpoint_path,
    device="GPU"
)

# Run performance tests
# ... (use functions from test_performance_profile.py)
```

## Deployment Workflow

### 1. Commit Changes
```bash
git add test_performance_profile.py test_memory_profile.py test_gpu_utilization.py run_performance_tests.sh
git commit -m "Add Task 13: Performance optimization tests"
git push
```

### 2. Deploy to gremlin-1
```bash
bash force_update_gremlin1.sh
```

This script:
- Fetches latest git commit
- Updates flake.nix with new commit hash
- Rebuilds NixOS system
- Restarts exo service
- Checks service status

### 3. Verify Deployment
```bash
ssh root@10.1.1.12 "systemctl status exo"
ssh root@10.1.1.12 "journalctl -u exo -n 50 --no-pager"
```

### 4. Copy Test Files
```bash
scp test_performance_profile.py root@10.1.1.12:/tmp/
scp test_memory_profile.py root@10.1.1.12:/tmp/
scp test_gpu_utilization.py root@10.1.1.12:/tmp/
scp run_performance_tests.sh root@10.1.1.12:/tmp/
```

### 5. Run Tests
```bash
ssh root@10.1.1.12
cd /tmp

# Set up environment (if needed)
export PYTHONNOUSERSITE='true'
export EXO_TINYGRAD_ENABLED='true'

# Run all tests
bash run_performance_tests.sh

# Or run individual tests
python3 test_performance_profile.py
python3 test_memory_profile.py
python3 test_gpu_utilization.py
```

## Expected Performance Metrics

### Generation Speed (Requirement 8.1)
- **Target**: >10 tokens/sec on Intel Arc A770
- **Prefill**: Should be faster (processing multiple tokens at once)
- **Generation**: Token-by-token, should meet target

### Memory Usage (Requirement 8.2, 8.5)
- **Model (3B)**: ~12 GB in float32, ~6 GB in float16
- **KV cache**: Grows linearly with sequence length
  - Per token: ~0.5-1 MB (depends on batch size and config)
- **Cache benefit**: >1.5x speedup expected

### GPU Utilization (Requirement 8.3, 6.5)
- **CPU vs GPU**: >1.2x speedup expected
- **Fallbacks**: Should not occur for standard operations
- **Overhead**: <20% is good

## Common Issues and Solutions

### Issue 1: Model Not Found
**Symptom**: FileNotFoundError for checkpoint path
**Solution**:
```bash
# Check model location
ssh root@10.1.1.12 "ls -la ~/.cache/huggingface/hub/"

# Download model if needed
ssh root@10.1.1.12
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
```

### Issue 2: GPU Not Detected
**Symptom**: "GPU: ✗ Not available"
**Solution**:
```bash
# Check GPU
ssh root@10.1.1.12 "intel_gpu_top"

# Check tinygrad device
ssh root@10.1.1.12
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# Set device explicitly
export DEVICE=GPU
```

### Issue 3: Import Errors
**Symptom**: ModuleNotFoundError for exo modules
**Solution**:
```bash
# Use direct module loading (already in test scripts)
# Or set PYTHONPATH
export PYTHONPATH="/nix/store/.../exo-0.3.0/lib/python3.13/site-packages:$PYTHONPATH"
```

### Issue 4: Slow Performance
**Symptom**: TPS < 10 tokens/sec
**Debug steps**:
1. Check GPU utilization: `intel_gpu_top`
2. Check for CPU fallbacks in test output
3. Verify tinygrad JIT is working
4. Check system load: `htop`
5. Review kernel execution overhead

### Issue 5: Memory Issues
**Symptom**: OOM errors or excessive memory usage
**Debug steps**:
1. Check available GPU memory: `intel_gpu_top`
2. Reduce batch size
3. Reduce max sequence length
4. Check for memory leaks in cache

### Issue 6: KV Cache Not Working
**Symptom**: No speedup with cache
**Debug steps**:
1. Verify cache is being updated (check logs)
2. Check cache sequence length grows
3. Verify cache is passed between forward passes
4. Check for cache clearing bugs

## Debugging Commands

### Check Service Logs
```bash
# Recent logs
ssh root@10.1.1.12 "journalctl -u exo -n 100 --no-pager"

# Follow logs
ssh root@10.1.1.12 "journalctl -u exo -f"

# Logs with timestamp
ssh root@10.1.1.12 "journalctl -u exo --since '10 minutes ago'"
```

### Check GPU Status
```bash
# GPU utilization
ssh root@10.1.1.12 "intel_gpu_top"

# GPU info
ssh root@10.1.1.12 "lspci | grep VGA"
```

### Check System Resources
```bash
# CPU and memory
ssh root@10.1.1.12 "htop"

# Disk space
ssh root@10.1.1.12 "df -h"

# Process list
ssh root@10.1.1.12 "ps aux | grep exo"
```

### Test API Endpoint
```bash
# Check state
curl -s 'http://10.1.1.12:52415/state' | python3 -m json.tool

# Test generation
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## File Locations

### Source Files
- `src/exo/worker/engines/tinygrad/llama_transformer.py` - Core implementation
- `src/exo/worker/engines/tinygrad/model_loader.py` - Weight loading
- `src/exo/worker/engines/tinygrad/generator.py` - Text generation
- `src/exo/worker/engines/tinygrad/tinygrad_backend.py` - Backend integration

### Test Files
- `test_performance_profile.py` - Generation speed profiling
- `test_memory_profile.py` - Memory usage profiling
- `test_gpu_utilization.py` - GPU utilization verification
- `run_performance_tests.sh` - Run all tests

### Documentation
- `.kiro/specs/tinygrad-llama-transformer/TASK_13_COMPLETE.md` - Task completion summary
- `.kiro/specs/tinygrad-llama-transformer/IMPLEMENTATION_CONTEXT.md` - Implementation context
- `DEBUGGING_CONTEXT_TASK13.md` - This file

### Deployment
- `force_update_gremlin1.sh` - Deploy to gremlin-1
- `flake.nix` - NixOS configuration

## Performance Benchmarks

### Expected Results (3B Model on Intel Arc A770)

**Generation Speed**:
- Prefill: 50-100 tokens/sec
- Generation: 10-20 tokens/sec
- Overall: 15-30 tokens/sec

**Memory Usage**:
- Model: ~6 GB (float16)
- KV cache: ~50 MB for 100 tokens
- Total: ~7 GB

**GPU Utilization**:
- GPU vs CPU speedup: 2-5x
- GPU utilization: 60-90%
- Kernel overhead: 10-15%

### Comparison with Other Backends

**MLX (Apple Silicon)**:
- Generation: 20-40 tokens/sec on M1 Max
- Memory: Similar

**CUDA (NVIDIA)**:
- Generation: 30-60 tokens/sec on RTX 3090
- Memory: Similar

**Target for tinygrad/Intel Arc**:
- Generation: >10 tokens/sec (requirement)
- Competitive with other backends

## Next Steps After Testing

### If Tests Pass
1. Document performance results
2. Create benchmark report
3. Update README with performance metrics
4. Consider optimizations (Flash Attention, quantization)

### If Tests Fail
1. Review error messages and logs
2. Check this debugging guide
3. Verify environment setup
4. Test with smaller model (1B) first
5. Profile specific bottlenecks
6. Consider filing issues

## Requirements Checklist

- [ ] **8.1**: Generation speed >10 tokens/sec
- [ ] **8.2**: Memory usage monitored and optimized
- [ ] **8.3**: GPU utilization verified
- [ ] **8.5**: KV cache reduces computation
- [ ] **6.5**: Device support (CPU, GPU) verified

## Contact and Support

**Hardware**: gremlin-1 (root@10.1.1.12)
**Service**: exo.service (systemd)
**API**: http://10.1.1.12:52415
**Model**: meta-llama/Llama-3.2-3B-Instruct

For issues, check:
1. Service logs: `journalctl -u exo`
2. This debugging guide
3. Implementation context: `IMPLEMENTATION_CONTEXT.md`
4. Task completion docs: `TASK_*_COMPLETE.md`

## Summary

Task 13 adds comprehensive performance profiling tools to measure and optimize the tinygrad Llama transformer implementation. The three test scripts provide detailed insights into generation speed, memory usage, and GPU utilization. Use this guide to deploy, test, and debug the implementation on gremlin-1 with real model weights.

# End-to-End Validation Plan

## Objective
Verify that all completed tasks (1-13) from the tinygrad-llama-transformer implementation work correctly on gremlin-1 with real model weights.

## Validation Date
February 12, 2026

## Target System
- **Hardware**: gremlin-1 (Intel Arc A770, 16GB VRAM)
- **OS**: NixOS
- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Backend**: tinygrad with GPU support

## Validation Phases

### Phase 1: Deployment Verification
**Goal**: Ensure latest code is deployed and service is running

**Steps**:
1. Deploy latest commit to gremlin-1
2. Verify exo service starts successfully
3. Check service logs for errors
4. Verify API endpoint responds

**Commands**:
```bash
# Deploy
bash force_update_gremlin1.sh

# Verify service
ssh root@10.1.1.12 "systemctl status exo"

# Check logs
ssh root@10.1.1.12 "journalctl -u exo -n 100 --no-pager"

# Test API
curl -s 'http://10.1.1.12:52415/state' | python3 -m json.tool
```

**Success Criteria**:
- ✓ Service is active and running
- ✓ No critical errors in logs
- ✓ API returns valid JSON response
- ✓ Tinygrad backend is detected

### Phase 2: Component Validation
**Goal**: Verify individual transformer components work correctly

**Tests**:
- Task 2: Core components (RMSNorm, Embedding, Linear)
- Task 3: Rotary Position Embeddings
- Task 4: Multi-head Attention with GQA
- Task 5: MLP with SwiGLU
- Task 6: Transformer Layer

**Test Script**: `test_llama_simple.py`

**Commands**:
```bash
# Copy test to gremlin-1
scp test_llama_simple.py root@10.1.1.12:/tmp/

# Run component tests
ssh root@10.1.1.12 "cd /tmp && python3 test_llama_simple.py"
```

**Success Criteria**:
- ✓ All component shapes are correct
- ✓ No NaN or Inf values in outputs
- ✓ Components run on GPU (not CPU fallback)

### Phase 3: Weight Loading Validation
**Goal**: Verify model weights load correctly from HuggingFace

**Tests**:
- Task 9: Weight loading from safetensors
- Task 11: Configuration parsing

**Test Script**: `test_weight_loading.py`

**Commands**:
```bash
# Copy test to gremlin-1
scp test_weight_loading.py root@10.1.1.12:/tmp/

# Run weight loading test
ssh root@10.1.1.12 "cd /tmp && python3 test_weight_loading.py"
```

**Success Criteria**:
- ✓ Model downloads successfully (or uses cached version)
- ✓ All weights load without errors
- ✓ Weight shapes match model configuration
- ✓ No missing or unexpected weights

### Phase 4: Generation Validation
**Goal**: Verify end-to-end text generation works correctly

**Tests**:
- Task 7: KV cache system
- Task 8: Complete transformer forward pass
- Task 10: Backend integration

**Test Script**: `test_llama_validation.py`

**Commands**:
```bash
# Copy test to gremlin-1
scp test_llama_validation.py root@10.1.1.12:/tmp/

# Run generation test
ssh root@10.1.1.12 "cd /tmp && python3 test_llama_validation.py"
```

**Success Criteria**:
- ✓ Model generates coherent text
- ✓ Output is deterministic with fixed seed
- ✓ KV cache reduces computation time
- ✓ No errors during generation

### Phase 5: Performance Validation
**Goal**: Verify performance meets requirements

**Tests**:
- Task 13.1: Generation speed profiling
- Task 13.2: Memory usage optimization
- Task 13.3: GPU utilization verification

**Test Scripts**: 
- `test_performance_profile.py`
- `test_memory_profile.py`
- `test_gpu_utilization.py`

**Commands**:
```bash
# Copy all performance tests
scp test_performance_profile.py test_memory_profile.py test_gpu_utilization.py run_performance_tests.sh root@10.1.1.12:/tmp/

# Run all performance tests
ssh root@10.1.1.12 "cd /tmp && bash run_performance_tests.sh"
```

**Success Criteria**:
- ✓ Generation speed >10 tokens/sec (Requirement 8.1)
- ✓ KV cache provides >1.5x speedup (Requirement 8.5)
- ✓ GPU speedup >1.2x vs CPU (Requirement 8.3)
- ✓ Memory usage is reasonable (<8GB for 3B model)
- ✓ No CPU fallbacks detected

### Phase 6: API Integration Validation
**Goal**: Verify transformer works through exo API

**Tests**:
- OpenAI-compatible chat completions endpoint
- Streaming responses
- Multiple concurrent requests

**Commands**:
```bash
# Test chat completion
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Test streaming
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "max_tokens": 100,
    "stream": true
  }'
```

**Success Criteria**:
- ✓ API returns valid responses
- ✓ Generated text is coherent
- ✓ Streaming works correctly
- ✓ Response times are acceptable

### Phase 7: Stress Testing
**Goal**: Verify stability under load

**Tests**:
- Long sequence generation (>1000 tokens)
- Multiple concurrent requests
- Memory leak detection
- Error recovery

**Commands**:
```bash
# Long sequence test
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Write a long story about a dragon"}],
    "max_tokens": 1000
  }'

# Monitor memory during generation
ssh root@10.1.1.12 "watch -n 1 'intel_gpu_top | head -20'"
```

**Success Criteria**:
- ✓ Long sequences complete without errors
- ✓ Memory usage stays stable (no leaks)
- ✓ GPU utilization remains high
- ✓ Service remains responsive

## Validation Checklist

### Core Functionality
- [ ] Service deploys and starts successfully
- [ ] Tinygrad backend initializes with GPU support
- [ ] Model weights load from HuggingFace
- [ ] Configuration parsing works for Llama-3.2-3B
- [ ] All transformer components work correctly
- [ ] KV cache system functions properly
- [ ] Text generation produces coherent output
- [ ] API endpoints respond correctly

### Performance Requirements
- [ ] Generation speed >10 tokens/sec (Req 8.1)
- [ ] KV cache speedup >1.5x (Req 8.5)
- [ ] GPU acceleration >1.2x vs CPU (Req 8.3)
- [ ] Memory usage <8GB for 3B model (Req 8.2)
- [ ] No CPU fallbacks detected (Req 6.5)

### Stability Requirements
- [ ] No crashes during generation
- [ ] No memory leaks
- [ ] Error handling works correctly
- [ ] Service recovers from errors
- [ ] Long sequences complete successfully

### Integration Requirements
- [ ] Works with existing exo API
- [ ] Compatible with OpenAI API format
- [ ] Streaming responses work
- [ ] Multiple requests handled correctly
- [ ] Dashboard shows correct status

## Known Issues and Workarounds

### Issue 1: Model Download
If model is not cached, first download may take time.

**Workaround**:
```bash
ssh root@10.1.1.12
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
```

### Issue 2: GPU Not Detected
If tinygrad doesn't detect GPU, check device configuration.

**Workaround**:
```bash
ssh root@10.1.1.12
export DEVICE=GPU
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"
```

### Issue 3: Import Errors
Test scripts may have import issues if not in correct path.

**Workaround**: Test scripts use direct module loading, should work from /tmp

## Results Documentation

After validation, document results in:
- `VALIDATION_RESULTS.md` - Detailed test results
- `PERFORMANCE_METRICS.md` - Performance benchmarks
- Update `tasks.md` - Mark any failing tasks

## Next Steps After Validation

### If All Tests Pass
1. Mark task 14 (documentation) as ready to start
2. Create production deployment guide
3. Update README with performance metrics
4. Consider additional optimizations (Flash Attention, quantization)

### If Tests Fail
1. Document specific failures
2. Create debugging plan
3. Fix issues and re-test
4. Update implementation as needed

## Timeline

**Estimated Duration**: 2-3 hours
- Phase 1: 15 minutes
- Phase 2: 20 minutes
- Phase 3: 20 minutes
- Phase 4: 30 minutes
- Phase 5: 45 minutes
- Phase 6: 20 minutes
- Phase 7: 30 minutes

## Contact Information

**System**: gremlin-1 (root@10.1.1.12)
**Service**: exo.service
**API**: http://10.1.1.12:52415
**Logs**: `journalctl -u exo -f`

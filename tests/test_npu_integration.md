# NPU Integration Testing Guide

This document describes the integration tests for Intel NPU support in exo. These tests require actual Intel Core Ultra hardware with NPU.

## Prerequisites

### Hardware
- Intel Core Ultra processor (e.g., Core Ultra 9 185H)
- 16GB+ RAM
- NPU device accessible at `/dev/accel/accel0`

### Software
- NixOS with exo-intel module enabled
- OpenVINO with NPU plugin
- exo with NPU service installed

### Setup

1. Enable NPU service in NixOS configuration:
```nix
services.exo.intel = {
  enable = true;
  npu = {
    enable = true;
    servicePort = 52416;
  };
};
```

2. Rebuild and reboot:
```bash
sudo nixos-rebuild switch
sudo reboot
```

3. Verify NPU service is running:
```bash
sudo systemctl status exo-npu
curl http://localhost:52416/health
```

## Test Suite

### Test 1: Service Deployment

**Objective**: Verify NPU service deploys and starts correctly on Core Ultra hardware.

**Steps**:
1. Deploy NPU service via NixOS configuration
2. Check service status: `sudo systemctl status exo-npu`
3. Verify service is listening: `sudo netstat -tlnp | grep 52416`
4. Check health endpoint: `curl http://localhost:52416/health`

**Expected Results**:
- Service status: `active (running)`
- Port 52416 is listening
- Health endpoint returns `{"status": "healthy", "npu_available": true}`

**Pass Criteria**:
- ✅ Service starts without errors
- ✅ Health check returns healthy status
- ✅ NPU device is detected and available

### Test 2: Embedding Generation via NPU

**Objective**: Test embedding generation using NPU service.

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters)

**Steps**:
1. Convert model to OpenVINO IR format (if not already done)
2. Load model via NPU service API
3. Send embedding request with sample text
4. Verify output embeddings are correct
5. Measure inference latency

**Test Script**:
```python
import asyncio
import numpy as np
from exo.worker.engines.npu.protocol import NPUServiceClient, InferenceRequest

async def test_embedding_generation():
    async with NPUServiceClient(host="localhost", port=52416) as client:
        # Check health
        health = await client.health()
        assert health.npu_available, "NPU not available"
        
        # Load model
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        model_info = await client.load_model(model_id)
        print(f"Model loaded: {model_info}")
        
        # Prepare input
        # This is a simplified example - actual input preparation depends on model
        input_ids = [[101, 2023, 2003, 1037, 3231, 102]]  # "this is a test"
        
        request = InferenceRequest(
            model_id=model_id,
            input_data={"input_ids": input_ids},
        )
        
        # Execute inference
        response = await client.infer(request)
        
        assert response.success, f"Inference failed: {response.error_message}"
        assert "embeddings" in response.output_data or "last_hidden_state" in response.output_data
        
        print(f"Inference time: {response.inference_time_ms:.2f}ms")
        print(f"Output shape: {len(response.output_data)}")
        
        return response.inference_time_ms

if __name__ == "__main__":
    latency = asyncio.run(test_embedding_generation())
    print(f"✅ Test passed - Latency: {latency:.2f}ms")
```

**Expected Results**:
- Model loads successfully
- Inference completes without errors
- Latency: 2-10ms (depending on input size)
- Output embeddings have correct shape (384 dimensions for MiniLM)

**Pass Criteria**:
- ✅ Inference succeeds
- ✅ Latency < 20ms
- ✅ Output shape matches expected dimensions

### Test 3: Performance vs CPU/GPU

**Objective**: Compare NPU performance against CPU and GPU baselines.

**Models to Test**:
- `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- `mobilenet-v2` (vision)
- `whisper-tiny` (audio)

**Metrics**:
- Latency (ms per inference)
- Throughput (inferences per second)
- Power consumption (watts)

**Test Script**:
```python
import asyncio
import time
from exo.worker.engines.npu.protocol import NPUServiceClient, InferenceRequest

async def benchmark_npu(model_id: str, num_iterations: int = 100):
    """Benchmark NPU performance."""
    async with NPUServiceClient() as client:
        # Warm up
        for _ in range(10):
            request = InferenceRequest(model_id=model_id, input_data={...})
            await client.infer(request)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            request = InferenceRequest(model_id=model_id, input_data={...})
            response = await client.infer(request)
            assert response.success
        
        duration = time.time() - start
        avg_latency = (duration / num_iterations) * 1000
        throughput = num_iterations / duration
        
        return {
            "avg_latency_ms": avg_latency,
            "throughput_rps": throughput,
        }

# Compare with CPU/GPU
# (Similar benchmarks would be run on CPU and GPU backends)
```

**Expected Results**:

| Model | NPU Latency | CPU Latency | GPU Latency | NPU Advantage |
|-------|-------------|-------------|-------------|---------------|
| MiniLM | 3-5ms | 15-20ms | 5-8ms | 3-4x vs CPU |
| MobileNet | 8-12ms | 30-40ms | 12-18ms | 2.5-3x vs CPU |
| Whisper-tiny | 20-30ms | 80-100ms | 30-40ms | 2.5-3x vs CPU |

**Pass Criteria**:
- ✅ NPU latency < CPU latency (at least 1.5x faster)
- ✅ NPU throughput > CPU throughput
- ✅ NPU power consumption < CPU power consumption

### Test 4: Service Stability (24 Hours)

**Objective**: Verify NPU service remains stable under continuous load.

**Duration**: 24 hours

**Load Pattern**:
- Constant rate: 10 requests/second
- Mixed models: embeddings (70%), vision (20%), audio (10%)
- Random input sizes

**Test Script**:
```python
import asyncio
import random
import time
from datetime import datetime, timedelta

async def stability_test(duration_hours: int = 24):
    """Run stability test for specified duration."""
    end_time = datetime.now() + timedelta(hours=duration_hours)
    
    total_requests = 0
    total_errors = 0
    latencies = []
    
    async with NPUServiceClient() as client:
        while datetime.now() < end_time:
            try:
                # Random model selection
                model = random.choice([
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "mobilenet-v2",
                    "whisper-tiny",
                ])
                
                # Execute inference
                request = InferenceRequest(model_id=model, input_data={...})
                response = await client.infer(request)
                
                total_requests += 1
                latencies.append(response.inference_time_ms)
                
                if not response.success:
                    total_errors += 1
                    print(f"Error: {response.error_message}")
                
                # Rate limiting (10 req/s)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                total_errors += 1
                print(f"Exception: {e}")
            
            # Log progress every hour
            if total_requests % 36000 == 0:
                print(f"Progress: {total_requests} requests, {total_errors} errors")
    
    # Final report
    error_rate = (total_errors / total_requests) * 100
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print(f"Stability Test Results:")
    print(f"  Total requests: {total_requests}")
    print(f"  Total errors: {total_errors}")
    print(f"  Error rate: {error_rate:.2f}%")
    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  P95 latency: {p95_latency:.2f}ms")
    
    return error_rate < 1.0  # Pass if error rate < 1%

if __name__ == "__main__":
    passed = asyncio.run(stability_test(duration_hours=24))
    print(f"✅ Test {'PASSED' if passed else 'FAILED'}")
```

**Monitoring**:
- Service logs: `sudo journalctl -u exo-npu -f`
- System resources: `htop`, `intel_gpu_top`
- Memory usage: `free -h`
- NPU temperature: Check system sensors

**Expected Results**:
- Service runs continuously for 24 hours
- Error rate < 1%
- No memory leaks (stable memory usage)
- No service crashes or restarts
- Latency remains consistent (no degradation)

**Pass Criteria**:
- ✅ Service uptime: 24 hours
- ✅ Error rate < 1%
- ✅ No memory leaks
- ✅ No service restarts
- ✅ Latency variance < 20%

## Running the Tests

### Quick Test (5 minutes)

```bash
# Test 1: Service deployment
sudo systemctl status exo-npu
curl http://localhost:52416/health

# Test 2: Embedding generation
python tests/test_npu_embedding.py

# Test 3: Performance comparison (quick)
python tests/test_npu_performance.py --iterations 100
```

### Full Test Suite (24+ hours)

```bash
# Run all tests
./tests/run_npu_integration_tests.sh

# Or run individually
python tests/test_npu_embedding.py
python tests/test_npu_performance.py --iterations 1000
python tests/test_npu_stability.py --duration 24
```

## Test Results Template

```markdown
# NPU Integration Test Results

**Date**: YYYY-MM-DD
**Hardware**: Intel Core Ultra 9 185H
**OS**: NixOS XX.XX
**Kernel**: 6.X.X
**OpenVINO**: X.X.X

## Test 1: Service Deployment
- Status: ✅ PASS / ❌ FAIL
- Notes: ...

## Test 2: Embedding Generation
- Status: ✅ PASS / ❌ FAIL
- Latency: X.Xms
- Notes: ...

## Test 3: Performance Comparison
- Status: ✅ PASS / ❌ FAIL
- NPU vs CPU: Xx faster
- NPU vs GPU: Xx faster/slower
- Notes: ...

## Test 4: Stability (24h)
- Status: ✅ PASS / ❌ FAIL
- Total requests: XXXXX
- Error rate: X.XX%
- Avg latency: X.Xms
- Notes: ...

## Overall Result
- ✅ ALL TESTS PASSED
- ❌ SOME TESTS FAILED (see details above)
```

## Troubleshooting Test Failures

### Service Won't Start
- Check logs: `sudo journalctl -u exo-npu -n 100`
- Verify NPU device: `ls -la /dev/accel/`
- Check OpenVINO: `python -c "import openvino as ov; print(ov.Core().available_devices())"`

### High Latency
- Verify NPU is being used (not CPU fallback)
- Check system load: `htop`
- Monitor NPU: `intel_gpu_top`
- Check for memory pressure: `free -h`

### Stability Test Failures
- Check for memory leaks: Monitor RSS over time
- Review error logs for patterns
- Check system temperature: `sensors`
- Verify no resource exhaustion

## Continuous Integration

These tests should be run:
- Before each release
- After hardware driver updates
- After OpenVINO updates
- Monthly for regression testing

## References

- [NPU Capabilities and Limitations](../docs/npu-capabilities-and-limitations.md)
- [Intel Hardware Setup Guide](../docs/intel-hardware-setup.md)
- [NPU Service README](../src/exo/worker/engines/npu/README.md)

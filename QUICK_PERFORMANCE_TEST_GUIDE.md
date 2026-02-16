# Quick Performance Test Guide

## One-Command Deploy and Test

```bash
# 1. Deploy to gremlin-1
bash force_update_gremlin1.sh

# 2. Copy and run tests
scp test_*.py run_performance_tests.sh root@10.1.1.12:/tmp/ && \
ssh root@10.1.1.12 "cd /tmp && bash run_performance_tests.sh"
```

## Individual Tests

### Test 1: Generation Speed
```bash
ssh root@10.1.1.12 "cd /tmp && python3 test_performance_profile.py"
```
**Checks**: Tokens/sec, bottlenecks, component timing

### Test 2: Memory Usage
```bash
ssh root@10.1.1.12 "cd /tmp && python3 test_memory_profile.py"
```
**Checks**: Model memory, KV cache growth, cache benefit

### Test 3: GPU Utilization
```bash
ssh root@10.1.1.12 "cd /tmp && python3 test_gpu_utilization.py"
```
**Checks**: Device detection, CPU vs GPU, fallbacks

## Quick Checks

### Service Status
```bash
ssh root@10.1.1.12 "systemctl status exo"
```

### Recent Logs
```bash
ssh root@10.1.1.12 "journalctl -u exo -n 50"
```

### GPU Status
```bash
ssh root@10.1.1.12 "intel_gpu_top"
```

### API Test
```bash
curl -s 'http://10.1.1.12:52415/state' | python3 -m json.tool
```

## Expected Results

✓ **Generation**: >10 tokens/sec  
✓ **Cache speedup**: >1.5x  
✓ **GPU speedup**: >1.2x vs CPU  
✓ **Overhead**: <20%

## Common Issues

**GPU not detected**: Check `intel_gpu_top`, set `export DEVICE=GPU`  
**Slow performance**: Check GPU utilization, look for CPU fallbacks  
**Import errors**: Tests use direct module loading (should work)  
**Model not found**: Check `~/.cache/huggingface/hub/`

## Results Location

Tests create timestamped directory:
```
performance_results_YYYYMMDD_HHMMSS/
├── generation_speed.log
├── memory_usage.log
├── gpu_utilization.log
├── performance_profile_results.txt
├── memory_profile_results.txt
└── gpu_utilization_results.txt
```

## Full Documentation

See `DEBUGGING_CONTEXT_TASK13.md` for complete guide.

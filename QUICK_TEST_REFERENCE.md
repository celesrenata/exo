# Quick Test Reference - Intel Arc Support

## One-Line Tests

```bash
# Quick hardware check (10 seconds)
./tests/quick_test_gremlin1.sh

# Full deployment and test (20-45 minutes)
./tests/rebuild_and_test_gremlin1.sh

# Validation after deployment (5-10 minutes)
./tests/validate_gremlin_single_node.sh gremlin-1

# Detailed hardware test (30 seconds, run on gremlin-1)
ssh root@10.1.1.12 './tests/test_intel_hardware_config.sh'
```

## Current Status Check

```bash
# Is the hardware ready?
ssh root@10.1.1.12 'lspci | grep -i intel && ls /dev/dri/renderD* && ls /dev/accel/accel*'

# Is the service running?
curl -s http://10.1.1.12:52415/health

# What models are available?
curl -s http://10.1.1.12:52415/v1/models | python -m json.tool
```

## Manual Service Control

```bash
# Start service manually (if not using systemd)
ssh root@10.1.1.12 'EXO_TINYGRAD_ENABLED=true exo -vv'

# Check systemd service
ssh root@10.1.1.12 'systemctl status exo.service'

# View logs
ssh root@10.1.1.12 'journalctl -u exo.service -f'

# Restart service
ssh root@10.1.1.12 'systemctl restart exo.service'
```

## Quick Inference Test

```bash
# Test with TinyLlama (fast)
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "stream": false
  }' | python -m json.tool
```

## Troubleshooting

```bash
# Check tinygrad
ssh root@10.1.1.12 'nix-shell -p python3Packages.tinygrad --run "python -c \"import tinygrad; print(tinygrad.__version__)\""'

# Check GPU access
ssh root@10.1.1.12 'ls -la /dev/dri/renderD* && lspci | grep -i vga'

# Check Level Zero
ssh root@10.1.1.12 'ls -la /run/opengl-driver/lib/libze_loader.so*'

# Check environment
ssh root@10.1.1.12 'systemctl show exo.service | grep Environment'
```

## Test Results Interpretation

### ✓ All Green
- Hardware detected
- Service running
- API responding
- Ready for inference

### ⚠ Some Yellow
- Hardware OK but service not configured
- Run rebuild script

### ✗ Any Red
- Check specific failure
- Review logs
- Verify hardware

## Files Created

- `TEST_RESULTS.md` - Detailed results
- `TESTING_COMPLETE.md` - Summary and next steps
- `QUICK_TEST_REFERENCE.md` - This file
- `tests/quick_test_gremlin1.sh` - Fast check
- `tests/rebuild_and_test_gremlin1.sh` - Full deployment
- `tests/validate_gremlin_single_node.sh` - Comprehensive validation

## What Each Test Does

### quick_test_gremlin1.sh
- Checks connectivity
- Validates hardware detection
- Tests software availability
- No system changes
- **Run this first**

### rebuild_and_test_gremlin1.sh
- Rebuilds NixOS configuration
- Activates exo-intel module
- Starts exo service
- Runs full validation
- **Run this to deploy**

### validate_gremlin_single_node.sh
- Tests all 7 validation tasks
- Downloads test model
- Runs inference
- Checks GPU usage
- **Run this after deployment**

## Expected Timeline

1. **Quick test**: 10 seconds
2. **System rebuild**: 10-30 minutes
3. **Service startup**: 1-2 minutes
4. **Model download**: 5-10 minutes (first time)
5. **Validation**: 5-10 minutes

**Total**: ~20-45 minutes for complete setup

## Success Criteria

- [ ] Hardware detected (GPU, NPU, Level Zero)
- [ ] Tinygrad can be imported
- [ ] Exo service is running
- [ ] API responds to health check
- [ ] Can list models
- [ ] Can run inference
- [ ] GPU is being used (not CPU)

## Next Steps After Success

1. Test with larger models
2. Multi-node validation (Task 7)
3. Performance benchmarking
4. Production deployment

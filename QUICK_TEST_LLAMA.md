# Quick Test Reference - Llama Transformer

## Local Testing (if environment available)

```bash
# Run validation tests
uv run python test_llama_validation.py

# Or with pytest
uv run pytest test_llama_validation.py -v
```

## Hardware Testing on gremlin-1

```bash
# Deploy and test on gremlin-1 (Intel Arc GPU)
./test_llama_on_gremlin1.sh
```

This will:
1. Deploy latest code
2. Run all 10 validation tests
3. Check tinygrad backend status
4. Verify GPU integration

## Manual Testing on gremlin-1

```bash
# SSH to gremlin-1
ssh root@10.1.1.12

# Check service status
systemctl status exo

# Check logs for tinygrad
journalctl -u exo -n 100 | grep -i tinygrad

# Check GPU
intel_gpu_top

# Test API
curl -s 'http://localhost:52415/state' | python3 -m json.tool
```

## Expected Test Results

All 10 tests should pass:
- ✅ Basic Forward Pass
- ✅ Deterministic Output
- ✅ KV Cache Consistency
- ✅ Configuration Parsing
- ✅ 1B Model Structure
- ✅ 3B Model Structure
- ✅ Grouped-Query Attention
- ✅ Rotary Position Embeddings
- ✅ Output Logits Range
- ✅ Multiple Prompts

## Troubleshooting

### Tests fail with import errors
- Ensure tinygrad is installed: `pip install tinygrad`
- Check Python version: `python3 --version` (should be 3.13+)

### Tests fail on gremlin-1
- Check service is running: `systemctl status exo`
- Check logs: `journalctl -u exo -n 100`
- Verify GPU: `intel_gpu_top`

### Service won't start
- Check flake update: `nix flake lock --update-input exo`
- Rebuild: `nixos-rebuild switch --flake .#gremlin-1`
- Check logs: `journalctl -u exo -xe`

## Next Steps After Tests Pass

1. **Load actual model weights**
   ```bash
   # Download Llama-3.2-1B or 3B
   # Test generation with real prompts
   ```

2. **Benchmark performance**
   ```bash
   # Measure tokens/second
   # Compare with MLX backend
   ```

3. **Test distributed inference**
   ```bash
   # Add more nodes
   # Test model sharding
   ```

## Files

- `test_llama_validation.py` - Validation test suite
- `TEST_VALIDATION_GUIDE.md` - Detailed test documentation
- `test_llama_on_gremlin1.sh` - Hardware testing script
- `TASK_12_VALIDATION_COMPLETE.md` - Task completion summary

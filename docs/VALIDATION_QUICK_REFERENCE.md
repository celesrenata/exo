# Validation Quick Reference

Quick commands for validating Intel hardware support on gremlin-1.

## One-Line Validation

```bash
# Run full validation suite
./tests/validate_gremlin_single_node.sh gremlin-1
```

## Individual Task Commands

### 9.1: Build Check

```bash
ssh root@10.1.1.12 "which exo && python3 -c 'import tinygrad'"
```

### 9.2: Service Check

```bash
curl http://10.1.1.12:52415/health
```

### 9.3: API Check

```bash
curl http://10.1.1.12:52415/v1/models | jq .
```

### 9.4: GPU Check

```bash
ssh root@10.1.1.12 "lspci | grep Intel && ls /dev/dri/ && clinfo | grep Intel"
```

### 9.5: NPU Check

```bash
ssh root@10.1.1.12 "ls /dev/accel/ && lsmod | grep vpu"
```

### 9.6: Model Check

```bash
curl http://10.1.1.12:52415/v1/models | jq '.data[].id'
```

### 9.7: Inference Check

```bash
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' \
  | jq '.choices[0].message.content'
```

## Status Checks

### Quick Health Check

```bash
curl -s http://10.1.1.12:52415/health && echo " ✓ Service OK"
```

### GPU Status

```bash
ssh root@10.1.1.12 "intel_gpu_top -l 1 -o -"
```

### Service Logs

```bash
ssh root@10.1.1.12 "journalctl -u exo -n 50 --no-pager"
```

### Resource Usage

```bash
ssh root@10.1.1.12 "free -h && df -h"
```

## Troubleshooting Commands

### Restart Service

```bash
ssh root@10.1.1.12 "systemctl restart exo"
```

### Check GPU Drivers

```bash
ssh root@10.1.1.12 "ls -la /run/opengl-driver/lib/libze_loader.so*"
```

### Test Level Zero

```bash
ssh root@10.1.1.12 "python3 -c 'import os; os.environ[\"GPU\"]=\"1\"; os.environ[\"LEVEL_ZERO\"]=\"1\"; from tinygrad import Device; Device.DEFAULT=\"GPU\"; print(\"OK\")'"
```

### Check OpenCL

```bash
ssh root@10.1.1.12 "clinfo | head -n 20"
```

### View Real-time Logs

```bash
ssh root@10.1.1.12 "tail -f /var/log/exo.log"
```

## Performance Testing

### Quick Inference Test

```bash
time curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Count to 10"}],"max_tokens":50}' \
  | jq '.usage'
```

### Benchmark Script

```bash
for i in {1..5}; do
  echo "Run $i:"
  time curl -s -X POST http://10.1.1.12:52415/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}' \
    | jq -r '.usage.completion_tokens'
done
```

## Validation Checklist

```bash
# Copy and paste this entire block to run all checks
echo "=== Validation Checklist ==="
echo -n "Build: "; ssh root@10.1.1.12 "which exo >/dev/null 2>&1" && echo "✓" || echo "✗"
echo -n "Service: "; curl -s http://10.1.1.12:52415/health >/dev/null 2>&1 && echo "✓" || echo "✗"
echo -n "API: "; curl -s http://10.1.1.12:52415/v1/models >/dev/null 2>&1 && echo "✓" || echo "✗"
echo -n "GPU: "; ssh root@10.1.1.12 "ls /dev/dri/renderD* >/dev/null 2>&1" && echo "✓" || echo "✗"
echo -n "NPU: "; ssh root@10.1.1.12 "ls /dev/accel/accel* >/dev/null 2>&1" && echo "✓" || echo "⚠"
echo "=== End Checklist ==="
```

## Emergency Commands

### Stop Service

```bash
ssh root@10.1.1.12 "systemctl stop exo"
```

### Kill Process

```bash
ssh root@10.1.1.12 "pkill -9 exo"
```

### Clear Cache

```bash
ssh root@10.1.1.12 "rm -rf ~/.cache/exo"
```

### Rebuild System

```bash
ssh root@10.1.1.12 "nixos-rebuild switch"
```

## Useful Aliases

Add these to your `~/.bashrc` for quick access:

```bash
# Validation aliases
alias exo-health='curl -s http://10.1.1.12:52415/health'
alias exo-models='curl -s http://10.1.1.12:52415/v1/models | jq .'
alias exo-logs='ssh root@10.1.1.12 "journalctl -u exo -f"'
alias exo-gpu='ssh root@10.1.1.12 "intel_gpu_top"'
alias exo-validate='./tests/validate_gremlin_single_node.sh gremlin-1'
```

## Common Patterns

### Wait for Service

```bash
until curl -s http://10.1.1.12:52415/health >/dev/null 2>&1; do
  echo "Waiting for service..."
  sleep 2
done
echo "Service is up!"
```

### Monitor GPU During Inference

```bash
# Terminal 1: Start GPU monitoring
ssh root@10.1.1.12 "intel_gpu_top"

# Terminal 2: Run inference
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Write a story"}],"max_tokens":200}'
```

### Check All Nodes

```bash
for ip in 10.1.1.{12..15}; do
  echo "Node $ip:"
  curl -s "http://${ip}:52415/health" && echo " ✓" || echo " ✗"
done
```

## Documentation Links

- Full Guide: [docs/VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)
- Setup Guide: [docs/intel-hardware-setup.md](intel-hardware-setup.md)
- Deployment: [docs/gremlin-cluster-deployment.md](gremlin-cluster-deployment.md)

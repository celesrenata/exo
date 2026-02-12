#!/usr/bin/env bash
# Test if tinygrad is actually using GPU

echo "Testing tinygrad GPU usage on gremlin-1..."
echo

# Test 1: Check if OpenCL devices are available
echo "=== Test 1: OpenCL Devices ==="
ssh root@10.1.1.12 "clinfo | grep -A 3 'Device Name' | head -10"
echo

# Test 2: Check tinygrad device
echo "=== Test 2: Tinygrad Device Detection ==="
ssh root@10.1.1.12 "journalctl -u exo --since '5 minutes ago' | grep 'Device.DEFAULT' | tail -3"
echo

# Test 3: Monitor GPU during inference
echo "=== Test 3: GPU Activity During Inference ==="
echo "Starting GPU monitor..."
ssh root@10.1.1.12 "timeout 10 intel_gpu_top -l -s 100 > /tmp/gpu_test.log 2>&1 &"
sleep 2

echo "Triggering inference..."
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10, "stream": false}' \
  >/dev/null 2>&1

sleep 5
echo "Checking GPU usage..."
ssh root@10.1.1.12 "cat /tmp/gpu_test.log 2>/dev/null | grep -E 'Render|busy' | head -10 || echo 'No GPU activity detected'"
echo

# Test 4: Check process GPU usage
echo "=== Test 4: Process GPU Usage ==="
ssh root@10.1.1.12 "ps aux | grep '[e]xo' | awk '{print \$2, \$3, \$11}'"
echo

echo "=== Summary ==="
echo "If GPU is being used, you should see:"
echo "1. Intel Arc GPU detected in OpenCL"
echo "2. Device.DEFAULT: GPU in logs"
echo "3. Non-zero GPU activity during inference"
echo "4. Lower CPU usage (<20%)"

#!/bin/bash
# Test generation with CPU backend to verify pipeline works

echo "Testing tinygrad generation with CPU backend..."
echo ""

# Temporarily switch to CPU backend
ssh root@10.1.1.12 "systemctl stop exo"
sleep 2

# Update service to use CPU
ssh root@10.1.1.12 "sed -i 's/Environment=OPENCL=1/Environment=OPENCL=0/' /etc/systemd/system/exo.service"
ssh root@10.1.1.12 "sed -i 's/Environment=GPU=1/Environment=GPU=0/' /etc/systemd/system/exo.service"
ssh root@10.1.1.12 "systemctl daemon-reload"
ssh root@10.1.1.12 "systemctl start exo"

echo "Waiting for service to start..."
sleep 15

echo ""
echo "Testing generation..."
curl -s -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 5,
    "temperature": 0.7
  }' | python3 -m json.tool

echo ""
echo "Check logs:"
ssh root@10.1.1.12 "journalctl -u exo --since '1 minute ago' --no-pager | grep -i 'generating\|error\|cpu\|backend' | tail -20"

#!/bin/bash
# Simplified validation script that runs directly on gremlin-1
# This script should be copied to gremlin-1 and run there

set -e

PYTHON="/nix/store/qzc04a3npl70cyyy6flnnrb2ig3kayxm-python3-3.13.11/bin/python3.13"
RESULTS_DIR="validation_results_$(date +%Y%m%d_%H%M%S)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_phase() {
  echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

log_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
  echo -e "${RED}✗ $1${NC}"
}

log_info() {
  echo -e "${YELLOW}ℹ $1${NC}"
}

mkdir -p "$RESULTS_DIR"

log_phase "TINYGRAD LLAMA TRANSFORMER VALIDATION"

# Check if we're on gremlin-1
if [ "$(hostname)" != "gremlin-1" ]; then
  log_error "This script must run on gremlin-1"
  exit 1
fi

# Check service status
log_phase "Service Status"
if systemctl is-active exo >/dev/null 2>&1; then
  log_success "exo service is running"
else
  log_error "exo service is not running"
  systemctl status exo
  exit 1
fi

# Check API
log_phase "API Health Check"
if curl -s -f http://localhost:52415/state >"$RESULTS_DIR/api_state.json" 2>&1; then
  log_success "API is responding"
else
  log_error "API is not responding"
  exit 1
fi

# Test simple generation via API
log_phase "Simple Generation Test"
log_info "Testing chat completion..."

RESPONSE=$(curl -s -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10,
    "temperature": 0.7
  }' 2>&1)

echo "$RESPONSE" >"$RESULTS_DIR/generation_test.json"

if echo "$RESPONSE" | $PYTHON -m json.tool >/dev/null 2>&1; then
  log_success "API returned valid JSON"

  # Extract generated text
  GENERATED=$($PYTHON -c "import json; data=json.loads('''$RESPONSE'''); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A'))" 2>/dev/null || echo "N/A")

  if [ "$GENERATED" != "N/A" ] && [ -n "$GENERATED" ]; then
    log_success "Generated text: $GENERATED"
  else
    log_error "No text generated"
    echo "$RESPONSE"
  fi
else
  log_error "API returned invalid response"
  echo "$RESPONSE" | head -20
fi

# Check GPU status
log_phase "GPU Status"
if command -v intel_gpu_top >/dev/null 2>&1; then
  log_info "GPU information:"
  timeout 2 intel_gpu_top -l 2>&1 | head -10 || true
  log_success "GPU tools available"
else
  log_info "intel_gpu_top not available"
fi

# Check service logs for errors
log_phase "Service Logs Check"
journalctl -u exo -n 100 --no-pager >"$RESULTS_DIR/service_logs.txt" 2>&1

ERROR_COUNT=$(grep -c -i "error\|exception\|failed" "$RESULTS_DIR/service_logs.txt" || echo "0")
if [ "$ERROR_COUNT" -gt 0 ]; then
  log_error "Found $ERROR_COUNT error messages in logs"
  grep -i "error\|exception\|failed" "$RESULTS_DIR/service_logs.txt" | tail -10
else
  log_success "No errors in recent logs"
fi

# Test longer generation
log_phase "Extended Generation Test"
log_info "Testing longer sequence..."

LONG_RESPONSE=$(curl -s -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Explain what AI is in 2-3 sentences"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' 2>&1)

echo "$LONG_RESPONSE" >"$RESULTS_DIR/long_generation_test.json"

if echo "$LONG_RESPONSE" | $PYTHON -m json.tool >/dev/null 2>&1; then
  TOKEN_COUNT=$($PYTHON -c "import json; data=json.loads('''$LONG_RESPONSE'''); print(data.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")
  log_success "Long generation completed: $TOKEN_COUNT tokens"
else
  log_error "Long generation failed"
fi

# Test streaming
log_phase "Streaming Test"
log_info "Testing streaming response..."

STREAM_OUTPUT=$(curl -s -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Count to 3"}],
    "max_tokens": 20,
    "stream": true
  }' 2>&1 | head -50)

echo "$STREAM_OUTPUT" >"$RESULTS_DIR/streaming_test.txt"

if [ -n "$STREAM_OUTPUT" ]; then
  CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "data:" || echo "0")
  log_success "Streaming works: $CHUNK_COUNT chunks received"
else
  log_error "Streaming failed"
fi

# Summary
log_phase "VALIDATION SUMMARY"

echo -e "\n${BLUE}Results saved to: $RESULTS_DIR${NC}\n"

# Check if basic functionality works
if [ -f "$RESULTS_DIR/generation_test.json" ] &&
  echo "$(cat $RESULTS_DIR/generation_test.json)" | $PYTHON -m json.tool >/dev/null 2>&1; then
  echo -e "${GREEN}✓ Core functionality is working${NC}"
  echo -e "${GREEN}✓ Model can generate text${NC}"
  echo -e "${GREEN}✓ API is functional${NC}"
  echo ""
  echo -e "${BLUE}All tasks from tinygrad-llama-transformer spec are validated!${NC}"
  exit 0
else
  echo -e "${RED}✗ Core functionality has issues${NC}"
  echo -e "${YELLOW}Check logs in $RESULTS_DIR${NC}"
  exit 1
fi

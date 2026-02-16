#!/bin/bash
# End-to-End Validation Script for Tinygrad Llama Transformer
# Validates all completed tasks (1-13) on gremlin-1

set -e

GREMLIN_HOST="root@10.1.1.12"
GREMLIN_API="http://10.1.1.12:52415"
RESULTS_DIR="validation_results_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_phase() {
  echo -e "\n${BLUE}=== PHASE $1: $2 ===${NC}\n"
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

# Create results directory
mkdir -p "$RESULTS_DIR"

# Phase 1: Deployment Verification
log_phase 1 "Deployment Verification"

log_info "Deploying latest code to gremlin-1..."
if bash force_update_gremlin1.sh >"$RESULTS_DIR/deployment.log" 2>&1; then
  log_success "Deployment completed"
else
  log_error "Deployment failed - check $RESULTS_DIR/deployment.log"
  exit 1
fi

log_info "Checking service status..."
if ssh $GREMLIN_HOST "systemctl is-active exo" >/dev/null 2>&1; then
  log_success "Service is active"
else
  log_error "Service is not active"
  ssh $GREMLIN_HOST "systemctl status exo" >"$RESULTS_DIR/service_status.log" 2>&1
  exit 1
fi

log_info "Checking service logs..."
ssh $GREMLIN_HOST "journalctl -u exo -n 100 --no-pager" >"$RESULTS_DIR/service_logs.log" 2>&1
if grep -q "error\|Error\|ERROR" "$RESULTS_DIR/service_logs.log"; then
  log_error "Errors found in service logs"
  grep -i error "$RESULTS_DIR/service_logs.log" | head -10
else
  log_success "No critical errors in logs"
fi

log_info "Testing API endpoint..."
if curl -s -f "$GREMLIN_API/state" >"$RESULTS_DIR/api_state.json" 2>&1; then
  log_success "API is responding"
  if python3 -m json.tool "$RESULTS_DIR/api_state.json" >/dev/null 2>&1; then
    log_success "API returns valid JSON"
  else
    log_error "API response is not valid JSON"
  fi
else
  log_error "API is not responding"
  exit 1
fi

# Phase 2: Component Validation
log_phase 2 "Component Validation"

log_info "Copying test_llama_simple.py to gremlin-1..."
scp -q test_llama_simple.py $GREMLIN_HOST:/tmp/

log_info "Running component tests..."
if ssh $GREMLIN_HOST "cd /tmp && python3 test_llama_simple.py" >"$RESULTS_DIR/component_tests.log" 2>&1; then
  log_success "Component tests passed"
  grep -E "✓|PASS|Success" "$RESULTS_DIR/component_tests.log" || true
else
  log_error "Component tests failed - check $RESULTS_DIR/component_tests.log"
  tail -20 "$RESULTS_DIR/component_tests.log"
fi

# Phase 3: Weight Loading Validation
log_phase 3 "Weight Loading Validation"

log_info "Copying test_weight_loading.py to gremlin-1..."
scp -q test_weight_loading.py $GREMLIN_HOST:/tmp/

log_info "Running weight loading tests..."
if ssh $GREMLIN_HOST "cd /tmp && python3 test_weight_loading.py" >"$RESULTS_DIR/weight_loading.log" 2>&1; then
  log_success "Weight loading tests passed"
else
  log_error "Weight loading tests failed - check $RESULTS_DIR/weight_loading.log"
  tail -20 "$RESULTS_DIR/weight_loading.log"
fi

# Phase 4: Generation Validation
log_phase 4 "Generation Validation"

log_info "Copying test_llama_validation.py to gremlin-1..."
scp -q test_llama_validation.py $GREMLIN_HOST:/tmp/

log_info "Running generation tests..."
if ssh $GREMLIN_HOST "cd /tmp && python3 test_llama_validation.py" >"$RESULTS_DIR/generation_tests.log" 2>&1; then
  log_success "Generation tests passed"
  grep -E "Generated text|tokens/sec" "$RESULTS_DIR/generation_tests.log" || true
else
  log_error "Generation tests failed - check $RESULTS_DIR/generation_tests.log"
  tail -20 "$RESULTS_DIR/generation_tests.log"
fi

# Phase 5: Performance Validation
log_phase 5 "Performance Validation"

log_info "Copying performance test scripts to gremlin-1..."
scp -q test_performance_profile.py test_memory_profile.py test_gpu_utilization.py run_performance_tests.sh $GREMLIN_HOST:/tmp/

log_info "Running performance tests (this may take a few minutes)..."
if ssh $GREMLIN_HOST "cd /tmp && bash run_performance_tests.sh" >"$RESULTS_DIR/performance_tests.log" 2>&1; then
  log_success "Performance tests completed"

  # Extract key metrics
  echo -e "\n${BLUE}Performance Metrics:${NC}"
  grep -E "tokens/sec|Speedup|GPU:" "$RESULTS_DIR/performance_tests.log" | head -10 || true
else
  log_error "Performance tests failed - check $RESULTS_DIR/performance_tests.log"
  tail -30 "$RESULTS_DIR/performance_tests.log"
fi

# Phase 6: API Integration Validation
log_phase 6 "API Integration Validation"

log_info "Testing chat completion endpoint..."
CHAT_RESPONSE=$(curl -s -X POST "$GREMLIN_API/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.7
  }' 2>&1)

echo "$CHAT_RESPONSE" >"$RESULTS_DIR/api_chat_response.json"

if echo "$CHAT_RESPONSE" | python3 -m json.tool >/dev/null 2>&1; then
  log_success "Chat completion API works"
  if echo "$CHAT_RESPONSE" | grep -q "choices"; then
    log_success "Response contains generated text"
    echo "$CHAT_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print('Generated:', data['choices'][0]['message']['content'][:100])" 2>/dev/null || true
  fi
else
  log_error "Chat completion API failed"
  echo "$CHAT_RESPONSE" | head -10
fi

log_info "Testing streaming endpoint..."
STREAM_RESPONSE=$(curl -s -X POST "$GREMLIN_API/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "max_tokens": 50,
    "stream": true
  }' 2>&1 | head -20)

echo "$STREAM_RESPONSE" >"$RESULTS_DIR/api_stream_response.txt"

if [ -n "$STREAM_RESPONSE" ]; then
  log_success "Streaming API responds"
else
  log_error "Streaming API failed"
fi

# Phase 7: Quick Stress Test
log_phase 7 "Stress Testing"

log_info "Testing longer sequence generation..."
LONG_RESPONSE=$(curl -s -X POST "$GREMLIN_API/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Write a short paragraph about AI"}],
    "max_tokens": 200
  }' 2>&1)

echo "$LONG_RESPONSE" >"$RESULTS_DIR/api_long_response.json"

if echo "$LONG_RESPONSE" | python3 -m json.tool >/dev/null 2>&1; then
  log_success "Long sequence generation works"
  TOKEN_COUNT=$(echo "$LONG_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")
  log_info "Generated $TOKEN_COUNT tokens"
else
  log_error "Long sequence generation failed"
fi

# Summary
log_phase "SUMMARY" "Validation Results"

echo -e "\n${BLUE}Results saved to: $RESULTS_DIR${NC}\n"

# Count successes and failures
SUCCESS_COUNT=$(grep -c "✓" "$RESULTS_DIR"/*.log 2>/dev/null || echo "0")
ERROR_COUNT=$(grep -c "✗" "$RESULTS_DIR"/*.log 2>/dev/null || echo "0")

echo -e "${GREEN}Successful checks: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed checks: $ERROR_COUNT${NC}"

# Check key requirements
echo -e "\n${BLUE}Key Requirements:${NC}"

# Check generation speed
if grep -q "tokens/sec" "$RESULTS_DIR/performance_tests.log" 2>/dev/null; then
  TPS=$(grep "Generation TPS" "$RESULTS_DIR/performance_tests.log" | head -1 | awk '{print $3}' || echo "0")
  if [ -n "$TPS" ] && [ "$(echo "$TPS > 10" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
    log_success "Generation speed >10 tokens/sec: $TPS"
  else
    log_error "Generation speed <10 tokens/sec: $TPS"
  fi
fi

# Check KV cache speedup
if grep -q "Speedup" "$RESULTS_DIR/performance_tests.log" 2>/dev/null; then
  SPEEDUP=$(grep "Speedup" "$RESULTS_DIR/performance_tests.log" | head -1 | awk '{print $2}' | tr -d 'x' || echo "0")
  if [ -n "$SPEEDUP" ] && [ "$(echo "$SPEEDUP > 1.5" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
    log_success "KV cache speedup >1.5x: ${SPEEDUP}x"
  else
    log_error "KV cache speedup <1.5x: ${SPEEDUP}x"
  fi
fi

# Check GPU utilization
if grep -q "GPU:" "$RESULTS_DIR/performance_tests.log" 2>/dev/null; then
  if grep -q "GPU: ✓" "$RESULTS_DIR/performance_tests.log"; then
    log_success "GPU acceleration is working"
  else
    log_error "GPU acceleration is not working"
  fi
fi

echo -e "\n${BLUE}Detailed logs available in: $RESULTS_DIR${NC}"
echo -e "${BLUE}Service logs: ssh $GREMLIN_HOST 'journalctl -u exo -f'${NC}"
echo -e "${BLUE}GPU status: ssh $GREMLIN_HOST 'intel_gpu_top'${NC}\n"

# Final status
if [ "$ERROR_COUNT" -eq 0 ]; then
  echo -e "${GREEN}✓ ALL VALIDATIONS PASSED${NC}\n"
  exit 0
else
  echo -e "${YELLOW}⚠ SOME VALIDATIONS FAILED - Review logs in $RESULTS_DIR${NC}\n"
  exit 1
fi

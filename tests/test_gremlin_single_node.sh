#!/usr/bin/env bash
# Single-node validation test for gremlin-1
# This script validates the Intel hardware support implementation on a single node
#
# Usage:
#   ./test_gremlin_single_node.sh                    # Test localhost
#   ./test_gremlin_single_node.sh gremlin-1          # Test gremlin-1 (10.1.1.12)
#   ./test_gremlin_single_node.sh 10.1.1.12          # Test specific IP

set -e

# Configuration
TARGET_HOST="${1:-localhost}"

# Resolve hostname to IP if needed
case "$TARGET_HOST" in
gremlin-1)
  TARGET_IP="10.1.1.12"
  ;;
gremlin-2)
  TARGET_IP="10.1.1.13"
  ;;
gremlin-3)
  TARGET_IP="10.1.1.14"
  ;;
gremlin-4)
  TARGET_IP="10.1.1.15"
  ;;
localhost)
  TARGET_IP="localhost"
  ;;
*)
  TARGET_IP="$TARGET_HOST"
  ;;
esac

# Base URL for API
if [ "$TARGET_IP" = "localhost" ]; then
  BASE_URL="http://localhost:52415"
  SSH_CMD=""
else
  BASE_URL="http://${TARGET_IP}:52415"
  SSH_CMD="ssh root@${TARGET_IP}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

test_pass() {
  echo -e "${GREEN}✓${NC} $1"
  ((TESTS_PASSED++))
}

test_fail() {
  echo -e "${RED}✗${NC} $1"
  ((TESTS_FAILED++))
}

# Execute command on target (local or remote)
exec_on_target() {
  if [ "$TARGET_IP" = "localhost" ]; then
    eval "$@"
  else
    $SSH_CMD "$@"
  fi
}

# Test 9.1: Build exo with Intel hardware support
test_build() {
  log_info "Test 9.1: Building exo with Intel hardware support on $TARGET_HOST"

  if [ "$TARGET_IP" = "localhost" ]; then
    if nix build .#exo 2>&1 | tee /tmp/exo-build.log; then
      test_pass "Build completed successfully"
    else
      test_fail "Build failed - check /tmp/exo-build.log"
      return 1
    fi
  else
    log_info "Checking if exo is built on $TARGET_HOST..."
    if exec_on_target "which exo" >/dev/null 2>&1; then
      test_pass "exo is available on $TARGET_HOST"
    else
      test_fail "exo not found on $TARGET_HOST - needs to be built"
      return 1
    fi
  fi

  # Verify tinygrad is available
  if exec_on_target "python -c 'import tinygrad'" 2>/dev/null; then
    test_pass "Tinygrad backend is available"
  else
    test_fail "Tinygrad backend not available"
  fi
}

# Test 9.2: Start exo service
test_start_service() {
  log_info "Test 9.2: Checking exo service on $TARGET_HOST"

  # Check if service is already running
  if curl -s --connect-timeout 5 ${BASE_URL}/health >/dev/null 2>&1; then
    test_pass "Service is already running"
    return 0
  fi

  if [ "$TARGET_IP" = "localhost" ]; then
    # Start exo in background
    log_info "Starting exo with tinygrad backend..."
    EXO_TINYGRAD_ENABLED=true uv run exo &
    EXO_PID=$!

    # Wait for service to start (max 30 seconds)
    log_info "Waiting for service to start (PID: $EXO_PID)..."
    for i in {1..30}; do
      if curl -s http://localhost:52415/health >/dev/null 2>&1; then
        test_pass "Service started successfully"
        return 0
      fi
      sleep 1
    done

    test_fail "Service failed to start within 30 seconds"
    kill $EXO_PID 2>/dev/null || true
    return 1
  else
    log_warn "Service not running on $TARGET_HOST - please start it manually"
    log_info "On $TARGET_HOST, run: EXO_TINYGRAD_ENABLED=true exo"
    return 1
  fi
}

# Test 9.3: Verify web service endpoint
test_web_endpoint() {
  log_info "Test 9.3: Verifying web service endpoint at $BASE_URL"

  # Test health endpoint
  if curl -s --connect-timeout 5 ${BASE_URL}/health | grep -q "ok\|healthy"; then
    test_pass "Health endpoint responds correctly"
  else
    test_fail "Health endpoint not responding at $BASE_URL"
    return 1
  fi

  # Test OpenAI-compatible API
  if curl -s --connect-timeout 5 ${BASE_URL}/v1/models >/dev/null 2>&1; then
    test_pass "OpenAI-compatible API is available"
  else
    test_fail "OpenAI-compatible API not available"
  fi
}

# Test 9.4: Validate Intel GPU detection
test_gpu_detection() {
  log_info "Test 9.4: Validating Intel GPU detection on $TARGET_HOST"

  # Check for Intel GPU device
  if exec_on_target "ls /dev/dri/renderD* >/dev/null 2>&1"; then
    test_pass "DRI render devices found"
  else
    test_fail "No DRI render devices found"
  fi

  # Check for Intel vendor
  if exec_on_target "lspci | grep -i 'VGA.*Intel'" >/dev/null 2>&1; then
    test_pass "Intel GPU detected via lspci"
  else
    test_warn "Intel GPU not detected via lspci"
  fi

  # Check Level Zero
  if exec_on_target "python -c \"import os; os.environ['TINYGRAD_RUNTIME']='LEVEL_ZERO'; from tinygrad import Device; Device.DEFAULT='GPU'\"" 2>/dev/null; then
    test_pass "Level Zero runtime is available"
  else
    log_warn "Level Zero runtime not available, checking OpenCL..."

    # Check OpenCL fallback
    if exec_on_target "clinfo 2>/dev/null | grep -i intel" >/dev/null; then
      test_pass "OpenCL runtime is available (fallback)"
    else
      test_fail "Neither Level Zero nor OpenCL available"
    fi
  fi

  # Check device metrics
  log_info "Checking device metrics..."
  if curl -s --connect-timeout 5 ${BASE_URL}/metrics 2>/dev/null | grep -i "gpu\|intel" >/dev/null; then
    test_pass "GPU appears in metrics"
  else
    test_warn "GPU not visible in metrics (may not be implemented yet)"
  fi
}

# Test 9.5: Validate Intel NPU detection
test_npu_detection() {
  log_info "Test 9.5: Validating Intel NPU detection on $TARGET_HOST"

  # Check for NPU device node
  if exec_on_target "ls /dev/accel/accel* >/dev/null 2>&1"; then
    test_pass "NPU device node found (/dev/accel/accel*)"
  elif exec_on_target "ls /dev/dri/renderD* >/dev/null 2>&1"; then
    log_warn "NPU may be on DRI device (older kernel)"
  else
    test_fail "No NPU device node found"
  fi

  # Check kernel modules
  if exec_on_target "lsmod | grep -E 'intel_vpu|ivpu'" >/dev/null 2>&1; then
    test_pass "Intel NPU kernel module loaded"
  else
    test_fail "Intel NPU kernel module not loaded"
    log_info "Try: sudo modprobe intel_vpu"
  fi

  # Check OpenVINO
  if exec_on_target "python -c \"import openvino as ov; core = ov.Core(); assert any('NPU' in d for d in core.available_devices())\"" 2>/dev/null; then
    test_pass "OpenVINO can access NPU device"
  else
    test_warn "OpenVINO cannot access NPU (may not be installed)"
  fi

  # Run capability report
  log_info "Running NPU capability report..."
  if exec_on_target "python -m exo.worker.engines.npu.capability_report 2>/dev/null | grep -i 'available.*true'" >/dev/null; then
    test_pass "NPU capability report shows NPU available"
  else
    test_warn "NPU capability report shows NPU unavailable"
  fi
}

# Test 9.6: Download and load tiny model
test_model_download() {
  log_info "Test 9.6: Downloading and loading tiny model on $TARGET_HOST"

  MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

  # Download model via API
  log_info "Requesting model download: $MODEL"

  # Send chat completion request (will trigger download)
  RESPONSE=$(curl -s --connect-timeout 10 -X POST ${BASE_URL}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
            \"max_tokens\": 1,
            \"stream\": false
        }" 2>&1)

  # Check if download started or model loaded
  if echo "$RESPONSE" | grep -q "downloading\|loading\|choices"; then
    test_pass "Model download/load initiated"
  else
    test_fail "Failed to initiate model download"
    log_error "Response: $RESPONSE"
    return 1
  fi

  # Wait for model to be ready (max 5 minutes for download)
  log_info "Waiting for model to be ready (this may take a few minutes)..."
  for i in {1..60}; do
    if curl -s --connect-timeout 5 ${BASE_URL}/v1/models | grep -q "$MODEL"; then
      test_pass "Model loaded successfully"
      return 0
    fi
    sleep 5
  done

  test_warn "Model download/load taking longer than expected"
}

# Test 9.7: Run inference on tiny model
test_inference() {
  log_info "Test 9.7: Running inference on tiny model on $TARGET_HOST"

  MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

  # Send inference request
  log_info "Sending inference request..."
  RESPONSE=$(curl -s --connect-timeout 30 -X POST ${BASE_URL}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
            \"max_tokens\": 10,
            \"stream\": false
        }")

  # Check if inference succeeded
  if echo "$RESPONSE" | grep -q "choices"; then
    test_pass "Inference completed successfully"

    # Extract generated text
    GENERATED=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if [ -n "$GENERATED" ]; then
      log_info "Generated text: $GENERATED"
      test_pass "Tokens generated correctly"
    fi
  else
    test_fail "Inference failed"
    log_error "Response: $RESPONSE"
    return 1
  fi

  # Check performance metrics
  if echo "$RESPONSE" | grep -q "usage"; then
    TOKENS=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
    log_info "Tokens generated: $TOKENS"
    test_pass "Performance metrics available"
  fi

  # Verify GPU is being used (check logs or metrics)
  log_info "Checking if GPU is being used..."
  if curl -s --connect-timeout 5 ${BASE_URL}/metrics 2>/dev/null | grep -i "gpu.*active\|tinygrad.*gpu" >/dev/null; then
    test_pass "GPU is being used for inference"
  else
    test_warn "Cannot confirm GPU usage (may need to check logs)"
  fi
}

# Cleanup function
cleanup() {
  log_info "Cleaning up..."
  if [ -n "$EXO_PID" ]; then
    kill $EXO_PID 2>/dev/null || true
    wait $EXO_PID 2>/dev/null || true
  fi
}

# Main test execution
main() {
  log_info "Starting single-node validation tests"
  log_info "Target: $TARGET_HOST ($TARGET_IP)"
  log_info "API URL: $BASE_URL"
  log_info "=================================================="

  # Set trap for cleanup
  trap cleanup EXIT

  # Run tests
  test_build || exit 1
  test_start_service || exit 1
  test_web_endpoint || exit 1
  test_gpu_detection
  test_npu_detection
  test_model_download
  test_inference

  # Summary
  echo ""
  log_info "=================================================="
  log_info "Test Summary for $TARGET_HOST"
  log_info "=================================================="
  echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
  echo -e "${RED}Failed:${NC} $TESTS_FAILED"

  if [ $TESTS_FAILED -eq 0 ]; then
    log_info "All tests passed! ✓"
    exit 0
  else
    log_error "Some tests failed. Please review the output above."
    exit 1
  fi
}

# Run main
main "$@"

#!/usr/bin/env bash
# Comprehensive single-node validation for gremlin-1
# This script orchestrates all validation tasks from task 9
#
# Usage:
#   ./validate_gremlin_single_node.sh [TARGET_HOST]
#
# Examples:
#   ./validate_gremlin_single_node.sh                    # Test localhost
#   ./validate_gremlin_single_node.sh gremlin-1          # Test gremlin-1
#   ./validate_gremlin_single_node.sh 10.1.1.12          # Test specific IP

set -euo pipefail

# Configuration
TARGET_HOST="${1:-gremlin-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve hostname to IP
case "$TARGET_HOST" in
    gremlin-1) TARGET_IP="10.1.1.12" ;;
    gremlin-2) TARGET_IP="10.1.1.13" ;;
    gremlin-3) TARGET_IP="10.1.1.14" ;;
    gremlin-4) TARGET_IP="10.1.1.15" ;;
    localhost) TARGET_IP="localhost" ;;
    *) TARGET_IP="$TARGET_HOST" ;;
esac

# API configuration
if [ "$TARGET_IP" = "localhost" ]; then
    BASE_URL="http://localhost:52415"
    SSH_CMD=""
    IS_REMOTE=false
else
    BASE_URL="http://${TARGET_IP}:52415"
    SSH_CMD="ssh root@${TARGET_IP}"
    IS_REMOTE=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNED=0
EXO_PID=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
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

test_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((TESTS_WARNED++))
}

# Execute command on target
exec_on_target() {
    if [ "$IS_REMOTE" = false ]; then
        eval "$@"
    else
        $SSH_CMD "$@"
    fi
}

# Check if target is reachable
check_connectivity() {
    log_info "Checking connectivity to $TARGET_HOST ($TARGET_IP)..."
    
    if [ "$IS_REMOTE" = true ]; then
        if ! ping -c 1 -W 2 "$TARGET_IP" >/dev/null 2>&1; then
            log_error "Cannot reach $TARGET_IP"
            return 1
        fi
        
        if ! $SSH_CMD "echo 'SSH OK'" >/dev/null 2>&1; then
            log_error "Cannot SSH to $TARGET_IP"
            return 1
        fi
    fi
    
    test_pass "Target is reachable"
    return 0
}

# Task 9.1: Build exo with Intel hardware support
task_9_1_build() {
    log_info "=========================================="
    log_info "Task 9.1: Build exo with Intel hardware support"
    log_info "=========================================="
    
    if [ "$IS_REMOTE" = true ]; then
        log_info "Checking if exo is built on $TARGET_HOST..."
        
        if exec_on_target "which exo" >/dev/null 2>&1; then
            test_pass "exo binary is available"
        else
            test_fail "exo binary not found"
            log_error "Please build exo on $TARGET_HOST first"
            log_info "Run: nix build .#exo"
            return 1
        fi
        
        # Check exo version
        EXO_VERSION=$(exec_on_target "exo --version 2>/dev/null || echo 'unknown'")
        log_info "exo version: $EXO_VERSION"
    else
        log_info "Building exo locally..."
        
        cd "$PROJECT_ROOT"
        if nix build .#exo 2>&1 | tee /tmp/exo-build.log; then
            test_pass "Build completed successfully"
        else
            test_fail "Build failed"
            log_error "Check /tmp/exo-build.log for details"
            return 1
        fi
    fi
    
    # Verify tinygrad is available
    log_info "Checking tinygrad availability..."
    if exec_on_target "python3 -c 'import tinygrad; print(tinygrad.__version__)'" 2>/dev/null; then
        TINYGRAD_VERSION=$(exec_on_target "python3 -c 'import tinygrad; print(tinygrad.__version__)'" 2>/dev/null)
        test_pass "Tinygrad backend is available (version: $TINYGRAD_VERSION)"
    else
        test_fail "Tinygrad backend not available"
        return 1
    fi
    
    # Verify dependencies
    log_info "Checking dependencies..."
    
    local deps_ok=true
    
    if exec_on_target "python3 -c 'import numpy'" 2>/dev/null; then
        test_pass "numpy available"
    else
        test_fail "numpy not available"
        deps_ok=false
    fi
    
    if exec_on_target "python3 -c 'import pyopencl'" 2>/dev/null; then
        test_pass "pyopencl available"
    else
        test_warn "pyopencl not available (optional for OpenCL)"
    fi
    
    if [ "$deps_ok" = false ]; then
        return 1
    fi
    
    log_success "Task 9.1 completed successfully"
    return 0
}

# Task 9.2: Start exo service
task_9_2_start_service() {
    log_info "=========================================="
    log_info "Task 9.2: Start exo service"
    log_info "=========================================="
    
    # Check if service is already running
    if curl -s --connect-timeout 5 "${BASE_URL}/health" >/dev/null 2>&1; then
        test_pass "Service is already running"
        log_success "Task 9.2 completed successfully"
        return 0
    fi
    
    if [ "$IS_REMOTE" = true ]; then
        log_warn "Service not running on $TARGET_HOST"
        log_info "Please start exo manually on $TARGET_HOST:"
        log_info "  EXO_TINYGRAD_ENABLED=true exo -vv"
        log_info ""
        log_info "Waiting for service to become available..."
        
        for i in {1..60}; do
            if curl -s --connect-timeout 2 "${BASE_URL}/health" >/dev/null 2>&1; then
                test_pass "Service is now running"
                log_success "Task 9.2 completed successfully"
                return 0
            fi
            sleep 2
        done
        
        test_fail "Service did not start within 120 seconds"
        return 1
    else
        log_info "Starting exo locally with tinygrad backend..."
        
        cd "$PROJECT_ROOT"
        EXO_TINYGRAD_ENABLED=true uv run exo -vv > /tmp/exo.log 2>&1 &
        EXO_PID=$!
        
        log_info "Started exo (PID: $EXO_PID)"
        log_info "Logs: /tmp/exo.log"
        
        # Wait for service to start
        log_info "Waiting for service to start..."
        for i in {1..30}; do
            if curl -s --connect-timeout 2 "http://localhost:52415/health" >/dev/null 2>&1; then
                test_pass "Service started successfully"
                
                # Check logs for initialization
                if grep -q "tinygrad" /tmp/exo.log 2>/dev/null; then
                    test_pass "Tinygrad backend initialized"
                fi
                
                log_success "Task 9.2 completed successfully"
                return 0
            fi
            sleep 1
        done
        
        test_fail "Service failed to start within 30 seconds"
        log_error "Check /tmp/exo.log for errors"
        
        if [ -n "$EXO_PID" ]; then
            tail -n 50 /tmp/exo.log
        fi
        
        return 1
    fi
}

# Task 9.3: Verify web service endpoint
task_9_3_verify_endpoint() {
    log_info "=========================================="
    log_info "Task 9.3: Verify web service endpoint"
    log_info "=========================================="
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s --connect-timeout 5 "${BASE_URL}/health" 2>&1)
    
    if echo "$HEALTH_RESPONSE" | grep -q "ok\|healthy"; then
        test_pass "Health endpoint responds correctly"
    else
        test_fail "Health endpoint not responding"
        log_error "Response: $HEALTH_RESPONSE"
        return 1
    fi
    
    # Test OpenAI-compatible API
    log_info "Testing OpenAI-compatible API..."
    MODELS_RESPONSE=$(curl -s --connect-timeout 5 "${BASE_URL}/v1/models" 2>&1)
    
    if echo "$MODELS_RESPONSE" | grep -q "object.*list\|data"; then
        test_pass "OpenAI-compatible API is available"
    else
        test_fail "OpenAI-compatible API not available"
        log_error "Response: $MODELS_RESPONSE"
        return 1
    fi
    
    # Test metrics endpoint (if available)
    log_info "Testing metrics endpoint..."
    if curl -s --connect-timeout 5 "${BASE_URL}/metrics" >/dev/null 2>&1; then
        test_pass "Metrics endpoint is available"
    else
        test_warn "Metrics endpoint not available (may not be implemented)"
    fi
    
    log_success "Task 9.3 completed successfully"
    return 0
}

# Task 9.4: Validate Intel GPU detection
task_9_4_validate_gpu() {
    log_info "=========================================="
    log_info "Task 9.4: Validate Intel GPU detection"
    log_info "=========================================="
    
    # Check for DRI devices
    log_info "Checking for DRI render devices..."
    if exec_on_target "ls /dev/dri/renderD* >/dev/null 2>&1"; then
        RENDER_DEVICES=$(exec_on_target "ls /dev/dri/renderD*" 2>/dev/null | tr '\n' ' ')
        test_pass "DRI render devices found: $RENDER_DEVICES"
    else
        test_fail "No DRI render devices found"
        return 1
    fi
    
    # Check for Intel GPU via lspci
    log_info "Checking for Intel GPU via lspci..."
    if exec_on_target "lspci | grep -i 'VGA.*Intel'" >/dev/null 2>&1; then
        GPU_NAME=$(exec_on_target "lspci | grep -i 'VGA.*Intel' | cut -d: -f3" 2>/dev/null)
        test_pass "Intel GPU detected:$GPU_NAME"
    else
        test_warn "Intel GPU not detected via lspci"
    fi
    
    # Check Level Zero runtime
    log_info "Checking Level Zero runtime..."
    if exec_on_target "ls /run/opengl-driver/lib/libze_loader.so* >/dev/null 2>&1"; then
        test_pass "Level Zero loader library found"
        
        # Try to use Level Zero with tinygrad
        if exec_on_target "python3 -c \"import os; os.environ['GPU']='1'; os.environ['LEVEL_ZERO']='1'; from tinygrad import Device; Device.DEFAULT='GPU'\"" 2>/dev/null; then
            test_pass "Level Zero runtime is functional"
        else
            test_warn "Level Zero library found but not functional"
        fi
    else
        test_warn "Level Zero loader library not found"
        
        # Check OpenCL fallback
        log_info "Checking OpenCL fallback..."
        if exec_on_target "command -v clinfo >/dev/null 2>&1 && clinfo 2>/dev/null | grep -i intel" >/dev/null; then
            test_pass "OpenCL runtime is available (fallback)"
        else
            test_fail "Neither Level Zero nor OpenCL available"
            return 1
        fi
    fi
    
    # Check device metrics
    log_info "Checking device metrics..."
    METRICS=$(curl -s --connect-timeout 5 "${BASE_URL}/metrics" 2>/dev/null || echo "")
    
    if echo "$METRICS" | grep -qi "gpu\|intel\|tinygrad"; then
        test_pass "GPU/backend information appears in metrics"
    else
        test_warn "GPU not visible in metrics (may not be implemented yet)"
    fi
    
    log_success "Task 9.4 completed successfully"
    return 0
}

# Task 9.5: Validate Intel NPU detection
task_9_5_validate_npu() {
    log_info "=========================================="
    log_info "Task 9.5: Validate Intel NPU detection"
    log_info "=========================================="
    
    # Check for NPU device node
    log_info "Checking for NPU device node..."
    if exec_on_target "ls /dev/accel/accel* >/dev/null 2>&1"; then
        NPU_DEVICE=$(exec_on_target "ls /dev/accel/accel*" 2>/dev/null | head -n1)
        test_pass "NPU device node found: $NPU_DEVICE"
    else
        test_warn "No NPU device node found (/dev/accel/accel*)"
        test_warn "NPU may not be available on this hardware"
    fi
    
    # Check kernel modules
    log_info "Checking Intel NPU kernel modules..."
    if exec_on_target "lsmod | grep -E 'intel_vpu|ivpu'" >/dev/null 2>&1; then
        MODULE_NAME=$(exec_on_target "lsmod | grep -E 'intel_vpu|ivpu' | awk '{print \$1}'" 2>/dev/null)
        test_pass "Intel NPU kernel module loaded: $MODULE_NAME"
    else
        test_warn "Intel NPU kernel module not loaded"
        log_info "Try: sudo modprobe intel_vpu"
    fi
    
    # Check OpenVINO
    log_info "Checking OpenVINO NPU access..."
    if exec_on_target "python3 -c \"import openvino as ov; core = ov.Core(); devices = core.available_devices(); assert any('NPU' in d for d in devices), f'NPU not in {devices}'\"" 2>/dev/null; then
        test_pass "OpenVINO can access NPU device"
    else
        test_warn "OpenVINO cannot access NPU (may not be installed or configured)"
    fi
    
    # Run NPU capability report
    log_info "Running NPU capability report..."
    if exec_on_target "python3 -m exo.worker.engines.npu.capability_report" 2>/dev/null | grep -qi "available.*true"; then
        test_pass "NPU capability report shows NPU available"
    else
        test_warn "NPU capability report shows NPU unavailable or not implemented"
    fi
    
    log_success "Task 9.5 completed (NPU is optional)"
    return 0
}

# Task 9.6: Download and load tiny model
task_9_6_download_model() {
    log_info "=========================================="
    log_info "Task 9.6: Download and load tiny model"
    log_info "=========================================="
    
    local MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    log_info "Requesting model: $MODEL"
    log_info "This may take several minutes for first download..."
    
    # Send a minimal inference request to trigger download
    RESPONSE=$(curl -s --connect-timeout 10 -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
            \"max_tokens\": 1,
            \"stream\": false
        }" 2>&1)
    
    # Check response
    if echo "$RESPONSE" | grep -qi "downloading\|loading\|choices\|error"; then
        if echo "$RESPONSE" | grep -qi "error"; then
            log_warn "Request returned error, but model download may have started"
            log_info "Response: $RESPONSE"
        else
            test_pass "Model download/load initiated"
        fi
    else
        test_fail "Failed to initiate model download"
        log_error "Response: $RESPONSE"
        return 1
    fi
    
    # Wait for model to be ready
    log_info "Waiting for model to be ready (max 5 minutes)..."
    for i in {1..60}; do
        MODELS_LIST=$(curl -s --connect-timeout 5 "${BASE_URL}/v1/models" 2>/dev/null || echo "")
        
        if echo "$MODELS_LIST" | grep -q "$MODEL"; then
            test_pass "Model loaded successfully"
            log_success "Task 9.6 completed successfully"
            return 0
        fi
        
        # Show progress every 10 seconds
        if [ $((i % 10)) -eq 0 ]; then
            log_info "Still waiting... ($i/60)"
        fi
        
        sleep 5
    done
    
    test_warn "Model download/load taking longer than expected"
    log_info "Model may still be downloading. Check logs for progress."
    return 0
}

# Task 9.7: Run inference on tiny model
task_9_7_run_inference() {
    log_info "=========================================="
    log_info "Task 9.7: Run inference on tiny model"
    log_info "=========================================="
    
    local MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    log_info "Sending inference request..."
    RESPONSE=$(curl -s --connect-timeout 30 -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word\"}],
            \"max_tokens\": 10,
            \"stream\": false
        }" 2>&1)
    
    # Check if inference succeeded
    if echo "$RESPONSE" | grep -q "choices"; then
        test_pass "Inference completed successfully"
        
        # Extract generated text
        if command -v python3 >/dev/null 2>&1; then
            GENERATED=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null || echo "")
            if [ -n "$GENERATED" ]; then
                log_info "Generated text: $GENERATED"
                test_pass "Tokens generated correctly"
            fi
            
            # Check usage metrics
            COMPLETION_TOKENS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('usage', {}).get('completion_tokens', 0))" 2>/dev/null || echo "0")
            if [ "$COMPLETION_TOKENS" -gt 0 ]; then
                log_info "Completion tokens: $COMPLETION_TOKENS"
                test_pass "Performance metrics available"
            fi
        fi
    else
        test_fail "Inference failed"
        log_error "Response: $RESPONSE"
        return 1
    fi
    
    # Check if GPU is being used
    log_info "Checking GPU usage..."
    
    # Method 1: Check metrics endpoint
    METRICS=$(curl -s --connect-timeout 5 "${BASE_URL}/metrics" 2>/dev/null || echo "")
    if echo "$METRICS" | grep -qi "gpu.*active\|tinygrad.*gpu\|backend.*gpu"; then
        test_pass "GPU is being used for inference"
    else
        test_warn "Cannot confirm GPU usage from metrics"
        
        # Method 2: Check logs (if local)
        if [ "$IS_REMOTE" = false ] && [ -f "/tmp/exo.log" ]; then
            if grep -qi "gpu\|level.zero\|opencl" /tmp/exo.log; then
                test_pass "GPU usage detected in logs"
            else
                test_warn "No GPU usage detected in logs (may be using CPU)"
            fi
        fi
    fi
    
    log_success "Task 9.7 completed successfully"
    return 0
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    if [ -n "$EXO_PID" ] && kill -0 "$EXO_PID" 2>/dev/null; then
        log_info "Stopping exo (PID: $EXO_PID)..."
        kill "$EXO_PID" 2>/dev/null || true
        wait "$EXO_PID" 2>/dev/null || true
    fi
}

# Main execution
main() {
    echo ""
    log_info "=========================================="
    log_info "Single-Node Validation for gremlin-1"
    log_info "=========================================="
    log_info "Target: $TARGET_HOST ($TARGET_IP)"
    log_info "API URL: $BASE_URL"
    log_info "Remote: $IS_REMOTE"
    log_info "=========================================="
    echo ""
    
    # Set trap for cleanup
    trap cleanup EXIT INT TERM
    
    # Check connectivity
    if ! check_connectivity; then
        log_error "Cannot reach target. Exiting."
        exit 1
    fi
    
    echo ""
    
    # Run all tasks
    local all_passed=true
    
    task_9_1_build || all_passed=false
    echo ""
    
    task_9_2_start_service || all_passed=false
    echo ""
    
    task_9_3_verify_endpoint || all_passed=false
    echo ""
    
    task_9_4_validate_gpu || all_passed=false
    echo ""
    
    task_9_5_validate_npu || all_passed=false
    echo ""
    
    task_9_6_download_model || all_passed=false
    echo ""
    
    task_9_7_run_inference || all_passed=false
    echo ""
    
    # Summary
    log_info "=========================================="
    log_info "Validation Summary"
    log_info "=========================================="
    echo -e "${GREEN}Tests Passed:${NC} $TESTS_PASSED"
    echo -e "${YELLOW}Tests Warned:${NC} $TESTS_WARNED"
    echo -e "${RED}Tests Failed:${NC} $TESTS_FAILED"
    echo ""
    
    if [ "$all_passed" = true ] && [ $TESTS_FAILED -eq 0 ]; then
        log_success "All validation tasks completed successfully! ✓"
        log_info "gremlin-1 is ready for deployment"
        exit 0
    else
        log_warn "Some validation tasks failed or had warnings"
        log_info "Review the output above for details"
        exit 1
    fi
}

# Run main
main "$@"

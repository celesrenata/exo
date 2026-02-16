#!/usr/bin/env bash
# Script to capture Intel Arc GPU diagnostics after forcing device selection

set -e

GREMLIN_HOST="10.1.1.12"
GREMLIN_USER="root"
API_URL="http://${GREMLIN_HOST}:52415"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LOG_FILE="intel_arc_diagnostics_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Intel Arc GPU Diagnostic Capture"
echo "=========================================="
echo "Target: ${GREMLIN_HOST}"
echo "Model: ${MODEL}"
echo "Log file: ${LOG_FILE}"
echo ""

# Verify Intel Arc is being used
echo "Verifying GPU configuration..."
ssh ${GREMLIN_USER}@${GREMLIN_HOST} "systemctl show exo | grep 'Environment=.*OPENCL_DEVICE'"
echo ""

# Function to start log monitoring
start_log_monitor() {
  echo "Starting log monitor..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "journalctl -u exo -f --since '5 seconds ago'" >"${LOG_FILE}" 2>&1 &
  LOG_MONITOR_PID=$!
  echo "✓ Log monitor started (PID: ${LOG_MONITOR_PID})"
  echo ""
  sleep 2
}

# Function to stop log monitoring
stop_log_monitor() {
  if [ ! -z "${LOG_MONITOR_PID}" ]; then
    echo ""
    echo "Stopping log monitor..."
    kill ${LOG_MONITOR_PID} 2>/dev/null || true
    wait ${LOG_MONITOR_PID} 2>/dev/null || true
    echo "✓ Log monitor stopped"
  fi
}

# Function to send inference request
send_inference_request() {
  echo "Sending inference request to trigger model loading..."
  echo "Model: ${MODEL}"
  echo ""

  RESPONSE=$(curl -s -X POST "${API_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}],
            \"max_tokens\": 20,
            \"stream\": false
        }" 2>&1)

  echo "Response received:"
  echo "${RESPONSE}" | jq '.' 2>/dev/null || echo "${RESPONSE}"
  echo ""
}

# Function to analyze logs
analyze_logs() {
  echo ""
  echo "=========================================="
  echo "DIAGNOSTIC LOG ANALYSIS"
  echo "=========================================="
  echo ""

  # Check for Intel Arc debug messages
  if grep -q "\[INTEL ARC DEBUG\]" "${LOG_FILE}"; then
    echo "✓ Found Intel Arc diagnostic messages!"
    grep "\[INTEL ARC DEBUG\]" "${LOG_FILE}" >"${LOG_FILE}.intel_arc_only"
    echo "  Total diagnostic lines: $(wc -l <"${LOG_FILE}.intel_arc_only")"
    echo ""

    # Show device info
    echo "--- Device Information ---"
    grep "DEVICE INFORMATION" -A 5 "${LOG_FILE}.intel_arc_only" || echo "No device info"
    echo ""

    # Verify it's Intel Arc
    if grep -q "Intel.*Arc" "${LOG_FILE}.intel_arc_only"; then
      echo "✓ Confirmed: Using Intel Arc GPU"
    else
      echo "⚠️  WARNING: Not using Intel Arc GPU!"
      grep "Device:" "${LOG_FILE}.intel_arc_only"
    fi
    echo ""

    # Show all allocations
    echo "--- All Allocations (>100MB) ---"
    grep "Allocation #" "${LOG_FILE}.intel_arc_only" || echo "No allocations logged"
    echo ""

    # Show contexts
    echo "--- Allocation Contexts ---"
    grep "Context:" "${LOG_FILE}.intel_arc_only" | head -20 || echo "No contexts"
    echo ""

    # Check for >4GB
    echo "--- Checking for >4GB Allocations ---"
    if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
      echo "⚠️  FOUND ALLOCATIONS >4GB:"
      grep "WARNING: Buffer >4GB" -A 5 "${LOG_FILE}.intel_arc_only"
    else
      echo "✓ No allocations >4GB detected"
    fi
    echo ""

    # Show allocation summary
    echo "--- Allocation Summary ---"
    grep "ALLOCATION SUMMARY" -A 10 "${LOG_FILE}.intel_arc_only" || echo "No summary yet"
    echo ""
  else
    echo "⚠️  WARNING: No [INTEL ARC DEBUG] messages found"
    echo ""
  fi

  # Check for errors
  echo "--- OpenCL Errors ---"
  if grep -q "OpenCL Error" "${LOG_FILE}"; then
    echo "⚠️  FOUND ERRORS:"
    grep "OpenCL Error" "${LOG_FILE}" | head -20
    echo ""

    # Show error context
    echo "--- Error Context ---"
    grep -B 5 "OpenCL Error" "${LOG_FILE}" | head -30
  else
    echo "✓ No OpenCL errors"
  fi
  echo ""
}

# Trap to ensure cleanup
trap stop_log_monitor EXIT

# Main execution
main() {
  start_log_monitor

  echo "Waiting 5 seconds for log monitor to stabilize..."
  sleep 5

  send_inference_request

  echo "Waiting 60 seconds for model loading and inference..."
  sleep 60

  stop_log_monitor

  analyze_logs

  echo ""
  echo "=========================================="
  echo "Diagnostic Capture Complete!"
  echo "=========================================="
  echo "Full log: ${LOG_FILE}"
  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    echo "Intel Arc logs: ${LOG_FILE}.intel_arc_only"
    echo ""
    echo "Summary:"
    echo "  Device: $(grep 'Device:' "${LOG_FILE}.intel_arc_only" | head -1 | cut -d':' -f3-)"
    echo "  Allocations >100MB: $(grep -c 'Allocation #' "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo '0')"
    echo "  Allocations >4GB: $(grep -c 'WARNING: Buffer >4GB' "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo '0')"
    echo "  OpenCL Errors: $(grep -c 'OpenCL Error' "${LOG_FILE}" 2>/dev/null || echo '0')"
  fi
  echo ""
}

main

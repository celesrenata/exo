#!/usr/bin/env bash
# Script to capture diagnostics from a fresh model load
# Restarts service and monitors from the beginning

set -e

GREMLIN_HOST="10.1.1.12"
GREMLIN_USER="root"
API_URL="http://${GREMLIN_HOST}:52415"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LOG_FILE="intel_arc_fresh_load_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Intel Arc GPU Fresh Model Load Diagnostics"
echo "=========================================="
echo "Target: ${GREMLIN_HOST}"
echo "Model: ${MODEL}"
echo "Log file: ${LOG_FILE}"
echo ""

# Function to restart service
restart_service() {
  echo "Restarting exo service to force fresh model load..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "systemctl restart exo"
  echo "✓ Service restarted"
  echo "Waiting 5 seconds for service to stabilize..."
  sleep 5
  echo ""
}

# Function to start log monitoring
start_log_monitor() {
  echo "Starting log monitor from service start..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "journalctl -u exo -f --since '10 seconds ago'" >"${LOG_FILE}" 2>&1 &
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

  # Send a simple chat completion request
  RESPONSE=$(curl -s -X POST "${API_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
            \"max_tokens\": 10,
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
    echo "✓ Found Intel Arc diagnostic messages"
    grep "\[INTEL ARC DEBUG\]" "${LOG_FILE}" >"${LOG_FILE}.intel_arc_only"
    echo "  Total diagnostic lines: $(wc -l <"${LOG_FILE}.intel_arc_only")"
    echo ""

    # Show device info
    echo "--- Device Information ---"
    grep "DEVICE INFORMATION" -A 5 "${LOG_FILE}.intel_arc_only" || echo "No device info"
    echo ""

    # Show allocations
    echo "--- Large Allocations (>100MB) ---"
    grep "Allocation #" "${LOG_FILE}.intel_arc_only" || echo "No large allocations"
    echo ""

    # Check for >4GB
    echo "--- Checking for >4GB Allocations ---"
    if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
      echo "⚠️  FOUND:"
      grep "WARNING: Buffer >4GB" -A 5 "${LOG_FILE}.intel_arc_only"
    else
      echo "✓ No allocations >4GB"
    fi
    echo ""
  else
    echo "⚠️  WARNING: No [INTEL ARC DEBUG] messages found"
    echo ""
  fi

  # Check for errors
  echo "--- OpenCL Errors ---"
  if grep -q "OpenCL Error" "${LOG_FILE}"; then
    echo "⚠️  FOUND ERRORS:"
    grep "OpenCL Error" "${LOG_FILE}" | head -10
  else
    echo "✓ No OpenCL errors"
  fi
  echo ""

  # Show model loading activity
  echo "--- Model Loading Activity ---"
  grep -i "loading\|download\|weight\|model" "${LOG_FILE}" | grep -v "DEBUG.*NodeGathered" | head -20
  echo ""
}

# Trap to ensure cleanup
trap stop_log_monitor EXIT

# Main execution
main() {
  restart_service
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
  echo "Diagnostic capture complete!"
  echo "=========================================="
  echo "Full log: ${LOG_FILE}"
  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    echo "Intel Arc logs: ${LOG_FILE}.intel_arc_only"
  fi
  echo ""
}

main

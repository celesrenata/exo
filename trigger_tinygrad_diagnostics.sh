#!/usr/bin/env bash
# Script to trigger tinygrad model loading and capture diagnostics

set -e

GREMLIN_HOST="10.1.1.12"
GREMLIN_USER="root"
API_URL="http://${GREMLIN_HOST}:52415"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
INSTANCE_ID="28edb645-7a7d-468f-8252-718c90741b69" # TinygradRing instance
LOG_FILE="intel_arc_tinygrad_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Intel Arc GPU Tinygrad Diagnostic Capture"
echo "=========================================="
echo "Target: ${GREMLIN_HOST}"
echo "Model: ${MODEL}"
echo "Instance ID: ${INSTANCE_ID}"
echo "Log file: ${LOG_FILE}"
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

# Function to create instance
create_instance() {
  echo "Creating Tinygrad instance..."
  RESPONSE=$(curl -s -X POST "${API_URL}/instance/create" \
    -H "Content-Type: application/json" \
    -d "{
            \"model_id\": \"${MODEL}\",
            \"instance_id\": \"${INSTANCE_ID}\"
        }" 2>&1)

  echo "Create response:"
  echo "${RESPONSE}" | jq '.' 2>/dev/null || echo "${RESPONSE}"
  echo ""
}

# Function to send inference request
send_inference_request() {
  echo "Sending inference request..."
  RESPONSE=$(curl -s -X POST "${API_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}],
            \"max_tokens\": 20,
            \"stream\": false
        }" 2>&1)

  echo "Inference response:"
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

    # Show all diagnostic output
    echo "--- All Intel Arc Debug Messages ---"
    cat "${LOG_FILE}.intel_arc_only"
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
  else
    echo "✓ No OpenCL errors"
  fi
  echo ""
}

# Function to create summary report
create_report() {
  REPORT_FILE="intel_arc_diagnostic_report_$(date +%Y%m%d_%H%M%S).md"

  echo "Creating diagnostic report: ${REPORT_FILE}"

  cat >"${REPORT_FILE}" <<EOF
# Intel Arc GPU Diagnostic Report - Task 3

## Test Information

- **Date**: $(date)
- **Target**: ${GREMLIN_HOST}
- **Model**: ${MODEL}
- **Instance**: TinygradRing (${INSTANCE_ID})
- **Log File**: ${LOG_FILE}

## Diagnostic Logs

### Intel Arc Debug Messages

\`\`\`
EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    cat "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}"
  else
    echo "No [INTEL ARC DEBUG] messages captured" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

### OpenCL Errors

\`\`\`
EOF

  if grep -q "OpenCL Error" "${LOG_FILE}"; then
    grep "OpenCL Error" "${LOG_FILE}" >>"${REPORT_FILE}"
  else
    echo "No OpenCL errors detected" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

## Analysis

### Requirements Coverage

- **Requirement 1.1**: Log allocations >100MB - $(grep -c "Allocation #" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") allocations logged
- **Requirement 1.2**: Log buffer sizes - $(grep -c "bytes" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") size logs
- **Requirement 1.3**: Warn on >4GB - $(grep -c "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") warnings
- **Requirement 2.1**: Capture call stack - $(grep -c "Context:" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") context captures
- **Requirement 2.2**: Include tensor shape - $(grep -c "shape=" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") shape logs
- **Requirement 2.3**: Include data type - $(grep -c "dtype=" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") dtype logs

### Findings

EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
      echo "- ⚠️  **CRITICAL**: Allocations >4GB detected" >>"${REPORT_FILE}"
    else
      echo "- ✓ No allocations >4GB detected" >>"${REPORT_FILE}"
    fi

    LARGE_COUNT=$(grep -c "Allocation #" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
    echo "- Found ${LARGE_COUNT} allocations >100MB" >>"${REPORT_FILE}"
  else
    echo "- ⚠️  No diagnostic data captured - patch may not be active" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF

### Next Steps

- [ ] Review allocation patterns
- [ ] Identify problematic allocations
- [ ] Proceed to Task 4: Analyze diagnostic results

## Raw Logs

Full logs available in: \`${LOG_FILE}\`
EOF

  echo "✓ Report created: ${REPORT_FILE}"
  echo ""
}

# Trap to ensure cleanup
trap stop_log_monitor EXIT

# Main execution
main() {
  start_log_monitor

  echo "Waiting 3 seconds for log monitor to stabilize..."
  sleep 3

  create_instance

  echo "Waiting 10 seconds for instance creation..."
  sleep 10

  send_inference_request

  echo "Waiting 60 seconds for model loading and inference..."
  sleep 60

  stop_log_monitor

  analyze_logs
  create_report

  echo ""
  echo "=========================================="
  echo "Diagnostic capture complete!"
  echo "=========================================="
  echo "Full log: ${LOG_FILE}"
  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    echo "Intel Arc logs: ${LOG_FILE}.intel_arc_only"
  fi
  echo "Report: intel_arc_diagnostic_report_*.md"
  echo ""
}

main

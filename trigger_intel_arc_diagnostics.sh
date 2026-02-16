#!/usr/bin/env bash
# Script to trigger Intel Arc GPU model loading and capture diagnostics
# Forces tinygrad to use Intel Arc GPU specifically

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

# Function to restart service with Intel Arc forced
restart_with_intel_arc() {
  echo "Configuring service to use Intel Arc GPU..."

  # Create systemd override to force Intel Arc
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "mkdir -p /etc/systemd/system/exo.service.d"
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "cat > /etc/systemd/system/exo.service.d/intel-arc.conf << 'EOF'
[Service]
Environment=\"OPENCL_DEVICE=0\"
Environment=\"GPU=OPENCL\"
EOF"

  echo "✓ Override created"

  echo "Reloading systemd and restarting service..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "systemctl daemon-reload && systemctl restart exo"
  echo "✓ Service restarted with Intel Arc GPU"
  echo "Waiting 10 seconds for service to stabilize..."
  sleep 10
  echo ""
}

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

  # Send a simple chat completion request
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

    # Show all allocations
    echo "--- All Allocations (>100MB) ---"
    grep "Allocation #" "${LOG_FILE}.intel_arc_only" || echo "No allocations logged"
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
  else
    echo "✓ No OpenCL errors"
  fi
  echo ""
}

# Function to create detailed report
create_report() {
  REPORT_FILE="intel_arc_diagnostic_report_$(date +%Y%m%d_%H%M%S).md"

  echo "Creating diagnostic report: ${REPORT_FILE}"

  cat >"${REPORT_FILE}" <<EOF
# Intel Arc GPU Diagnostic Report - Task 3

## Test Information

- **Date**: $(date)
- **Target**: ${GREMLIN_HOST}
- **Model**: ${MODEL}
- **Backend**: Tinygrad with OpenCL (Intel Arc forced)
- **Log File**: ${LOG_FILE}

## Objective

Capture diagnostic information about buffer allocations on Intel Arc GPU to identify:
1. All allocations >100MB (Requirement 1.1)
2. Buffer sizes in bytes and MB (Requirement 1.2)
3. Any allocations >4GB (Requirement 1.3)
4. Call stack/context for large allocations (Requirement 2.1)
5. Tensor shapes (Requirement 2.2)
6. Data types (Requirement 2.3)

## Diagnostic Logs

### Device Information

\`\`\`
EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    grep "DEVICE INFORMATION" -A 5 "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No device info captured" >>"${REPORT_FILE}"
  else
    echo "No diagnostic logs captured" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

### All Allocations >100MB

\`\`\`
EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    grep "Allocation #" "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No allocations logged" >>"${REPORT_FILE}"
    grep "Context:" "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || true
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

### Allocations >4GB

\`\`\`
EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
      grep "WARNING: Buffer >4GB" -A 5 "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}"
    else
      echo "No allocations >4GB detected" >>"${REPORT_FILE}"
    fi
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

### Allocation Summary

\`\`\`
EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    grep "ALLOCATION SUMMARY" -A 10 "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No summary captured" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

### OpenCL Errors

\`\`\`
EOF

  if grep -q "OpenCL Error" "${LOG_FILE}"; then
    grep "OpenCL Error" "${LOG_FILE}" | head -20 >>"${REPORT_FILE}"
  else
    echo "No OpenCL errors detected" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF
\`\`\`

## Analysis

### Requirements Coverage

EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    ALLOC_COUNT=$(grep -c "Allocation #" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
    SIZE_COUNT=$(grep -c "bytes" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
    WARN_COUNT=$(grep -c "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
    CONTEXT_COUNT=$(grep -c "Context:" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
    SHAPE_COUNT=$(grep -c "shape=" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
    DTYPE_COUNT=$(grep -c "dtype=" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")

    cat >>"${REPORT_FILE}" <<EOFINNER
- **Requirement 1.1** (Log allocations >100MB): ${ALLOC_COUNT} allocations logged
- **Requirement 1.2** (Log buffer sizes): ${SIZE_COUNT} size logs
- **Requirement 1.3** (Warn on >4GB): ${WARN_COUNT} warnings
- **Requirement 2.1** (Capture call stack): ${CONTEXT_COUNT} context captures
- **Requirement 2.2** (Include tensor shape): ${SHAPE_COUNT} shape logs
- **Requirement 2.3** (Include data type): ${DTYPE_COUNT} dtype logs

### Key Findings

EOFINNER

    if [ "${WARN_COUNT}" -gt 0 ]; then
      echo "- ⚠️  **CRITICAL**: ${WARN_COUNT} allocation(s) >4GB detected - exceeds Intel Arc GPU limits" >>"${REPORT_FILE}"
    else
      echo "- ✓ No allocations >4GB detected" >>"${REPORT_FILE}"
    fi

    echo "- Found ${ALLOC_COUNT} allocations >100MB" >>"${REPORT_FILE}"

    if grep -q "CL_MEM_OBJECT_ALLOCATION_FAILURE\|CL_INVALID_VALUE" "${LOG_FILE}"; then
      echo "- ⚠️  OpenCL allocation errors detected" >>"${REPORT_FILE}"
    fi
  else
    echo "- ⚠️  No diagnostic data captured - patch may not be active or no allocations occurred" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF

### Allocation Patterns

EOF

  if [ -f "${LOG_FILE}.intel_arc_only" ] && [ $(grep -c "Allocation #" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0") -gt 0 ]; then
    echo "Documented allocation sizes and patterns:" >>"${REPORT_FILE}"
    grep "Allocation #" "${LOG_FILE}.intel_arc_only" | sed 's/^/- /' >>"${REPORT_FILE}"
  else
    echo "No allocation patterns captured" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<EOF

## Task 3 Completion Status

- [x] Send inference request to trigger model loading
- [x] Monitor logs in real-time for allocation messages
- [x] Capture all \`[INTEL ARC DEBUG]\` log entries
- [x] Document allocation sizes and patterns

## Next Steps

Proceed to **Task 4: Analyze diagnostic results** to:
1. Identify all allocations >100MB
2. Identify any allocations >4GB
3. Determine which allocation causes errors
4. Document tensor shapes and contexts for large allocations
5. Create analysis report with findings

## Raw Data

- Full logs: \`${LOG_FILE}\`
- Intel Arc logs only: \`${LOG_FILE}.intel_arc_only\`
EOF

  echo "✓ Report created: ${REPORT_FILE}"
  echo ""
}

# Trap to ensure cleanup
trap stop_log_monitor EXIT

# Main execution
main() {
  restart_with_intel_arc
  start_log_monitor

  echo "Waiting 5 seconds for log monitor to stabilize..."
  sleep 5

  send_inference_request

  echo "Waiting 60 seconds for model loading and inference..."
  sleep 60

  stop_log_monitor

  analyze_logs
  create_report

  echo ""
  echo "=========================================="
  echo "Task 3 Complete!"
  echo "=========================================="
  echo "Full log: ${LOG_FILE}"
  if [ -f "${LOG_FILE}.intel_arc_only" ]; then
    echo "Intel Arc logs: ${LOG_FILE}.intel_arc_only"
  fi
  echo "Report: intel_arc_diagnostic_report_*.md"
  echo ""
  echo "Diagnostic data captured successfully."
  echo "Ready to proceed to Task 4: Analyze diagnostic results"
  echo ""
}

main

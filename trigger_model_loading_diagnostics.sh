#!/usr/bin/env bash
# Script to trigger model loading and capture Intel Arc diagnostic logs
# Task 3: Trigger model loading and capture diagnostics

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

# Function to check if service is running
check_service() {
  echo "Checking exo service status..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "systemctl is-active exo" || {
    echo "ERROR: exo service is not running"
    exit 1
  }
  echo "✓ Service is running"
  echo ""
}

# Function to clear old logs
clear_logs() {
  echo "Clearing old diagnostic logs..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "journalctl --rotate && journalctl --vacuum-time=1s" >/dev/null 2>&1
  echo "✓ Logs cleared"
  echo ""
}

# Function to start log monitoring in background
start_log_monitor() {
  echo "Starting log monitor..."
  ssh ${GREMLIN_USER}@${GREMLIN_HOST} "journalctl -u exo -f" >"${LOG_FILE}" 2>&1 &
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

# Function to extract and analyze diagnostic logs
analyze_logs() {
  echo ""
  echo "=========================================="
  echo "DIAGNOSTIC LOG ANALYSIS"
  echo "=========================================="
  echo ""

  # Extract all Intel Arc debug messages
  echo "Extracting [INTEL ARC DEBUG] messages..."
  grep "\[INTEL ARC DEBUG\]" "${LOG_FILE}" >"${LOG_FILE}.intel_arc_only" 2>/dev/null || {
    echo "WARNING: No [INTEL ARC DEBUG] messages found in logs"
    echo "This may indicate:"
    echo "  1. The patch was not applied correctly"
    echo "  2. No allocations occurred"
    echo "  3. The model did not load"
    return 1
  }

  echo "✓ Found $(wc -l <"${LOG_FILE}.intel_arc_only") diagnostic log lines"
  echo ""

  # Show device information
  echo "--- Device Information ---"
  grep "DEVICE INFORMATION" -A 5 "${LOG_FILE}.intel_arc_only" || echo "No device info found"
  echo ""

  # Show allocation summary
  echo "--- Allocation Summary ---"
  grep "ALLOCATION SUMMARY" -A 10 "${LOG_FILE}.intel_arc_only" || echo "No allocation summary found"
  echo ""

  # Show allocations >100MB
  echo "--- Large Allocations (>100MB) ---"
  grep "Allocation #" "${LOG_FILE}.intel_arc_only" | head -20
  echo ""

  # Check for >4GB allocations
  echo "--- Checking for >4GB Allocations ---"
  if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
    echo "⚠️  FOUND ALLOCATIONS >4GB:"
    grep "WARNING: Buffer >4GB" -A 5 "${LOG_FILE}.intel_arc_only"
  else
    echo "✓ No allocations >4GB detected"
  fi
  echo ""

  # Check for OpenCL errors
  echo "--- Checking for OpenCL Errors ---"
  if grep -q "CL_INVALID_VALUE\|OpenCL Error" "${LOG_FILE}"; then
    echo "⚠️  FOUND OPENCL ERRORS:"
    grep "CL_INVALID_VALUE\|OpenCL Error" "${LOG_FILE}" | head -10
  else
    echo "✓ No OpenCL errors detected"
  fi
  echo ""

  # Extract allocation sizes for analysis
  echo "--- Allocation Size Distribution ---"
  grep "Allocation #" "${LOG_FILE}.intel_arc_only" |
    sed -E 's/.*\(([0-9.]+) MB.*/\1/' |
    awk '{
            if ($1 < 500) small++
            else if ($1 < 1000) medium++
            else if ($1 < 2000) large++
            else if ($1 < 4000) xlarge++
            else huge++
            total++
        }
        END {
            print "  <500MB:     " small " allocations"
            print "  500MB-1GB:  " medium " allocations"
            print "  1GB-2GB:    " large " allocations"
            print "  2GB-4GB:    " xlarge " allocations"
            print "  >4GB:       " huge " allocations"
            print "  Total:      " total " allocations"
        }'
  echo ""
}

# Function to create analysis report
create_report() {
  REPORT_FILE="intel_arc_diagnostic_report_$(date +%Y%m%d_%H%M%S).md"

  echo "Creating diagnostic report: ${REPORT_FILE}"

  cat >"${REPORT_FILE}" <<'EOF'
# Intel Arc GPU Diagnostic Report

## Test Information

EOF

  echo "- **Date**: $(date)" >>"${REPORT_FILE}"
  echo "- **Target**: ${GREMLIN_HOST}" >>"${REPORT_FILE}"
  echo "- **Model**: ${MODEL}" >>"${REPORT_FILE}"
  echo "- **Log File**: ${LOG_FILE}" >>"${REPORT_FILE}"
  echo "" >>"${REPORT_FILE}"

  cat >>"${REPORT_FILE}" <<'EOF'
## Device Information

```
EOF

  grep "DEVICE INFORMATION" -A 5 "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No device info captured" >>"${REPORT_FILE}"

  cat >>"${REPORT_FILE}" <<'EOF'
```

## Allocation Summary

```
EOF

  grep "ALLOCATION SUMMARY" -A 10 "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No allocation summary captured" >>"${REPORT_FILE}"

  cat >>"${REPORT_FILE}" <<'EOF'
```

## Large Allocations (>100MB)

```
EOF

  grep "Allocation #" "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No large allocations found" >>"${REPORT_FILE}"

  cat >>"${REPORT_FILE}" <<'EOF'
```

## Allocations >4GB

```
EOF

  if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
    grep "WARNING: Buffer >4GB" -A 5 "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}"
  else
    echo "No allocations >4GB detected" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<'EOF'
```

## OpenCL Errors

```
EOF

  if grep -q "CL_INVALID_VALUE\|OpenCL Error" "${LOG_FILE}"; then
    grep "CL_INVALID_VALUE\|OpenCL Error" "${LOG_FILE}" | head -20 >>"${REPORT_FILE}"
  else
    echo "No OpenCL errors detected" >>"${REPORT_FILE}"
  fi

  cat >>"${REPORT_FILE}" <<'EOF'
```

## Allocation Context Details

```
EOF

  grep "Context:" "${LOG_FILE}.intel_arc_only" >>"${REPORT_FILE}" 2>/dev/null || echo "No context information captured" >>"${REPORT_FILE}"

  cat >>"${REPORT_FILE}" <<'EOF'
```

## Analysis

### Findings

EOF

  # Add automated findings
  if grep -q "WARNING: Buffer >4GB" "${LOG_FILE}.intel_arc_only"; then
    echo "- ⚠️  **CRITICAL**: Allocations >4GB detected - this exceeds Intel Arc GPU limits" >>"${REPORT_FILE}"
  fi

  if grep -q "CL_INVALID_VALUE" "${LOG_FILE}"; then
    echo "- ⚠️  **ERROR**: CL_INVALID_VALUE errors detected during allocation" >>"${REPORT_FILE}"
  fi

  LARGE_ALLOC_COUNT=$(grep -c "Allocation #" "${LOG_FILE}.intel_arc_only" 2>/dev/null || echo "0")
  echo "- Found ${LARGE_ALLOC_COUNT} allocations >100MB" >>"${REPORT_FILE}"

  cat >>"${REPORT_FILE}" <<'EOF'

### Recommendations

Based on the diagnostic data:

1. Review the largest allocations identified above
2. Determine which allocations exceed 4GB (if any)
3. Identify the tensor shapes and operations causing large allocations
4. Proceed to Phase 2: Buffer Splitting Implementation

### Next Steps

- [ ] Analyze allocation patterns
- [ ] Identify root cause of large allocations
- [ ] Design buffer splitting strategy
- [ ] Implement buffer splitting for allocations >3.5GB

EOF

  echo "✓ Report created: ${REPORT_FILE}"
  echo ""
}

# Trap to ensure log monitor is stopped on exit
trap stop_log_monitor EXIT

# Main execution
main() {
  check_service
  clear_logs
  start_log_monitor

  echo "Waiting 3 seconds for log monitor to stabilize..."
  sleep 3

  send_inference_request

  echo "Waiting 30 seconds for model loading and allocation to complete..."
  sleep 30

  stop_log_monitor

  analyze_logs
  create_report

  echo ""
  echo "=========================================="
  echo "Diagnostic capture complete!"
  echo "=========================================="
  echo "Log file: ${LOG_FILE}"
  echo "Intel Arc logs: ${LOG_FILE}.intel_arc_only"
  echo "Report: intel_arc_diagnostic_report_*.md"
  echo ""
  echo "To view the full logs:"
  echo "  cat ${LOG_FILE}"
  echo ""
  echo "To view only Intel Arc diagnostics:"
  echo "  cat ${LOG_FILE}.intel_arc_only"
  echo ""
}

main

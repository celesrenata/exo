#!/bin/bash
# Investigation script for EXO instance B6DAE586

echo "=== Investigating EXO Instance B6DAE586 ==="
echo "Timestamp: $(date)"
echo

# Function to run commands on gremlin-3
run_on_gremlin3() {
    echo "Running on gremlin-3: $1"
    ssh root@gremlin-3 "$1"
    echo
}

# Check if gremlin-3 is reachable
echo "1. Testing connectivity to gremlin-3..."
if ping -c 1 gremlin-3 >/dev/null 2>&1; then
    echo "✓ gremlin-3 is reachable"
else
    echo "✗ gremlin-3 is not reachable"
    exit 1
fi

# Get recent EXO logs from gremlin-3
echo "2. Getting recent EXO logs from gremlin-3..."
run_on_gremlin3 "journalctl -u exo --since '30 minutes ago' --no-pager | tail -50"

# Look for specific instance B6DAE586
echo "3. Searching for instance B6DAE586 in logs..."
run_on_gremlin3 "journalctl -u exo --since '2 hours ago' --no-pager | grep -i 'B6DAE586' | tail -20"

# Check for errors and failures
echo "4. Looking for recent errors on gremlin-3..."
run_on_gremlin3 "journalctl -u exo --since '30 minutes ago' --no-pager | grep -E '(ERROR|CRITICAL|Failed|Exception|Traceback)' | tail -20"

# Check EXO process status
echo "5. Checking EXO process status on gremlin-3..."
run_on_gremlin3 "systemctl status exo"

# Check system resources
echo "6. Checking system resources on gremlin-3..."
run_on_gremlin3 "free -h && echo && df -h | head -10"

# Check network connectivity from gremlin-3
echo "7. Checking network connectivity from gremlin-3..."
run_on_gremlin3 "netstat -tlnp | grep -E '(exo|python)' | head -10"

# Look for any recent crashes or restarts
echo "8. Checking for recent service restarts..."
run_on_gremlin3 "journalctl -u exo --since '1 hour ago' --no-pager | grep -E '(Started|Stopped|Restarted|systemd)'"

echo "=== Investigation Complete ==="
#!/usr/bin/env bash

# Quick deployment status checker for EXO race condition fixes
# This script provides a fast overview of deployment status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Status indicators
check_mark="✅"
cross_mark="❌"
warning_mark="⚠️"
info_mark="ℹ️"

echo -e "${BLUE}EXO Race Condition Fix - Deployment Status${NC}"
echo "=================================================="

# Check if race condition fixes are deployed
echo -e "\n${BLUE}1. Race Condition Fix Components${NC}"

components=(
    "src/exo/worker/runner/shutdown_coordinator.py"
    "src/exo/worker/runner/resource_manager.py"
    "src/exo/worker/runner/channel_manager.py"
    "src/exo/worker/runner/enhanced_queue_operations.py"
    "src/exo/worker/runner/error_handler.py"
)

all_components_present=true
for component in "${components[@]}"; do
    if [[ -f "$PROJECT_ROOT/$component" ]]; then
        echo -e "   ${check_mark} $component"
    else
        echo -e "   ${cross_mark} $component (MISSING)"
        all_components_present=false
    fi
done

if [[ "$all_components_present" == "true" ]]; then
    echo -e "   ${check_mark} All race condition fix components are present"
else
    echo -e "   ${cross_mark} Some components are missing - deployment may be incomplete"
fi

# Check service status
echo -e "\n${BLUE}2. Service Status${NC}"

if command -v systemctl &> /dev/null; then
    if systemctl is-active --quiet exo 2>/dev/null; then
        echo -e "   ${check_mark} EXO systemd service is running"
        
        # Check service uptime
        uptime=$(systemctl show exo --property=ActiveEnterTimestamp --value)
        if [[ -n "$uptime" ]]; then
            echo -e "   ${info_mark} Service started: $uptime"
        fi
    else
        echo -e "   ${cross_mark} EXO systemd service is not running"
    fi
else
    echo -e "   ${warning_mark} systemctl not available - cannot check service status"
fi

# Check if EXO processes are running
if pgrep -f "exo" > /dev/null; then
    process_count=$(pgrep -f "exo" | wc -l)
    echo -e "   ${check_mark} EXO processes running: $process_count"
else
    echo -e "   ${cross_mark} No EXO processes found"
fi

# Check API endpoints
echo -e "\n${BLUE}3. API Endpoint Status${NC}"

endpoints=(
    "http://localhost:52415/health:Health Check"
    "http://localhost:52415/state:API State"
    "http://localhost:52415/:Dashboard"
)

for endpoint_info in "${endpoints[@]}"; do
    IFS=':' read -r endpoint name <<< "$endpoint_info"
    
    if curl -s -f "$endpoint" > /dev/null 2>&1; then
        echo -e "   ${check_mark} $name ($endpoint)"
    else
        echo -e "   ${cross_mark} $name ($endpoint)"
    fi
done

# Check for recent race condition errors
echo -e "\n${BLUE}4. Recent Error Analysis${NC}"

if command -v journalctl &> /dev/null; then
    # Check last hour for race condition errors
    queue_closed_errors=$(journalctl -u exo --since "1 hour ago" 2>/dev/null | grep -c "Queue is closed" || echo "0")
    closed_resource_errors=$(journalctl -u exo --since "1 hour ago" 2>/dev/null | grep -c "ClosedResourceError" || echo "0")
    runner_failures=$(journalctl -u exo --since "1 hour ago" 2>/dev/null | grep -c "Runner.*exit.*code.*1" || echo "0")
    
    if [[ $queue_closed_errors -eq 0 ]]; then
        echo -e "   ${check_mark} No 'Queue is closed' errors in last hour"
    else
        echo -e "   ${cross_mark} Found $queue_closed_errors 'Queue is closed' errors in last hour"
    fi
    
    if [[ $closed_resource_errors -eq 0 ]]; then
        echo -e "   ${check_mark} No 'ClosedResourceError' errors in last hour"
    else
        echo -e "   ${cross_mark} Found $closed_resource_errors 'ClosedResourceError' errors in last hour"
    fi
    
    if [[ $runner_failures -eq 0 ]]; then
        echo -e "   ${check_mark} No runner failures in last hour"
    else
        echo -e "   ${cross_mark} Found $runner_failures runner failures in last hour"
    fi
    
    total_errors=$((queue_closed_errors + closed_resource_errors + runner_failures))
    if [[ $total_errors -eq 0 ]]; then
        echo -e "   ${check_mark} No race condition errors detected in last hour"
    else
        echo -e "   ${warning_mark} Total race condition errors in last hour: $total_errors"
    fi
else
    echo -e "   ${warning_mark} journalctl not available - cannot check error logs"
fi

# Check resource usage
echo -e "\n${BLUE}5. Resource Usage${NC}"

if pgrep -f "exo" > /dev/null; then
    exo_pid=$(pgrep -f "exo" | head -1)
    
    # Memory usage
    memory_usage=$(ps -o pid,ppid,cmd,%mem --sort=-%mem -C python | grep exo | head -1 | awk '{print $4}' 2>/dev/null || echo "N/A")
    if [[ "$memory_usage" != "N/A" ]]; then
        if (( $(echo "$memory_usage > 80" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "   ${warning_mark} Memory usage: ${memory_usage}% (HIGH)"
        else
            echo -e "   ${check_mark} Memory usage: ${memory_usage}%"
        fi
    else
        echo -e "   ${info_mark} Memory usage: N/A"
    fi
    
    # CPU usage
    cpu_usage=$(ps -o pid,ppid,cmd,%cpu --sort=-%cpu -C python | grep exo | head -1 | awk '{print $4}' 2>/dev/null || echo "N/A")
    if [[ "$cpu_usage" != "N/A" ]]; then
        if (( $(echo "$cpu_usage > 90" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "   ${warning_mark} CPU usage: ${cpu_usage}% (HIGH)"
        else
            echo -e "   ${check_mark} CPU usage: ${cpu_usage}%"
        fi
    else
        echo -e "   ${info_mark} CPU usage: N/A"
    fi
    
    # File descriptors
    if [[ -d "/proc/$exo_pid/fd" ]]; then
        fd_count=$(ls -1 /proc/$exo_pid/fd 2>/dev/null | wc -l)
        if [[ $fd_count -gt 1000 ]]; then
            echo -e "   ${warning_mark} File descriptors: $fd_count (HIGH)"
        else
            echo -e "   ${check_mark} File descriptors: $fd_count"
        fi
    else
        echo -e "   ${info_mark} File descriptors: N/A"
    fi
else
    echo -e "   ${info_mark} No EXO processes running - cannot check resource usage"
fi

# Check deployment artifacts
echo -e "\n${BLUE}6. Deployment Artifacts${NC}"

# Check for recent deployment logs
if [[ -d "$PROJECT_ROOT/deployment/logs" ]]; then
    recent_deploy_log=$(find "$PROJECT_ROOT/deployment/logs" -name "deploy_*.log" -mtime -1 2>/dev/null | head -1)
    if [[ -n "$recent_deploy_log" ]]; then
        echo -e "   ${check_mark} Recent deployment log found: $(basename "$recent_deploy_log")"
    else
        echo -e "   ${info_mark} No recent deployment logs found"
    fi
    
    # Check for backups
    if [[ -d "$PROJECT_ROOT/deployment/backups" ]]; then
        backup_count=$(ls -1 "$PROJECT_ROOT/deployment/backups" 2>/dev/null | wc -l)
        if [[ $backup_count -gt 0 ]]; then
            echo -e "   ${check_mark} Available backups: $backup_count"
            latest_backup=$(ls -1t "$PROJECT_ROOT/deployment/backups" 2>/dev/null | head -1)
            if [[ -n "$latest_backup" ]]; then
                echo -e "   ${info_mark} Latest backup: $latest_backup"
            fi
        else
            echo -e "   ${warning_mark} No backups found"
        fi
    else
        echo -e "   ${warning_mark} Backup directory not found"
    fi
else
    echo -e "   ${warning_mark} Deployment directory not found"
fi

# Overall status summary
echo -e "\n${BLUE}Overall Status Summary${NC}"
echo "======================"

# Determine overall status
overall_status="HEALTHY"
status_color="$GREEN"

# Check critical conditions
if [[ "$all_components_present" != "true" ]]; then
    overall_status="INCOMPLETE DEPLOYMENT"
    status_color="$RED"
elif ! systemctl is-active --quiet exo 2>/dev/null && ! pgrep -f "exo" > /dev/null; then
    overall_status="SERVICE DOWN"
    status_color="$RED"
elif ! curl -s -f http://localhost:52415/health > /dev/null 2>&1; then
    overall_status="API UNAVAILABLE"
    status_color="$YELLOW"
elif [[ $total_errors -gt 5 ]]; then
    overall_status="HIGH ERROR RATE"
    status_color="$YELLOW"
fi

echo -e "Status: ${status_color}${overall_status}${NC}"

# Provide recommendations
echo -e "\n${BLUE}Recommendations${NC}"
echo "==============="

case "$overall_status" in
    "HEALTHY")
        echo -e "${check_mark} System is operating normally"
        echo -e "${info_mark} Continue regular monitoring"
        ;;
    "INCOMPLETE DEPLOYMENT")
        echo -e "${cross_mark} Run deployment script: ./deployment/deploy-race-condition-fix.sh"
        ;;
    "SERVICE DOWN")
        echo -e "${cross_mark} Start EXO service: systemctl start exo"
        echo -e "${info_mark} Check logs: journalctl -u exo -f"
        ;;
    "API UNAVAILABLE")
        echo -e "${warning_mark} Check service logs: journalctl -u exo --since '10 minutes ago'"
        echo -e "${warning_mark} Verify network configuration and firewall settings"
        ;;
    "HIGH ERROR RATE")
        echo -e "${warning_mark} Review error logs: journalctl -u exo --since '1 hour ago' -p err"
        echo -e "${warning_mark} Consider running validation: ./deployment/monitoring/validate-deployment.sh"
        ;;
esac

# Quick action commands
echo -e "\n${BLUE}Quick Actions${NC}"
echo "============="
echo "Monitor logs:     journalctl -u exo -f"
echo "Check health:     curl http://localhost:52415/health"
echo "Run validation:   ./deployment/monitoring/validate-deployment.sh"
echo "View backups:     ls -la deployment/backups/"
echo "Deploy fixes:     ./deployment/deploy-race-condition-fix.sh"

echo ""
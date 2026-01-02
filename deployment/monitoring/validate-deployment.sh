#!/usr/bin/env bash

# Validation script for race condition fix deployment
# This script monitors and validates the deployment in production

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/deployment/logs/validation_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check service health
check_service_health() {
    log "Checking EXO service health..."
    
    local health_checks=0
    local health_passed=0
    
    # Check systemd service status
    ((health_checks++))
    if systemctl is-active --quiet exo 2>/dev/null; then
        success "✅ Systemd service is active"
        ((health_passed++))
    else
        error "❌ Systemd service is not active"
    fi
    
    # Check HTTP health endpoint
    ((health_checks++))
    if curl -s -f http://localhost:52415/health > /dev/null 2>&1; then
        success "✅ HTTP health endpoint responding"
        ((health_passed++))
    else
        error "❌ HTTP health endpoint not responding"
    fi
    
    # Check API state endpoint
    ((health_checks++))
    if curl -s -f http://localhost:52415/state > /dev/null 2>&1; then
        success "✅ API state endpoint responding"
        ((health_passed++))
    else
        error "❌ API state endpoint not responding"
    fi
    
    # Check dashboard accessibility
    ((health_checks++))
    if curl -s -f http://localhost:52415/ > /dev/null 2>&1; then
        success "✅ Dashboard is accessible"
        ((health_passed++))
    else
        error "❌ Dashboard is not accessible"
    fi
    
    log "Health check results: $health_passed/$health_checks passed"
    return $((health_checks - health_passed))
}

# Check for race condition errors in logs
check_race_condition_errors() {
    log "Checking for race condition errors in logs..."
    
    local error_count=0
    local log_period="1 hour ago"
    
    # Check for "Queue is closed" errors
    local queue_closed_errors=$(journalctl -u exo --since "$log_period" | grep -c "Queue is closed" || echo "0")
    if [[ $queue_closed_errors -gt 0 ]]; then
        error "❌ Found $queue_closed_errors 'Queue is closed' errors in the last hour"
        ((error_count += queue_closed_errors))
    else
        success "✅ No 'Queue is closed' errors found"
    fi
    
    # Check for ClosedResourceError
    local closed_resource_errors=$(journalctl -u exo --since "$log_period" | grep -c "ClosedResourceError" || echo "0")
    if [[ $closed_resource_errors -gt 0 ]]; then
        error "❌ Found $closed_resource_errors 'ClosedResourceError' errors in the last hour"
        ((error_count += closed_resource_errors))
    else
        success "✅ No 'ClosedResourceError' errors found"
    fi
    
    # Check for runner exit code 1
    local runner_failures=$(journalctl -u exo --since "$log_period" | grep -c "Runner.*exit.*code.*1" || echo "0")
    if [[ $runner_failures -gt 0 ]]; then
        error "❌ Found $runner_failures runner failures in the last hour"
        ((error_count += runner_failures))
    else
        success "✅ No runner failures found"
    fi
    
    # Check for shutdown coordination errors
    local shutdown_errors=$(journalctl -u exo --since "$log_period" | grep -c "shutdown.*error\|coordination.*failed" || echo "0")
    if [[ $shutdown_errors -gt 0 ]]; then
        error "❌ Found $shutdown_errors shutdown coordination errors in the last hour"
        ((error_count += shutdown_errors))
    else
        success "✅ No shutdown coordination errors found"
    fi
    
    log "Race condition error check: $error_count total errors found"
    return $error_count
}

# Test multinode functionality
test_multinode_functionality() {
    log "Testing multinode functionality..."
    
    # Run multinode validation script if available
    if [[ -f "$PROJECT_ROOT/validate_multinode_integration.py" ]]; then
        log "Running multinode integration test..."
        if python "$PROJECT_ROOT/validate_multinode_integration.py"; then
            success "✅ Multinode integration test passed"
            return 0
        else
            error "❌ Multinode integration test failed"
            return 1
        fi
    else
        warning "⚠️  Multinode integration test script not found"
        return 0
    fi
}

# Check resource usage
check_resource_usage() {
    log "Checking resource usage..."
    
    # Check memory usage
    local memory_usage=$(ps -o pid,ppid,cmd,%mem --sort=-%mem -C python | grep exo | head -1 | awk '{print $4}')
    if [[ -n "$memory_usage" ]]; then
        log "EXO memory usage: ${memory_usage}%"
        if (( $(echo "$memory_usage > 80" | bc -l) )); then
            warning "⚠️  High memory usage detected: ${memory_usage}%"
        else
            success "✅ Memory usage is normal: ${memory_usage}%"
        fi
    fi
    
    # Check CPU usage
    local cpu_usage=$(ps -o pid,ppid,cmd,%cpu --sort=-%cpu -C python | grep exo | head -1 | awk '{print $4}')
    if [[ -n "$cpu_usage" ]]; then
        log "EXO CPU usage: ${cpu_usage}%"
        if (( $(echo "$cpu_usage > 90" | bc -l) )); then
            warning "⚠️  High CPU usage detected: ${cpu_usage}%"
        else
            success "✅ CPU usage is normal: ${cpu_usage}%"
        fi
    fi
    
    # Check file descriptors
    local exo_pid=$(pgrep -f "exo" | head -1)
    if [[ -n "$exo_pid" ]]; then
        local fd_count=$(ls -1 /proc/$exo_pid/fd 2>/dev/null | wc -l)
        log "EXO file descriptors: $fd_count"
        if [[ $fd_count -gt 1000 ]]; then
            warning "⚠️  High file descriptor usage: $fd_count"
        else
            success "✅ File descriptor usage is normal: $fd_count"
        fi
    fi
}

# Performance monitoring
monitor_performance() {
    log "Monitoring performance metrics..."
    
    # Check response times
    local start_time=$(date +%s%N)
    if curl -s -f http://localhost:52415/health > /dev/null 2>&1; then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        log "Health endpoint response time: ${response_time}ms"
        if [[ $response_time -gt 5000 ]]; then
            warning "⚠️  Slow response time: ${response_time}ms"
        else
            success "✅ Response time is good: ${response_time}ms"
        fi
    fi
    
    # Check for memory leaks by monitoring over time
    local initial_memory=$(ps -o pid,ppid,cmd,rss --sort=-rss -C python | grep exo | head -1 | awk '{print $4}')
    if [[ -n "$initial_memory" ]]; then
        log "Initial memory usage: ${initial_memory}KB"
        
        # Wait and check again
        sleep 30
        local final_memory=$(ps -o pid,ppid,cmd,rss --sort=-rss -C python | grep exo | head -1 | awk '{print $4}')
        if [[ -n "$final_memory" ]]; then
            local memory_diff=$((final_memory - initial_memory))
            log "Memory change after 30s: ${memory_diff}KB"
            
            if [[ $memory_diff -gt 10000 ]]; then
                warning "⚠️  Potential memory leak detected: +${memory_diff}KB in 30s"
            else
                success "✅ Memory usage is stable"
            fi
        fi
    fi
}

# Generate validation report
generate_validation_report() {
    local report_file="$PROJECT_ROOT/deployment/logs/validation_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# EXO Race Condition Fix Validation Report

**Validation Date:** $(date)
**Validation Type:** Production Deployment Monitoring

## System Health Status

### Service Status
$(if systemctl is-active --quiet exo 2>/dev/null; then echo "✅ EXO service is running"; else echo "❌ EXO service is not running"; fi)
$(if curl -s -f http://localhost:52415/health > /dev/null 2>&1; then echo "✅ Health endpoint responding"; else echo "❌ Health endpoint not responding"; fi)
$(if curl -s -f http://localhost:52415/state > /dev/null 2>&1; then echo "✅ API endpoint responding"; else echo "❌ API endpoint not responding"; fi)
$(if curl -s -f http://localhost:52415/ > /dev/null 2>&1; then echo "✅ Dashboard accessible"; else echo "❌ Dashboard not accessible"; fi)

### Error Analysis (Last Hour)
- Queue Closed Errors: $(journalctl -u exo --since "1 hour ago" | grep -c "Queue is closed" || echo "0")
- Closed Resource Errors: $(journalctl -u exo --since "1 hour ago" | grep -c "ClosedResourceError" || echo "0")
- Runner Failures: $(journalctl -u exo --since "1 hour ago" | grep -c "Runner.*exit.*code.*1" || echo "0")
- Shutdown Errors: $(journalctl -u exo --since "1 hour ago" | grep -c "shutdown.*error\|coordination.*failed" || echo "0")

### Resource Usage
- Memory Usage: $(ps -o pid,ppid,cmd,%mem --sort=-%mem -C python | grep exo | head -1 | awk '{print $4}' || echo "N/A")%
- CPU Usage: $(ps -o pid,ppid,cmd,%cpu --sort=-%cpu -C python | grep exo | head -1 | awk '{print $4}' || echo "N/A")%
- File Descriptors: $(ls -1 /proc/$(pgrep -f "exo" | head -1)/fd 2>/dev/null | wc -l || echo "N/A")

### Performance Metrics
- Health Endpoint Response Time: Measured during validation
- Memory Stability: Monitored over 30-second period

## Recommendations

### If All Checks Pass ✅
- Continue monitoring for 24-48 hours
- Schedule regular validation runs
- Monitor logs for any new error patterns

### If Issues Found ❌
1. Review specific error messages in logs
2. Consider rollback if critical issues persist
3. Investigate root causes before redeployment
4. Contact development team if needed

## Monitoring Commands
\`\`\`bash
# Check service status
systemctl status exo

# Monitor logs in real-time
journalctl -u exo -f

# Check for race condition errors
journalctl -u exo --since "1 hour ago" | grep -E "Queue is closed|ClosedResourceError|Runner.*exit.*code.*1"

# Test endpoints
curl http://localhost:52415/health
curl http://localhost:52415/state

# Monitor resource usage
ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem -C python | grep exo
\`\`\`

EOF
    
    log "Validation report generated: $report_file"
}

# Main validation function
main() {
    log "Starting EXO deployment validation..."
    
    mkdir -p "$PROJECT_ROOT/deployment/logs"
    
    local total_checks=0
    local failed_checks=0
    
    # Run all validation checks
    if ! check_service_health; then
        ((failed_checks++))
    fi
    ((total_checks++))
    
    if ! check_race_condition_errors; then
        ((failed_checks++))
    fi
    ((total_checks++))
    
    if ! test_multinode_functionality; then
        ((failed_checks++))
    fi
    ((total_checks++))
    
    check_resource_usage
    monitor_performance
    generate_validation_report
    
    # Summary
    local passed_checks=$((total_checks - failed_checks))
    log "Validation summary: $passed_checks/$total_checks checks passed"
    
    if [[ $failed_checks -eq 0 ]]; then
        success "🎉 All validation checks passed!"
        echo ""
        echo "✅ EXO deployment validation successful!"
        echo "📊 All systems are operating normally"
        echo "🔍 Continue monitoring with regular validation runs"
        return 0
    else
        error "❌ $failed_checks validation checks failed"
        echo ""
        echo "⚠️  EXO deployment validation found issues!"
        echo "📋 Check the validation report for details"
        echo "🔧 Consider rollback if critical issues persist"
        return 1
    fi
}

# Handle script interruption
trap 'error "Validation interrupted. Check logs: $LOG_FILE"' INT TERM

# Run main function
main "$@"
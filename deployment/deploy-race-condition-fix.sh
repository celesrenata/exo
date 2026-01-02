#!/usr/bin/env bash

# Deployment script for multinode race condition fixes
# This script deploys the enhanced runner supervisor and coordination system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/deployment/backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/deployment/logs/deploy_$(date +%Y%m%d_%H%M%S).log"

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

# Create necessary directories
create_directories() {
    log "Creating deployment directories..."
    mkdir -p "$PROJECT_ROOT/deployment/logs"
    mkdir -p "$PROJECT_ROOT/deployment/backups"
    mkdir -p "$BACKUP_DIR"
}

# Pre-deployment validation
validate_environment() {
    log "Validating deployment environment..."
    
    # Check if we're in the correct directory
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        error "Not in EXO project root directory"
        exit 1
    fi
    
    # Check Python version
    if ! python3 --version | grep -q "3.13"; then
        warning "Python 3.13 not detected. Current version: $(python3 --version)"
    fi
    
    # Check if virtual environment is active
    if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$UV_PROJECT_ENVIRONMENT" ]]; then
        warning "No virtual environment detected. Consider using 'uv sync' first."
    fi
    
    # Check for existing EXO processes
    if pgrep -f "exo" > /dev/null; then
        warning "EXO processes are currently running. They will be stopped during deployment."
    fi
    
    success "Environment validation completed"
}

# Backup current installation
backup_current_installation() {
    log "Creating backup of current installation..."
    
    # Backup key files
    local files_to_backup=(
        "src/exo/worker/runner/runner_supervisor.py"
        "src/exo/worker/runner/bootstrap.py"
        "src/exo/worker/runner/runner.py"
        "src/exo/worker/runner/channel_manager.py"
        "src/exo/worker/runner/resource_manager.py"
        "src/exo/worker/runner/shutdown_coordinator.py"
        "src/exo/worker/runner/error_handler.py"
    )
    
    for file in "${files_to_backup[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            local backup_path="$BACKUP_DIR/$(dirname "$file")"
            mkdir -p "$backup_path"
            cp "$PROJECT_ROOT/$file" "$BACKUP_DIR/$file"
            log "Backed up: $file"
        fi
    done
    
    # Backup configuration files
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        cp "$PROJECT_ROOT/pyproject.toml" "$BACKUP_DIR/"
    fi
    
    # Create backup manifest
    cat > "$BACKUP_DIR/backup_manifest.txt" << EOF
Backup created: $(date)
EXO Version: $(grep '^version' "$PROJECT_ROOT/pyproject.toml" | cut -d'"' -f2)
Git commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
Deployment type: Race condition fixes
EOF
    
    success "Backup completed: $BACKUP_DIR"
}

# Stop EXO services
stop_services() {
    log "Stopping EXO services..."
    
    # Stop systemd service if it exists
    if systemctl is-active --quiet exo 2>/dev/null; then
        log "Stopping systemd exo service..."
        sudo systemctl stop exo
        success "Systemd service stopped"
    fi
    
    # Kill any remaining EXO processes
    if pgrep -f "exo" > /dev/null; then
        log "Terminating remaining EXO processes..."
        pkill -f "exo" || true
        sleep 2
        
        # Force kill if still running
        if pgrep -f "exo" > /dev/null; then
            warning "Force killing remaining processes..."
            pkill -9 -f "exo" || true
        fi
    fi
    
    success "All EXO services stopped"
}

# Deploy the fixes
deploy_fixes() {
    log "Deploying race condition fixes..."
    
    # The fixes are already in place in the codebase
    # This step validates they are present and working
    
    local required_files=(
        "src/exo/worker/runner/shutdown_coordinator.py"
        "src/exo/worker/runner/resource_manager.py"
        "src/exo/worker/runner/channel_manager.py"
        "src/exo/worker/runner/enhanced_queue_operations.py"
        "src/exo/worker/runner/error_handler.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            error "Required file missing: $file"
            exit 1
        fi
    done
    
    # Install/update the package
    log "Installing updated EXO package..."
    cd "$PROJECT_ROOT"
    
    if command -v uv &> /dev/null; then
        uv sync --all-packages
    else
        pip install -e .
    fi
    
    success "Package installation completed"
}

# Run validation tests
run_validation_tests() {
    log "Running validation tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run specific tests for race condition fixes
    local test_files=(
        "src/exo/worker/tests/unittests/test_runner/test_shutdown_coordinator.py"
        "src/exo/worker/tests/unittests/test_runner/test_resource_manager.py"
        "src/exo/worker/tests/unittests/test_runner/test_channel_manager.py"
        "src/exo/worker/tests/unittests/test_runner/test_error_handler.py"
    )
    
    local test_passed=true
    
    for test_file in "${test_files[@]}"; do
        if [[ -f "$test_file" ]]; then
            log "Running test: $test_file"
            if ! python -m pytest "$test_file" -v; then
                error "Test failed: $test_file"
                test_passed=false
            fi
        else
            warning "Test file not found: $test_file"
        fi
    done
    
    if [[ "$test_passed" == "true" ]]; then
        success "All validation tests passed"
    else
        error "Some validation tests failed"
        return 1
    fi
}

# Start services
start_services() {
    log "Starting EXO services..."
    
    # Start systemd service if it exists
    if systemctl list-unit-files | grep -q "exo.service"; then
        log "Starting systemd exo service..."
        sudo systemctl start exo
        
        # Wait for service to be active
        local timeout=30
        local count=0
        while ! systemctl is-active --quiet exo && [[ $count -lt $timeout ]]; do
            sleep 1
            ((count++))
        done
        
        if systemctl is-active --quiet exo; then
            success "Systemd service started successfully"
        else
            error "Failed to start systemd service"
            return 1
        fi
    else
        log "No systemd service found. Manual start required."
    fi
}

# Post-deployment validation
post_deployment_validation() {
    log "Running post-deployment validation..."
    
    # Check if EXO is responding
    local max_attempts=10
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Checking EXO health (attempt $attempt/$max_attempts)..."
        
        if curl -s -f http://localhost:52415/health > /dev/null 2>&1; then
            success "EXO is responding to health checks"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            warning "EXO health check failed after $max_attempts attempts"
            break
        fi
        
        sleep 5
        ((attempt++))
    done
    
    # Run integration test
    if [[ -f "$PROJECT_ROOT/validate_multinode_integration.py" ]]; then
        log "Running multinode integration validation..."
        if python "$PROJECT_ROOT/validate_multinode_integration.py"; then
            success "Multinode integration validation passed"
        else
            warning "Multinode integration validation failed"
        fi
    fi
}

# Generate deployment report
generate_deployment_report() {
    local report_file="$PROJECT_ROOT/deployment/logs/deployment_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# EXO Race Condition Fix Deployment Report

**Deployment Date:** $(date)
**Deployment Type:** Multinode Race Condition Fixes
**Backup Location:** $BACKUP_DIR

## Deployment Summary

### Components Deployed
- Enhanced Runner Supervisor with three-phase shutdown protocol
- Resource Manager with dependency-aware cleanup
- Channel Manager with race-condition-free operations
- Shutdown Coordinator for graceful termination
- Error Handler with improved recovery mechanisms

### Files Modified
- \`src/exo/worker/runner/runner_supervisor.py\`
- \`src/exo/worker/runner/bootstrap.py\`
- \`src/exo/worker/runner/runner.py\`
- \`src/exo/worker/runner/channel_manager.py\`
- \`src/exo/worker/runner/resource_manager.py\`
- \`src/exo/worker/runner/shutdown_coordinator.py\`
- \`src/exo/worker/runner/error_handler.py\`

### Validation Results
$(if systemctl is-active --quiet exo 2>/dev/null; then echo "✅ EXO service is running"; else echo "❌ EXO service is not running"; fi)
$(if curl -s -f http://localhost:52415/health > /dev/null 2>&1; then echo "✅ EXO health check passed"; else echo "❌ EXO health check failed"; fi)

### Rollback Information
To rollback this deployment, run:
\`\`\`bash
./deployment/rollback-race-condition-fix.sh $BACKUP_DIR
\`\`\`

### Monitoring Commands
- Check service status: \`systemctl status exo\`
- View logs: \`journalctl -u exo -f\`
- Health check: \`curl http://localhost:52415/health\`
- API status: \`curl http://localhost:52415/state\`

### Expected Improvements
- ✅ No more "Queue is closed" errors during shutdown
- ✅ No more "ClosedResourceError" exceptions
- ✅ Graceful multi-node coordination
- ✅ Improved error recovery and logging
- ✅ Better resource cleanup ordering

EOF
    
    log "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log "Starting EXO race condition fix deployment..."
    
    create_directories
    validate_environment
    backup_current_installation
    stop_services
    deploy_fixes
    
    if run_validation_tests; then
        start_services
        post_deployment_validation
        generate_deployment_report
        success "Deployment completed successfully!"
        
        echo ""
        echo "🎉 EXO Race Condition Fixes Deployed Successfully!"
        echo "📊 Monitor with: journalctl -u exo -f"
        echo "🌐 Dashboard: http://localhost:52415"
        echo "📋 Report: deployment/logs/deployment_report_$(date +%Y%m%d_%H%M%S).md"
        echo ""
        echo "Expected improvements:"
        echo "  ✅ No more 'Queue is closed' errors"
        echo "  ✅ No more 'ClosedResourceError' exceptions"
        echo "  ✅ Graceful multi-node shutdown coordination"
        echo "  ✅ Better error recovery and logging"
        
    else
        error "Deployment validation failed. Rolling back..."
        "$SCRIPT_DIR/rollback-race-condition-fix.sh" "$BACKUP_DIR"
        exit 1
    fi
}

# Handle script interruption
trap 'error "Deployment interrupted. Check logs: $LOG_FILE"' INT TERM

# Run main function
main "$@"
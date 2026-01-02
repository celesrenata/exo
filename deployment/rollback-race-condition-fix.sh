#!/usr/bin/env bash

# Rollback script for multinode race condition fixes
# This script restores the previous version in case of deployment issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${1:-}"
LOG_FILE="$PROJECT_ROOT/deployment/logs/rollback_$(date +%Y%m%d_%H%M%S).log"

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

# Usage information
usage() {
    echo "Usage: $0 <backup_directory>"
    echo ""
    echo "Rollback the EXO race condition fixes to a previous backup."
    echo ""
    echo "Arguments:"
    echo "  backup_directory    Path to the backup directory created during deployment"
    echo ""
    echo "Example:"
    echo "  $0 deployment/backups/20250102_143000"
    echo ""
    echo "Available backups:"
    if [[ -d "$PROJECT_ROOT/deployment/backups" ]]; then
        ls -1 "$PROJECT_ROOT/deployment/backups" | head -10
    else
        echo "  No backups found"
    fi
}

# Validate backup directory
validate_backup() {
    if [[ -z "$BACKUP_DIR" ]]; then
        error "Backup directory not specified"
        usage
        exit 1
    fi
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        error "Backup directory does not exist: $BACKUP_DIR"
        exit 1
    fi
    
    if [[ ! -f "$BACKUP_DIR/backup_manifest.txt" ]]; then
        error "Invalid backup directory (missing manifest): $BACKUP_DIR"
        exit 1
    fi
    
    log "Using backup directory: $BACKUP_DIR"
    log "Backup manifest:"
    cat "$BACKUP_DIR/backup_manifest.txt" | tee -a "$LOG_FILE"
}

# Create rollback backup
create_rollback_backup() {
    local rollback_backup_dir="$PROJECT_ROOT/deployment/backups/rollback_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$rollback_backup_dir"
    
    log "Creating rollback backup at: $rollback_backup_dir"
    
    # Backup current state before rollback
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
            local backup_path="$rollback_backup_dir/$(dirname "$file")"
            mkdir -p "$backup_path"
            cp "$PROJECT_ROOT/$file" "$rollback_backup_dir/$file"
        fi
    done
    
    # Create rollback manifest
    cat > "$rollback_backup_dir/rollback_manifest.txt" << EOF
Rollback backup created: $(date)
Original backup: $BACKUP_DIR
Git commit before rollback: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
Rollback reason: Manual rollback requested
EOF
    
    success "Rollback backup created: $rollback_backup_dir"
}

# Stop EXO services
stop_services() {
    log "Stopping EXO services for rollback..."
    
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

# Restore files from backup
restore_files() {
    log "Restoring files from backup..."
    
    # Find all files in backup
    local files_restored=0
    
    while IFS= read -r -d '' backup_file; do
        # Get relative path from backup directory
        local relative_path="${backup_file#$BACKUP_DIR/}"
        local target_file="$PROJECT_ROOT/$relative_path"
        
        # Skip manifest files
        if [[ "$relative_path" == *"manifest.txt" ]]; then
            continue
        fi
        
        # Create target directory if needed
        local target_dir="$(dirname "$target_file")"
        mkdir -p "$target_dir"
        
        # Restore file
        cp "$backup_file" "$target_file"
        log "Restored: $relative_path"
        ((files_restored++))
        
    done < <(find "$BACKUP_DIR" -type f -print0)
    
    success "Restored $files_restored files from backup"
}

# Reinstall package
reinstall_package() {
    log "Reinstalling EXO package..."
    
    cd "$PROJECT_ROOT"
    
    if command -v uv &> /dev/null; then
        uv sync --all-packages --force-reinstall
    else
        pip install -e . --force-reinstall
    fi
    
    success "Package reinstallation completed"
}

# Run rollback validation
run_rollback_validation() {
    log "Running rollback validation..."
    
    cd "$PROJECT_ROOT"
    
    # Run basic import test
    if python -c "import exo; print('EXO import successful')"; then
        success "EXO package imports successfully"
    else
        error "EXO package import failed"
        return 1
    fi
    
    # Run basic functionality test if available
    if [[ -f "$PROJECT_ROOT/test_basic_functionality.py" ]]; then
        log "Running basic functionality test..."
        if python "$PROJECT_ROOT/test_basic_functionality.py"; then
            success "Basic functionality test passed"
        else
            warning "Basic functionality test failed"
        fi
    fi
}

# Start services
start_services() {
    log "Starting EXO services after rollback..."
    
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

# Post-rollback validation
post_rollback_validation() {
    log "Running post-rollback validation..."
    
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
}

# Generate rollback report
generate_rollback_report() {
    local report_file="$PROJECT_ROOT/deployment/logs/rollback_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# EXO Race Condition Fix Rollback Report

**Rollback Date:** $(date)
**Backup Used:** $BACKUP_DIR
**Rollback Type:** Multinode Race Condition Fixes

## Rollback Summary

### Original Backup Information
$(cat "$BACKUP_DIR/backup_manifest.txt")

### Files Restored
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

### Post-Rollback Actions Required
1. Monitor system for stability
2. Investigate root cause of deployment issues
3. Plan corrective actions for next deployment attempt

### Monitoring Commands
- Check service status: \`systemctl status exo\`
- View logs: \`journalctl -u exo -f\`
- Health check: \`curl http://localhost:52415/health\`
- API status: \`curl http://localhost:52415/state\`

EOF
    
    log "Rollback report generated: $report_file"
}

# Main rollback function
main() {
    log "Starting EXO race condition fix rollback..."
    
    validate_backup
    create_rollback_backup
    stop_services
    restore_files
    reinstall_package
    
    if run_rollback_validation; then
        start_services
        post_rollback_validation
        generate_rollback_report
        success "Rollback completed successfully!"
        
        echo ""
        echo "🔄 EXO Race Condition Fixes Rolled Back Successfully!"
        echo "📊 Monitor with: journalctl -u exo -f"
        echo "🌐 Dashboard: http://localhost:52415"
        echo "📋 Report: deployment/logs/rollback_report_$(date +%Y%m%d_%H%M%S).md"
        echo ""
        echo "⚠️  System has been restored to previous state."
        echo "   Investigate deployment issues before attempting redeployment."
        
    else
        error "Rollback validation failed. Manual intervention required."
        exit 1
    fi
}

# Handle script interruption
trap 'error "Rollback interrupted. Check logs: $LOG_FILE"' INT TERM

# Run main function
main "$@"
#!/usr/bin/env bash
# Rebuild gremlin-1 with Intel hardware support and run tests
#
# This script:
# 1. Rebuilds gremlin-1 with the exo-intel module
# 2. Activates the new configuration
# 3. Starts the exo service
# 4. Runs validation tests
#
# Usage:
#   ./rebuild_and_test_gremlin1.sh

set -euo pipefail

# Configuration
TARGET_HOST="gremlin-1"
TARGET_IP="10.1.1.12"
SSH_CMD="ssh root@${TARGET_IP}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check connectivity
log_info "Step 1: Checking connectivity to $TARGET_HOST..."
if ! ping -c 1 -W 2 "$TARGET_IP" >/dev/null 2>&1; then
  log_error "Cannot reach $TARGET_IP"
  exit 1
fi

if ! $SSH_CMD "echo 'SSH OK'" >/dev/null 2>&1; then
  log_error "Cannot SSH to $TARGET_IP"
  exit 1
fi

log_success "Connected to $TARGET_HOST"

# Step 2: Check current configuration
log_info "Step 2: Checking current NixOS configuration..."
CURRENT_GEN=$($SSH_CMD "readlink /run/current-system | cut -d- -f2")
log_info "Current generation: $CURRENT_GEN"

# Step 3: Rebuild the system
log_info "Step 3: Rebuilding NixOS configuration with exo-intel module..."
log_warn "This may take 10-30 minutes depending on what needs to be built..."

if $SSH_CMD "cd /etc/nixos && nixos-rebuild switch --flake .#gremlin-1" 2>&1 | tee /tmp/nixos-rebuild.log; then
  log_success "System rebuilt successfully"
else
  log_error "System rebuild failed"
  log_error "Check /tmp/nixos-rebuild.log for details"
  exit 1
fi

# Step 4: Verify new generation
NEW_GEN=$($SSH_CMD "readlink /run/current-system | cut -d- -f2")
log_info "New generation: $NEW_GEN"

if [ "$CURRENT_GEN" = "$NEW_GEN" ]; then
  log_warn "Generation unchanged - configuration may already be active"
else
  log_success "System updated to new generation"
fi

# Step 5: Check if exo service is available
log_info "Step 5: Checking for exo systemd service..."
if $SSH_CMD "systemctl list-unit-files | grep -q exo.service"; then
  log_success "exo.service is available"

  # Check service status
  SERVICE_STATUS=$($SSH_CMD "systemctl is-active exo.service 2>/dev/null || echo 'inactive'")
  log_info "Service status: $SERVICE_STATUS"

  if [ "$SERVICE_STATUS" != "active" ]; then
    log_info "Starting exo service..."
    if $SSH_CMD "systemctl start exo.service"; then
      log_success "exo service started"
    else
      log_error "Failed to start exo service"
      $SSH_CMD "journalctl -u exo.service -n 50"
      exit 1
    fi
  fi
else
  log_warn "exo.service not found - may need manual configuration"
  log_info "You can start exo manually with: EXO_TINYGRAD_ENABLED=true exo -vv"
fi

# Step 6: Wait for service to be ready
log_info "Step 6: Waiting for exo API to be ready..."
for i in {1..30}; do
  if curl -s --connect-timeout 2 "http://${TARGET_IP}:52415/health" >/dev/null 2>&1; then
    log_success "exo API is responding"
    break
  fi

  if [ $i -eq 30 ]; then
    log_warn "API not responding after 30 seconds"
    log_info "Check service logs: ssh root@${TARGET_IP} journalctl -u exo.service -f"
  fi

  sleep 1
done

# Step 7: Run validation tests
log_info "Step 7: Running validation tests..."
log_info "=========================================="

if [ -f "tests/validate_gremlin_single_node.sh" ]; then
  ./tests/validate_gremlin_single_node.sh gremlin-1
else
  log_error "Validation script not found: tests/validate_gremlin_single_node.sh"
  exit 1
fi

log_info "=========================================="
log_success "Rebuild and test complete!"
log_info ""
log_info "Next steps:"
log_info "  - Check service logs: ssh root@${TARGET_IP} journalctl -u exo.service -f"
log_info "  - Access dashboard: http://${TARGET_IP}:52415"
log_info "  - Test API: curl http://${TARGET_IP}:52415/v1/models"

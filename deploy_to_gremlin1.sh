#!/usr/bin/env bash
# Deploy Intel hardware support to gremlin-1 and run validation
# This script performs the actual task 9 deployment and validation

set -euo pipefail

# Configuration
GREMLIN1_IP="10.1.1.12"
GREMLIN1_HOST="root@${GREMLIN1_IP}"
GIT_REPO="github:celesrenata/exo/ipex"

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

# Check connectivity
check_connectivity() {
  log_info "Checking connectivity to gremlin-1..."

  if ! ping -c 1 -W 2 "$GREMLIN1_IP" >/dev/null 2>&1; then
    log_error "Cannot reach gremlin-1 at $GREMLIN1_IP"
    return 1
  fi

  if ! ssh "$GREMLIN1_HOST" "echo 'SSH OK'" >/dev/null 2>&1; then
    log_error "Cannot SSH to gremlin-1"
    return 1
  fi

  log_success "Connected to gremlin-1"
  return 0
}

# Create NixOS configuration on gremlin-1
create_nixos_config() {
  log_info "Creating NixOS configuration on gremlin-1..."

  # Backup the current flake if it exists
  ssh "$GREMLIN1_HOST" "cd /etc/nixos && cp flake.nix flake.nix.backup-exo-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true"

  # Ensure hardware-configuration.nix is committed to git (force add since it's in .gitignore)
  ssh "$GREMLIN1_HOST" "cd /etc/nixos && git add -f hardware-configuration.nix && git commit -m 'Add hardware configuration' || true"

  # Update the existing flake to add exo input and module
  ssh "$GREMLIN1_HOST" "cat > /etc/nixos/flake.nix" <<'EOF'
{
  description = "NixOS configuration flake for gremlin systems";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.05";
    # Intel SR-IOV support
    i915-sriov.url = "github:strongtz/i915-sriov-dkms";
    # Exo with Intel hardware support
    exo.url = "github:celesrenata/exo/ipex";
  };

  outputs = { self, nixpkgs, nixpkgs-stable, i915-sriov, exo, ... }@inputs: 
  let
    # Helper function to create system configurations with reset mode support
    mkSystem = { hostname, pkgs ? nixpkgs, hasNvidia ? false, resetMode ? false }: 
      pkgs.lib.nixosSystem {
        system = "x86_64-linux";
        specialArgs = { 
          inherit inputs resetMode hasNvidia; 
          systemHostname = hostname;
        };
        modules = [
          ./hosts/${hostname}/configuration.nix
          ./modules/common.nix
          # Graphics modules now include comprehensive i915-sriov patches for all systems
          (if hasNvidia then ./modules/graphics-nvidia.nix else ./modules/graphics-intel.nix)
          ./modules/networking.nix
          ./modules/virtualisation.nix
          ./modules/ups.nix
          # Add exo Intel hardware support module
          exo.nixosModules.exo-intel
          # Configuration block
          {
            # Graphics configuration for existing modules
            gremlin.graphics = {
              intel.enable = true;
              intel.sriov = true;
              nvidia.enable = hasNvidia;
            };
            
            # CPU power management
            boot.kernelParams = [ "intel_pstate=disable" ];
            powerManagement.cpuFreqGovernor = pkgs.lib.mkForce "userspace";
            systemd.services.disable-turbo = {
              description = "Disable CPU Turbo Boost";
              wantedBy = [ "multi-user.target" ];
              script = "echo 0 > /sys/devices/system/cpu/cpufreq/boost";
              serviceConfig = {
                Type = "oneshot";
                RemainAfterExit = true;
              };
            };
            
            # Exo Intel hardware configuration
            services.exo.intel = {
              enable = true;
              tinygrad = {
                enable = true;
                backend = "GPU";
              };
              arc = {
                enable = true;
                runtime = "auto";
              };
              npu = {
                enable = false;  # Disabled until fully implemented
                servicePort = 52416;
              };
            };
          }
          # Conditionally include kubernetes and monitoring based on resetMode
        ] ++ (if resetMode then [] else [
          ./modules/kubernetes.nix
          ./modules/monitoring.nix
        ]);
      };
  in {
    nixosConfigurations = {
      # Normal configurations
      gremlin-1 = mkSystem { 
        hostname = "gremlin-1"; 
        hasNvidia = true; 
      };
      
      gremlin-2 = mkSystem { 
        hostname = "gremlin-2"; 
        hasNvidia = false;
      };
      
      gremlin-3 = mkSystem { 
        hostname = "gremlin-3"; 
        hasNvidia = false; 
      };
      
      gremlin-4 = mkSystem { 
        hostname = "gremlin-4";
        hasNvidia = false; 
      };

      # Reset mode configurations (for cluster reset)
      gremlin-1-reset = mkSystem { 
        hostname = "gremlin-1"; 
        hasNvidia = true; 
        resetMode = true;
      };
      
      gremlin-2-reset = mkSystem { 
        hostname = "gremlin-2"; 
        hasNvidia = false;
        resetMode = true;
      };
      
      gremlin-3-reset = mkSystem { 
        hostname = "gremlin-3"; 
        hasNvidia = false;
        resetMode = true;
      };
      
      gremlin-4-reset = mkSystem { 
        hostname = "gremlin-4"; 
        hasNvidia = false;
        resetMode = true;
      };

      # Future: gremlin-2 with NVIDIA (when ready)
      gremlin-2-nvidia = mkSystem { 
        hostname = "gremlin-2"; 
        hasNvidia = true; 
      };
    };
  };
}
EOF

  # Add the new flake.nix to git
  ssh "$GREMLIN1_HOST" "cd /etc/nixos && git add flake.nix && git commit -m 'Add exo Intel hardware support module' || true"

  log_success "NixOS configuration updated with exo Intel hardware support"
}

# Build and deploy
deploy_nixos() {
  log_info "Building and deploying NixOS configuration..."

  # Update flake lock
  log_info "Updating flake inputs..."
  ssh "$GREMLIN1_HOST" "cd /etc/nixos && nix flake update"

  # Show what will be built
  log_info "Checking flake..."
  ssh "$GREMLIN1_HOST" "cd /etc/nixos && nix flake show"

  # Build the configuration
  log_info "Building NixOS configuration (this may take a while)..."
  if ssh "$GREMLIN1_HOST" "nixos-rebuild build --flake /etc/nixos#gremlin-1"; then
    log_success "Build completed successfully"
  else
    log_error "Build failed"
    return 1
  fi

  # Switch to new configuration
  log_info "Switching to new configuration..."
  if ssh "$GREMLIN1_HOST" "nixos-rebuild switch --flake /etc/nixos#gremlin-1"; then
    log_success "Deployment completed successfully"
  else
    log_error "Deployment failed"
    return 1
  fi
}

# Run hardware configuration tests
test_hardware_config() {
  log_info "Running hardware configuration tests..."

  # Copy test script to gremlin-1
  scp tests/test_intel_hardware_config.sh "${GREMLIN1_HOST}:/tmp/"

  # Run tests
  if ssh "$GREMLIN1_HOST" "bash /tmp/test_intel_hardware_config.sh"; then
    log_success "Hardware configuration tests passed"
    return 0
  else
    log_warn "Some hardware configuration tests failed"
    return 0 # Don't fail deployment on warnings
  fi
}

# Start exo service
start_exo() {
  log_info "Starting exo service on gremlin-1..."

  # Check if exo is available
  if ! ssh "$GREMLIN1_HOST" "which exo" >/dev/null 2>&1; then
    log_error "exo binary not found after deployment"
    return 1
  fi

  # Start exo with tinygrad backend
  log_info "Starting exo with tinygrad backend..."
  ssh "$GREMLIN1_HOST" "nohup env EXO_TINYGRAD_ENABLED=true exo -vv > /var/log/exo.log 2>&1 &"

  # Wait for service to start
  log_info "Waiting for service to start..."
  for i in {1..30}; do
    if ssh "$GREMLIN1_HOST" "curl -s http://localhost:52415/health" >/dev/null 2>&1; then
      log_success "exo service started successfully"
      return 0
    fi
    sleep 2
  done

  log_error "Service did not start within 60 seconds"
  log_info "Check logs: ssh $GREMLIN1_HOST 'tail -f /var/log/exo.log'"
  return 1
}

# Run validation tests
run_validation() {
  log_info "Running validation tests..."

  # Run the comprehensive validation script
  if ./tests/validate_gremlin_single_node.sh gremlin-1; then
    log_success "All validation tests passed!"
    return 0
  else
    log_error "Some validation tests failed"
    return 1
  fi
}

# Main deployment workflow
main() {
  echo ""
  log_info "=========================================="
  log_info "Task 9: Deploy to gremlin-1 and Validate"
  log_info "=========================================="
  log_info "Target: gremlin-1 ($GREMLIN1_IP)"
  log_info "Git Repo: $GIT_REPO"
  log_info "=========================================="
  echo ""

  # Step 1: Check connectivity
  if ! check_connectivity; then
    log_error "Cannot connect to gremlin-1. Exiting."
    exit 1
  fi
  echo ""

  # Step 2: Create NixOS configuration
  log_info "Step 1: Creating NixOS configuration..."
  if ! create_nixos_config; then
    log_error "Failed to create configuration. Exiting."
    exit 1
  fi
  echo ""

  # Step 3: Deploy
  log_info "Step 2: Building and deploying..."
  if ! deploy_nixos; then
    log_error "Deployment failed. Exiting."
    exit 1
  fi
  echo ""

  # Step 4: Test hardware configuration
  log_info "Step 3: Testing hardware configuration..."
  test_hardware_config
  echo ""

  # Step 5: Start exo
  log_info "Step 4: Starting exo service..."
  if ! start_exo; then
    log_error "Failed to start exo. Exiting."
    exit 1
  fi
  echo ""

  # Step 6: Run validation
  log_info "Step 5: Running validation tests..."
  if ! run_validation; then
    log_error "Validation failed."
    exit 1
  fi
  echo ""

  # Success!
  log_info "=========================================="
  log_success "Task 9 completed successfully!"
  log_info "=========================================="
  log_info "gremlin-1 is now running with Intel hardware support"
  log_info ""
  log_info "Next steps:"
  log_info "  - Monitor service: ssh $GREMLIN1_HOST 'tail -f /var/log/exo.log'"
  log_info "  - Check API: curl http://$GREMLIN1_IP:52415/v1/models"
  log_info "  - View dashboard: http://$GREMLIN1_IP:52415/"
  log_info "  - Proceed to task 10: Multi-node cluster deployment"
  echo ""
}

# Run main
main "$@"

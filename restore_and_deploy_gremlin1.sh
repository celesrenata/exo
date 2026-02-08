#!/usr/bin/env bash
# Restore original flake and add exo-intel module properly

set -euo pipefail

GREMLIN1_HOST="root@10.1.1.12"

echo "=== Restoring and deploying exo-intel to gremlin-1 ==="
echo ""

# Step 1: Update the flake to add exo-intel module
echo "Step 1: Adding exo-intel module to existing flake..."
ssh "$GREMLIN1_HOST" "cat > /etc/nixos/flake.nix" <<'EOF'
{
  description = "NixOS configuration flake for gremlin systems";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.11";
    exo.url = "github:celesrenata/exo/ipex";
  };

  outputs = { self, exo, nixpkgs, nixpkgs-stable, ... }@inputs: 
  let
    # Helper function to create system configurations
    mkSystem = { hostname, pkgs ? nixpkgs, intel ? true, nvidia ? false, sriov ? true, resetMode ? false, exoIntel ? false }: 
      pkgs.lib.nixosSystem {
        system = "x86_64-linux";
        specialArgs = { 
          inherit inputs resetMode nixpkgs-stable; 
          systemHostname = hostname;
        };
        modules = [
          { nixpkgs.config.allowUnfree = true; }
          ./hosts/${hostname}/configuration.nix
          ./modules/common.nix
          ./modules/graphics.nix
          ./modules/networking.nix
          ./modules/virtualisation.nix
          ./modules/ups.nix 
          {
            gremlin.graphics = {
              intel.enable = intel;
              intel.sriov = intel && sriov;
              nvidia.enable = nvidia;
            };
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
          }
          # External Modules
          exo.nixosModules.default
          # Conditionally include kubernetes and monitoring based on resetMode
        ] ++ (if resetMode then [] else [
          ./modules/kubernetes.nix
          ./modules/monitoring.nix
        ]) ++ (if exoIntel then [
          # Add exo Intel hardware support
          exo.nixosModules.exo-intel
          {
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
                enable = false;
                servicePort = 52416;
              };
            };
          }
        ] else []);
      };
  in {
    nixosConfigurations = {
      # Normal configurations
      gremlin-1 = mkSystem { 
        hostname = "gremlin-1"; 
        intel = true;
        nvidia = true;
        sriov = true;
        exoIntel = true;  # Enable exo Intel support
      };
      
      gremlin-2 = mkSystem { 
        hostname = "gremlin-2"; 
        intel = true;
        nvidia = false;
        sriov = true;
      };
      
      gremlin-3 = mkSystem { 
        hostname = "gremlin-3"; 
        intel = true;
        nvidia = false;
        sriov = true;
      };
      
      gremlin-4 = mkSystem { 
        hostname = "gremlin-4";
        intel = true;
        nvidia = false;
        sriov = true;
      };

      # Reset mode configurations (for cluster reset)
      gremlin-1-reset = mkSystem { 
        hostname = "gremlin-1"; 
        intel = true;
        nvidia = true;
        sriov = true;
        resetMode = true;
      };
      
      gremlin-2-reset = mkSystem { 
        hostname = "gremlin-2"; 
        intel = true;
        nvidia = false;
        sriov = true;
        resetMode = true;
      };
      
      gremlin-3-reset = mkSystem { 
        hostname = "gremlin-3"; 
        intel = true;
        nvidia = false;
        sriov = true;
        resetMode = true;
      };
      
      gremlin-4-reset = mkSystem { 
        hostname = "gremlin-4"; 
        intel = true;
        nvidia = false;
        sriov = true;
        resetMode = true;
      };

      # Future: gremlin-2 with NVIDIA (when ready)
      gremlin-2-nvidia = mkSystem { 
        hostname = "gremlin-2"; 
        intel = true;
        nvidia = true;
        sriov = true;
      };
    };
  };
}
EOF

echo "✓ Flake updated"
echo ""

# Step 2: Commit the change
echo "Step 2: Committing changes..."
ssh "$GREMLIN1_HOST" "cd /etc/nixos && git add flake.nix && git commit -m 'Add exo-intel module to gremlin-1 configuration'"
echo "✓ Changes committed"
echo ""

# Step 3: Update flake inputs
echo "Step 3: Updating flake inputs..."
ssh "$GREMLIN1_HOST" "cd /etc/nixos && nix flake update"
echo "✓ Flake inputs updated"
echo ""

# Step 4: Build (don't switch yet)
echo "Step 4: Building configuration..."
if ssh "$GREMLIN1_HOST" "nixos-rebuild build --flake /etc/nixos#gremlin-1"; then
    echo "✓ Build successful"
    echo ""
    echo "Configuration built successfully!"
    echo ""
    echo "To apply the configuration, run:"
    echo "  ssh root@10.1.1.12 'nixos-rebuild switch --flake /etc/nixos#gremlin-1'"
    echo ""
    echo "Or to test it first:"
    echo "  ssh root@10.1.1.12 'nixos-rebuild test --flake /etc/nixos#gremlin-1'"
else
    echo "✗ Build failed"
    exit 1
fi

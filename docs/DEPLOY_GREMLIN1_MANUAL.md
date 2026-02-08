# Manual Deployment to gremlin-1

This guide provides step-by-step instructions for deploying Intel hardware support to gremlin-1 and running task 9 validation.

## Prerequisites

- Code committed and pushed to `github:celesrenata/exo/ipex`
- SSH access to gremlin-1 (root@10.1.1.12)
- gremlin-1 running NixOS

## Step 1: Create NixOS Configuration on gremlin-1

SSH to gremlin-1:

```bash
ssh root@10.1.1.12
```

Create `/etc/nixos/flake.nix`:

```bash
cat > /etc/nixos/flake.nix << 'EOF'
{
  description = "gremlin-1 NixOS configuration with Intel hardware support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:celesrenata/exo/ipex";
  };

  outputs = { self, nixpkgs, exo, ... }: {
    nixosConfigurations.gremlin-1 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./hardware-configuration.nix
        exo.nixosModules.exo-intel
        {
          networking.hostName = "gremlin-1";
          
          # Enable Intel hardware support
          services.exo.intel = {
            enable = true;
            
            arc = {
              enable = true;
              runtime = "auto";  # Auto-detect Level Zero or OpenCL
            };
            
            npu = {
              enable = true;  # Experimental NPU support
              servicePort = 52416;
            };
          };
          
          # Additional packages for testing
          environment.systemPackages = with nixpkgs.legacyPackages.x86_64-linux; [
            intel-gpu-tools
            clinfo
            pciutils
            usbutils
          ];
          
          # Ensure graphics support
          hardware.graphics.enable = true;
        }
      ];
    };
  };
}
EOF
```

## Step 2: Update Flake Inputs

```bash
cd /etc/nixos
nix flake update
```

Verify the flake:

```bash
nix flake show
```

You should see:

```
github:celesrenata/exo/ipex
└───nixosModules
    └───exo-intel: NixOS module
```

## Step 3: Build Configuration

Build first to check for errors:

```bash
nixos-rebuild build --flake /etc/nixos#gremlin-1
```

This will download and build all dependencies. It may take 10-20 minutes.

## Step 4: Deploy Configuration

If build succeeds, switch to the new configuration:

```bash
nixos-rebuild switch --flake /etc/nixos#gremlin-1
```

## Step 5: Verify Installation

Check that exo is available:

```bash
which exo
exo --version
```

Check tinygrad:

```bash
python3 -c "import tinygrad; print(tinygrad.__version__)"
```

Check hardware:

```bash
# GPU
lspci | grep Intel
ls -la /dev/dri/

# Level Zero
ls -la /run/opengl-driver/lib/libze_loader.so*

# OpenCL
clinfo | grep Intel

# NPU (optional)
ls -la /dev/accel/
lsmod | grep vpu
```

## Step 6: Start exo Service

Start exo with tinygrad backend:

```bash
# Start in background
nohup env EXO_TINYGRAD_ENABLED=true exo -vv > /var/log/exo.log 2>&1 &

# Check it started
curl http://localhost:52415/health

# View logs
tail -f /var/log/exo.log
```

## Step 7: Run Validation from esnixi

From your development machine (esnixi), run the validation:

```bash
# From esnixi
./tests/validate_gremlin_single_node.sh gremlin-1
```

This will run all 7 validation subtasks:
- 9.1: Build verification
- 9.2: Service startup
- 9.3: API endpoints
- 9.4: GPU detection
- 9.5: NPU detection
- 9.6: Model download
- 9.7: Inference test

## Expected Results

All tests should pass:

```
[INFO] ==========================================
[INFO] Validation Summary
[INFO] ==========================================
Tests Passed: 25+
Tests Warned: 2-3 (NPU is optional)
Tests Failed: 0

[SUCCESS] All validation tasks completed successfully! ✓
[INFO] gremlin-1 is ready for deployment
```

## Troubleshooting

### Build Fails

```bash
# Check flake syntax
nix flake check /etc/nixos

# Try updating nixpkgs
nix flake update nixpkgs

# Check for specific errors
nixos-rebuild build --flake /etc/nixos#gremlin-1 --show-trace
```

### exo Not Found

```bash
# Check if it's in the path
echo $PATH

# Try running directly
/nix/store/*/bin/exo --version

# Rebuild
nixos-rebuild switch --flake /etc/nixos#gremlin-1
```

### Service Won't Start

```bash
# Check logs
tail -f /var/log/exo.log

# Check dependencies
python3 -c "import tinygrad, numpy"

# Check GPU access
ls -la /dev/dri/
groups  # Should include 'video'

# Try running in foreground
EXO_TINYGRAD_ENABLED=true exo -vv
```

### GPU Not Detected

```bash
# Check hardware
lspci | grep Intel

# Check drivers
ls -la /run/opengl-driver/lib/

# Load kernel module
modprobe i915

# Check Level Zero
python3 << EOF
import os
os.environ['GPU'] = '1'
os.environ['LEVEL_ZERO'] = '1'
from tinygrad import Device
Device.DEFAULT = 'GPU'
print("Level Zero OK")
EOF
```

### Validation Fails

```bash
# Run individual tests
ssh root@10.1.1.12 "which exo"
curl http://10.1.1.12:52415/health
curl http://10.1.1.12:52415/v1/models

# Check service is running
ssh root@10.1.1.12 "ps aux | grep exo"

# Check logs
ssh root@10.1.1.12 "tail -100 /var/log/exo.log"
```

## Rollback

If something goes wrong:

```bash
# On gremlin-1
nixos-rebuild switch --rollback

# Or boot into previous generation
nixos-rebuild boot --rollback
reboot
```

## Alternative: Automated Deployment

You can also use the automated deployment script:

```bash
# From esnixi
./deploy_to_gremlin1.sh
```

This script automates all the steps above.

## Next Steps

After successful validation:

1. Document performance metrics
2. Run extended stability tests
3. Proceed to task 10: Multi-node cluster deployment
4. Deploy to gremlin-2, gremlin-3, gremlin-4

## Quick Commands Reference

```bash
# SSH to gremlin-1
ssh root@10.1.1.12

# Check service
curl http://10.1.1.12:52415/health

# View logs
ssh root@10.1.1.12 "tail -f /var/log/exo.log"

# Run validation
./tests/validate_gremlin_single_node.sh gremlin-1

# Stop service
ssh root@10.1.1.12 "pkill exo"

# Restart service
ssh root@10.1.1.12 "nohup env EXO_TINYGRAD_ENABLED=true exo -vv > /var/log/exo.log 2>&1 &"
```

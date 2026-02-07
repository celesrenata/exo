# gremlin-1 Deployment Instructions

Manual deployment instructions for Intel hardware support on gremlin-1.

## Prerequisites

- SSH access to root@gremlin-1 (10.1.1.12)
- Git changes committed and pushed to `github:celesrenata/exo/ipex`
- gremlin-1 running NixOS

## Step 1: Copy Flake Configuration to gremlin-1

On gremlin-1, create the flake configuration:

```bash
# SSH to gremlin-1
ssh root@10.1.1.12

# Backup existing configuration if it exists
cp /etc/nixos/flake.nix /etc/nixos/flake.nix.backup 2>/dev/null || true

# Create new flake.nix
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
          
          # Additional packages for testing and monitoring
          environment.systemPackages = with pkgs; [
            intel-gpu-tools  # intel_gpu_top for GPU monitoring
            clinfo           # OpenCL info
            pciutils         # lspci
            usbutils         # lsusb
            htop             # System monitoring
            curl             # API testing
            jq               # JSON parsing
          ];
          
          # Ensure graphics support is enabled
          hardware.graphics.enable = true;
          
          # Allow unfree packages if needed
          nixpkgs.config.allowUnfree = true;
        }
      ];
    };
  };
}
EOF
```

## Step 2: Update Flake Inputs

```bash
# Still on gremlin-1
cd /etc/nixos

# Update flake lock to get latest from git
nix flake update

# Verify the flake is valid
nix flake show

# Check what will be built
nix flake check
```

## Step 3: Build the Configuration

```bash
# Build without switching (test first)
nixos-rebuild build --flake /etc/nixos#gremlin-1

# If build succeeds, check what changed
nix store diff-closures /run/current-system ./result
```

## Step 4: Deploy the Configuration

```bash
# Switch to new configuration
nixos-rebuild switch --flake /etc/nixos#gremlin-1

# Verify exo is available
which exo
exo --version

# Check tinygrad
python3 -c "import tinygrad; print(f'tinygrad {tinygrad.__version__}')"
```

## Step 5: Verify Hardware Configuration

```bash
# Check Intel GPU
lspci | grep -i "VGA.*Intel"
ls -la /dev/dri/

# Check Level Zero
ls -la /run/opengl-driver/lib/libze_loader.so*

# Check OpenCL
clinfo | grep -i intel

# Check NPU (if available)
ls -la /dev/accel/
lsmod | grep -E "intel_vpu|ivpu"
```

## Step 6: Start exo Service

```bash
# Start exo with tinygrad backend
EXO_TINYGRAD_ENABLED=true exo -vv > /var/log/exo.log 2>&1 &

# Save the PID
echo $! > /var/run/exo.pid

# Wait for service to start
sleep 5

# Check if running
curl http://localhost:52415/health
```

## Step 7: Run Validation Tests

From your local machine (esnixi), run the validation:

```bash
# Run comprehensive validation
./tests/validate_gremlin_single_node.sh gremlin-1
```

Or manually test from gremlin-1:

```bash
# On gremlin-1

# Test health endpoint
curl http://localhost:52415/health

# Test models endpoint
curl http://localhost:52415/v1/models | jq .

# Test inference
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10,
    "stream": false
  }' | jq .
```

## Step 8: Monitor GPU Usage

```bash
# On gremlin-1

# Monitor GPU in real-time
intel_gpu_top

# Check logs
tail -f /var/log/exo.log

# Check system resources
htop
```

## Troubleshooting

### Build Fails

```bash
# Check flake syntax
nix flake check

# Update nixpkgs
nix flake update nixpkgs

# Try with verbose output
nixos-rebuild build --flake /etc/nixos#gremlin-1 --show-trace
```

### Service Won't Start

```bash
# Check logs
tail -n 100 /var/log/exo.log

# Check if port is in use
netstat -tlnp | grep 52415

# Kill existing process
pkill -9 exo

# Restart
EXO_TINYGRAD_ENABLED=true exo -vv > /var/log/exo.log 2>&1 &
```

### GPU Not Detected

```bash
# Check kernel module
lsmod | grep i915

# Load if needed
modprobe i915

# Check device permissions
ls -la /dev/dri/
groups  # Should include 'video'
```

### Rollback if Needed

```bash
# Rollback to previous configuration
nixos-rebuild switch --rollback

# Or boot into previous generation
nixos-rebuild boot --rollback
reboot
```

## Validation Checklist

- [ ] Flake configuration created at `/etc/nixos/flake.nix`
- [ ] Flake inputs updated (`nix flake update`)
- [ ] Configuration builds successfully
- [ ] Configuration deployed (`nixos-rebuild switch`)
- [ ] exo binary available (`which exo`)
- [ ] tinygrad package available
- [ ] Intel GPU detected
- [ ] Level Zero or OpenCL available
- [ ] exo service starts without errors
- [ ] Health endpoint responds
- [ ] API endpoints accessible
- [ ] Model downloads successfully
- [ ] Inference works correctly
- [ ] GPU is being used (not CPU fallback)

## Next Steps

After successful validation:

1. Document any issues encountered
2. Record performance metrics
3. Let service run for stability testing (24+ hours)
4. Proceed to task 10: Multi-node cluster deployment

## Quick Commands Reference

```bash
# On gremlin-1
cd /etc/nixos
nix flake update
nixos-rebuild switch --flake /etc/nixos#gremlin-1
EXO_TINYGRAD_ENABLED=true exo -vv > /var/log/exo.log 2>&1 &
curl http://localhost:52415/health

# From esnixi
./tests/validate_gremlin_single_node.sh gremlin-1
```

## Files Reference

- Configuration: `/etc/nixos/flake.nix` (on gremlin-1)
- Logs: `/var/log/exo.log` (on gremlin-1)
- Validation script: `./tests/validate_gremlin_single_node.sh` (on esnixi)
- Git repository: `github:celesrenata/exo/ipex`

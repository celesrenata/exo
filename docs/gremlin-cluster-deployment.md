# Gremlin Cluster Deployment Guide

This guide covers deploying the Intel hardware support to the gremlin cluster (gremlin-1 through gremlin-4).

## Overview

The gremlin cluster consists of 4 Intel Core Ultra nodes:
- **gremlin-1**: 10.1.1.12
- **gremlin-2**: 10.1.1.13
- **gremlin-3**: 10.1.1.14
- **gremlin-4**: 10.1.1.15

Each node has:
- Intel Core Ultra 9 185H processor
- Intel Arc iGPU
- Intel NPU
- 32GB RAM

## Prerequisites

1. SSH access to all gremlin nodes as root
2. Git repository with exo Intel hardware support
3. NixOS installed on all nodes
4. Network connectivity between nodes

## Deployment Steps

### Step 1: Prepare Git Repository

Ensure all changes are committed and pushed:

```bash
# On development machine (esnixi)
cd /path/to/exo

# Check status
git status

# Commit any pending changes
git add .
git commit -m "Intel hardware support implementation"

# Tag release
git tag -a v0.1.0-intel -m "Intel hardware support release"

# Push to remote
git push origin ipex
git push origin v0.1.0-intel
```

### Step 2: Test on gremlin-1 (Single Node)

First, test on gremlin-1 with local flake:

```bash
# SSH to gremlin-1
ssh root@10.1.1.12

# If using local drive mapping, test first
cd /path/to/exo
nix build .#exo

# Start exo
EXO_TINYGRAD_ENABLED=true exo
```

Run validation tests from esnixi:

```bash
# From esnixi (192.168.42.254)
./tests/test_gremlin_single_node.sh gremlin-1
```

### Step 3: Update gremlin-1 to Use Git Flake

Once local testing passes, update to use git flake:

#### Option A: GitHub Repository

```nix
# /etc/nixos/configuration.nix on gremlin-1
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:exo-explore/exo/ipex";  # Update with your repo
  };

  outputs = { nixpkgs, exo, ... }: {
    nixosConfigurations.gremlin-1 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./hardware-configuration.nix
        exo.nixosModules.exo-intel
        {
          networking.hostName = "gremlin-1";
          
          services.exo.intel = {
            enable = true;
            arc = {
              enable = true;
              runtime = "auto";
            };
            npu = {
              enable = true;
              servicePort = 52416;
            };
          };
        }
      ];
    };
  };
}
```

#### Option B: Local Git Repository

If using a local git server or shared filesystem:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "git+file:///shared/exo?ref=ipex";  # Local git repo
  };
  # ... rest same as above
}
```

#### Rebuild gremlin-1

```bash
# On gremlin-1
nixos-rebuild switch

# Verify
systemctl status exo  # If using systemd service
# OR
EXO_TINYGRAD_ENABLED=true exo  # Manual start

# Test from esnixi
./tests/test_gremlin_single_node.sh gremlin-1
```

### Step 4: Deploy to gremlin-2, 3, 4

Once gremlin-1 is working with git flake, deploy to other nodes:

#### Update Configuration for Each Node

Create or update `/etc/nixos/configuration.nix` on each node:

**gremlin-2** (10.1.1.13):
```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:exo-explore/exo/ipex";
  };

  outputs = { nixpkgs, exo, ... }: {
    nixosConfigurations.gremlin-2 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./hardware-configuration.nix
        exo.nixosModules.exo-intel
        {
          networking.hostName = "gremlin-2";
          
          services.exo.intel = {
            enable = true;
            arc.enable = true;
            npu.enable = true;
          };
        }
      ];
    };
  };
}
```

**gremlin-3** (10.1.1.14) and **gremlin-4** (10.1.1.15): Same as gremlin-2, just change hostname.

#### Deploy Script

Use this script to deploy to all nodes:

```bash
#!/usr/bin/env bash
# deploy_gremlin_cluster.sh

NODES=("gremlin-2:10.1.1.13" "gremlin-3:10.1.1.14" "gremlin-4:10.1.1.15")

for node_info in "${NODES[@]}"; do
    NODE_NAME=$(echo "$node_info" | cut -d: -f1)
    NODE_IP=$(echo "$node_info" | cut -d: -f2)
    
    echo "Deploying to $NODE_NAME ($NODE_IP)..."
    
    # Copy configuration
    scp /path/to/configuration.nix "root@${NODE_IP}:/etc/nixos/"
    
    # Rebuild
    ssh "root@${NODE_IP}" "nixos-rebuild switch"
    
    # Verify
    ssh "root@${NODE_IP}" "systemctl status exo || echo 'Service not enabled'"
    
    echo "$NODE_NAME deployed successfully"
    echo ""
done

echo "All nodes deployed!"
```

### Step 5: Start exo on All Nodes

Start exo service on each node:

```bash
# On each gremlin node
ssh root@10.1.1.12 "EXO_TINYGRAD_ENABLED=true exo &"
ssh root@10.1.1.13 "EXO_TINYGRAD_ENABLED=true exo &"
ssh root@10.1.1.14 "EXO_TINYGRAD_ENABLED=true exo &"
ssh root@10.1.1.15 "EXO_TINYGRAD_ENABLED=true exo &"
```

Or if using systemd service:

```bash
ssh root@10.1.1.12 "systemctl start exo"
ssh root@10.1.1.13 "systemctl start exo"
ssh root@10.1.1.14 "systemctl start exo"
ssh root@10.1.1.15 "systemctl start exo"
```

### Step 6: Verify Cluster Formation

Check that nodes discover each other:

```bash
# Check logs on each node
ssh root@10.1.1.12 "journalctl -u exo -f | grep -i 'peer\|discover\|cluster'"

# Check cluster status via API
curl http://10.1.1.12:52415/cluster

# Check dashboard
open http://10.1.1.12:52415/
```

### Step 7: Run Cluster Tests

From esnixi, run the cluster validation tests:

```bash
# Quick test (5 minutes)
./tests/test_gremlin_cluster.sh 5

# Standard test (60 minutes)
./tests/test_gremlin_cluster.sh 60

# Extended test (4 hours)
./tests/test_gremlin_cluster.sh 240
```

## Troubleshooting

### Node Not Accessible

```bash
# Check if node is up
ping 10.1.1.12

# Check if exo is running
ssh root@10.1.1.12 "ps aux | grep exo"

# Check logs
ssh root@10.1.1.12 "journalctl -u exo -n 100"
```

### Nodes Not Discovering Each Other

```bash
# Check network connectivity
ssh root@10.1.1.12 "ping 10.1.1.13"

# Check firewall
ssh root@10.1.1.12 "iptables -L"

# Check libp2p ports
ssh root@10.1.1.12 "netstat -tlnp | grep exo"
```

### GPU Not Detected

```bash
# Check GPU device
ssh root@10.1.1.12 "ls -la /dev/dri/"

# Check Level Zero
ssh root@10.1.1.12 "python -c 'from tinygrad import Device; Device.DEFAULT=\"GPU\"'"

# Check OpenCL
ssh root@10.1.1.12 "clinfo | grep Intel"
```

### NPU Not Detected

```bash
# Check NPU device
ssh root@10.1.1.12 "ls -la /dev/accel/"

# Check kernel module
ssh root@10.1.1.12 "lsmod | grep intel_vpu"

# Load module if needed
ssh root@10.1.1.12 "modprobe intel_vpu"
```

### Build Failures

```bash
# Check nix build
ssh root@10.1.1.12 "nix build github:exo-explore/exo/ipex#exo"

# Check flake
ssh root@10.1.1.12 "nix flake show github:exo-explore/exo/ipex"

# Update flake lock
ssh root@10.1.1.12 "cd /etc/nixos && nix flake update"
```

## Monitoring

### Check Cluster Health

```bash
# From esnixi
for ip in 10.1.1.{12..15}; do
    echo "Checking gremlin node at $ip..."
    curl -s "http://${ip}:52415/health" | jq .
done
```

### Monitor Performance

```bash
# Check GPU usage on each node
ssh root@10.1.1.12 "intel_gpu_top"

# Check system resources
ssh root@10.1.1.12 "htop"

# Check exo metrics
curl http://10.1.1.12:52415/metrics
```

### View Logs

```bash
# Real-time logs
ssh root@10.1.1.12 "journalctl -u exo -f"

# Recent errors
ssh root@10.1.1.12 "journalctl -u exo -p err -n 50"

# Search logs
ssh root@10.1.1.12 "journalctl -u exo | grep -i 'error\|fail\|crash'"
```

## Rollback

If deployment fails, rollback to previous configuration:

```bash
# On each node
ssh root@10.1.1.12 "nixos-rebuild switch --rollback"

# Or boot into previous generation
ssh root@10.1.1.12 "nixos-rebuild boot --rollback && reboot"
```

## Next Steps

After successful deployment:

1. Run extended stability tests (24+ hours)
2. Benchmark performance across cluster
3. Test model sharding with large models
4. Monitor for memory leaks or crashes
5. Document any issues or optimizations

## Configuration Examples

### Minimal Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    arc.enable = true;
  };
}
```

### Full Configuration

```nix
{
  services.exo.intel = {
    enable = true;
    
    arc = {
      enable = true;
      runtime = "level-zero";  # or "opencl" or "auto"
    };
    
    npu = {
      enable = true;
      servicePort = 52416;
    };
  };
  
  # Optional: Add monitoring tools
  environment.systemPackages = with pkgs; [
    intel-gpu-tools
    clinfo
  ];
}
```

## References

- [Intel Hardware Setup Guide](intel-hardware-setup.md)
- [Single Node Test Script](../tests/test_gremlin_single_node.sh)
- [Cluster Test Script](../tests/test_gremlin_cluster.sh)
- [NixOS Configuration Example](examples/nixos-intel-config.nix)

# EXO NixOS Installation Guide

This guide provides step-by-step instructions for installing and configuring EXO distributed AI inference system on NixOS using the official flake.

## Prerequisites

### System Requirements

- **Operating System**: NixOS 23.05 or later
- **Architecture**: x86_64-linux or aarch64-linux
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large models)
- **Storage**: 50GB+ free space for model cache
- **Network**: Ethernet connection (bonded interfaces recommended for multi-node setups)

### Hardware Requirements

EXO supports various hardware accelerators:

- **NVIDIA GPUs**: GTX 1060 or newer, RTX series recommended
- **AMD GPUs**: RX 6000 series or newer with ROCm support
- **Intel Arc GPUs**: A-series with IPEX support
- **Apple Silicon**: M1/M2/M3 with MLX framework (macOS only)
- **CPU-only**: Any modern x86_64 or ARM64 processor

### Network Requirements

- **Single Node**: Standard Ethernet connection
- **Multi-Node Cluster**: 
  - Gigabit Ethernet minimum (10GbE recommended)
  - Bonded interfaces for high bandwidth
  - RDMA over Thunderbolt 5 (optional, for ultra-low latency)

## Installation Methods

### Method 1: Flake Import (Recommended)

This is the recommended method for most users as it provides automatic updates and easy configuration management.

#### Step 1: Add Flake Input

Add the EXO flake to your system's `flake.nix`:

```nix
{
  description = "My NixOS Configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    
    # Add EXO flake
    exo = {
      url = "github:exo-explore/exo";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, exo, ... }: {
    nixosConfigurations.your-hostname = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";  # or "aarch64-linux"
      modules = [
        ./configuration.nix
        # Import EXO NixOS module
        exo.nixosModules.default
      ];
    };
  };
}
```

#### Step 2: Configure EXO Service

Add EXO configuration to your `configuration.nix`:

```nix
{ config, pkgs, ... }:

{
  # Enable EXO service
  services.exo = {
    enable = true;
    mode = "auto";  # Automatically determine master/worker role
    
    # Network configuration
    networking = {
      # Use existing bonded interface (optional)
      bondInterface = "bond0";
      rdmaEnabled = true;
      openFirewall = true;
    };
    
    # Hardware configuration
    hardware = {
      autoDetect = true;  # Automatically detect GPU drivers
      memoryLimit = "80%";  # Use up to 80% of system memory
    };
    
    # Dashboard configuration
    dashboard = {
      enable = true;
      port = 8080;  # Web interface port
    };
  };
}
```

#### Step 3: Rebuild System

Apply the configuration:

```bash
sudo nixos-rebuild switch --flake .#your-hostname
```

### Method 2: Direct Package Installation

For users who prefer manual package management:

#### Step 1: Install Packages

```bash
# Install EXO with automatic hardware detection
nix profile install github:exo-explore/exo#exo-complete

# Or install specific hardware variant:
# nix profile install github:exo-explore/exo#exo-cuda    # NVIDIA GPUs
# nix profile install github:exo-explore/exo#exo-rocm    # AMD GPUs
# nix profile install github:exo-explore/exo#exo-mlx     # Apple Silicon
# nix profile install github:exo-explore/exo#exo-cpu     # CPU-only
```

#### Step 2: Manual Service Setup

Create systemd service files manually (not recommended - use flake method instead).

## Configuration Options

### Basic Configuration

```nix
services.exo = {
  enable = true;
  mode = "auto";  # "master", "worker", or "auto"
  apiPort = 52415;  # API server port
  
  # Package selection (usually auto-detected)
  package = pkgs.exo-complete;  # or exo-cuda, exo-rocm, etc.
};
```

### Network Configuration

```nix
services.exo.networking = {
  # Use specific network interface
  bondInterface = "bond0";  # null to auto-detect
  
  # Enable RDMA over Thunderbolt
  rdmaEnabled = true;
  
  # Discovery and communication ports
  discoveryPort = 52416;
  apiPort = 52415;
  
  # Firewall configuration
  openFirewall = true;  # Automatically open required ports
  
  # Advanced networking
  networkNamespace = null;  # Optional network isolation
  bindAddress = "0.0.0.0";  # Bind to all interfaces
};
```

### Hardware Configuration

```nix
services.exo.hardware = {
  # Automatic hardware detection
  autoDetect = true;
  
  # Override automatic detection
  preferredAccelerator = null;  # "cuda", "rocm", "mlx", "cpu"
  
  # Memory management
  memoryLimit = "80%";  # Percentage or absolute value like "32GB"
  
  # GPU-specific settings
  cuda = {
    enable = true;  # Auto-detected if null
    devices = [ 0 1 ];  # GPU device indices, null for all
  };
  
  rocm = {
    enable = true;  # Auto-detected if null
    devices = [ 0 1 ];  # GPU device indices, null for all
  };
};
```

### Security Configuration

```nix
services.exo.security = {
  # Service user and group
  user = "exo";
  group = "exo";
  
  # Data directories
  dataDir = "/var/lib/exo";
  configDir = "/etc/exo";
  cacheDir = "/var/cache/exo";
  
  # Systemd sandboxing
  sandboxing = {
    enable = true;
    privateNetwork = false;  # Set to true for network isolation
    protectSystem = "strict";
    protectHome = true;
  };
};
```

### Dashboard Configuration

```nix
services.exo.dashboard = {
  enable = true;
  port = 8080;
  
  # SSL/TLS configuration
  ssl = {
    enable = false;  # Set to true for HTTPS
    certificatePath = "/path/to/cert.pem";
    keyPath = "/path/to/key.pem";
  };
  
  # Authentication (optional)
  auth = {
    enable = false;
    method = "basic";  # "basic" or "oauth"
    users = {
      admin = "password_hash";
    };
  };
};
```

### Logging Configuration

```nix
services.exo.logging = {
  level = "info";  # "debug", "info", "warn", "error"
  
  # Systemd journal integration
  journal = {
    enable = true;
    structured = true;  # JSON structured logging
  };
  
  # File logging (optional)
  file = {
    enable = false;
    path = "/var/log/exo/exo.log";
    maxSize = "100MB";
    maxFiles = 10;
  };
  
  # Performance monitoring
  metrics = {
    enable = true;
    prometheus = {
      enable = false;
      port = 9090;
    };
  };
};
```

## Post-Installation Setup

### Verify Installation

1. **Check Service Status**:
   ```bash
   sudo systemctl status exo-master exo-worker exo-api
   ```

2. **View Logs**:
   ```bash
   sudo journalctl -u exo-master -f
   ```

3. **Test API**:
   ```bash
   curl http://localhost:52415/v1/models
   ```

4. **Access Dashboard**:
   Open `http://localhost:8080` in your browser

### Initial Configuration

1. **Set Model Cache Location** (optional):
   ```bash
   sudo mkdir -p /var/cache/exo/models
   sudo chown exo:exo /var/cache/exo/models
   ```

2. **Configure Firewall** (if not using `openFirewall = true`):
   ```bash
   sudo firewall-cmd --permanent --add-port=52415/tcp  # API
   sudo firewall-cmd --permanent --add-port=52416/tcp  # Discovery
   sudo firewall-cmd --permanent --add-port=8080/tcp   # Dashboard
   sudo firewall-cmd --reload
   ```

3. **Test GPU Detection**:
   ```bash
   # Check detected hardware
   sudo -u exo exo --list-devices
   
   # Test inference
   curl -X POST http://localhost:52415/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "llama-3.2-1b", "messages": [{"role": "user", "content": "Hello!"}]}'
   ```

## Troubleshooting

### Common Issues

#### Service Won't Start

1. **Check system requirements**:
   ```bash
   # Verify NixOS version
   nixos-version
   
   # Check available memory
   free -h
   
   # Check disk space
   df -h /var/lib/exo
   ```

2. **Check configuration syntax**:
   ```bash
   sudo nixos-rebuild dry-build --flake .#your-hostname
   ```

3. **View detailed logs**:
   ```bash
   sudo journalctl -u exo-master -n 50
   ```

#### GPU Not Detected

1. **Verify GPU drivers**:
   ```bash
   # NVIDIA
   nvidia-smi
   
   # AMD
   rocm-smi
   
   # Intel
   intel_gpu_top
   ```

2. **Check hardware detection**:
   ```bash
   sudo -u exo exo --debug --list-devices
   ```

3. **Force specific accelerator**:
   ```nix
   services.exo.hardware.preferredAccelerator = "cuda";  # or "rocm", "cpu"
   ```

#### Network Issues

1. **Check port availability**:
   ```bash
   sudo netstat -tlnp | grep -E ':(52415|52416|8080)'
   ```

2. **Test network connectivity**:
   ```bash
   # Test API endpoint
   curl -v http://localhost:52415/health
   
   # Test discovery
   sudo tcpdump -i any port 52416
   ```

3. **Check firewall rules**:
   ```bash
   sudo iptables -L | grep -E '(52415|52416|8080)'
   ```

#### Memory Issues

1. **Check memory usage**:
   ```bash
   sudo systemctl status exo-worker
   free -h
   ```

2. **Adjust memory limits**:
   ```nix
   services.exo.hardware.memoryLimit = "50%";  # Reduce memory usage
   ```

3. **Monitor memory usage**:
   ```bash
   sudo journalctl -u exo-worker -f | grep -i memory
   ```

### Getting Help

1. **Check logs first**:
   ```bash
   sudo journalctl -u exo-master -u exo-worker -u exo-api --since "1 hour ago"
   ```

2. **Enable debug logging**:
   ```nix
   services.exo.logging.level = "debug";
   ```

3. **Collect system information**:
   ```bash
   # System info
   uname -a
   nixos-version
   
   # Hardware info
   lscpu
   lsmem
   lspci | grep -i vga
   
   # Network info
   ip addr show
   ip route show
   ```

4. **Report issues**:
   - GitHub Issues: https://github.com/exo-explore/exo/issues
   - Include logs, system info, and configuration
   - Describe expected vs actual behavior

## Next Steps

- [K3s Integration Guide](nixos-k3s-integration.md) - Integrate with existing Kubernetes clusters
- [Hardware Compatibility](nixos-hardware-compatibility.md) - Detailed hardware requirements and optimization
- [Performance Tuning](nixos-performance-tuning.md) - Optimize for your specific hardware setup
- [Multi-Node Setup](nixos-multi-node.md) - Configure distributed clusters

## Security Considerations

### Production Deployment

1. **Enable HTTPS**:
   ```nix
   services.exo.dashboard.ssl.enable = true;
   ```

2. **Configure authentication**:
   ```nix
   services.exo.dashboard.auth.enable = true;
   ```

3. **Network isolation**:
   ```nix
   services.exo.security.sandboxing.privateNetwork = true;
   ```

4. **Regular updates**:
   ```bash
   # Update flake inputs
   nix flake update
   sudo nixos-rebuild switch --flake .#your-hostname
   ```

### Monitoring

1. **Enable metrics collection**:
   ```nix
   services.exo.logging.metrics.enable = true;
   ```

2. **Set up log monitoring**:
   ```bash
   # Monitor for errors
   sudo journalctl -u exo-master -f | grep -i error
   ```

3. **Health checks**:
   ```bash
   # Automated health check script
   #!/bin/bash
   curl -f http://localhost:52415/health || systemctl restart exo-master
   ```
# EXO NixOS Modules

This directory contains NixOS modules for the EXO distributed AI inference system.

## Modules

### exo-service.nix
The main service module that provides:
- Core EXO service configuration options
- Systemd service definitions for master, worker, and API components
- User and permission management
- Logging and monitoring integration
- Security sandboxing and isolation
- Network and firewall configuration

### exo-hardware.nix
Hardware detection and configuration module that provides:
- Automatic GPU detection (NVIDIA, AMD, Intel)
- Hardware-specific package selection
- Device permissions and udev rules
- Driver integration and optimization

## Usage

To use these modules in your NixOS configuration:

```nix
{
  imports = [
    # Import EXO modules
    ./path/to/exo/nix/modules/exo-service.nix
    # Hardware module is automatically imported by service module
  ];

  # Enable EXO service
  services.exo = {
    enable = true;
    mode = "auto";  # or "master" or "worker"
    
    # Network configuration
    networking = {
      bondInterface = "bond0";  # optional
      rdmaEnabled = true;
      openFirewall = true;
    };
    
    # Hardware configuration
    hardware = {
      autoDetect = true;
      preferredAccelerator = null;  # auto-detect
      memoryLimit = "80%";
    };
    
    # K3s integration (optional)
    k3s = {
      integration = false;
      serviceDiscovery = true;
    };
    
    # Dashboard configuration
    dashboard = {
      enable = true;
      ssl.enable = false;  # set to true for HTTPS
    };
  };
}
```

## Configuration Options

See the module files for complete configuration options. Key options include:

- `services.exo.enable`: Enable the EXO service
- `services.exo.mode`: Operation mode (master/worker/auto)
- `services.exo.apiPort`: API server port (default: 52415)
- `services.exo.networking.*`: Network configuration options
- `services.exo.hardware.*`: Hardware detection and configuration
- `services.exo.k3s.*`: Kubernetes integration options
- `services.exo.security.*`: Security and sandboxing options
- `services.exo.logging.*`: Logging and monitoring configuration

## Security

The modules implement comprehensive security measures:
- Dedicated system user and group
- Systemd sandboxing and isolation
- Minimal file system permissions
- Device access controls
- Network namespace isolation (optional)
- AppArmor profiles (when enabled)
- Comprehensive logging and monitoring

## Monitoring

Built-in monitoring includes:
- Systemd journal integration
- Performance metrics collection
- Log analysis and alerting
- Health checks and status monitoring
- Prometheus metrics export (optional)
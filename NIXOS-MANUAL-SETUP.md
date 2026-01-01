# Manual NixOS EXO Setup Instructions

If the automatic script doesn't work, follow these manual steps:

## Option 1: Using the NixOS Module (Recommended)

1. **Copy the EXO configuration:**
   ```bash
   sudo cp nixos-exo-config.nix /etc/nixos/exo.nix
   ```

2. **Edit your main configuration.nix:**
   ```bash
   sudo nano /etc/nixos/configuration.nix
   ```

3. **Add the import to your imports section:**
   ```nix
   imports = [
     ./hardware-configuration.nix
     ./exo.nix  # Add this line
     # ... other imports
   ];
   ```

4. **Rebuild NixOS:**
   ```bash
   sudo nixos-rebuild switch
   ```

## Option 2: Direct Service Configuration

If you prefer to add the service directly to your configuration.nix:

```nix
{ config, pkgs, ... }:

let
  exoFlake = builtins.getFlake "path:/home/celes/sources/celesrenata/exo";
  exoPackage = exoFlake.packages.${pkgs.system}.exo-cpu;
in
{
  # Import the EXO module
  imports = [
    exoFlake.nixosModules.default
    # ... your other imports
  ];

  # Enable EXO service
  services.exo = {
    enable = true;
    package = exoPackage;
    accelerator = "cpu";
    port = 52415;
    openFirewall = true;
  };

  # Override environment variables for MLX fixes
  systemd.services.exo.environment = {
    MLX_DISABLE = "1";
    EXO_INFERENCE_ENGINE = "cpu";
    DASHBOARD_DIR = "${exoPackage}/share/exo/dashboard";
    PYTHONPATH = "${exoPackage}/lib/python3.13/site-packages";
  };
}
```

## Option 3: Quick Service Override (Temporary Fix)

If you just want to fix the current service quickly:

```bash
# Stop the current service
sudo systemctl stop exo

# Create a service override
sudo mkdir -p /etc/systemd/system/exo.service.d
sudo tee /etc/systemd/system/exo.service.d/override.conf << EOF
[Service]
Environment="MLX_DISABLE=1"
Environment="EXO_INFERENCE_ENGINE=cpu"
Environment="DASHBOARD_DIR=$(readlink -f ./result)/share/exo/dashboard"
ExecStart=
ExecStart=$(readlink -f ./result)/bin/exo
EOF

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl start exo
```

## Verification

After applying any of these methods:

1. **Check service status:**
   ```bash
   systemctl status exo
   ```

2. **Monitor logs:**
   ```bash
   sudo journalctl -u exo -f
   ```

3. **Access dashboard:**
   Open http://localhost:52415 in your browser

## Expected Behavior

- ✅ No more "ModuleNotFoundError: No module named 'mlx'" errors
- ✅ Service should show "MLX not available" message and continue with CPU inference
- ✅ Dashboard should be accessible
- ✅ Models should load and run on CPU
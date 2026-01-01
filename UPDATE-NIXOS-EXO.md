# Update NixOS EXO Configuration

Your NixOS configuration already has EXO configured! You just need to update it to use your local fixed version.

## Method 1: Update the flake input (Recommended)

1. **Navigate to your NixOS configuration directory:**
   ```bash
   cd ~/sources/nixos
   ```

2. **Update the EXO input in your flake.nix to point to your local repository:**
   
   Change this line (around line 47):
   ```nix
   exo.url = "github:celesrenata/exo";
   ```
   
   To this:
   ```nix
   exo.url = "path:/home/celes/sources/celesrenata/exo";
   ```

3. **Update your flake lock:**
   ```bash
   nix flake update exo
   ```

4. **Rebuild your system:**
   ```bash
   sudo nixos-rebuild switch --flake .#esnixi
   ```

## Method 2: Override the EXO service environment (Quick fix)

If you want to keep using the GitHub version but fix the MLX issue, add this to your `esnixi/exo.nix` file:

```nix
{ config, pkgs, ... }:

{
  services.exo = {
    enable = true;
    accelerator = "cpu";  # Force CPU inference
    port = 52415;
    openFirewall = true;
  };

  # Override environment variables to fix MLX issues
  systemd.services.exo = {
    environment = {
      MLX_DISABLE = "1";
      EXO_INFERENCE_ENGINE = "cpu";
    };
  };
}
```

## Method 3: Temporary service override (Immediate fix)

If you need an immediate fix without rebuilding:

```bash
# Stop the service
sudo systemctl stop exo

# Create service override directory
sudo mkdir -p /etc/systemd/system/exo.service.d

# Create override configuration
sudo tee /etc/systemd/system/exo.service.d/mlx-fix.conf << 'EOF'
[Service]
Environment="MLX_DISABLE=1"
Environment="EXO_INFERENCE_ENGINE=cpu"
EOF

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl start exo
```

## Verification

After applying any method:

1. **Check service status:**
   ```bash
   systemctl status exo
   ```

2. **Monitor logs for the fix:**
   ```bash
   sudo journalctl -u exo -f
   ```

   You should see:
   - ✅ "MLX not available: MLX disabled by environment variable" 
   - ✅ No more "ModuleNotFoundError: No module named 'mlx'" errors
   - ✅ Service continues running with CPU inference

3. **Access dashboard:**
   - Open http://localhost:52415

## Recommended Approach

**Method 1** is recommended because:
- Uses your fixed EXO version with proper MLX handling
- Provides the most robust solution
- Ensures you get all the latest fixes from your repository

The current error you're seeing indicates the service is using the old GitHub version that doesn't have the MLX fixes yet.
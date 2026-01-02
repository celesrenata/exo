# Update NixOS to Use Fixed EXO from GitHub

The MLX compatibility fixes have been pushed to GitHub. Here's how to update your NixOS system:

## Quick Update

```bash
cd ~/sources/nixos
nix flake update exo
sudo nixos-rebuild switch --flake .#esnixi
```

## What the Fix Does

The latest commit fixes the runner to:
- ✅ **Not exit when MLX is unavailable** if using CPU inference
- ✅ **Only exit if specifically trying to use MLX** when it's not available
- ✅ **Log informative messages** about inference engine selection
- ✅ **Allow CPU inference to work properly** on Linux systems

## Expected Behavior After Update

1. **Service starts successfully** without MLX errors
2. **Logs show**: `MLX not available, using cpu inference engine`
3. **No more**: `MLX inference engine is not available on Linux` errors
4. **Runner continues** instead of exiting
5. **Models load and run** on CPU inference

## Verification

After rebuilding:

```bash
# Check service status
systemctl status exo

# Monitor logs (should see CPU inference working)
sudo journalctl -u exo -f

# Access dashboard
curl http://localhost:52415
```

## Your NixOS Config Should Work Now

Since your configuration already has:
```nix
services.exo = {
  enable = true;
  accelerator = "cpu";  # This now works correctly
};
```

The accelerator parameter will:
1. Select the `exo-cpu` package
2. Set `EXO_INFERENCE_ENGINE=cpu`
3. Set `MLX_DISABLE=1`
4. Allow the runner to continue with CPU inference

No manual overrides needed - the accelerator parameter now works as intended!
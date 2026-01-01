# EXO Accelerator Parameter Fix

## The Problem
The NixOS module had an `accelerator` parameter that wasn't actually being used. The package selection was hardcoded to `exo-cpu` regardless of the accelerator setting.

## The Fix
Now the accelerator parameter works correctly:

### Package Selection
```nix
# Before (broken)
default = pkgs.exo-cpu or (mkExoPackage { inherit pkgs; system = pkgs.system; accelerator = "cpu"; });

# After (fixed)  
default = pkgs."exo-${cfg.accelerator}" or (mkExoPackage { inherit pkgs; system = pkgs.system; accelerator = cfg.accelerator; });
```

### Environment Variables
```nix
# Automatically sets the right environment based on accelerator choice
EXO_INFERENCE_ENGINE = cfg.accelerator;

# Special handling for CPU and MLX-on-Linux
} // (if cfg.accelerator == "cpu" || (pkgs.stdenv.isLinux && cfg.accelerator == "mlx") then {
  EXO_INFERENCE_ENGINE = "cpu";
  MLX_DISABLE = "1";
} else {});
```

## How to Use

In your NixOS configuration:

```nix
services.exo = {
  enable = true;
  accelerator = "cpu";  # This now actually works!
  port = 52415;
  openFirewall = true;
};
```

## What This Fixes

- ✅ **accelerator = "cpu"** → Uses `exo-cpu` package + sets `MLX_DISABLE=1`
- ✅ **accelerator = "cuda"** → Uses `exo-cuda` package + sets `EXO_INFERENCE_ENGINE=cuda`
- ✅ **accelerator = "mlx" on Linux** → Uses `exo-cpu` package + sets `MLX_DISABLE=1` (fallback)
- ✅ **accelerator = "mlx" on macOS** → Uses `exo-mlx` package + sets `EXO_INFERENCE_ENGINE=mlx`

## For Your Current Setup

Since your NixOS config likely already has:
```nix
services.exo = {
  enable = true;
  accelerator = "cpu";  # This should now work properly
};
```

You just need to:
1. Update your flake to use the local EXO repository (with this fix)
2. Rebuild your system

The accelerator parameter will now properly select the CPU package and set the right environment variables to disable MLX.
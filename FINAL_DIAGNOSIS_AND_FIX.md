# Final Diagnosis and Fix

## Root Cause Identified

The issue is a **path mismatch** between where models are stored and where EXO is looking for them:

### Current Situation
- **Models are stored in**: `/var/lib/exo/exo/models/`
- **EXO is looking in**: `/var/lib/exo/models/`

### Why This Happens
1. The systemd service sets `XDG_DATA_HOME=/var/lib/exo`
2. EXO's constants.py calculates `EXO_MODELS_DIR = EXO_DATA_HOME / "models"`
3. This results in `EXO_MODELS_DIR = /var/lib/exo/models`
4. But models are actually stored in `/var/lib/exo/exo/models/`

### Evidence
```bash
# Models exist here:
$ find /var/lib/exo -name "*microsoft*"
/var/lib/exo/exo/models/microsoft--DialoGPT-medium

# But EXO looks here:
# /var/lib/exo/models/microsoft--DialoGPT-medium (doesn't exist)
```

## The Fix

There are two possible solutions:

### Option 1: Fix the systemd service configuration (Recommended)
Update the flake.nix to set the correct XDG_DATA_HOME:

```nix
# In flake.nix, change:
XDG_DATA_HOME = "/var/lib/exo";

# To:
XDG_DATA_HOME = "/var/lib";
```

This way:
- `EXO_DATA_HOME = /var/lib/exo` (from XDG_DATA_HOME + "exo")
- `EXO_MODELS_DIR = /var/lib/exo/models`
- Models will be stored in `/var/lib/exo/models/` (matching where EXO looks)

### Option 2: Create a symlink (Quick fix)
```bash
sudo ln -s /var/lib/exo/exo/models /var/lib/exo/models
```

## Why This Causes "loading → loaded → failed"

1. **loading**: EXO starts the LoadModel task
2. **loaded**: Status notification works (our previous fix)
3. **failed**: `initialize_engine` fails because model files don't exist at the expected path
4. The failure is persistent (not transient) because the path issue won't resolve itself

## Implementation

The recommended fix is to update the flake.nix systemd service configuration to use the correct XDG_DATA_HOME path.
# Dashboard Update Needed for Tinygrad Ring Option

## Issue
The "Tinygrad Ring" option is not visible in the dashboard UI on gremlin-1.

## Root Cause
The dashboard on gremlin-1 is using an older version that was built before the Tinygrad Ring option was added to the code.

## Verification
✓ **Code Status**: The "Tinygrad Ring" button IS in the source code (dashboard/src/routes/+page.svelte, lines 2582-2603)
✓ **Local Build**: Dashboard was rebuilt successfully and includes the Tinygrad Ring option
✗ **Deployment**: gremlin-1 is serving an older dashboard version from the Nix store

## How Dashboard Loading Works

Exo finds the dashboard by:
1. Looking for `dashboard/build/` relative to the Python module location
2. On NixOS with Nix packages, this means it looks in the Nix store package
3. The current package on gremlin-1 was built before Tinygrad Ring was added

## Solutions

### Option 1: Rebuild Exo on Gremlin-1 (Recommended)
This will rebuild the entire exo package with the updated dashboard.

```bash
# On gremlin-1
cd /path/to/exo/source
nix build .#exo

# Or use NixOS rebuild (if using the exo-intel module)
cd /etc/nixos
nixos-rebuild switch --flake .#gremlin-1
```

### Option 2: Set Dashboard Directory (Quick Test)
Point exo to use the locally built dashboard:

```bash
# Copy dashboard to gremlin-1
scp -r dashboard/build root@10.1.1.12:/tmp/dashboard-build

# Restart exo with custom dashboard path
ssh root@10.1.1.12 'pkill -f "exo -vv"'
ssh root@10.1.1.12 'EXO_TINYGRAD_ENABLED=true EXO_DASHBOARD_DIR=/tmp/dashboard-build exo -vv'
```

### Option 3: Use Local Exo (Development)
Run exo locally where the dashboard is already built:

```bash
# In your local exo directory
EXO_TINYGRAD_ENABLED=true uv run exo -vv
# Access at http://localhost:52415
```

## Current Status

### What's Working ✓
- Exo service is running on gremlin-1
- API is responding correctly
- Models endpoint works
- Tinygrad backend is enabled
- Hardware is detected (Intel GPU, NPU, Level Zero)

### What Needs Update ⚠
- Dashboard UI (cosmetic issue only)
- The Tinygrad Ring option exists in code but isn't visible because gremlin-1 is serving an old dashboard

## Quick Verification

After updating, verify the Tinygrad Ring option is visible:

1. Open http://10.1.1.12:52415 in your browser
2. Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)
3. Look at the "Instance Type" section in the right sidebar
4. You should see three buttons:
   - MLX Ring
   - MLX RDMA
   - **Tinygrad Ring** ← This should now be visible

## Technical Details

### Dashboard Location Logic
```python
# From src/exo/utils/dashboard_path.py
def find_dashboard() -> Path:
    # Searches up from Python module location
    # Looks for: dashboard/build/index.html
    # On NixOS: /nix/store/xxx-exo-0.3.0/dashboard/build
```

### Environment Variable Override
```bash
export EXO_DASHBOARD_DIR=/path/to/dashboard/build
```

## Files Created

- `update_dashboard_gremlin1.sh` - Script to copy dashboard (won't work with Nix packages)
- `DASHBOARD_UPDATE_NEEDED.md` - This file
- `GREMLIN1_TEST_RESULTS.md` - Test results showing exo is working

## Recommendation

**Use Option 1** (rebuild exo) for a proper deployment. This ensures:
- Dashboard is part of the Nix package
- Consistent deployment
- No manual file copying needed
- Proper version tracking

The core functionality is working - this is just a UI update to show the Tinygrad Ring option that's already implemented in the code.

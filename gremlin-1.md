# Gremlin-1 Deployment Process

## Overview
This document describes the process for deploying code changes to gremlin-1, a remote NixOS machine used for testing Intel hardware support with the exo distributed AI inference system.

## Quick Reference

### Deploy Latest Changes
```bash
bash force_update_gremlin1.sh
```

This script automatically:
1. Gets the latest commit from `git log --oneline -1`
2. Updates the flake on gremlin-1 to use that commit
3. Rebuilds the NixOS system
4. Restarts the exo service
5. Shows service status and logs

### Check Service Status
```bash
ssh root@10.1.1.12 "systemctl status exo"
```

### View Logs
```bash
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '5 minutes ago' | tail -50"
```

### Check for Errors
```bash
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '5 minutes ago' | grep -E 'error|exception|failed|traceback' -i | tail -30"
```

### Test Model Loading
```bash
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 20,
    "stream": false
  }'
```

### Check Runner Status
```bash
curl -s 'http://10.1.1.12:52415/state' | python3 -c "import sys, json; data=json.load(sys.stdin); runner = list(data['runners'].values())[0] if data.get('runners') else None; print('Runner status:', list(runner.keys())[0] if runner else 'No runner')"
```

## Deployment Workflow

### 1. Make Code Changes
Edit the necessary files in the local repository.

### 2. Commit Changes
```bash
git add -A
git commit -m "Description of changes"
```

### 3. Push to GitHub
```bash
git push origin ipex
```

**Important**: The commit MUST be pushed to GitHub before deploying to gremlin-1, as the NixOS flake pulls from the GitHub repository.

### 4. Deploy to Gremlin-1
```bash
bash force_update_gremlin1.sh
```

Wait for the script to complete. It will:
- Show the commit being deployed
- Display build progress
- Restart the service
- Show initial status

### 5. Verify Deployment

Check the logs for the specific functionality you're testing:

```bash
# For tinygrad backend issues
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '2 minutes ago' | grep -E 'Tinygrad|Device selection|Loading|GPU|OpenCL' | tail -20"

# For runner issues
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '2 minutes ago' | grep -E 'runner|CreateInstance|Loading' -i | tail -20"

# For model loading issues
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '2 minutes ago' | grep -E 'model|loading|ready|failed' -i | tail -20"
```

## Common Issues and Debugging

### Issue: "No commit found for SHA"
**Cause**: The commit hasn't been pushed to GitHub yet, or GitHub hasn't processed it.

**Solution**:
1. Ensure you've pushed: `git push origin ipex`
2. Wait 5-10 seconds for GitHub to process
3. Try deploying again

### Issue: Service fails to start
**Cause**: Various - check logs for specific error.

**Solution**:
```bash
# Check full error
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '5 minutes ago' | tail -100"

# Check for Python errors
ssh root@10.1.1.12 "journalctl -u exo --no-pager --since '5 minutes ago' | grep -A10 'Traceback'"
```

### Issue: Model fails to load (preparing → loading → failed)
**Cause**: Usually an error in the runner code during model initialization.

**Solution**:
1. Check logs around "Loading tinygrad model" message
2. Look for exceptions or assertion errors
3. Common issues:
   - Undefined variables (e.g., `device_caps`)
   - Missing imports
   - Incorrect variable names

### Issue: Model loads but inference fails (ready → running → failed)
**Cause**: Error during inference execution.

**Solution**:
1. Check logs for "runner running" message
2. Look for assertion errors or exceptions immediately after
3. Common issues:
   - Model variables not properly assigned (e.g., `tinygrad_model` is None)
   - Missing tokenizer
   - Device mismatch

### Issue: Model not recognized (WARNING: TODO: we should send a notification)
**Cause**: Model isn't in the supported models list or isn't being properly registered.

**Solution**:
1. Check available models: `curl -s http://10.1.1.12:52415/v1/models | python3 -m json.tool`
2. Verify the model ID matches exactly
3. Check if model is marked as `is_custom: true` (may need special handling)

## Environment Variables on Gremlin-1

The exo service on gremlin-1 is configured with:
- `OPENCL=1` - Force OpenCL runtime for GPU
- `TINYGRAD_BACKEND=GPU` - Use GPU backend for tinygrad

These are set in the systemd service file.

## Key Files

### Local Repository
- `force_update_gremlin1.sh` - Deployment script
- `src/exo/worker/runner/runner.py` - Main runner logic
- `src/exo/worker/runner/bootstrap.py` - Runner initialization
- `src/exo/worker/engines/tinygrad/` - Tinygrad backend implementation

### On Gremlin-1
- `/etc/nixos/flake.nix` - NixOS configuration
- `/etc/systemd/system/exo.service` - Service definition
- Service runs as root
- Logs via journalctl

## Testing Checklist

After deploying changes:

1. ✓ Service starts successfully
2. ✓ No errors in initial logs
3. ✓ Dashboard accessible at http://10.1.1.12:52415
4. ✓ Model appears in `/v1/models` endpoint
5. ✓ Runner can be created (check state endpoint)
6. ✓ Model loads (status: preparing → loading → ready)
7. ✓ Inference works (status: ready → running → complete)
8. ✓ Response is returned successfully

## Tips

- Always check logs immediately after deployment to catch startup errors
- Use `grep` with multiple patterns to filter relevant log lines
- The `-A` and `-B` flags with grep show context around matches
- Wait 10-15 seconds after sending a request before checking runner status
- If stuck, check the full logs without filtering to see the complete picture
- Remember: changes must be committed AND pushed before deploying

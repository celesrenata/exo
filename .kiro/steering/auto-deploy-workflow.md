# Auto-Deploy Workflow

## Post-Push Deployment to Gremlin-1

When pushing code changes to the repository, automatically deploy to the gremlin-1 test machine.

### Workflow

After executing `git push origin ipex`:

1. **Wait 5 seconds** to allow GitHub to process the push
2. **Run deployment script**: `bash force_update_gremlin1.sh`

### Implementation

```bash
# After git push
sleep 5
bash force_update_gremlin1.sh
```

### What This Does

The `force_update_gremlin1.sh` script:
- Pulls latest code from GitHub on gremlin-1
- Rebuilds the NixOS configuration with new code
- Restarts the exo service
- Verifies the service is running
- Shows service status and logs

### When to Use

Apply this workflow when:
- Pushing to the `ipex` branch
- Working on PyTorch+IPEX backend development
- Making changes that need testing on Intel Arc GPU hardware
- Deploying to the gremlin-1 remote test machine (10.1.1.12)

### Verification

After deployment completes:
- Dashboard available at: http://10.1.1.12:52415
- Check service status: `ssh root@10.1.1.12 "systemctl status exo"`
- View logs: `ssh root@10.1.1.12 "journalctl -u exo -f"`

### Note

The 5-second wait ensures GitHub has processed the push before gremlin-1 attempts to pull the changes.

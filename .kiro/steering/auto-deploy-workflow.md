---
inclusion: manual
---

# Auto-Deploy Workflow

## Post-Push Deployment

When pushing code changes to the repository, follow this workflow:

1. **Push changes** to the remote repository
2. **Wait 5 seconds** to allow GitHub to process the push
3. **Run deployment script** to update gremlin-1 test machine

### Implementation

After executing `git push`, automatically:
```bash
sleep 5
./force_update_gremlin1.sh
```

This ensures that:
- GitHub has time to process the push
- The remote test machine (gremlin-1) gets the latest code
- Testing can begin immediately with the new changes

### Usage

This workflow applies when:
- Pushing to the `ipex` branch
- Working on PyTorch+IPEX backend development
- Testing changes on the gremlin-1 remote machine

The `force_update_gremlin1.sh` script will:
- Pull the latest code from GitHub
- Rebuild the NixOS configuration
- Restart the exo service
- Make the changes available for testing

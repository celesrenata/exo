---
inclusion: always
---

# Gremlin Cluster Deployment

## Quick Deploy (All 4 Nodes)

```bash
bash deploy_cluster.sh
```

This pushes the current branch, deploys to gremlin-1 first (master), then gremlin-2/3/4 in parallel.

## Deploy Specific Nodes

```bash
bash deploy_cluster.sh gremlin-1              # just the master
bash deploy_cluster.sh gremlin-2 gremlin-3    # specific workers
```

## What It Does

1. Pushes current branch to GitHub
2. SSHs to each node, updates the nix flake input to the current commit
3. Runs `nixos-rebuild switch` (rebuilds Rust + Python + dashboard)
4. Restarts the exo service
5. Verifies the service is running

## Deployment Order

- gremlin-1 deploys first (it's the master and runs the API)
- gremlin-2, 3, 4 deploy in parallel after gremlin-1 succeeds

## Checking Status

```bash
# Cluster state (from master)
curl -s http://10.1.1.12:52415/state | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Nodes: {len(d.get(\"topology\",{}).get(\"nodes\",{}))}')"

# Individual node
ssh root@10.1.1.12 "systemctl status exo --no-pager | head -10"
ssh root@10.1.1.14 "journalctl -u exo -n 20 --no-pager"
```

## Node Addresses

| Node | IP | Role |
|------|------|------|
| gremlin-1 | 10.1.1.12 | Master, API, NVIDIA + Intel iGPU |
| gremlin-2 | 10.1.1.13 | Worker, Intel iGPU |
| gremlin-3 | 10.1.1.14 | Worker, Intel iGPU |
| gremlin-4 | 10.1.1.15 | Worker, Intel iGPU |

## Prerequisites

- SSH access as root to all gremlin nodes
- Current branch pushed to GitHub (script handles this)
- Each gremlin's `/etc/nixos/flake.nix` has an `exo` input pointing to `github:celesrenata/exo`

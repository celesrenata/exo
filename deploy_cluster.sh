#!/bin/bash
set -euo pipefail

# Deploy exo to the gremlin cluster.
#
# Workflow:
#   1. Push current branch to GitHub
#   2. Deploy to each node sequentially (update flake, rebuild, restart, verify)
#
# Usage:
#   bash deploy_cluster.sh              # deploy to all 4 nodes
#   bash deploy_cluster.sh gremlin-1    # deploy to gremlin-1 only
#   bash deploy_cluster.sh gremlin-2 gremlin-3  # deploy to specific nodes

BRANCH=$(git branch --show-current)
COMMIT=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short HEAD)

declare -A HOSTS=(
  [gremlin-1]="root@10.1.1.12"
  [gremlin-2]="root@10.1.1.13"
  [gremlin-3]="root@10.1.1.14"
  [gremlin-4]="root@10.1.1.15"
)

# Determine which nodes to deploy
if [ $# -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("gremlin-1" "gremlin-2" "gremlin-3" "gremlin-4")
fi

echo "=== Deploying exo cluster ==="
echo "Branch: $BRANCH"
echo "Commit: $SHORT_COMMIT ($COMMIT)"
echo "Targets: ${TARGETS[*]}"
echo ""

# Step 1: Push to GitHub
echo ">>> Pushing $BRANCH to origin..."
if ! git push origin "$BRANCH" 2>&1; then
  echo "ERROR: git push failed"
  exit 1
fi
echo ""

# Wait for GitHub to process
sleep 3

# Deploy function — sequential, verbose, with verification
deploy_node() {
  local NODE=$1
  local HOST=${HOSTS[$NODE]}

  if [ -z "$HOST" ]; then
    echo "[$NODE] ERROR: Unknown node"
    return 1
  fi

  # Step A: Update flake input
  echo "[$NODE] Updating flake input to $SHORT_COMMIT..."
  if ! ssh -o ConnectTimeout=10 "$HOST" "cd /etc/nixos && nix flake update exo --override-input exo github:celesrenata/exo/$COMMIT 2>&1"; then
    echo "[$NODE] ERROR: flake update failed"
    return 1
  fi

  # Step B: Verify the lock file has the correct commit
  local LOCK_REV
  LOCK_REV=$(ssh -o ConnectTimeout=10 "$HOST" "grep -A10 'celesrenata' /etc/nixos/flake.lock | grep rev | head -1 | grep -o '[0-9a-f]\{40\}'")
  if [ "$LOCK_REV" != "$COMMIT" ]; then
    echo "[$NODE] ERROR: flake.lock has $LOCK_REV, expected $COMMIT"
    echo "[$NODE] Retrying flake update..."
    ssh -o ConnectTimeout=10 "$HOST" "cd /etc/nixos && nix flake update exo --override-input exo github:celesrenata/exo/$COMMIT 2>&1"
    LOCK_REV=$(ssh -o ConnectTimeout=10 "$HOST" "grep -A10 'celesrenata' /etc/nixos/flake.lock | grep rev | head -1 | grep -o '[0-9a-f]\{40\}'")
    if [ "$LOCK_REV" != "$COMMIT" ]; then
      echo "[$NODE] ERROR: flake.lock still wrong after retry ($LOCK_REV)"
      return 1
    fi
  fi
  echo "[$NODE] ✓ flake.lock verified: $SHORT_COMMIT"

  # Step C: Rebuild NixOS
  echo "[$NODE] Rebuilding NixOS (this takes a while on first build)..."
  if ! ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 "$HOST" "cd /etc/nixos && nixos-rebuild switch --flake .#$NODE 2>&1 | tee /tmp/nixos-rebuild.log | tail -10"; then
    echo "[$NODE] ERROR: nixos-rebuild failed. Last 20 lines:"
    ssh -o ConnectTimeout=10 "$HOST" "tail -20 /tmp/nixos-rebuild.log" 2>/dev/null
    return 1
  fi
  echo "[$NODE] ✓ NixOS rebuilt"

  # Step D: Restart exo service
  echo "[$NODE] Restarting exo service..."
  ssh -o ConnectTimeout=10 "$HOST" "systemctl restart exo"
  sleep 5

  # Step E: Verify service is running
  local STATUS
  STATUS=$(ssh -o ConnectTimeout=10 "$HOST" "systemctl is-active exo" 2>/dev/null || echo "failed")
  if [ "$STATUS" = "active" ]; then
    echo "[$NODE] ✓ exo is running"
  else
    echo "[$NODE] ✗ exo failed to start! Logs:"
    ssh -o ConnectTimeout=10 "$HOST" "journalctl -u exo -n 15 --no-pager" 2>/dev/null
    return 1
  fi
  echo ""
}

# Deploy nodes sequentially — gremlin-1 first, then the rest
FAILED_NODES=()

for NODE in "${TARGETS[@]}"; do
  if ! deploy_node "$NODE"; then
    FAILED_NODES+=("$NODE")
    echo ">>> $NODE FAILED — continuing with remaining nodes"
    echo ""
  fi
done

# Summary
echo "=== Deployment Summary ==="
echo "Commit: $SHORT_COMMIT"
SUCCEEDED=$((${#TARGETS[@]} - ${#FAILED_NODES[@]}))
echo "Succeeded: $SUCCEEDED/${#TARGETS[@]}"
if [ ${#FAILED_NODES[@]} -gt 0 ]; then
  echo "Failed: ${FAILED_NODES[*]}"
fi
echo ""
echo "Dashboard: http://10.1.1.12:52415"
echo ""

# Final cluster check
echo ">>> Checking cluster state..."
sleep 10
ssh -o ConnectTimeout=5 root@10.1.1.12 "nix-shell -p jq --run 'curl -s http://localhost:52415/state | jq \"{nodes: (.topology.nodes | length), instances: (.instances | length), runners: (.runners | length)}\"'" 2>/dev/null || echo "API not responding"

if [ ${#FAILED_NODES[@]} -gt 0 ]; then
  exit 1
fi

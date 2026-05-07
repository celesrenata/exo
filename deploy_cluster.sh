#!/bin/bash
set -e

# Deploy exo to the gremlin cluster.
#
# Workflow:
#   1. Push current branch to GitHub
#   2. Deploy to gremlin-1 (rebuild NixOS, restart exo)
#   3. Deploy to gremlin-2, gremlin-3, gremlin-4 in parallel
#
# Usage:
#   bash deploy_cluster.sh              # deploy to all 4 nodes
#   bash deploy_cluster.sh gremlin-1    # deploy to gremlin-1 only
#   bash deploy_cluster.sh gremlin-2 gremlin-3  # deploy to specific nodes

BRANCH=$(git branch --show-current)
COMMIT=$(git log --oneline -1 | awk '{print $1}')

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
echo "Commit: $COMMIT"
echo "Targets: ${TARGETS[*]}"
echo ""

# Step 1: Push to GitHub
echo ">>> Pushing $BRANCH to origin..."
git push origin "$BRANCH" 2>&1 | tail -5
echo ""

# Wait for GitHub to process
sleep 3

# Step 2: Deploy function
deploy_node() {
  local NODE=$1
  local HOST=${HOSTS[$NODE]}

  if [ -z "$HOST" ]; then
    echo "[$NODE] ERROR: Unknown node"
    return 1
  fi

  echo "[$NODE] Updating flake input to commit $COMMIT..."
  ssh "$HOST" "cd /etc/nixos && nix flake lock --update-input exo --override-input exo github:celesrenata/exo/$COMMIT" 2>&1 | tail -3

  echo "[$NODE] Rebuilding NixOS..."
  ssh "$HOST" "cd /etc/nixos && nixos-rebuild switch --flake .#$NODE" 2>&1 | grep -E "building|activating|Done|switching" | tail -5

  echo "[$NODE] Restarting exo service..."
  ssh "$HOST" "systemctl restart exo"

  echo "[$NODE] Waiting for startup..."
  sleep 5

  echo "[$NODE] Checking status..."
  local STATUS
  STATUS=$(ssh "$HOST" "systemctl is-active exo" 2>/dev/null || echo "failed")
  if [ "$STATUS" = "active" ]; then
    echo "[$NODE] ✓ exo is running"
  else
    echo "[$NODE] ✗ exo failed to start!"
    ssh "$HOST" "journalctl -u exo -n 10 --no-pager" 2>/dev/null | tail -5
    return 1
  fi
  echo ""
}

# Step 3: Deploy gremlin-1 first (it's the master)
if printf '%s\n' "${TARGETS[@]}" | grep -q "^gremlin-1$"; then
  deploy_node "gremlin-1"
  # Remove gremlin-1 from remaining targets
  REMAINING=()
  for t in "${TARGETS[@]}"; do
    [ "$t" != "gremlin-1" ] && REMAINING+=("$t")
  done
else
  REMAINING=("${TARGETS[@]}")
fi

# Step 4: Deploy remaining nodes in parallel
if [ ${#REMAINING[@]} -gt 0 ]; then
  echo ">>> Deploying ${REMAINING[*]} in parallel..."
  PIDS=()
  for NODE in "${REMAINING[@]}"; do
    deploy_node "$NODE" &
    PIDS+=($!)
  done

  # Wait for all parallel deployments
  FAILED=0
  for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
      FAILED=$((FAILED + 1))
    fi
  done

  if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED node(s) failed to deploy"
  fi
fi

echo "=== Deployment complete ==="
echo ""
echo "Dashboard: http://10.1.1.12:52415"
echo "Check cluster: curl -s http://10.1.1.12:52415/state | python3 -c \"import sys,json; d=json.load(sys.stdin); print(f'Nodes: {len(d.get(\\\"topology\\\",{}).get(\\\"nodes\\\",{}))}')\" 2>/dev/null || echo 'API not ready yet'"

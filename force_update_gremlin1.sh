#!/bin/bash
set -e

echo "=== Force Update gremlin-1 to Latest Commit ==="
echo ""

GREMLIN_HOST="root@10.1.1.12"
LATEST_COMMIT="88276688"

echo "Latest commit: $LATEST_COMMIT"
echo ""

echo "Step 1: Force update flake to specific commit..."
ssh $GREMLIN_HOST "cd /etc/nixos && nix flake lock --update-input exo --override-input exo github:celesrenata/exo/$LATEST_COMMIT"

echo ""
echo "Step 2: Rebuild NixOS system..."
ssh $GREMLIN_HOST "cd /etc/nixos && nixos-rebuild switch --flake .#gremlin-1" 2>&1 | grep -E "building|copying.*exo|Done|store.*exo" | tail -20

echo ""
echo "Step 3: Check exo package..."
ssh $GREMLIN_HOST "which exo"

echo ""
echo "Step 4: Restart exo service..."
ssh $GREMLIN_HOST "systemctl restart exo"

echo ""
echo "Step 5: Wait for startup..."
sleep 10

echo ""
echo "Step 6: Check service status..."
ssh $GREMLIN_HOST "systemctl status exo --no-pager -l | head -30"

echo ""
echo "Step 7: Check logs..."
ssh $GREMLIN_HOST "journalctl -u exo -n 50 --no-pager | grep -E 'hello from|Tinygrad|backend|error|Dashboard' || echo 'No relevant log entries yet'"

echo ""
echo "=== Update Complete ==="
echo ""
echo "Test with:"
echo "  curl -s 'http://10.1.1.12:52415/state' | python3 -c \"import sys, json; data=json.load(sys.stdin); runner = list(data['runners'].values())[0] if data.get('runners') else None; print('Runner status:', list(runner.keys())[0] if runner else 'No runner')\""

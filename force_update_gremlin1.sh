#!/bin/bash
set -e

echo "=== Force Update gremlin-1 to Latest Commit ==="
echo ""

GREMLIN_HOST="root@10.1.1.12"
LATEST_COMMIT="c04177a2"

echo "Latest commit: $LATEST_COMMIT"
echo ""

echo "Step 1: Force update flake to specific commit..."
ssh $GREMLIN_HOST "cd /etc/nixos && nix flake lock --update-input exo --override-input exo github:celesrenata/exo/$LATEST_COMMIT"

echo ""
echo "Step 2: Rebuild NixOS system..."
ssh $GREMLIN_HOST "cd /etc/nixos && nixos-rebuild switch --flake .#gremlin-1" 2>&1 | grep -E "building|copying.*exo|Done|store.*exo" | tail -20

echo ""
echo "Step 3: Find new exo package path..."
NEW_PACKAGE=$(ssh $GREMLIN_HOST "ls -dt /nix/store/*-exo-0.3.0 2>/dev/null | head -1")
echo "New package: $NEW_PACKAGE"

echo ""
echo "Step 4: Kill old exo process..."
ssh $GREMLIN_HOST "pkill -f '.exo-wrapped' || true"

echo ""
echo "Step 5: Start new exo process..."
ssh $GREMLIN_HOST "cd /tmp && EXO_TINYGRAD_ENABLED=true nohup $NEW_PACKAGE/bin/exo -vv > /tmp/exo.log 2>&1 &"

echo ""
echo "Step 6: Wait for startup..."
sleep 10

echo ""
echo "Step 7: Check if exo is running..."
ssh $GREMLIN_HOST "tail -30 /tmp/exo.log | grep -E 'hello from|Tinygrad|backend|error' || echo 'No relevant log entries yet'"

echo ""
echo "=== Update Complete ==="
echo ""
echo "Test with:"
echo "  curl -s 'http://10.1.1.12:52415/state' | python3 -c \"import sys, json; data=json.load(sys.stdin); runner = list(data['runners'].values())[0] if data.get('runners') else None; print('Runner status:', list(runner.keys())[0] if runner else 'No runner')\""

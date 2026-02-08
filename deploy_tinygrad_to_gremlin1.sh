#!/bin/bash
set -e

echo "=== Deploying TinygradRing Support to gremlin-1 ==="
echo ""

GREMLIN_HOST="root@10.1.1.12"

echo "Step 1: Update flake lock on gremlin-1 to pull latest exo code..."
ssh $GREMLIN_HOST 'cd /etc/nixos && nix flake lock --update-input exo --refresh'

echo ""
echo "Step 2: Rebuild NixOS system with updated exo..."
ssh $GREMLIN_HOST 'cd /etc/nixos && nixos-rebuild switch --flake .#gremlin-1'

echo ""
echo "Step 3: Wait for exo service to restart..."
sleep 10

echo ""
echo "Step 4: Check exo service status..."
ssh $GREMLIN_HOST 'systemctl status exo --no-pager -l' || true

echo ""
echo "Step 5: Test TinygradRing placement API..."
echo "Testing with tinyllama-1b-cpu model..."
ssh $GREMLIN_HOST 'curl -s "http://localhost:52415/instance/previews?model_id=tinyllama-1b-cpu" | python3 -c "import sys, json; data=json.load(sys.stdin); tinygrad=[p for p in data[\"previews\"] if \"Tinygrad\" in p.get(\"instance_meta\",\"\")]; print(\"TinygradRing previews found:\", len(tinygrad)); print(json.dumps(tinygrad[:2], indent=2) if tinygrad else \"No TinygradRing previews\")"'

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Open dashboard at http://10.1.1.12:52415"
echo "2. Select a model"
echo "3. Choose 'Tinygrad Ring' as instance type"
echo "4. Verify placement configurations appear"

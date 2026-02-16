#!/bin/bash
set -e

echo "=== Patching apply.py on gremlin-1 ==="
echo ""

GREMLIN_HOST="root@10.1.1.12"

echo "Step 1: Copy fixed apply.py to gremlin-1..."
scp src/exo/shared/apply.py $GREMLIN_HOST:/tmp/apply.py

echo ""
echo "Step 2: Find current exo installation..."
EXO_PATH=$(ssh $GREMLIN_HOST "ls -dt /nix/store/*-exo-0.3.0 2>/dev/null | head -1")
echo "Found exo at: $EXO_PATH"

if [ -z "$EXO_PATH" ]; then
  echo "ERROR: Could not find exo installation"
  exit 1
fi

echo ""
echo "Step 3: Backup original apply.py..."
ssh $GREMLIN_HOST "cp $EXO_PATH/lib/python3.13/site-packages/exo/shared/apply.py /tmp/apply.py.backup"

echo ""
echo "Step 4: Replace apply.py..."
ssh $GREMLIN_HOST "cp /tmp/apply.py $EXO_PATH/lib/python3.13/site-packages/exo/shared/apply.py"

echo ""
echo "Step 5: Kill old exo process..."
ssh $GREMLIN_HOST "pkill -f '.exo-wrapped' || true"

echo ""
echo "Step 6: Clear log file..."
ssh $GREMLIN_HOST "> /tmp/exo.log"

echo ""
echo "Step 7: Start new exo process..."
ssh $GREMLIN_HOST "cd /tmp && EXO_TINYGRAD_ENABLED=true nohup $EXO_PATH/bin/exo -vv > /tmp/exo.log 2>&1 &"

echo ""
echo "Step 8: Wait for startup..."
sleep 10

echo ""
echo "Step 9: Check logs for errors..."
ssh $GREMLIN_HOST "tail -50 /tmp/exo.log"

echo ""
echo "=== Patch Complete ==="
echo ""
echo "Monitor logs with:"
echo "  ssh $GREMLIN_HOST 'tail -f /tmp/exo.log'"

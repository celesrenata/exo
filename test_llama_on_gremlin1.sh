#!/bin/bash
set -e

echo "=== Test Llama Transformer on gremlin-1 ==="
echo ""

GREMLIN_HOST="root@10.1.1.12"
LATEST_COMMIT=$(git log --oneline -1 | awk '{print $1}')

echo "Latest commit: $LATEST_COMMIT"
echo ""

# Step 1: Deploy latest code
echo "Step 1: Deploying latest code to gremlin-1..."
./force_update_gremlin1.sh

echo ""
echo "Step 2: Copy validation test to gremlin-1..."
scp test_llama_validation.py $GREMLIN_HOST:/tmp/

echo ""
echo "Step 3: Run validation tests on gremlin-1..."
ssh $GREMLIN_HOST "cd /tmp && python3 test_llama_validation.py" 2>&1 | tee /tmp/gremlin1_validation_results.txt

echo ""
echo "Step 4: Check if tests passed..."
if grep -q "All validation tests passed" /tmp/gremlin1_validation_results.txt; then
  echo "✅ All validation tests PASSED on gremlin-1!"

  echo ""
  echo "Step 5: Test with actual model loading (if available)..."
  ssh $GREMLIN_HOST "curl -s 'http://localhost:52415/state' | python3 -c \"import sys, json; data=json.load(sys.stdin); runner = list(data['runners'].values())[0] if data.get('runners') else None; print('Runner backend:', runner.get('backend') if runner else 'No runner'); print('Runner status:', list(runner.keys())[0] if runner else 'No runner')\""

  echo ""
  echo "Step 6: Check tinygrad backend is available..."
  ssh $GREMLIN_HOST "journalctl -u exo -n 100 --no-pager | grep -i 'tinygrad\|backend\|llama' | tail -20"

else
  echo "❌ Some validation tests FAILED on gremlin-1"
  echo ""
  echo "Failed test output:"
  grep -A 5 "Test failed\|Error\|Traceback" /tmp/gremlin1_validation_results.txt || echo "No detailed error found"
  exit 1
fi

echo ""
echo "=== Testing Complete ==="
echo ""
echo "Summary:"
echo "  - Validation tests: PASSED"
echo "  - Backend: tinygrad"
echo "  - Hardware: Intel Arc GPU"
echo ""
echo "Next steps:"
echo "  1. Test with actual model: curl -X POST http://10.1.1.12:52415/v1/chat/completions ..."
echo "  2. Monitor performance: ssh $GREMLIN_HOST 'journalctl -u exo -f'"
echo "  3. Check GPU usage: ssh $GREMLIN_HOST 'intel_gpu_top'"

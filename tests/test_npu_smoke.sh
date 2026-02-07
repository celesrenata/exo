#!/usr/bin/env bash
# NPU smoke test runner script

set -e

echo "=========================================="
echo "Intel NPU Smoke Test"
echo "=========================================="
echo ""

# Check if running on appropriate hardware
echo "Checking system information..."
CPU_MODEL=$(cat /proc/cpuinfo | grep "model name" | head -n1 | cut -d: -f2 | xargs)
echo "CPU: $CPU_MODEL"

if [[ ! "$CPU_MODEL" =~ "Core Ultra" ]]; then
    echo "⚠️  Warning: This does not appear to be a Core Ultra processor"
    echo "   NPU may not be available"
fi

echo ""
echo "Checking kernel version..."
KERNEL_VERSION=$(uname -r)
echo "Kernel: $KERNEL_VERSION"

echo ""
echo "Checking for NPU kernel modules..."
if lsmod | grep -qE "intel_vpu|ivpu"; then
    echo "✅ NPU kernel module loaded:"
    lsmod | grep -E "intel_vpu|ivpu"
else
    echo "❌ No NPU kernel module loaded"
    echo "   Try: sudo modprobe intel_vpu"
fi

echo ""
echo "Checking for NPU device nodes..."
if [ -e /dev/accel/accel0 ]; then
    echo "✅ Found /dev/accel/accel0"
    ls -la /dev/accel/accel0
elif ls /dev/dri/renderD* >/dev/null 2>&1; then
    echo "⚠️  No /dev/accel/accel0, but found render devices:"
    ls -la /dev/dri/renderD*
else
    echo "❌ No NPU device nodes found"
fi

echo ""
echo "=========================================="
echo "Running NPU capability report..."
echo "=========================================="
echo ""

uv run python -m exo.worker.engines.npu.capability_report

echo ""
echo "=========================================="
echo "Running NPU smoke test..."
echo "=========================================="
echo ""

uv run python -m exo.worker.engines.npu.smoke_test

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NPU smoke test PASSED"
else
    echo "❌ NPU smoke test FAILED"
fi
echo "=========================================="

exit $EXIT_CODE

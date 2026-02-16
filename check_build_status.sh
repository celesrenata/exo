#!/usr/bin/env bash
#
# Check the status of the PyTorch + IPEX build
#

echo "=========================================="
echo "PyTorch + IPEX Build Status"
echo "=========================================="
echo ""

# Check if build process is running
if pgrep -f "build_pytorch_ipex.sh" > /dev/null; then
    echo "Status: BUILDING"
    echo ""
    
    # Show recent build output
    if [ -f "build_logs/pytorch_build.log" ]; then
        echo "Recent PyTorch build output:"
        echo "----------------------------"
        tail -20 build_logs/pytorch_build.log
    fi
    
    if [ -f "build_logs/ipex_build.log" ]; then
        echo ""
        echo "Recent IPEX build output:"
        echo "-------------------------"
        tail -20 build_logs/ipex_build.log
    fi
else
    echo "Status: NOT RUNNING"
    echo ""
    
    # Check if build completed successfully
    if [ -f "build_logs/pytorch_build.log" ] && grep -q "✓ PyTorch build completed" build_logs/pytorch_build.log; then
        echo "✓ PyTorch build: COMPLETED"
    elif [ -f "build_logs/pytorch_build.log" ]; then
        echo "✗ PyTorch build: FAILED or INCOMPLETE"
    else
        echo "? PyTorch build: NOT STARTED"
    fi
    
    if [ -f "build_logs/ipex_build.log" ] && grep -q "✓ IPEX build completed" build_logs/ipex_build.log; then
        echo "✓ IPEX build: COMPLETED"
    elif [ -f "build_logs/ipex_build.log" ]; then
        echo "✗ IPEX build: FAILED or INCOMPLETE"
    else
        echo "? IPEX build: NOT STARTED"
    fi
fi

echo ""
echo "=========================================="
echo ""
echo "To monitor build in real-time:"
echo "  tail -f build_logs/pytorch_build.log"
echo ""
echo "To restart build if failed:"
echo "  ./build_pytorch_ipex.sh"
echo ""

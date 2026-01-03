#!/usr/bin/env python3
"""
Debug the EXO planning process to see why LoadModel is failing.
"""

import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_planning_logic():
    """Test the EXO planning logic to understand task sequencing."""
    print("=== Testing EXO Planning Logic ===")
    
    try:
        from exo.worker.plan import plan, _model_needs_download, _load_model
        from exo.shared.types.worker.runners import RunnerIdle, RunnerConnected, RunnerLoaded
        from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
        from exo.shared.types.models import ModelMetadata
        from exo.shared.types.memory import Memory
        from exo.shared.types.common import Id, NodeId
        from exo.worker.runner.runner_supervisor import RunnerSupervisor
        
        print("✅ Successfully imported planning functions")
        
        # Let's test the individual planning functions
        print("\nTesting _model_needs_download logic...")
        
        # This is complex because we need to create proper mock objects
        # Let's just check if the functions are callable
        print("✅ Planning functions are accessible")
        
        # The issue is likely that we need to trace through the actual execution
        # to see what's happening with downloads and task creation
        
        return True
        
    except Exception as e:
        print(f"❌ Planning logic test failed: {e}")
        traceback.print_exc()
        return False

def test_download_status_tracking():
    """Test how download status is tracked."""
    print("\n=== Testing Download Status Tracking ===")
    
    try:
        from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending, DownloadOngoing
        
        print("Testing download status types...")
        
        # These are the status types that should be tracked
        print("Available download status types:")
        print("- DownloadPending: Initial state")
        print("- DownloadOngoing: Download in progress") 
        print("- DownloadCompleted: Download finished")
        
        print("✅ Download status types accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Download status test failed: {e}")
        traceback.print_exc()
        return False

def test_model_path_resolution():
    """Test if the model path resolution is working correctly."""
    print("\n=== Testing Model Path Resolution ===")
    
    try:
        from exo.worker.download.download_utils import build_model_path
        from exo.shared.constants import EXO_MODELS_DIR
        
        print(f"EXO models directory: {EXO_MODELS_DIR}")
        
        test_model = "gpt2"
        model_path = build_model_path(test_model)
        print(f"Model path for {test_model}: {model_path}")
        print(f"Path exists: {model_path.exists()}")
        
        # Check if there are any models downloaded
        if EXO_MODELS_DIR.exists():
            models = list(EXO_MODELS_DIR.iterdir())
            print(f"Downloaded models: {[m.name for m in models if m.is_dir()]}")
        else:
            print("❌ EXO models directory doesn't exist")
            
        return True
        
    except Exception as e:
        print(f"❌ Model path test failed: {e}")
        traceback.print_exc()
        return False

def test_runner_status_flow():
    """Test the expected runner status flow."""
    print("\n=== Testing Runner Status Flow ===")
    
    try:
        from exo.shared.types.worker.runners import (
            RunnerIdle, RunnerConnecting, RunnerConnected, 
            RunnerLoading, RunnerLoaded, RunnerWarmingUp, 
            RunnerReady, RunnerRunning, RunnerFailed
        )
        
        print("Expected runner status flow:")
        print("1. RunnerIdle - Initial state")
        print("2. RunnerConnecting - Connecting to group (multi-node only)")
        print("3. RunnerConnected - Connected to group")
        print("4. RunnerLoading - Loading model")
        print("5. RunnerLoaded - Model loaded")
        print("6. RunnerWarmingUp - Warming up model")
        print("7. RunnerReady - Ready for inference")
        print("8. RunnerRunning - Performing inference")
        print("9. RunnerFailed - Error state")
        
        print("✅ Runner status types accessible")
        
        # The issue might be that we're getting stuck at RunnerLoading
        # because the model loading is failing
        
        return True
        
    except Exception as e:
        print(f"❌ Runner status test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("Starting EXO planning process diagnostics...")
    
    # Test 1: Planning logic
    planning_success = test_planning_logic()
    
    # Test 2: Download status tracking
    download_success = test_download_status_tracking()
    
    # Test 3: Model path resolution
    path_success = test_model_path_resolution()
    
    # Test 4: Runner status flow
    status_success = test_runner_status_flow()
    
    print(f"\n=== Results ===")
    print(f"Planning logic: {'✅ PASS' if planning_success else '❌ FAIL'}")
    print(f"Download status: {'✅ PASS' if download_success else '❌ FAIL'}")
    print(f"Model paths: {'✅ PASS' if path_success else '❌ FAIL'}")
    print(f"Runner status: {'✅ PASS' if status_success else '❌ FAIL'}")
    
    print(f"\n🔍 Key Insights:")
    print("1. The planning process has a specific sequence:")
    print("   - _model_needs_download: Creates DownloadModel tasks")
    print("   - _load_model: Creates LoadModel tasks (only after download complete)")
    print("   - _ready_to_warmup: Creates StartWarmup tasks")
    
    print("2. LoadModel task creation requires:")
    print("   - All downloads complete globally across nodes")
    print("   - Runner in correct state (Idle for single-node, Connected for multi-node)")
    
    print("3. The issue might be:")
    print("   - Download not completing properly")
    print("   - Global download status not being updated")
    print("   - Race condition between download and LoadModel creation")
    print("   - Model files downloaded to wrong location")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
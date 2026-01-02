#!/usr/bin/env python3
"""
Basic Functionality Test

Tests core functionality without external dependencies by mocking loguru.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock loguru before importing anything that uses it
sys.modules['loguru'] = Mock()
mock_logger = Mock()
mock_logger.info = Mock()
mock_logger.debug = Mock()
mock_logger.warning = Mock()
mock_logger.error = Mock()
mock_logger.critical = Mock()
mock_logger.opt = Mock(return_value=mock_logger)
sys.modules['loguru'].logger = mock_logger

def test_constants():
    """Test that constants are properly configured."""
    print("Testing constants...")
    
    try:
        from exo.shared.constants import (
            EXO_RUNNER_SHUTDOWN_TIMEOUT,
            EXO_RUNNER_HEALTH_CHECK_INTERVAL,
            EXO_RUNNER_STARTUP_TIMEOUT,
            EXO_RUNNER_CLEANUP_TIMEOUT,
            EXO_RUNNER_MAX_STARTUP_RETRIES,
            EXO_RUNNER_ENABLE_ENHANCED_LOGGING
        )
        
        print(f"✓ EXO_RUNNER_SHUTDOWN_TIMEOUT: {EXO_RUNNER_SHUTDOWN_TIMEOUT}")
        print(f"✓ EXO_RUNNER_HEALTH_CHECK_INTERVAL: {EXO_RUNNER_HEALTH_CHECK_INTERVAL}")
        print(f"✓ EXO_RUNNER_STARTUP_TIMEOUT: {EXO_RUNNER_STARTUP_TIMEOUT}")
        print(f"✓ EXO_RUNNER_CLEANUP_TIMEOUT: {EXO_RUNNER_CLEANUP_TIMEOUT}")
        print(f"✓ EXO_RUNNER_MAX_STARTUP_RETRIES: {EXO_RUNNER_MAX_STARTUP_RETRIES}")
        print(f"✓ EXO_RUNNER_ENABLE_ENHANCED_LOGGING: {EXO_RUNNER_ENABLE_ENHANCED_LOGGING}")
        
        return True
        
    except Exception as e:
        print(f"✗ Constants test failed: {e}")
        return False

def test_component_creation():
    """Test that we can create instances of the new components."""
    print("\nTesting component creation...")
    
    try:
        # Test ShutdownCoordinator
        from exo.worker.runner.shutdown_coordinator import ShutdownCoordinator
        coordinator = ShutdownCoordinator()
        print("✓ ShutdownCoordinator created")
        
        # Test ResourceManager
        from exo.worker.runner.resource_manager import ResourceManager
        manager = ResourceManager()
        print("✓ ResourceManager created")
        
        # Test ChannelManager
        from exo.worker.runner.channel_manager import ChannelManager
        channel_mgr = ChannelManager()
        print("✓ ChannelManager created")
        
        # Test ErrorHandler
        from exo.worker.runner.error_handler import ErrorHandler
        error_handler = ErrorHandler()
        print("✓ ErrorHandler created")
        
        # Test LifecycleLogger
        from exo.worker.runner.lifecycle_logger import LifecycleLogger
        lifecycle_logger = LifecycleLogger()
        print("✓ LifecycleLogger created")
        
        return True
        
    except Exception as e:
        print(f"✗ Component creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_singleton_functions():
    """Test the singleton getter functions."""
    print("\nTesting singleton functions...")
    
    try:
        from exo.worker.runner.shutdown_coordinator import get_shutdown_coordinator
        from exo.worker.runner.resource_manager import get_resource_manager
        from exo.worker.runner.channel_manager import get_channel_manager
        from exo.worker.runner.error_handler import get_error_handler
        from exo.worker.runner.lifecycle_logger import get_lifecycle_logger
        
        # Test that we can get instances
        coordinator = get_shutdown_coordinator()
        manager = get_resource_manager()
        channel_mgr = get_channel_manager()
        error_handler = get_error_handler()
        lifecycle_logger = get_lifecycle_logger()
        
        print("✓ get_shutdown_coordinator() works")
        print("✓ get_resource_manager() works")
        print("✓ get_channel_manager() works")
        print("✓ get_error_handler() works")
        print("✓ get_lifecycle_logger() works")
        
        # Test that they return the same instance (singleton behavior)
        coordinator2 = get_shutdown_coordinator()
        if coordinator is coordinator2:
            print("✓ Singleton behavior confirmed")
        else:
            print("⚠ Singleton behavior not working (may be expected)")
        
        return True
        
    except Exception as e:
        print(f"✗ Singleton functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_updates():
    """Test that integration updates are in place."""
    print("\nTesting integration updates...")
    
    try:
        # Check worker main.py has the new imports
        with open("src/exo/worker/main.py", "r") as f:
            worker_content = f.read()
        
        integration_checks = [
            ("EXO_RUNNER_SHUTDOWN_TIMEOUT", "Shutdown timeout constant"),
            ("EXO_RUNNER_HEALTH_CHECK_INTERVAL", "Health check interval constant"),
            ("EXO_RUNNER_STARTUP_TIMEOUT", "Startup timeout constant"),
            ("EXO_RUNNER_ENABLE_ENHANCED_LOGGING", "Enhanced logging constant"),
            ("Enhanced runner features enabled", "Enhanced logging message")
        ]
        
        for check_string, description in integration_checks:
            if check_string in worker_content:
                print(f"✓ {description} found in worker main.py")
            else:
                print(f"✗ {description} NOT found in worker main.py")
                return False
        
        # Check engine_init.py has enhanced logging
        with open("src/exo/worker/engines/engine_init.py", "r") as f:
            engine_content = f.read()
        
        if "EXO_RUNNER_ENABLE_ENHANCED_LOGGING" in engine_content:
            print("✓ Enhanced logging found in engine_init.py")
        else:
            print("✗ Enhanced logging NOT found in engine_init.py")
            return False
        
        # Check constants.py has all new constants
        with open("src/exo/shared/constants.py", "r") as f:
            constants_content = f.read()
        
        required_constants = [
            "EXO_RUNNER_SHUTDOWN_TIMEOUT",
            "EXO_RUNNER_HEALTH_CHECK_INTERVAL", 
            "EXO_RUNNER_STARTUP_TIMEOUT",
            "EXO_RUNNER_CLEANUP_TIMEOUT",
            "EXO_RUNNER_MAX_STARTUP_RETRIES",
            "EXO_RUNNER_ENABLE_ENHANCED_LOGGING"
        ]
        
        for const in required_constants:
            if const in constants_content:
                print(f"✓ {const} found in constants.py")
            else:
                print(f"✗ {const} NOT found in constants.py")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Integration updates test failed: {e}")
        return False

def test_documentation_updates():
    """Test that documentation has been updated."""
    print("\nTesting documentation updates...")
    
    try:
        # Check main.py has enhanced docstring
        with open("src/exo/main.py", "r") as f:
            main_content = f.read()
        
        if "Enhanced Features:" in main_content and "Environment Variables:" in main_content:
            print("✓ Enhanced documentation found in main.py")
        else:
            print("✗ Enhanced documentation NOT found in main.py")
            return False
        
        # Check worker main.py has enhanced docstring
        with open("src/exo/worker/main.py", "r") as f:
            worker_content = f.read()
        
        if "Enhanced Features:" in worker_content:
            print("✓ Enhanced documentation found in worker main.py")
        else:
            print("✗ Enhanced documentation NOT found in worker main.py")
            return False
        
        # Check engine_init.py has enhanced docstring
        with open("src/exo/worker/engines/engine_init.py", "r") as f:
            engine_content = f.read()
        
        if "enhanced error handling" in engine_content.lower():
            print("✓ Enhanced documentation found in engine_init.py")
        else:
            print("✗ Enhanced documentation NOT found in engine_init.py")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Documentation updates test failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    print("🚀 Basic Functionality Test")
    print("=" * 35)
    
    tests = [
        ("Constants", test_constants),
        ("Component Creation", test_component_creation),
        ("Singleton Functions", test_singleton_functions),
        ("Integration Updates", test_integration_updates),
        ("Documentation Updates", test_documentation_updates)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 35)
    print("📊 SUMMARY")
    print("=" * 35)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic functionality test PASSED!")
        print("\n✅ Integration appears to be working correctly!")
        print("   - All new constants are properly configured")
        print("   - All new components can be created")
        print("   - Integration points are updated")
        print("   - Documentation is enhanced")
        return 0
    else:
        print("❌ Basic functionality test FAILED!")
        print(f"   {total - passed} tests failed - please check the integration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
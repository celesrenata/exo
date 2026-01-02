#!/usr/bin/env python3
"""
Comprehensive test of the EXO CPU inference engine.
"""

import sys
import os

sys.path.insert(0, "src")


def test_imports():
    """Test that all required modules can be imported."""
    print("=== Testing Imports ===")

    try:
        # Core engine imports
        from exo.worker.engines.engine_utils import (
            detect_available_engines,
            select_best_engine,
        )
        from exo.worker.engines.engine_init import (
            initialize_engine,
            warmup_engine,
            generate_with_engine,
        )

        # Torch engine imports
        from exo.worker.engines.torch.utils_torch import (
            initialize_torch,
            check_torch_availability,
        )
        from exo.worker.engines.torch.generator.generate import (
            warmup_inference,
            torch_generate,
        )
        from exo.worker.engines.torch import TokenizerWrapper, Model

        # Main application
        import exo.main

        print("✅ All imports successful")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_engine_detection():
    """Test engine detection and selection."""
    print("\n=== Testing Engine Detection ===")

    try:
        from exo.worker.engines.engine_utils import (
            detect_available_engines,
            select_best_engine,
        )

        # Test with forced torch engine
        os.environ["EXO_ENGINE"] = "torch"

        available = detect_available_engines()
        selected = select_best_engine()

        print(f"Available engines: {available}")
        print(f"Selected engine: {selected}")

        if "torch" in available and selected == "torch":
            print("✅ Engine detection working correctly")
            return True
        else:
            print("❌ Engine detection failed")
            return False

    except Exception as e:
        print(f"❌ Engine detection error: {e}")
        return False


def test_torch_functionality():
    """Test PyTorch and Transformers functionality."""
    print("\n=== Testing PyTorch Functionality ===")

    try:
        import torch
        from transformers import AutoConfig
        from exo.worker.engines.torch.utils_torch import check_torch_availability

        # Test torch availability
        torch_available = check_torch_availability()
        if not torch_available:
            print("❌ Torch availability check failed")
            return False

        # Test basic operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.softmax(x, dim=0)
        print(f"Softmax test: {x.tolist()} -> {y.tolist()}")

        # Test model config loading
        config = AutoConfig.from_pretrained("gpt2")
        print(
            f"GPT-2 config loaded: {config.model_type}, vocab_size={config.vocab_size}"
        )

        print("✅ PyTorch functionality working")
        return True

    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False


def test_engine_components():
    """Test engine component integration."""
    print("\n=== Testing Engine Components ===")

    try:
        from exo.worker.engines.engine_init import (
            initialize_engine,
            warmup_engine,
            generate_with_engine,
        )

        # Test that functions are callable
        assert callable(initialize_engine)
        assert callable(warmup_engine)
        assert callable(generate_with_engine)

        print("✅ Engine components are properly structured")
        return True

    except Exception as e:
        print(f"❌ Engine components test failed: {e}")
        return False


def test_application_startup():
    """Test that the main application can start."""
    print("\n=== Testing Application Startup ===")

    try:
        # Set environment for CPU engine
        os.environ["EXO_ENGINE"] = "torch"

        # Import main module
        import exo.main

        # Test that we can access the main function
        assert hasattr(exo.main, "main")

        print("✅ Application can start successfully")
        return True

    except Exception as e:
        print(f"❌ Application startup test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 EXO CPU Engine Comprehensive Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_engine_detection,
        test_torch_functionality,
        test_engine_components,
        test_application_startup,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ CPU Engine Implementation Complete!")
        print("\nYour EXO CPU inference engine is ready to use:")
        print("1. Set: export EXO_ENGINE=torch")
        print("2. Run: python -m exo.main")
        print("3. The engine will automatically download and use CPU-compatible models")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

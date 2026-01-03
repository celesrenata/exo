#!/usr/bin/env python3
"""
Debug script to investigate model loading failures.
"""

import sys
import os
import traceback
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_engine_selection():
    """Test engine selection and availability."""
    print("=== Testing Engine Selection ===")
    
    try:
        from exo.worker.engines.engine_utils import detect_available_engines, select_best_engine, get_engine_info
        
        available = detect_available_engines()
        print(f"Available engines: {available}")
        
        selected = select_best_engine()
        print(f"Selected engine: {selected}")
        
        info = get_engine_info()
        print(f"Engine info: {info}")
        
        return selected
        
    except Exception as e:
        print(f"❌ Engine selection failed: {e}")
        traceback.print_exc()
        return None

def test_torch_initialization():
    """Test basic PyTorch functionality."""
    print("\n=== Testing PyTorch Initialization ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x + 1
        print(f"Basic tensor test: {x} + 1 = {y}")
        
        # Test model loading components
        from transformers import AutoConfig, AutoTokenizer
        print("Transformers library available")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_model_path_resolution():
    """Test model path resolution."""
    print("\n=== Testing Model Path Resolution ===")
    
    try:
        from exo.worker.download.download_utils import build_model_path
        
        # Test with a common model ID
        test_model_id = "microsoft/DialoGPT-medium"
        model_path = build_model_path(test_model_id)
        print(f"Model path for {test_model_id}: {model_path}")
        print(f"Path exists: {model_path.exists()}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Model path resolution failed: {e}")
        traceback.print_exc()
        return None

def test_bound_instance_creation():
    """Test creating a minimal bound instance for testing."""
    print("\n=== Testing Bound Instance Creation ===")
    
    try:
        from exo.shared.types.worker.instances import BoundInstance
        from exo.shared.types.worker.shards import ShardMetadata
        from exo.shared.types.models import ModelMeta
        from exo.shared.types.memory import Memory
        
        # Create minimal test objects
        model_meta = ModelMeta(
            model_id="microsoft/DialoGPT-medium",
            storage_size=Memory.from_float_gb(1.0)
        )
        
        shard_metadata = ShardMetadata(
            model_meta=model_meta,
            start_layer=0,
            end_layer=12,
            n_layers=12,
            device_rank=0
        )
        
        # This is a simplified version - real BoundInstance needs more fields
        print("✅ Basic type creation successful")
        return shard_metadata
        
    except Exception as e:
        print(f"❌ Bound instance creation failed: {e}")
        traceback.print_exc()
        return None

def test_model_config_loading():
    """Test loading model configuration."""
    print("\n=== Testing Model Config Loading ===")
    
    try:
        from transformers import AutoConfig
        
        # Try to load a small model config
        test_model_id = "microsoft/DialoGPT-medium"
        
        print(f"Attempting to load config for {test_model_id}...")
        config = AutoConfig.from_pretrained(
            test_model_id,
            trust_remote_code=True,
        )
        
        print(f"✅ Config loaded successfully")
        print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
        print(f"Vocab size: {getattr(config, 'vocab_size', 0)}")
        print(f"Hidden size: {getattr(config, 'hidden_size', 0)}")
        
        return config
        
    except Exception as e:
        print(f"❌ Model config loading failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Run all diagnostic tests."""
    print("Starting EXO model loading diagnostics...")
    
    # Test 1: Engine selection
    selected_engine = test_engine_selection()
    if not selected_engine:
        print("💥 Cannot proceed - engine selection failed")
        return 1
    
    # Test 2: PyTorch functionality
    if not test_torch_initialization():
        print("💥 Cannot proceed - PyTorch not working")
        return 1
    
    # Test 3: Model path resolution
    model_path = test_model_path_resolution()
    if not model_path:
        print("💥 Cannot proceed - model path resolution failed")
        return 1
    
    # Test 4: Bound instance creation
    shard_metadata = test_bound_instance_creation()
    if not shard_metadata:
        print("💥 Cannot proceed - bound instance creation failed")
        return 1
    
    # Test 5: Model config loading
    config = test_model_config_loading()
    if not config:
        print("💥 Model config loading failed - this might be the issue!")
        return 1
    
    print("\n🎉 All basic tests passed!")
    print("\nThe model loading failure might be due to:")
    print("1. Network connectivity issues during model download")
    print("2. Insufficient disk space for model storage")
    print("3. Memory issues during model loading")
    print("4. Specific model compatibility issues")
    print("5. Missing model files or corrupted downloads")
    
    print("\nTo debug further:")
    print("1. Check available disk space")
    print("2. Check network connectivity")
    print("3. Try with a smaller model")
    print("4. Check the EXO logs for specific error messages")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
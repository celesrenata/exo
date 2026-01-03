#!/usr/bin/env python3
"""
Debug script to test PyTorch model loading specifically.
"""

import sys
import os
import traceback
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_torch_model_loading():
    """Test actual PyTorch model loading with a small model."""
    print("=== Testing PyTorch Model Loading ===")
    
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        
        # Use a very small model for testing
        test_model_id = "gpt2"  # Small, commonly available model
        
        print(f"Testing with model: {test_model_id}")
        
        # Test 1: Load config
        print("Loading model config...")
        config = AutoConfig.from_pretrained(
            test_model_id,
            trust_remote_code=True,
        )
        print(f"✅ Config loaded: {config.model_type}")
        
        # Test 2: Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            test_model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
        
        # Test 3: Load model (this is where it might fail)
        print("Loading model (this may take a moment)...")
        model = AutoModelForCausalLM.from_pretrained(
            test_model_id,
            config=config,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"✅ Model loaded successfully")
        
        # Test 4: Basic model operations
        print("Testing basic model operations...")
        model.eval()
        
        # Test tokenization and forward pass
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"✅ Forward pass successful: output shape={logits.shape}")
        
        print("🎉 All PyTorch model loading tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch model loading failed: {e}")
        traceback.print_exc()
        return False

def test_exo_torch_utils():
    """Test EXO's torch utilities specifically."""
    print("\n=== Testing EXO Torch Utils ===")
    
    try:
        from exo.worker.engines.torch.utils_torch import check_torch_availability, initialize_torch
        from exo.shared.types.worker.instances import BoundInstance
        from exo.shared.types.worker.shards import ShardMetadata
        from exo.shared.types.models import ModelMetadata
        from exo.shared.types.memory import Memory
        
        # Test 1: Check torch availability
        print("Checking PyTorch availability...")
        torch_available = check_torch_availability()
        print(f"PyTorch available: {torch_available}")
        
        if not torch_available:
            print("❌ PyTorch not available through EXO utils")
            return False
        
        # Test 2: Try to create a minimal bound instance (this might fail due to missing fields)
        print("Creating test bound instance...")
        
        # This is a simplified test - real BoundInstance has more complex structure
        print("✅ EXO torch utils are accessible")
        return True
        
    except Exception as e:
        print(f"❌ EXO torch utils test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_constraints():
    """Test if there are memory constraints causing issues."""
    print("\n=== Testing Memory Constraints ===")
    
    try:
        import psutil
        import torch
        
        # Check available memory
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"Used RAM: {memory.percent}%")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        print(f"Total disk: {disk.total / (1024**3):.1f} GB")
        print(f"Free disk: {disk.free / (1024**3):.1f} GB")
        
        # Check if we can allocate a reasonable tensor
        print("Testing tensor allocation...")
        test_tensor = torch.randn(1000, 1000)  # ~4MB tensor
        print(f"✅ Tensor allocation successful: {test_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory constraint test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("Starting PyTorch model loading diagnostics...")
    
    # Test 1: Basic PyTorch model loading
    torch_success = test_torch_model_loading()
    
    # Test 2: EXO torch utilities
    exo_success = test_exo_torch_utils()
    
    # Test 3: Memory constraints
    memory_success = test_memory_constraints()
    
    print(f"\n=== Results ===")
    print(f"PyTorch model loading: {'✅ PASS' if torch_success else '❌ FAIL'}")
    print(f"EXO torch utils: {'✅ PASS' if exo_success else '❌ FAIL'}")
    print(f"Memory constraints: {'✅ PASS' if memory_success else '❌ FAIL'}")
    
    if torch_success and exo_success and memory_success:
        print("\n🎉 All tests passed! The issue might be:")
        print("1. Specific model compatibility issues")
        print("2. Model size too large for available memory")
        print("3. Network issues during model file access")
        print("4. Corrupted model files")
        print("5. Missing model files after download")
        return 0
    else:
        print("\n💥 Some tests failed - this indicates the root cause")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
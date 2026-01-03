#!/usr/bin/env python3
"""
Test the torch loading fix.
"""

import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fixed_torch_loading():
    """Test PyTorch model loading without device_map."""
    print("=== Testing Fixed PyTorch Model Loading ===")
    
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        
        # Use a very small model for testing
        test_model_id = "gpt2"
        
        print(f"Testing with model: {test_model_id}")
        
        # Load config
        config = AutoConfig.from_pretrained(
            test_model_id,
            trust_remote_code=True,
        )
        print(f"✅ Config loaded: {config.model_type}")
        
        # Load model WITHOUT device_map
        print("Loading model without device_map...")
        model = AutoModelForCausalLM.from_pretrained(
            test_model_id,
            config=config,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            # Removed device_map="cpu" - this was causing the error
        )
        print(f"✅ Model loaded successfully")
        
        # Move to CPU explicitly if needed
        model = model.to('cpu')
        model.eval()
        print(f"✅ Model moved to CPU and set to eval mode")
        
        # Test tokenization and forward pass
        tokenizer = AutoTokenizer.from_pretrained(test_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"✅ Forward pass successful: output shape={logits.shape}")
        
        print("🎉 Fixed PyTorch model loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Fixed PyTorch model loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("Testing PyTorch loading fix...")
    
    success = test_fixed_torch_loading()
    
    if success:
        print("\n🎉 Fix successful! Model loading should now work.")
        return 0
    else:
        print("\n💥 Fix failed - still having issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
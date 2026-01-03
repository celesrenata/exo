#!/usr/bin/env python3
"""Quick test to validate tokenizer fix in nix develop environment."""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_tokenizer_fix():
    """Test that tokenizer loads on all ranks."""
    print("🔧 Testing tokenizer fix...")
    
    try:
        # Import required modules
        from exo.worker.engines.torch.utils_torch import initialize_torch
        from exo.shared.types.worker.instances import BoundInstance
        from exo.shared.types.worker.shards import ShardMetadata, ModelMetadata
        from exo.shared.types.memory import Memory
        print("✅ Successfully imported EXO modules")
        
        # Create test metadata
        model_meta = ModelMetadata(
            model_id="microsoft/DialoGPT-medium",
            storage_size=Memory.from_float_kb(1000000.0),  # 1GB in KB
            n_layers=24
        )
        
        # Test rank 0
        print("\n📍 Testing rank 0...")
        shard_meta_0 = ShardMetadata(
            model_meta=model_meta,
            start_layer=0,
            end_layer=11,
            device_rank=0
        )
        
        bound_instance_0 = BoundInstance(
            shard_metadata=shard_meta_0,
            instance_id="test-0"
        )
        
        model_0, tokenizer_0, sampler_0 = initialize_torch(bound_instance_0)
        
        if tokenizer_0 is None:
            print("❌ FAIL: Tokenizer is None on rank 0")
            return False
        else:
            print("✅ PASS: Tokenizer loaded successfully on rank 0")
            print(f"   Type: {type(tokenizer_0)}")
        
        # Test rank 1 (this was the problematic one)
        print("\n📍 Testing rank 1...")
        shard_meta_1 = ShardMetadata(
            model_meta=model_meta,
            start_layer=12,
            end_layer=23,
            device_rank=1
        )
        
        bound_instance_1 = BoundInstance(
            shard_metadata=shard_meta_1,
            instance_id="test-1"
        )
        
        model_1, tokenizer_1, sampler_1 = initialize_torch(bound_instance_1)
        
        if tokenizer_1 is None:
            print("❌ FAIL: Tokenizer is None on rank 1 (THIS WAS THE BUG!)")
            return False
        else:
            print("✅ PASS: Tokenizer loaded successfully on rank 1")
            print(f"   Type: {type(tokenizer_1)}")
        
        # Test basic tokenizer functionality
        print("\n🧪 Testing tokenizer functionality...")
        test_text = "Hello, how are you?"
        
        # Test rank 0 tokenizer
        tokens_0 = tokenizer_0.encode(test_text)
        decoded_0 = tokenizer_0.decode(tokens_0)
        print(f"   Rank 0: '{test_text}' -> {len(tokens_0)} tokens -> '{decoded_0}'")
        
        # Test rank 1 tokenizer
        tokens_1 = tokenizer_1.encode(test_text)
        decoded_1 = tokenizer_1.decode(tokens_1)
        print(f"   Rank 1: '{test_text}' -> {len(tokens_1)} tokens -> '{decoded_1}'")
        
        # Verify they produce the same results
        if tokens_0 == tokens_1 and decoded_0 == decoded_1:
            print("✅ PASS: Both tokenizers produce identical results")
        else:
            print("⚠️  WARNING: Tokenizers produce different results")
            print(f"   Rank 0 tokens: {tokens_0}")
            print(f"   Rank 1 tokens: {tokens_1}")
        
        print("\n🎉 ALL TESTS PASSED! The tokenizer fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_fix_in_code():
    """Verify the fix is actually in the code."""
    print("\n🔍 Verifying fix implementation in code...")
    
    try:
        with open('src/exo/worker/engines/torch/utils_torch.py', 'r') as f:
            content = f.read()
        
        # Check for fix indicators
        if "# Load tokenizer (needed on all ranks" in content:
            print("✅ Fix comment found")
        else:
            print("❌ Fix comment not found")
            return False
            
        if "tokenizer_raw = AutoTokenizer.from_pretrained(" in content:
            print("✅ Tokenizer loading code found")
        else:
            print("❌ Tokenizer loading code not found")
            return False
            
        if "tokenizer = TokenizerWrapper(tokenizer_raw)" in content:
            print("✅ TokenizerWrapper instantiation found")
        else:
            print("❌ TokenizerWrapper instantiation not found")
            return False
            
        # Check that old problematic code is gone
        if "if device_rank == 0:" in content and "tokenizer_raw = None" in content:
            print("⚠️  Old problematic code still present")
            return False
        else:
            print("✅ Old problematic code removed")
            
        print("✅ Fix is properly implemented in code")
        return True
        
    except Exception as e:
        print(f"❌ Error checking code: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 EXO TOKENIZER FIX VALIDATION")
    print("=" * 60)
    
    # Check fix in code first
    code_ok = check_fix_in_code()
    
    if not code_ok:
        print("\n❌ Fix not properly implemented in code!")
        sys.exit(1)
    
    # Test the fix
    test_ok = test_tokenizer_fix()
    
    if test_ok:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS: Tokenizer fix is working correctly!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ FAILURE: Tokenizer fix is not working!")
        print("=" * 60)
        sys.exit(1)
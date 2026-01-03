#!/usr/bin/env python3
"""
Check what model is actually being used and test it.
"""

import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=== Checking EXO Model Configuration ===")
    
    try:
        from exo.shared.models.model_cards import MODEL_CARDS
        
        print(f"Found {len(MODEL_CARDS)} model cards")
        
        # Show first few models
        for i, (short_id, card) in enumerate(list(MODEL_CARDS.items())[:3]):
            print(f"{i+1}. {short_id}")
            print(f"   Model ID: {card.model_id}")
            print(f"   Name: {card.name}")
            print(f"   Size: {card.metadata.storage_size}")
            print()
        
        # Test the first model (likely what EXO uses by default)
        first_model = list(MODEL_CARDS.values())[0]
        model_id = str(first_model.model_id)
        
        print(f"Testing first model: {model_id}")
        
        # Check if model exists locally
        from exo.worker.download.download_utils import build_model_path
        model_path = build_model_path(model_id)
        print(f"Model path: {model_path}")
        print(f"Exists: {model_path.exists()}")
        
        if model_path.exists():
            print("Files in model directory:")
            for file in sorted(model_path.iterdir()):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  {file.name} ({size_mb:.1f} MB)")
        
        # Try to load this model
        print(f"\nTesting model loading...")
        
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        try:
            # Test config
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            print(f"✅ Config: {config.model_type}")
            
            # Test tokenizer  
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            print(f"✅ Tokenizer: {tokenizer.vocab_size} tokens")
            
            # Test model loading
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                config=config,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"✅ Model loaded successfully")
            
            # Test inference
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            inputs = tokenizer("Hello", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            print(f"✅ Inference successful: {outputs.logits.shape}")
            
            print(f"\n🎉 Model {model_id} works perfectly!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading/testing failed: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ The model loading works fine in isolation.")
        print("The issue must be elsewhere in the EXO system.")
    else:
        print("\n❌ Found the root cause - model loading is failing.")
    
    sys.exit(0 if success else 1)
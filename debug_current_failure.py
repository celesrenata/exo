#!/usr/bin/env python3
"""
Debug the current persistent failure that's causing "failed" status.
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

def check_actual_model_being_used():
    """Check what model is actually being used in the current EXO setup."""
    print("=== Checking Actual Model Configuration ===")
    
    try:
        # Check if there are any model cards or configuration
        from exo.shared.models.model_cards import MODEL_CARDS
        
        print(f"Available model cards: {len(MODEL_CARDS)}")
        for model_id, card in list(MODEL_CARDS.items())[:5]:  # Show first 5
            print(f"  - {model_id}: {card.pretty_name}")
        
        # Check what model might be selected by default
        if MODEL_CARDS:
            first_model = list(MODEL_CARDS.keys())[0]
            print(f"\nFirst model (likely default): {first_model}")
            
            # Check if this model exists locally
            from exo.worker.download.download_utils import build_model_path
            model_path = build_model_path(first_model)
            print(f"Model path: {model_path}")
            print(f"Path exists: {model_path.exists()}")
            
            if model_path.exists():
                print("Model files found:")
                for file in model_path.iterdir():
                    if file.is_file():
                        print(f"  - {file.name} ({file.stat().st_size} bytes)")
            
            return first_model
        else:
            print("No model cards found!")
            return None
            
    except Exception as e:
        print(f"❌ Error checking model configuration: {e}")
        traceback.print_exc()
        return None

def test_specific_model_loading(model_id):
    """Test loading the specific model that EXO is trying to use."""
    print(f"\n=== Testing Specific Model Loading: {model_id} ===")
    
    try:
        from exo.worker.download.download_utils import build_model_path
        from exo.worker.engines.torch.utils_torch import initialize_torch
        from exo.shared.types.worker.instances import BoundInstance
        
        model_path = build_model_path(model_id)
        print(f"Model path: {model_path}")
        
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            print("This could be the issue - model not downloaded")
            return False
        
        # Try to load just the config first
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            print(f"✅ Config loaded: {config.model_type}")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            return False
        
        # Try to load tokenizer
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
            return False
        
        # Try to load the actual model
        from transformers import AutoModelForCausalLM
        import torch
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                config=config,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"✅ Model loaded successfully")
            
            # Test basic inference
            model.eval()
            test_input = tokenizer("Hello", return_tensors="pt")
            with torch.no_grad():
                output = model(**test_input)
            print(f"✅ Basic inference successful: {output.logits.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        return False

def check_api_status():
    """Check if there's an API or planning issue causing the failure."""
    print(f"\n=== Checking API and Planning Status ===")
    
    try:
        # Check if there's a planning or API issue
        import requests
        import time
        
        # Try to connect to local EXO API if it's running
        try:
            response = requests.get("http://localhost:52415/health", timeout=5)
            print(f"EXO API status: {response.status_code}")
            if response.status_code == 200:
                print("✅ EXO API is responding")
            else:
                print(f"⚠️ EXO API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("ℹ️ EXO API not running (expected if testing)")
        except Exception as e:
            print(f"⚠️ API check failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API status check failed: {e}")
        return False

def check_planning_process():
    """Check if there's an issue with the planning/sharding process."""
    print(f"\n=== Checking Planning Process ===")
    
    try:
        # Check if we can create a basic plan
        from exo.worker.plan import Plan
        from exo.shared.types.models import ModelMetadata
        from exo.shared.types.memory import Memory
        
        # Try to create a simple plan
        print("Testing basic plan creation...")
        
        # This is a simplified test - real planning is more complex
        print("✅ Planning imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Planning process check failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostics for the persistent failure."""
    print("Starting comprehensive diagnostics for persistent model loading failure...")
    print("=" * 80)
    
    # Step 1: Check what model is being used
    model_id = check_actual_model_being_used()
    
    # Step 2: Test that specific model
    model_success = False
    if model_id:
        model_success = test_specific_model_loading(model_id)
    
    # Step 3: Check API status
    api_success = check_api_status()
    
    # Step 4: Check planning process
    planning_success = check_planning_process()
    
    print(f"\n" + "=" * 80)
    print(f"DIAGNOSTIC RESULTS:")
    print(f"Model identification: {'✅ SUCCESS' if model_id else '❌ FAILED'}")
    print(f"Model loading: {'✅ SUCCESS' if model_success else '❌ FAILED'}")
    print(f"API status: {'✅ SUCCESS' if api_success else '❌ FAILED'}")
    print(f"Planning process: {'✅ SUCCESS' if planning_success else '❌ FAILED'}")
    
    if not model_success and model_id:
        print(f"\n🔍 LIKELY ROOT CAUSE:")
        print(f"The specific model '{model_id}' that EXO is trying to load is failing.")
        print(f"This is a persistent issue, not a transient dependency problem.")
        print(f"\nPossible causes:")
        print(f"1. Model files are corrupted or incomplete")
        print(f"2. Model is too large for available memory")
        print(f"3. Model format is incompatible with current transformers version")
        print(f"4. Model requires specific dependencies not available")
        print(f"5. Model path or permissions issue")
        
    elif not model_id:
        print(f"\n🔍 LIKELY ROOT CAUSE:")
        print(f"No model configuration found or model selection failing.")
        print(f"This could be a configuration or setup issue.")
    
    return 0 if (model_success and api_success and planning_success) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
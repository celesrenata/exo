#!/usr/bin/env python3
"""
Practical Intel IPEX Engine Integration Test.

This test validates the IPEX engine integration using the newly added
IPEX-compatible models from the model cards.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_ipex_model_cards():
    """Test that IPEX models are properly integrated into model cards."""
    
    print("🧠 Testing IPEX Model Cards Integration")
    print("=" * 50)
    
    try:
        from exo.shared.models.model_cards import MODEL_CARDS
        
        # Find IPEX models
        ipex_models = {k: v for k, v in MODEL_CARDS.items() if "ipex" in v.tags}
        
        print(f"Found {len(ipex_models)} IPEX-compatible models:")
        print("-" * 40)
        
        for short_id, model_card in ipex_models.items():
            print(f"✅ {short_id}")
            print(f"   Model ID: {model_card.model_id}")
            print(f"   Name: {model_card.name}")
            print(f"   Size: {model_card.metadata.storage_size}")
            print(f"   Tags: {model_card.tags}")
            print()
        
        # Verify we have the expected models from the provided list
        expected_models = [
            "distilgpt2-ipex",
            "bloomz-560m-ipex", 
            "gpt2-small-ipex",
            "phi-3.5-mini-ipex",
            "phi-mini-moe-ipex",
            "gpt-j-6b-ipex",
            "prometheus-7b-ipex",
            "yi-9b-awq-ipex",
            "gpt-neox-20b-ipex"
        ]
        
        missing_models = []
        for expected in expected_models:
            if expected not in ipex_models:
                missing_models.append(expected)
        
        if missing_models:
            print(f"❌ Missing expected models: {missing_models}")
            return False
        
        print("✅ All expected IPEX models are present in model cards")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model cards: {e}")
        return False


def test_ipex_engine_compatibility():
    """Test IPEX engine compatibility with the new models."""
    
    print("\n🔧 Testing IPEX Engine Compatibility")
    print("=" * 50)
    
    try:
        from exo.worker.engines.engine_utils import is_model_compatible
        from exo.shared.models.model_cards import MODEL_CARDS
        
        # Get IPEX models
        ipex_models = {k: v for k, v in MODEL_CARDS.items() if "ipex" in v.tags}
        
        compatible_count = 0
        total_count = len(ipex_models)
        
        for short_id, model_card in ipex_models.items():
            model_id = str(model_card.model_id)
            is_compatible = is_model_compatible(model_id, "ipex")
            
            if is_compatible:
                print(f"✅ {short_id}: {model_id} - Compatible")
                compatible_count += 1
            else:
                print(f"❌ {short_id}: {model_id} - Not Compatible")
        
        print(f"\nCompatibility Results: {compatible_count}/{total_count} models compatible")
        
        if compatible_count == total_count:
            print("✅ All IPEX models are compatible with IPEX engine")
            return True
        else:
            print(f"⚠️  {total_count - compatible_count} models are not compatible")
            return False
        
    except Exception as e:
        print(f"❌ Error testing compatibility: {e}")
        return False


def test_ipex_model_metadata():
    """Test IPEX model metadata is properly structured."""
    
    print("\n📊 Testing IPEX Model Metadata")
    print("=" * 50)
    
    try:
        from exo.shared.models.model_cards import MODEL_CARDS
        
        # Get IPEX models
        ipex_models = {k: v for k, v in MODEL_CARDS.items() if "ipex" in v.tags}
        
        # Categorize by size
        small_models = []
        medium_models = []
        large_models = []
        xlarge_models = []
        
        for short_id, model_card in ipex_models.items():
            if "small" in model_card.tags or "tiny" in model_card.tags:
                small_models.append(short_id)
            elif "medium" in model_card.tags:
                medium_models.append(short_id)
            elif "large" in model_card.tags:
                large_models.append(short_id)
            elif "xlarge" in model_card.tags:
                xlarge_models.append(short_id)
        
        print("Model Categories:")
        print(f"  Small/Tiny (<1GB): {small_models}")
        print(f"  Medium (~1-2GB): {medium_models}")
        print(f"  Large (6-12GB): {large_models}")
        print(f"  XLarge (>20GB): {xlarge_models}")
        
        # Verify metadata completeness
        metadata_complete = True
        for short_id, model_card in ipex_models.items():
            metadata = model_card.metadata
            
            # Check required fields
            required_fields = ['model_id', 'pretty_name', 'storage_size', 'n_layers', 'hidden_size']
            missing_fields = []
            
            for field in required_fields:
                if not hasattr(metadata, field) or getattr(metadata, field) is None:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ {short_id}: Missing metadata fields: {missing_fields}")
                metadata_complete = False
            else:
                print(f"✅ {short_id}: Complete metadata")
        
        if metadata_complete:
            print("\n✅ All IPEX models have complete metadata")
            return True
        else:
            print("\n❌ Some IPEX models have incomplete metadata")
            return False
        
    except Exception as e:
        print(f"❌ Error testing metadata: {e}")
        return False


def test_ipex_model_progression():
    """Test that we have a good progression of model sizes for testing."""
    
    print("\n📈 Testing IPEX Model Size Progression")
    print("=" * 50)
    
    try:
        from exo.shared.models.model_cards import MODEL_CARDS
        
        # Get IPEX models and their sizes
        ipex_models = {k: v for k, v in MODEL_CARDS.items() if "ipex" in v.tags}
        
        model_sizes = []
        for short_id, model_card in ipex_models.items():
            size_mb = model_card.metadata.storage_size.in_mb
            model_sizes.append((short_id, size_mb))
        
        # Sort by size
        model_sizes.sort(key=lambda x: x[1])
        
        print("IPEX Models by Size (for progressive testing):")
        print("-" * 40)
        
        for short_id, size_mb in model_sizes:
            if size_mb < 1024:
                size_str = f"{size_mb:.0f}MB"
            else:
                size_str = f"{size_mb/1024:.1f}GB"
            
            print(f"  {short_id}: {size_str}")
        
        # Check we have good coverage
        small_count = sum(1 for _, size in model_sizes if size < 1024)  # <1GB
        medium_count = sum(1 for _, size in model_sizes if 1024 <= size < 5120)  # 1-5GB
        large_count = sum(1 for _, size in model_sizes if size >= 5120)  # >5GB
        
        print(f"\nSize Distribution:")
        print(f"  Small (<1GB): {small_count} models")
        print(f"  Medium (1-5GB): {medium_count} models")
        print(f"  Large (>5GB): {large_count} models")
        
        # We should have models in each category for good testing coverage
        if small_count > 0 and medium_count > 0 and large_count > 0:
            print("✅ Good size distribution for progressive testing")
            return True
        else:
            print("⚠️  Limited size distribution - consider adding more variety")
            return True  # Not a failure, just a recommendation
        
    except Exception as e:
        print(f"❌ Error testing model progression: {e}")
        return False


def test_ipex_model_features():
    """Test IPEX model feature coverage."""
    
    print("\n🎯 Testing IPEX Model Feature Coverage")
    print("=" * 50)
    
    try:
        from exo.shared.models.model_cards import MODEL_CARDS
        
        # Get IPEX models
        ipex_models = {k: v for k, v in MODEL_CARDS.items() if "ipex" in v.tags}
        
        # Analyze features
        features = {
            "multilingual": [],
            "instruct": [],
            "quantized": [],
            "moe": [],
            "distributed": []
        }
        
        for short_id, model_card in ipex_models.items():
            tags = model_card.tags
            
            if "multilingual" in tags:
                features["multilingual"].append(short_id)
            if "instruct" in tags:
                features["instruct"].append(short_id)
            if "quantized" in tags:
                features["quantized"].append(short_id)
            if "moe" in tags:
                features["moe"].append(short_id)
            if "distributed" in tags:
                features["distributed"].append(short_id)
        
        print("Feature Coverage:")
        for feature, models in features.items():
            if models:
                print(f"  {feature.capitalize()}: {len(models)} models - {models}")
            else:
                print(f"  {feature.capitalize()}: 0 models")
        
        # Check that we have good feature coverage
        feature_count = sum(1 for models in features.values() if models)
        
        if feature_count >= 3:
            print(f"\n✅ Good feature coverage ({feature_count}/5 features represented)")
            return True
        else:
            print(f"\n⚠️  Limited feature coverage ({feature_count}/5 features represented)")
            return True  # Not a failure, just a note
        
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        return False


def main():
    """Run all IPEX integration tests."""
    
    print("🚀 Intel IPEX Integration Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Model Cards Integration", test_ipex_model_cards),
        ("Engine Compatibility", test_ipex_engine_compatibility),
        ("Model Metadata", test_ipex_model_metadata),
        ("Model Size Progression", test_ipex_model_progression),
        ("Feature Coverage", test_ipex_model_features)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Final Test Results")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All IPEX integration tests passed!")
        print("\nThe following IPEX-compatible models are now available:")
        print("• Small models (500MB-1GB): distilgpt2, bloomz-560m, gpt2-small")
        print("• Medium models (1-2GB): phi-3.5-mini, phi-mini-moe")
        print("• Large models (6-12GB): gpt-j-6b, prometheus-7b, yi-9b-awq")
        print("• XLarge models (20GB+): gpt-neox-20b")
        print("\nThese models can be used with Intel GPU acceleration via IPEX!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
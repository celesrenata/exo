#!/usr/bin/env python3
"""Test weight loading implementation for Llama transformer."""

import sys

sys.path.insert(0, "src")

from exo.worker.engines.tinygrad.llama_transformer import (
    LlamaConfig,
    LlamaTransformer,
    create_weight_name_mapping,
    get_parameter_from_name,
)


def test_weight_name_mapping():
    """Test that weight name mapping creates correct mappings."""
    # Create a small test config
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    # Create model
    model = LlamaTransformer(config)

    # Test weight name mapping
    weight_map = create_weight_name_mapping(model)
    print(f"✓ Created weight mapping with {len(weight_map)} weights")

    # Test getting parameters by name
    embed_weight = get_parameter_from_name(model, "model.embed_tokens.weight")
    print(f"✓ Retrieved embedding weight: shape {embed_weight.shape}")

    layer_0_q = get_parameter_from_name(model, "model.layers.0.self_attn.q_proj.weight")
    print(f"✓ Retrieved layer 0 Q projection: shape {layer_0_q.shape}")

    # Test that all expected weights are in the mapping
    expected_weights = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for i in range(config.num_hidden_layers):
        expected_weights.extend(
            [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
            ]
        )

    all_present = all(name in weight_map for name in expected_weights)
    print(f"✓ All {len(expected_weights)} expected weights present: {all_present}")

    assert all_present, "Not all expected weights are present in mapping"
    assert len(weight_map) == len(expected_weights), (
        f"Expected {len(expected_weights)} weights, got {len(weight_map)}"
    )

    print()
    print("✅ Weight loading implementation test passed!")


if __name__ == "__main__":
    test_weight_name_mapping()

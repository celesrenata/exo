# Bugfix Requirements Document

## Introduction

Qwen3.5-4B and Qwen3.6-27B models produce whitespace/garbage output during tensor-parallel inference on the gremlin cluster. The root cause is that the native HuggingFace linear attention layers' `Cache` object is not maintaining recurrent state across autoregressive decode steps. Without persistent state, each token is processed in isolation by the Gated DeltaNet layers (75% of all layers), resulting in incoherent output.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the `_forward_linear_attn_layer()` method delegates to a native HuggingFace linear attention layer with `cache_params=self._native_cache` THEN the system produces whitespace/garbage tokens because the Cache object does not persist recurrent state between decode steps

1.2 WHEN `_create_native_cache()` constructs a `Cache(layers=cache_layers)` with `LinearAttentionLayer()` instances THEN the system creates cache layers that are not properly initialized with the model's dimensions (batch size, num_v_heads, key_head_dim, value_head_dim), so `has_previous_state()` never returns True

1.3 WHEN the native layer's `forward()` is called with `cache_params` and `attention_mask` keyword arguments THEN the system passes arguments that do not match the native layer's actual forward signature in transformers 5.7.0, causing the cache to be silently ignored or the state to not be written back

1.4 WHEN multiple decode steps are executed sequentially for Qwen3.5/3.6 models THEN the system treats each token independently (no accumulated recurrent state) and produces whitespace instead of coherent text

### Expected Behavior (Correct)

2.1 WHEN the `_forward_linear_attn_layer()` method delegates to a native HuggingFace linear attention layer THEN the system SHALL pass cache parameters that match the native layer's forward signature in transformers 5.7.0, and the recurrent state SHALL persist across decode steps

2.2 WHEN `_create_native_cache()` constructs the cache THEN the system SHALL initialize the cache with the correct dimensions and configuration so that `has_previous_state()` returns True after the first forward call

2.3 WHEN the native layer's `forward()` is called THEN the system SHALL use the correct keyword argument names and types as defined by the transformers 5.7.0 API for Gated DeltaNet linear attention layers

2.4 WHEN multiple decode steps are executed sequentially for Qwen3.5/3.6 models THEN the system SHALL accumulate recurrent state of shape `(B, num_v_heads, key_head_dim, value_head_dim)` across steps and produce coherent text output

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the model is Phi-4 or Qwen2.5-7B-Instruct (models without linear attention layers) THEN the system SHALL CONTINUE TO produce coherent text output via the full attention path with KV cache

3.2 WHEN a layer is a full attention layer (25% of Qwen3.5/3.6 layers) THEN the system SHALL CONTINUE TO use the existing KV cache mechanism with QKV projections, RoPE, and SDPA

3.3 WHEN tensor parallelism is applied to MLP layers (gate/up/down projections) THEN the system SHALL CONTINUE TO shard and all-reduce correctly regardless of the linear attention cache fix

3.4 WHEN the custom fallback implementation in `_forward_linear_attn_layer()` is used (when native layers are not available) THEN the system SHALL CONTINUE TO maintain its own `_linear_attn_states` dict for conv and recurrent state

3.5 WHEN linear attention weights are loaded THEN the system SHALL CONTINUE TO keep them redundant (not sharded) on all ranks as defined by `_LINEAR_ATTN_PATTERN`

---

## Bug Condition

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type ForwardCall
  OUTPUT: boolean
  
  // Returns true when the forward call targets a native linear attention layer
  // during autoregressive decode (where state persistence is required)
  RETURN X.layer_type = "linear_attention"
     AND X.layer_idx IN native_linear_attn_layers
     AND X.decode_step > 0
END FUNCTION
```

## Property Specification

```pascal
// Property: Fix Checking — Cache State Persistence
FOR ALL X WHERE isBugCondition(X) DO
  state_before ← get_cache_state(X.layer_idx, X.decode_step - 1)
  result ← F'(X)
  state_after ← get_cache_state(X.layer_idx, X.decode_step)
  ASSERT state_after ≠ zeros(B, num_v_heads, key_head_dim, value_head_dim)
  ASSERT state_after ≠ state_before  // state was updated
  ASSERT result ≠ whitespace_tokens  // output is coherent
END FOR
```

## Preservation Goal

```pascal
// Property: Preservation Checking — Non-linear-attention layers unchanged
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
END FOR
```

This ensures that for all non-buggy inputs (full attention layers, non-hybrid models, first decode step), the fixed code behaves identically to the original.

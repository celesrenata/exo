# Bugfix Requirements Document

## Introduction

Tensor parallel inference on 4 gremlin nodes running Qwen3.5-27B produces garbage output (whitespace tokens only). The root cause is that the streaming loader path creates `TensorParallelShard` with a raw state dict instead of a HuggingFace model object, so `_extract_native_rotary_emb()` finds nothing. The manual 1D RoPE fallback runs instead of the required 3D Multi-Resolution RoPE (MRoPE), corrupting all 16 full attention layers and causing the model to converge to a degenerate fixed point. A secondary issue exists where `q_proj.bias` sharding uses incorrect shard sizes for doubled Q projections (Q + gate).

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the streaming loader creates a TensorParallelShard with `model=sharded_state_dict` (a dict) and `pre_sharded=True` THEN the system skips `_extract_native_rotary_emb()` and sets `_native_rotary_emb = None`

1.2 WHEN `_native_rotary_emb` is None and a Qwen3.5-27B full attention layer executes `_apply_rotary_pos_emb()` THEN the system applies standard 1D RoPE instead of 3D MRoPE with `mrope_section=[11, 11, 10]`, `partial_rotary_factor=0.25`, and `mrope_interleaved=true`

1.3 WHEN 1D RoPE is applied to all 16 full attention layers (every 4th layer: 3, 7, 11, ..., 63) THEN the system produces degenerate logits (top5_ids=[220, 198, 13, 16, 271]) with constant hidden state norm (113.5) across all decode steps

1.4 WHEN a Qwen3.5 MRoPE model is loaded via the streaming path and native rotary embedding is unavailable THEN the system silently falls back to incorrect RoPE without any error or warning to the operator

1.5 WHEN `q_proj.bias` exists on a doubled Q projection (Q + gate) and the bias is sharded THEN the system uses `heads_per_rank * head_dim` instead of `heads_per_rank * head_dim * 2` for the shard size, corrupting gate bias values

### Expected Behavior (Correct)

2.1 WHEN the streaming loader creates a TensorParallelShard for a Qwen3.5 model THEN the system SHALL instantiate `Qwen3_5RotaryEmbedding` from the model config and pass it to the shard via a `native_rotary_emb` constructor parameter

2.2 WHEN `_native_rotary_emb` is available and a full attention layer executes `_apply_rotary_pos_emb()` THEN the system SHALL apply 3D MRoPE using the native rotary embedding module with correct `mrope_section`, `partial_rotary_factor`, and interleaved layout

2.3 WHEN tensor parallel inference runs on Qwen3.5-27B with correct MRoPE THEN the system SHALL produce coherent text output with varying hidden state norms across decode steps and diverse top-5 predicted tokens

2.4 WHEN a Qwen3.5 MRoPE model is detected (config has `mrope_section`) but native rotary embedding cannot be instantiated THEN the system SHALL raise a RuntimeError with a descriptive message instead of silently using the broken manual fallback

2.5 WHEN `q_proj.bias` exists on a doubled Q projection (Q + gate) and the bias is sharded THEN the system SHALL derive the shard size from the actual tensor shape (matching the weight sharding logic) to correctly split both Q and gate bias portions

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a model is loaded via the non-streaming path (full HuggingFace model object) THEN the system SHALL CONTINUE TO extract native rotary embedding from the model's `model.rotary_emb` attribute

3.2 WHEN a model does not use MRoPE (standard 1D RoPE models like Llama, Mistral) THEN the system SHALL CONTINUE TO apply the manual partial-RoPE fallback correctly

3.3 WHEN native linear attention layers (Qwen3_5GatedDeltaNet) are loaded via the streaming path THEN the system SHALL CONTINUE TO instantiate and attach them to the shard correctly

3.4 WHEN tensor parallel sharding splits `q_proj.weight` for doubled Q projections THEN the system SHALL CONTINUE TO use `heads_per_rank * head_dim * 2` as the shard size for weights

3.5 WHEN pipeline parallel inference runs on Qwen3.5-4B (non-streaming path) THEN the system SHALL CONTINUE TO produce coherent output using the native rotary embedding extracted from the HuggingFace model object

3.6 WHEN the streaming loader processes MLP weights (gate_up_proj, down_proj) for hybrid models with varying intermediate sizes THEN the system SHALL CONTINUE TO shard them correctly based on actual tensor dimensions

---

## Bug Condition (Formal)

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type ModelLoadConfig
  OUTPUT: boolean

  // The bug triggers when ALL of these hold:
  // 1. Model uses MRoPE (has mrope_section in config)
  // 2. Loading via streaming path (state dict, not HF model object)
  // 3. No native rotary_emb is passed to TensorParallelShard
  RETURN X.model_has_mrope = true
     AND X.load_path = "streaming"
     AND X.native_rotary_emb = None
END FUNCTION
```

```pascal
// Property: Fix Checking — MRoPE models get native rotary embedding
FOR ALL X WHERE isBugCondition(X) DO
  shard ← create_tensor_parallel_shard(X)
  ASSERT shard._native_rotary_emb IS NOT None
  ASSERT type(shard._native_rotary_emb).__name__ CONTAINS "RotaryEmbedding"

  // Forward pass produces non-degenerate output
  result ← shard.forward(input_ids, position_ids)
  ASSERT result.hidden_state_norms VARY across decode steps
  ASSERT result.top5_token_ids != [220, 198, 13, 16, 271]
END FOR
```

```pascal
// Property: Preservation Checking — non-MRoPE models unchanged
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)
  // Manual RoPE fallback still works for standard models
  // HF model path still extracts native rotary_emb from model object
END FOR
```

# Bug: Pipeline Parallel Multi-Turn Conversation Broken

## Summary

The pipeline parallel inference engine (PyTorch XPU, Qwen3.5-4B across 4 gremlin nodes) does not correctly handle multi-turn conversations. When a conversation has multiple messages, the model responds to the FIRST user message instead of the LATEST one. It appears to repeat the assistant's previous response verbatim.

## Reproduction

```bash
curl -s -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {"role": "user", "content": "Hello world"},
      {"role": "assistant", "content": "Hello! How can I help you today?"},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "stream": false,
    "enable_thinking": false
  }'
```

**Expected**: "The capital of France is Paris."
**Actual**: "Hello! How can I help you today?" (repeats the previous assistant response)

Single-turn works correctly:
```bash
curl -s -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"stream":false,"enable_thinking":false}'
```
**Result**: "4" (correct)

## Root Cause Analysis

The pipeline parallel prefill processes the full tokenized prompt across 4 nodes. For multi-turn conversations, the chat template produces a longer token sequence containing all messages. The bug is in how the pipeline handles this longer prefill:

1. `reset_state()` IS called before each new request (verified deployed)
2. The chat template IS applied correctly (the full multi-turn prompt is tokenized)
3. The prefill processes the tokens through the pipeline
4. **The model generates tokens that match the PREVIOUS assistant response** — suggesting the prefill is not correctly updating the KV cache with the full context, OR the model is attending only to the first portion of the prefill

## Key Files

- **Engine entry point**: `src/exo/worker/engines/pytorch_xpu/engine.py` — `_build_generator()` builds the prompt and calls `pipeline_parallel_generate()`
- **Pipeline generator (rank 0)**: `src/exo/worker/engines/pytorch_xpu/pipeline_generator.py` — `pipeline_parallel_generate()` handles prefill and decode
- **Pipeline shard**: `src/exo/worker/engines/pytorch_xpu/pipeline_parallel_shard.py` — `forward()` and `_forward_layer()` process tokens through layers
- **KV Cache**: `pipeline_parallel_shard.py` — `self._hf_cache` (DynamicCache or GatedDeltaNetCache)
- **Reset**: `pipeline_parallel_shard.py` — `reset_state()` clears the cache

## What Works

- Single-turn conversations (1 user message → correct response)
- `reset_state()` between requests (no cross-request contamination)
- `enable_thinking=False` (no thinking tokens generated)
- Token generation mechanics (sampling, EOS detection)

## What's Broken

- Multi-turn conversations: model responds to first message, ignores subsequent messages
- The generated output is the EXACT text of the previous assistant message — suggesting the model is "continuing" from where the first turn ended rather than processing the full multi-turn prompt

## Hypothesis

The prefill phase processes the full multi-turn token sequence, but the pipeline's handling of the KV cache during prefill may be incorrect:

1. **Chunked prefill issue**: If the prefill is chunked (processed in segments), the later chunks (containing the second user message) might not be correctly updating the KV cache positions
2. **Position IDs**: The rotary embeddings might not be getting correct position IDs for the full sequence length after prefill
3. **Cache sequence length tracking**: After prefill, the cache might report a shorter sequence length than the actual prefilled tokens, causing decode to attend only to the first portion

## Cluster Info

- 4 nodes: gremlin-1 (10.1.1.12), gremlin-2 (10.1.1.13), gremlin-3 (10.1.1.14), gremlin-4 (10.1.1.15)
- Model: Qwen/Qwen3.5-4B (32 layers, pipeline-sharded across 4 nodes, 8 layers each)
- Backend: PyTorch XPU on Intel Arc Meteor Lake-P iGPUs
- Branch: `twenty-tps-research`
- Last known good single-turn: commit `f4e802e9` and later

## Debugging Steps

1. Add logging in `pipeline_parallel_generate()` to print the prompt token count and the prefill output shape
2. After prefill completes, log `model.get_cache_seq_length()` to verify the full prompt was cached
3. Check if the first decode token after prefill is the correct next token (not a repeat of earlier content)
4. Compare the tokenized multi-turn prompt vs single-turn to verify the chat template is correct
5. Check position IDs during prefill — are they sequential from 0 to prompt_length-1?

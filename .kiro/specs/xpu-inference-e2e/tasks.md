# Tasks

## 1. Fix TensorParallelShard Weight Key Mapping

- [x] 1.1 Add `_detect_architecture()` method to TensorParallelShard that inspects `sharded_state_dict` keys and returns the model architecture (qwen_llama or phi)
- [x] 1.2 Modify `shard_weights()` to detect fused QKV weights (`qkv_proj.weight`) and split them into separate `q_proj.weight`, `k_proj.weight`, `v_proj.weight` entries in `sharded_state_dict`
- [x] 1.3 Update `_get_weight()` to raise a descriptive KeyError with the missing key name and available keys with the same layer prefix
- [x] 1.4 Add LayerNorm support alongside RMSNorm — detect `input_layernorm.bias` presence and apply `F.layer_norm` instead of `_rms_norm` when bias is present
- [x] 1.5 Write unit tests for architecture detection, fused QKV splitting, and key consistency

## 2. Fix Placement Engine Instance Type Override

- [x] 2.1 Modify the single-node override in `src/exo/master/placement.py` (lines 206–211) to preserve `InstanceMeta.PyTorchXPURing` — only force MlxRing when the requested instance_meta is not PyTorchXPURing
- [x] 2.2 Write a unit test that verifies single-node placement with `InstanceMeta.PyTorchXPURing` produces a `PyTorchXPURingInstance`

## 3. Create Model Cards for Validation Models

- [x] 3.1 Create `resources/inference_model_cards/Qwen--Qwen3.5-4B.toml` with `model_id = "Qwen/Qwen3.5-4B"`, `n_layers = 36`, `hidden_size = 2560`, `num_key_value_heads = 4`, `supports_tensor = true`, `tasks = ["TextGeneration"]`, `family = "qwen"`
- [x] 3.2 Create `resources/inference_model_cards/microsoft--Phi-4.toml` with `model_id = "microsoft/Phi-4"`, `n_layers = 40`, `hidden_size = 6144`, `num_key_value_heads = 10`, `supports_tensor = true`, `tasks = ["TextGeneration"]`, `family = "phi"`
- [x] 3.3 Create `resources/inference_model_cards/Qwen--Qwen2.5-7B-Instruct.toml` with `model_id = "Qwen/Qwen2.5-7B-Instruct"`, `n_layers = 28`, `hidden_size = 3584`, `num_key_value_heads = 4`, `supports_tensor = true`, `tasks = ["TextGeneration"]`, `family = "qwen"`

## 4. Deploy and Validate Qwen3.5-4B

- [x] 4.1 Run `bash deploy_cluster.sh` to deploy code changes to all 4 gremlin nodes
- [x] 4.2 Verify cluster topology shows 4 nodes: `curl -s http://10.1.1.12:52415/state | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Nodes: {len(d[\"topology\"][\"nodes\"])}')"` — expected output: `Nodes: 4`
- [x] 4.3 Submit inference request and verify text output:
  ```bash
  curl -s --max-time 120 http://10.1.1.12:52415/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "Say hello world and explain what you are."}], "max_tokens": 100, "temperature": 0.7}'
  ```
  Expected: JSON response with `choices[0].message.content` containing ≥10 coherent English tokens
- [x] 4.4 Verify instance type is PyTorchXPURing: `curl -s http://10.1.1.12:52415/state | python3 -c "import sys,json; d=json.load(sys.stdin); instances=d.get('instances',{}); print([v.get('type','unknown') for v in instances.values()])"` — expected: list containing `"PyTorchXPURing"`

## 5. Deploy and Validate Phi-4

- [ ] 5.1 Submit inference request and verify text output:
  ```bash
  curl -s --max-time 180 http://10.1.1.12:52415/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "microsoft/Phi-4", "messages": [{"role": "user", "content": "Say hello world and explain what you are."}], "max_tokens": 100, "temperature": 0.7}'
  ```
  Expected: JSON response with `choices[0].message.content` containing ≥10 coherent English tokens
- [x] 5.2 Verify instance type is PyTorchXPURing in /state API

## 6. Deploy and Validate Qwen2.5-7B-Instruct

- [ ] 6.1 Submit inference request and verify text output:
  ```bash
  curl -s --max-time 180 http://10.1.1.12:52415/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Say hello world and explain what you are."}], "max_tokens": 100, "temperature": 0.7}'
  ```
  Expected: JSON response with `choices[0].message.content` containing ≥10 coherent English tokens
- [ ] 6.2 Verify instance type is PyTorchXPURing in /state API

## 7. Property-Based Tests

- [x] 7.1 Write Hypothesis property test: Key Preservation — for any generated state dict, `shard_weights()` preserves all input keys in `sharded_state_dict`
  - Tag: `Feature: xpu-inference-e2e, Property 1: Key preservation in shard_weights()`
  - Minimum 100 iterations
- [x] 7.2 Write Hypothesis property test: Forward/Shard Key Consistency — for any generated architecture-specific state dict, `forward()` accesses only keys that exist in `sharded_state_dict`
  - Tag: `Feature: xpu-inference-e2e, Property 2: Forward/shard key consistency`
  - Minimum 100 iterations
- [x] 7.3 Write Hypothesis property test: Fused QKV Split Round-Trip — for any fused QKV weight, splitting into Q/K/V shards across all ranks and concatenating reconstructs the original
  - Tag: `Feature: xpu-inference-e2e, Property 3: Fused QKV split round-trip`
  - Minimum 100 iterations
- [x] 7.4 Write Hypothesis property test: Bias Shard Dimension Consistency — for any weight-bias pair, column-parallel bias shard dim 0 equals weight shard dim 0
  - Tag: `Feature: xpu-inference-e2e, Property 4: Bias shard dimension consistency`
  - Minimum 100 iterations

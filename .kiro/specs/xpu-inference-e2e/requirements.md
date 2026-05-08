# Requirements Document

## Introduction

This spec covers getting PyTorch XPU inference working end-to-end on the 4-node gremlin cluster (10.1.1.12–15) with Intel Arc Meteor Lake-P iGPUs. The cluster currently connects, elects a master, places instances, loads models, and reaches "ready" state, but generation hangs due to a weight key mapping mismatch in `TensorParallelShard.forward()`. Additionally, the dashboard incorrectly shows "MlxRing" for XPU instances, and single-node placement forces `MlxRing` even when the PyTorchXPU backend is active. This spec addresses the weight key mapping fix, the instance type display fix, the single-node bypass of tensor parallelism, and end-to-end validation with three ungated HuggingFace models.

## Glossary

- **TensorParallelShard**: The class in `tensor_parallel_shard.py` that holds sharded weights and executes the tensor-parallel forward pass across multiple ranks.
- **TPShardConfig**: Configuration dataclass defining how weight matrices are split across tensor-parallel ranks.
- **State_Dict**: A Python dictionary mapping parameter name strings (e.g., `model.layers.0.self_attn.q_proj.weight`) to PyTorch tensors, as returned by `model.state_dict()`.
- **HuggingFace_Model**: A pretrained transformer model loaded via the `transformers` library's `AutoModelForCausalLM.from_pretrained()`.
- **Weight_Key**: A string key in the State_Dict that identifies a specific parameter (e.g., `model.layers.0.self_attn.q_proj.weight`).
- **Placement_Engine**: The module (`placement.py`) that determines which instance type and sharding strategy to use for a given model on the cluster.
- **PyTorchXPURingInstance**: The instance type representing a PyTorch XPU distributed ring topology, used for Gloo-based collective operations.
- **MlxRingInstance**: The instance type representing an MLX ring topology, used on macOS/Apple Silicon.
- **Gloo_Backend**: PyTorch's CPU-based distributed backend used for collective operations (broadcast, all_reduce) when NCCL is unavailable.
- **Gremlin_Cluster**: The 4-node cluster (gremlin-1 through gremlin-4, IPs 10.1.1.12–15) with Intel Arc Meteor Lake-P iGPUs.
- **Deploy_Script**: `deploy_cluster.sh`, which pushes code to GitHub, rebuilds NixOS on each node, and restarts the exo service.
- **Local_Generator**: The code path used for single-node (world_size=1) inference that bypasses tensor parallelism and uses HuggingFace's native generation.

## Requirements

### Requirement 1: Weight Key Mapping Compatibility

**User Story:** As a cluster operator, I want TensorParallelShard to correctly map HuggingFace model state dict keys for Qwen, Llama, GLM, and Phi architectures, so that the forward pass executes without KeyError exceptions.

#### Acceptance Criteria

1. WHEN a HuggingFace Qwen-architecture model is loaded, THE TensorParallelShard SHALL shard all weight keys present in the model's State_Dict without raising KeyError.
2. WHEN a HuggingFace Llama-architecture model is loaded, THE TensorParallelShard SHALL shard all weight keys present in the model's State_Dict without raising KeyError.
3. WHEN a HuggingFace Phi-architecture model is loaded, THE TensorParallelShard SHALL shard all weight keys present in the model's State_Dict without raising KeyError.
4. WHEN the forward pass accesses a weight via `_get_weight(key)`, THE TensorParallelShard SHALL use the same key format that was stored during `shard_weights()`.
5. FOR ALL valid State_Dict keys produced by `model.state_dict()`, THE TensorParallelShard SHALL store them in `sharded_state_dict` using the identical key string (no renaming or prefix stripping).
6. IF a required weight key is missing from the sharded_state_dict, THEN THE TensorParallelShard SHALL raise a descriptive error including the missing key name and a list of available keys with similar prefixes.

### Requirement 2: Tensor-Parallel Inference Across 4 Nodes

**User Story:** As a cluster operator, I want all model inference to use tensor parallelism across all 4 gremlin nodes (world_size=4), so that the distributed XPU pipeline is validated end-to-end with real multi-node communication.

#### Acceptance Criteria

1. WHEN a model is placed, THE Placement_Engine SHALL create a 4-node tensor-parallel instance (world_size=4, min_nodes=4) using PyTorchXPURingInstance.
2. THE TensorParallelShard SHALL execute the forward pass with Gloo-based CPU-staged all_reduce across all 4 ranks.
3. THE Gloo process group SHALL initialize with all 4 nodes connected before model loading begins.
4. WHEN world_size equals 4, THE Model_Loader SHALL create a TensorParallelShard with weights sharded across 4 ranks.

### Requirement 3: Correct Instance Type for XPU Placement

**User Story:** As a cluster operator, I want the placement engine and dashboard to correctly identify and display "PyTorchXPURing" when the XPU backend is active, so that I can verify the correct backend is in use.

#### Acceptance Criteria

1. WHEN a single-node placement is requested with InstanceMeta.PyTorchXPURing, THE Placement_Engine SHALL create a PyTorchXPURingInstance (not force MlxRingInstance).
2. WHEN the cluster state API returns instance information, THE API SHALL report the instance type as "PyTorchXPURing" for PyTorchXPURingInstance instances.
3. WHEN the dashboard displays running instances, THE Dashboard SHALL show "PyTorchXPURing" for instances using the PyTorch XPU backend.
4. WHEN auto-placement selects the PyTorchXPU backend for a model, THE Placement_Engine SHALL preserve InstanceMeta.PyTorchXPURing through the single-node code path.

### Requirement 4: End-to-End Generation with Qwen3.5-4B

**User Story:** As a cluster operator, I want to validate that Qwen/Qwen3.5-4B produces coherent text output via the API on the gremlin cluster, so that I can confirm the XPU inference pipeline works for small Qwen models while using all 4 nodes.

#### Acceptance Criteria

1. WHEN a "hello world" prompt is submitted to the API with model Qwen/Qwen3.5-4B, THE Gremlin_Cluster SHALL return a text response containing coherent English tokens.
2. THE Gremlin_Cluster SHALL load Qwen/Qwen3.5-4B across all 4 nodes using tensor-parallel sharding (world_size=4, min_nodes=4).
3. THE API response SHALL contain at least 10 generated tokens beyond the prompt.
4. THE Gremlin_Cluster SHALL complete generation within 120 seconds of prompt submission.
5. THE Deploy_Script SHALL be used to deploy the code changes before validation.

### Requirement 5: End-to-End Generation with Phi-4

**User Story:** As a cluster operator, I want to validate that microsoft/Phi-4 produces coherent text output via the API on the gremlin cluster, so that I can confirm the XPU inference pipeline works for Phi-architecture models.

#### Acceptance Criteria

1. WHEN a "hello world" prompt is submitted to the API with model microsoft/Phi-4, THE Gremlin_Cluster SHALL return a text response containing coherent English tokens.
2. THE Gremlin_Cluster SHALL load microsoft/Phi-4 across all 4 nodes using tensor-parallel sharding (world_size=4, min_nodes=4).
3. THE API response SHALL contain at least 10 generated tokens beyond the prompt.
4. THE Gremlin_Cluster SHALL complete generation within 180 seconds of prompt submission.

5. THE Deploy_Script SHALL be used to deploy the code changes before validation.

### Requirement 6: End-to-End Generation with Qwen2.5-7B-Instruct

**User Story:** As a cluster operator, I want to validate that Qwen/Qwen2.5-7B-Instruct produces coherent text output via the API on the gremlin cluster, so that I can confirm the XPU inference pipeline works for medium-sized instruction-tuned models.

#### Acceptance Criteria

1. WHEN a "hello world" prompt is submitted to the API with model Qwen/Qwen2.5-7B-Instruct, THE Gremlin_Cluster SHALL return a text response containing coherent English tokens.
2. THE Gremlin_Cluster SHALL load Qwen/Qwen2.5-7B-Instruct across all 4 nodes using tensor-parallel sharding (world_size=4, min_nodes=4).
3. THE API response SHALL contain at least 10 generated tokens beyond the prompt.
4. THE Gremlin_Cluster SHALL complete generation within 180 seconds of prompt submission.
5. THE Deploy_Script SHALL be used to deploy the code changes before validation.

### Requirement 7: Architecture-Agnostic Forward Pass

**User Story:** As a developer, I want the TensorParallelShard forward pass to handle architectural differences between Qwen, Llama, and Phi models (different layer naming, presence/absence of biases, different normalization), so that adding new model architectures does not require modifying the forward pass logic.

#### Acceptance Criteria

1. THE TensorParallelShard SHALL detect the model architecture from the State_Dict key patterns (presence of `model.layers.N.self_attn.q_proj.weight` for Qwen/Llama, `model.layers.N.self_attn.qkv_proj.weight` for Phi).
2. WHEN a model uses fused QKV projections (single `qkv_proj.weight`), THE TensorParallelShard SHALL split the fused weight into separate Q, K, V shards.
3. WHEN a model has bias terms on projections, THE TensorParallelShard SHALL shard biases using the same slicing logic as the corresponding weight.
4. WHEN a model does not have bias terms on projections, THE TensorParallelShard SHALL handle the absence without error (return None for optional biases).
5. THE TensorParallelShard SHALL support both RMSNorm (Qwen, Llama) and LayerNorm (Phi) by detecting which normalization weights are present.

### Requirement 8: Deployment Verification

**User Story:** As a cluster operator, I want deployment verification to confirm that the exo service is running with the PyTorchXPU backend active on all nodes, so that I can trust the cluster is correctly configured before running inference.

#### Acceptance Criteria

1. WHEN deploy_cluster.sh completes, THE Gremlin_Cluster SHALL have the exo service running on all 4 nodes.
2. WHEN the cluster reaches "ready" state, THE API at 10.1.1.12:52415 SHALL report all 4 nodes in the topology.
3. WHEN a model is placed, THE API state SHALL show the instance type as "PyTorchXPURing" (not "MlxRing").
4. IF a node fails to start the exo service after deployment, THEN THE Deploy_Script SHALL report the failure with the node name and error output.

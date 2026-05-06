# Implementation Plan: Tensor Parallelism for PyTorch XPU over Thunderbolt 4

## Overview

Implement tensor parallelism as a parallel code path alongside the existing pipeline parallelism. The implementation adds TB4 topology discovery, tensor-parallel weight sharding, all-reduce-based forward pass, and a generation pipeline where all ranks compute simultaneously. A NixOS module configures TB4 networking declaratively. The existing pipeline-parallel code remains untouched.

## Tasks

- [x] 1. Create NixOS Thunderbolt 4 networking module
  - [x] 1.1 Create `nix/thunderbolt-net.nix` with module options and kernel module loading
    - Define `services.exo.thunderbolt.enable` option
    - Define `services.exo.thunderbolt.subnet` option (default "10.4.0.0/24")
    - Define `services.exo.thunderbolt.nodeAssignments` option (hostname → list of TB4 IPs)
    - Define `services.exo.thunderbolt.firewallPorts` option (from/to for Gloo ephemeral range)
    - Load `thunderbolt-net` and `thunderbolt` kernel modules via `boot.kernelModules`
    - Authorize Thunderbolt devices for networking (TB4 security levels)
    - _Requirements: 1.1, 1.3, 12.1, 12.2, 12.5_

  - [x] 1.2 Implement systemd-networkd configuration for TB4 interfaces
    - Configure udev rules to match Thunderbolt network devices
    - Assign static IPs to TB4 interfaces as they appear using systemd-networkd
    - Ensure TB4 subnet is separate from existing 10.1.1.x ethernet subnet
    - Handle multiple TB4 ports per node with unique IPs
    - _Requirements: 1.2, 1.3, 12.3_

  - [x] 1.3 Implement firewall rules and systemd service for TB4 readiness
    - Open Gloo TCP port range (49152–65535) and ICMP on TB4 interfaces
    - Create systemd service that waits for TB4 interfaces and logs status
    - Add dependency target so exo service starts after TB4 interfaces are up
    - Handle link-down events gracefully (log without crashing)
    - _Requirements: 1.4, 1.5, 1.6, 12.4, 12.6_

  - [x] 1.4 Write NixOS module evaluation tests
    - Test module evaluation with valid configuration (nix-instantiate)
    - Verify kernel module declarations present
    - Verify firewall rules include Gloo port range
    - Verify systemd-networkd configuration generated for TB4 interfaces
    - Verify service dependency ordering (TB4 interfaces before exo service)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6_

- [x] 2. Implement TB4 topology discovery module
  - [x] 2.1 Create `src/exo/worker/engines/pytorch_xpu/tb4_topology.py` with data models
    - Define `TB4Peer` frozen dataclass (node_ip, interface_name, bandwidth_gbps)
    - Define `TB4Topology` frozen dataclass (topology_type, local_interfaces, local_ips, peers, all_node_ips)
    - Implement `is_available` property (True if at least 2 nodes reachable)
    - Implement `world_size` property (number of nodes in TB4 group)
    - _Requirements: 2.1, 2.6_

  - [x] 2.2 Implement `discover_tb4_topology()` async function
    - Accept tb4_subnet, expected_nodes, and probe_timeout_seconds parameters
    - Enumerate active TB4 network interfaces on the local node
    - Probe configured TB4 subnet addresses for reachability (ICMP or TCP connect)
    - Build reachability graph from probe results
    - Classify topology as "mesh", "ring", "partial", or "unavailable"
    - Return TB4Topology with all discovered information
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_

  - [x] 2.3 Implement `select_tb4_interface()` function
    - For mesh topology: return any TB4 interface (Gloo handles routing)
    - For ring topology: select interface connected to the most peers
    - Return None if TB4 is unavailable
    - _Requirements: 2.1, 2.2_

  - [x] 2.4 Write property test for topology classification — Property 1
    - **Property 1: Topology Classification Correctness**
    - Generate random adjacency matrices for 2–8 nodes
    - Verify "mesh" iff all pairs connected, "ring" iff each node has exactly 2 neighbors forming a cycle, "partial" iff at least 2 reachable but neither mesh nor ring, "unavailable" iff fewer than 2 reachable
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**

  - [x] 2.5 Write unit tests for TB4 topology discovery
    - Test mesh detection with 4 fully-connected nodes
    - Test ring detection with 4 nodes in a cycle
    - Test partial topology with incomplete connections
    - Test unavailable when fewer than 2 nodes reachable
    - Test select_tb4_interface returns correct interface for each topology type
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Extend distributed.py with tensor-parallel process group management
  - [x] 4.1 Add `TensorParallelGroupConfig` dataclass to `distributed.py`
    - Define rank, world_size, master_addr, master_port, tb4_interface_name fields
    - Set backend to "gloo" (literal)
    - Set init_timeout_seconds default to 60 (shorter than ethernet — TB4 is local)
    - Set allreduce_timeout_seconds default to 30
    - _Requirements: 4.1, 9.1, 9.5_

  - [x] 4.2 Implement `init_tensor_parallel_group()` function
    - Set GLOO_SOCKET_IFNAME to the TB4 interface name
    - Initialize a named Gloo process group for tensor parallelism (separate from pipeline group)
    - Use env:// rendezvous with TB4 master_addr and master_port
    - Support coexistence with existing pipeline-parallel process group
    - _Requirements: 4.1, 4.4, 4.6, 9.1, 9.6_

  - [x] 4.3 Implement `verify_tensor_parallel_group()` function
    - Each rank contributes its rank ID via all-reduce (sum)
    - Verify result equals world_size * (world_size - 1) / 2
    - Return True if verification passes, False otherwise
    - _Requirements: 9.3_

  - [x] 4.4 Write property test for rank derivation consistency — Property 6
    - **Property 6: Rank Derivation Consistency**
    - Generate random instance configs with N nodes and rank assignments
    - Verify rank assignments are unique, complete [0, N), and rank 0 maps to MASTER_ADDR
    - Produce same rank for a given node regardless of which node performs derivation
    - **Validates: Requirements 9.2**

  - [x] 4.5 Write unit tests for tensor-parallel process group
    - Test init_tensor_parallel_group sets correct environment variables
    - Test verify_tensor_parallel_group with expected sum formula
    - Test separate process groups for TP (TB4) and PP (ethernet) coexistence
    - Test initialization timeout behavior
    - _Requirements: 4.1, 4.6, 9.1, 9.3, 9.5, 9.6_

- [x] 5. Implement TensorParallelShard with weight sharding
  - [x] 5.1 Create `src/exo/worker/engines/pytorch_xpu/tensor_parallel_shard.py` with TPShardConfig
    - Define `TPShardConfig` frozen dataclass (rank, world_size, hidden_size, num_attention_heads, head_dim, intermediate_size, num_key_value_heads, allreduce_timeout_seconds)
    - Validate divisibility: raise ValueError if num_attention_heads % world_size != 0
    - Validate divisibility: raise ValueError if intermediate_size % world_size != 0
    - _Requirements: 3.6, 7.2_

  - [x] 5.2 Implement `TensorParallelShard.__init__()` and `shard_weights()`
    - Load full model on CPU, extract this rank's weight slices, discard the rest
    - QKV projection: slice along output dim (heads_per_rank heads)
    - Attention output: slice along input dim (head_dim * heads_per_rank cols)
    - MLP gate/up: slice along output dim (intermediate_per_rank cols)
    - MLP down: slice along input dim (intermediate_per_rank cols)
    - Keep embedding, layer norms, and lm_head redundant (not sharded)
    - Move only this rank's shards to target device
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.3 Implement `_column_parallel_linear()` and `_row_parallel_linear()` methods
    - Column-parallel: F.linear with column-parallel weight shard, no communication
    - Row-parallel: F.linear with row-parallel weight shard, followed by all_reduce (sum)
    - _Requirements: 4.2, 4.3, 5.1, 5.3_

  - [x] 5.4 Implement `_all_reduce()` method with timeout handling
    - In-place all-reduce (sum) over the tensor-parallel process group
    - Raise RuntimeError with layer index, tensor shape, timeout value, and rank on timeout
    - _Requirements: 4.2, 4.3, 4.5, 11.1_

  - [x] 5.5 Implement `forward()` method for tensor-parallel forward pass
    - For each layer: LayerNorm (redundant) → column-parallel QKV → local attention → row-parallel output → all_reduce → residual → LayerNorm → column-parallel gate/up → activation → row-parallel down → all_reduce → residual
    - Execute embedding lookup redundantly on all ranks
    - Execute final lm_head redundantly on all ranks
    - Maintain head-parallel KV cache (only this rank's assigned heads)
    - Return (logits_or_hidden_states, kv_cache)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.5_

  - [x] 5.6 Write property test for weight shard shapes — Property 2
    - **Property 2: Weight Shard Shape Correctness**
    - Generate random valid model configs (hidden_size ∈ [64, 8192], num_heads ∈ [4, 128], world_size ∈ [2, 8], constrained to divisibility)
    - Verify all shard shapes match the formula from the design document
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

  - [x] 5.7 Write property test for divisibility validation — Property 3
    - **Property 3: Divisibility Validation**
    - Generate random (num_heads, world_size) pairs
    - Verify ValueError raised iff num_heads % world_size != 0
    - **Validates: Requirements 3.6, 7.2**

  - [x] 5.8 Write property test for forward pass equivalence — Property 4
    - **Property 4: Tensor-Parallel Forward Pass Equivalence**
    - Generate random input tensors and weight matrices
    - Compute full linear operation and sharded+summed operation
    - Verify they produce the same result (within floating-point tolerance for bf16)
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 6.2**

  - [x] 5.9 Write unit tests for TensorParallelShard
    - Test KV cache shape matches head-parallel assignment
    - Test all-reduce timeout produces RuntimeError with layer context
    - Test weight memory reduction (approximately 1/world_size for parallelized layers)
    - _Requirements: 3.5, 4.5, 6.5_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement tensor-parallel generation pipeline
  - [x] 7.1 Create `src/exo/worker/engines/pytorch_xpu/tensor_parallel_generator.py` with imports and constants
    - Import TensorParallelShard, torch.distributed, GenerationResponse types
    - Define TERMINATION_SENTINEL constant (reuse from distributed_generator or define locally)
    - Define TPPerformanceMetrics dataclass with all-reduce latencies, throughput, bandwidth utilization
    - _Requirements: 10.1, 10.2, 10.3_

  - [x] 7.2 Implement `tensor_parallel_generate()` function (rank 0 orchestrator)
    - Tokenize prompt, broadcast token_ids to all ranks
    - Prefill phase: forward all tokens through TensorParallelShard (all-reduce happens internally)
    - Sample first token from logits (all ranks have identical logits due to all-reduce)
    - Decode loop: broadcast token → forward (1 token with KV cache) → sample → yield GenerationResponse
    - Terminate on EOS or max_tokens: broadcast TERMINATION_SENTINEL to all ranks
    - Collect performance metrics (all-reduce latencies, tokens/sec, bandwidth utilization)
    - Yield GenerationResponse with performance summary on completion
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 10.1, 10.2, 10.3, 10.4_

  - [x] 7.3 Implement `tensor_parallel_worker_loop()` function (non-rank-0 nodes)
    - Receive token broadcasts from rank 0
    - Execute forward pass (all-reduce happens inside TensorParallelShard.forward())
    - Wait for next token broadcast
    - Exit cleanly on TERMINATION_SENTINEL
    - Much simpler than pipeline worker loop — no explicit activation sending
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

  - [x] 7.4 Implement error handling and graceful shutdown
    - On all-reduce failure: detect communication error, transition to error state
    - Rank 0 broadcasts TERMINATION_SENTINEL on error (best-effort)
    - Yield GenerationResponse with finish_reason="error"
    - Release KV caches on all ranks
    - Log performance warning if all-reduce latency exceeds threshold (default 10ms)
    - _Requirements: 11.1, 11.2, 10.5_

  - [x] 7.5 Write unit tests for tensor-parallel generation
    - Test TERMINATION_SENTINEL broadcast on EOS/max_tokens
    - Test token broadcast from rank 0 to all other ranks
    - Test performance metrics calculation (bandwidth utilization formula)
    - Test error handling on all-reduce timeout
    - _Requirements: 6.3, 6.4, 6.6, 10.1, 10.3, 11.1, 11.2_

- [x] 8. Implement placement and runner integration
  - [x] 8.1 Define `TensorParallelInstance` dataclass in shared types
    - Define instance_id, model_id, tp_world_size, tb4_master_addr, tb4_master_port
    - Define rank_assignments (dict[NodeId, int]) and tb4_interface_by_node
    - Optional pipeline_group_id, pipeline_rank, pipeline_world_size for hybrid mode
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 8.2 Extend runner ConnectToGroup handler for tensor-parallel instances
    - Discover TB4 topology
    - Select TB4 interface via select_tb4_interface()
    - Initialize TP process group over TB4 using init_tensor_parallel_group()
    - Verify with test all-reduce via verify_tensor_parallel_group()
    - Transition to RunnerConnected on success
    - On failure: attempt fallback to ethernet-based TP, log warning about degraded performance
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 8.3 Extend placement module for tensor-parallel instance creation
    - Check model card `supports_tensor` field
    - Verify attention_heads divisible by requested world_size
    - Check TB4 topology availability from topology discovery
    - Validate symmetric tensor-parallel groups (all groups same size)
    - Fall back to pipeline parallelism if TP not feasible
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.4, 8.5_

  - [x] 8.4 Write property test for symmetric group validation — Property 5
    - **Property 5: Symmetric Tensor-Parallel Group Validation**
    - Generate random lists of group sizes
    - Verify acceptance iff all sizes are equal
    - **Validates: Requirements 8.4**

  - [x] 8.5 Write unit tests for placement and runner integration
    - Test model card `supports_tensor` field validation
    - Test fallback from TB4 to ethernet on init failure
    - Test placement rejects incompatible models (non-divisible heads)
    - Test placement falls back to pipeline parallelism when TP unavailable
    - _Requirements: 7.2, 7.5, 8.5, 9.4_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement TB4 link health monitoring and graceful degradation
  - [x] 10.1 Add periodic TB4 link health checking to topology discoverer
    - Check TB4 interface status periodically (configurable, default 60 seconds)
    - Report link-down events to cluster state
    - Report link-up events when TB4 link is restored
    - _Requirements: 11.4, 11.5_

  - [x] 10.2 Implement graceful degradation on TB4 link failure
    - Detect TB4 link failure during generation (all-reduce communication error)
    - Terminate current generation with error, transition instance to failed state
    - Enable master to re-place model using pipeline parallelism over ethernet
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 10.3 Write unit tests for link health monitoring
    - Test link-down detection and reporting
    - Test link-up restoration reporting
    - Test graceful degradation flow (TP failure → PP fallback)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 11. Update model cards and wire everything together
  - [x] 11.1 Update Qwen3.5-4B model card for tensor parallelism
    - Set `supports_tensor = true` in the model card TOML
    - Verify 32 attention heads divisible by 4 (cluster size)
    - _Requirements: 7.3_

  - [x] 11.2 Update Qwen3.5-2B model card for tensor parallelism
    - Set `supports_tensor = true` in the model card TOML
    - Verify 20 attention heads divisible by 4 (cluster size)
    - _Requirements: 7.4_

  - [x] 11.3 Wire tensor-parallel generation into runner TextGeneration handler
    - Add dispatch branch for TensorParallelInstance in runner
    - If rank == 0: call tensor_parallel_generate()
    - If rank != 0: call tensor_parallel_worker_loop()
    - Import all new modules at the dispatch point
    - _Requirements: 8.1, 8.2, 8.3_

- [x] 12. Write integration tests for multi-process tensor parallelism
  - [x] 12.1 Write multi-process integration tests using torch.multiprocessing.spawn
    - Test 4-process tensor-parallel forward pass (verify identical outputs on all ranks)
    - Test TB4 process group initialization with real Gloo backend (2+ processes)
    - Test token broadcast and all-reduce coordination
    - Test TERMINATION_SENTINEL propagation and clean exit
    - Test error propagation (inject error on one rank, verify all terminate)
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 6.1, 6.3, 6.4_

  - [x] 12.2 Write integration test for hybrid parallelism coexistence
    - Test TP group all-reduce + PP group send/recv coexistence
    - Verify separate process groups don't interfere
    - _Requirements: 4.6, 8.3_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests require `torch.multiprocessing.spawn()` for real multi-process Gloo groups
- The existing pipeline-parallel code (`distributed_generator.py`, `TransformerShard`) remains untouched
- Run tests with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest`
- Type checking: `uv run basedpyright`
- Linting: `uv run ruff check`

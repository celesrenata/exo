# Implementation Plan: Distributed GPU Sharding

## Overview

This plan implements pipeline-parallel distributed inference across 4 heterogeneous NixOS gremlin nodes using PyTorch's Gloo backend with CPU tensor staging. Tasks are ordered so foundational types and detection come first, followed by communication primitives, memory budgeting, placement extensions, runner dispatch, and finally integration wiring. All code uses native PyTorch XPU (2.11+) — no IPEX dependency.

## Tasks

- [x] 1. Extend MemoryUsage type with GPU memory architecture info
  - [x] 1.1 Add GpuMemoryInfo model to `src/exo/shared/types/profiling.py`
    - Create `GpuMemoryInfo(CamelCaseModel)` with fields: `device_type: Literal["cuda", "xpu", "cpu"]`, `memory_architecture: Literal["Shared", "Discrete"]`, `gpu_total_memory: Memory`, `gpu_available_memory: Memory`
    - Add optional `gpu_info: GpuMemoryInfo | None = None` field to the existing `MemoryUsage` class
    - Ensure backward compatibility: existing macOS/CPU-only nodes continue to work with `gpu_info=None`
    - _Requirements: 3.6, 12.4_

  - [x] 1.2 Write unit tests for GpuMemoryInfo and extended MemoryUsage
    - Test that `MemoryUsage` can be constructed with and without `gpu_info`
    - Test serialization/deserialization round-trip with `gpu_info` present and absent
    - Test `from_psutil` still works without GPU info
    - _Requirements: 3.6_

- [x] 2. Implement GPU Detector module
  - [x] 2.1 Create `src/exo/worker/engines/pytorch_ipex/gpu_detector.py`
    - Implement `GpuMemoryArchitecture` enum with `Shared` and `Discrete` values
    - Implement `GpuInfo` frozen dataclass with fields: `name`, `device_type`, `device_index`, `memory_architecture`, `total_memory_bytes`, `available_memory_bytes`
    - Implement `NodeGpuReport` frozen dataclass with fields: `gpus`, `has_gpu`, `primary_device_type`
    - Implement `detect_gpus() -> NodeGpuReport` function:
      - Use `torch.cuda.is_available()` / `torch.cuda.get_device_properties()` for NVIDIA detection
      - Use `torch.xpu.is_available()` / `torch.xpu.get_device_properties()` for Intel Arc detection (native PyTorch 2.11+, no IPEX)
      - Classify Intel integrated GPUs as `Shared` (shares system RAM), NVIDIA as `Discrete`
      - Fall back to CPU-only `NodeGpuReport` if no GPU detected or detection fails (log warning, don't crash)
      - Use `psutil.virtual_memory().total` as `total_memory_bytes` for shared-memory GPUs
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 11.1, 11.2_

  - [x] 2.2 Write property test for GPU detection and classification
    - **Property 4: GPU detection and classification**
    - Mock `torch.cuda` and `torch.xpu` availability and device properties with Hypothesis-generated GPU property combinations
    - Verify correct `device_type`, non-empty `name`, correct `memory_architecture` classification, and positive `total_memory_bytes`
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.6**

  - [x] 2.3 Write unit tests for GPU detector edge cases
    - Test: no GPU available → CPU-only report with `has_gpu=False`
    - Test: `ImportError` on `torch.cuda` → warning logged, CPU-only fallback
    - Test: NVIDIA GPU detected → `Discrete` architecture, `cuda` device_type
    - Test: Intel XPU detected → `Shared` architecture for integrated, `xpu` device_type
    - _Requirements: 3.5, 3.8_

- [x] 3. Implement Distributed Communicator module
  - [x] 3.1 Create `src/exo/worker/engines/pytorch_ipex/distributed.py`
    - Implement `ProcessGroupConfig` frozen dataclass with fields: `rank`, `world_size`, `master_addr`, `master_port`, `backend="gloo"`, `init_timeout_seconds=120`
    - Implement `CpuStagedTensor` frozen dataclass with fields: `cpu_tensor`, `original_dtype`, `original_shape`
    - Implement `init_process_group(config: ProcessGroupConfig) -> None`:
      - Set `MASTER_ADDR` and `MASTER_PORT` environment variables
      - Call `torch.distributed.init_process_group(backend="gloo", rank=config.rank, world_size=config.world_size, init_method="env://", timeout=timedelta(seconds=config.init_timeout_seconds))`
      - On failure: raise with error message containing rank, world_size, and backend ("gloo")
    - Implement `destroy_process_group(timeout_seconds: float = 5.0) -> None`:
      - Call `torch.distributed.destroy_process_group()` with timeout
      - On timeout: log warning, proceed without blocking
    - Implement `stage_to_cpu(tensor: torch.Tensor) -> CpuStagedTensor`:
      - Move tensor to CPU via `.to("cpu")`, preserve original dtype and shape
    - Implement `unstage_from_cpu(staged: CpuStagedTensor, target_device: str) -> torch.Tensor`:
      - Move CPU tensor to target device, verify dtype and shape match
    - Implement `send_activation(tensor: torch.Tensor, dst_rank: int) -> None`:
      - Stage to CPU, then `torch.distributed.send(cpu_tensor, dst=dst_rank)`
      - On failure: raise with source rank, dest rank, tensor shape in error message
    - Implement `recv_activation(shape, dtype, src_rank, target_device) -> torch.Tensor`:
      - Allocate CPU buffer, `torch.distributed.recv(buffer, src=src_rank)`, unstage to target device
      - On failure: raise with source rank, local rank, tensor shape in error message
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 2.1, 2.2, 2.3, 2.6, 2.7, 2a.1, 2a.4, 8.1, 8.2, 8.3, 8.4_

  - [x] 3.2 Write property test for CPU tensor staging round-trip
    - **Property 3: CPU tensor staging round-trip preserves dtype and shape**
    - Use Hypothesis to generate random tensors with varying shapes (1-4 dims, sizes 1-64) and dtypes (float16, bfloat16, float32)
    - Verify `unstage_from_cpu(stage_to_cpu(tensor), device)` produces tensor with identical dtype and shape
    - Verify staged tensor always resides on CPU device
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 3.3 Write property test for process group initialization correctness
    - **Property 1: Process group initialization correctness**
    - Use Hypothesis to generate random (rank, world_size) pairs where `0 <= rank < world_size`, random IP addresses, and random ports
    - Mock `torch.distributed.init_process_group`, verify it is called with `backend="gloo"`, correct rank, world_size, and `init_method="env://"`
    - Verify `MASTER_ADDR` and `MASTER_PORT` environment variables are set correctly
    - **Validates: Requirements 1.1, 1.3**

  - [x] 3.4 Write property test for communication failure reporting completeness
    - **Property 2: Communication failure reporting completeness**
    - Use Hypothesis to generate random failure scenarios with varying ranks, peer ranks, tensor shapes, and failure types (init timeout, send timeout, recv timeout, peer disconnect)
    - Verify error messages contain local rank, peer rank (if applicable), backend identifier ("gloo"), and tensor shape (for send/recv failures)
    - **Validates: Requirements 1.5, 2.6, 10.2**

  - [x] 3.5 Write unit tests for distributed communicator
    - Test: `init_process_group` success → no exception
    - Test: `init_process_group` timeout → error message contains rank, world_size, "gloo"
    - Test: `destroy_process_group` timeout → warning logged, no exception raised
    - Test: `send_activation` failure → error message contains src rank, dst rank, tensor shape
    - Test: `recv_activation` failure → error message contains src rank, local rank, tensor shape
    - _Requirements: 1.4, 1.5, 1.6, 8.1, 8.2, 8.4_

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Memory Budget Calculator
  - [x] 5.1 Create `src/exo/master/memory_budget.py`
    - Implement `MemoryBudget` frozen dataclass with fields: `total_pool: Memory`, `os_overhead: Memory`, `available_for_model: Memory`, `architecture: str`
    - Implement `calculate_memory_budget(total_memory, architecture, os_overhead_bytes=2*1024**3) -> MemoryBudget`:
      - For `Shared`: `available_for_model = total_memory - os_overhead`
      - For `Discrete`: `available_for_model = total_memory` (no overhead subtraction)
      - Return `MemoryBudget` with all fields populated
    - _Requirements: 12.1, 12.2, 12.3_

  - [x] 5.2 Write property test for memory budget calculation
    - **Property 8: Memory budget calculation for heterogeneous nodes**
    - Use Hypothesis to generate random (total_memory, architecture, overhead) tuples
    - Verify: Shared → `available = total - overhead`, Discrete → `available = total`
    - Verify: `available_for_model` is never negative (clamp to 0)
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**

  - [x] 5.3 Write property test for memory budget enforcement
    - **Property 9: Memory budget enforcement rejects over-committed placements**
    - Use Hypothesis to generate random (shard_memory_requirement, budget) pairs
    - Verify: when `shard_requirement > available_for_model` on a Shared node, placement is rejected
    - Verify: when `shard_requirement <= available_for_model`, placement is accepted
    - **Validates: Requirements 12.5**

  - [x] 5.4 Write unit tests for memory budget calculator
    - Test: Shared node with 32 GiB RAM → available = 30 GiB (32 - 2 overhead)
    - Test: Discrete node with 12 GiB VRAM → available = 12 GiB (no overhead)
    - Test: Custom overhead value respected
    - Test: Zero total memory → available = 0 (no negative values)
    - _Requirements: 12.1, 12.2, 12.3_

- [x] 6. Extend Placement module for PyTorch distributed
  - [x] 6.1 Create `get_pytorch_ring_hosts_by_node()` in `src/exo/master/placement_utils.py`
    - Unlike `get_mlx_ring_hosts_by_node()` which only resolves left/right neighbors, this function SHALL resolve ethernet IPs for ALL nodes in the cycle (full-mesh connectivity for `torch.distributed` rendezvous)
    - Use `_find_ip_prioritised()` to select ethernet IPs, prioritizing ethernet over wifi/unknown/thunderbolt
    - Each node's entry: `Host(ip="0.0.0.0", port=ephemeral_port)` for self, actual ethernet IP for all other nodes
    - Raise `ValueError` if any node lacks an ethernet IP address
    - _Requirements: 6.1, 6.3, 6.4, 6.5_

  - [x] 6.2 Write property test for host resolution completeness
    - **Property 6: Host resolution completeness with ethernet prioritization**
    - Use Hypothesis to generate random topologies with ethernet interfaces
    - Verify: `hosts_by_node` has entry for every node in cycle
    - Verify: selected IPs are ethernet when available
    - **Validates: Requirements 6.1, 6.3, 6.4**

  - [x] 6.3 Write property test for MASTER_ADDR derivation
    - **Property 7: MASTER_ADDR derivation from rank 0**
    - Use Hypothesis to generate random instance configurations with varying node counts and IP addresses
    - Verify: derived MASTER_ADDR is always rank 0's ethernet IP
    - Verify: derived MASTER_PORT always equals instance's `ephemeral_port`
    - **Validates: Requirements 6.2**

  - [x] 6.4 Integrate memory budgeting into placement for PyTorchIPEXRing
    - Modify `filter_cycles_by_memory()` in `src/exo/master/placement_utils.py` to use `MemoryBudget` calculations when `gpu_info` is present on `MemoryUsage`
    - For nodes with `Shared` memory architecture: use `calculate_memory_budget()` to get available memory instead of raw `ram_available`
    - For nodes with `Discrete` memory architecture: use GPU VRAM from `gpu_info.gpu_available_memory`
    - Log memory budget decisions for operator diagnostics
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [x] 6.5 Update `place_instance()` in `src/exo/master/placement.py` to use `get_pytorch_ring_hosts_by_node()`
    - Replace the current `get_mlx_ring_hosts_by_node()` call in the `InstanceMeta.PyTorchIPEXRing` branch with `get_pytorch_ring_hosts_by_node()`
    - Validate all nodes in candidate cycle have ethernet interfaces before selecting for PyTorchIPEXRing
    - _Requirements: 6.1, 6.3_

  - [x] 6.6 Write unit tests for placement extensions
    - Test: `get_pytorch_ring_hosts_by_node()` returns full-mesh hosts (not just left/right neighbors)
    - Test: ethernet-missing node → `ValueError` raised
    - Test: `filter_cycles_by_memory()` with shared-memory nodes subtracts OS overhead
    - Test: `place_instance()` for PyTorchIPEXRing uses `get_pytorch_ring_hosts_by_node()`
    - _Requirements: 6.1, 6.4, 6.5, 12.5_

- [x] 7. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Extend Info Gatherer for Linux GPU detection
  - [x] 8.1 Add Linux GPU detection loop to `src/exo/utils/info_gatherer/info_gatherer.py`
    - In `_monitor_memory_usage()`, on Linux: call `detect_gpus()` from `gpu_detector.py` and populate `GpuMemoryInfo` on the `MemoryUsage` object
    - Emit `MemoryUsage` with `gpu_info` field set when GPU is detected
    - On macOS: continue existing behavior with `gpu_info=None`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6_

  - [x] 8.2 Add Linux ethernet network interface detection to Info Gatherer
    - On Linux: use `psutil.net_if_addrs()` to detect ethernet interfaces and report them as `NetworkInterfaceInfo` with `interface_type="ethernet"`
    - Distinguish ethernet from wifi/loopback/virtual interfaces using interface naming conventions and `psutil.net_if_stats()`
    - _Requirements: 3.7_

  - [x] 8.3 Add Linux static node info to Info Gatherer
    - On Linux: read `/sys/class/dmi/id/product_name` for model identification in `StaticNodeInformation.gather()`
    - Fall back to "Unknown" if file not readable
    - _Requirements: 9.4_

  - [x] 8.4 Write unit tests for Info Gatherer Linux extensions
    - Test: Linux GPU detection loop populates `gpu_info` on `MemoryUsage`
    - Test: Linux ethernet detection returns interfaces with `interface_type="ethernet"`
    - Test: GPU detection failure → `gpu_info=None`, no crash
    - Test: Linux static node info reads product_name
    - _Requirements: 3.1, 3.7, 3.8_

- [x] 9. Remove IPEX dependency and refactor to native PyTorch XPU
  - [x] 9.1 Remove all `import intel_extension_for_pytorch` statements from `src/exo/worker/engines/pytorch_ipex/`
    - Audit all files in the `pytorch_ipex/` directory for IPEX imports
    - Replace `intel_extension_for_pytorch` calls with native `torch.xpu` equivalents
    - Update `device_manager.py` to use `torch.xpu.is_available()` and `torch.xpu.device_count()` instead of IPEX APIs
    - Update `model_loader.py` to remove IPEX optimization calls (e.g., `ipex.optimize()`)
    - Update `validate_ipex.py` or remove it if no longer needed
    - _Requirements: 11.1, 11.2, 11.5_

  - [x] 9.2 Update runner.py PyTorch backend imports to remove IPEX references
    - Remove IPEX-specific import paths in the `is_pytorch_ipex` branch of `runner.py`
    - Ensure the PyTorch backend path does not import any IPEX modules
    - _Requirements: 11.1, 7.4_

  - [x] 9.3 Write unit test verifying no IPEX imports remain
    - Grep the entire `src/` directory for `intel_extension_for_pytorch` — assert zero matches
    - Grep for `import ipex` — assert zero matches
    - Verify `torch.xpu.is_available()` is used instead of IPEX equivalents
    - _Requirements: 11.1, 11.5_

- [x] 10. Implement Runner task dispatch for PyTorch distributed
  - [x] 10.1 Implement ConnectToGroup handler for PyTorchIPEXRingInstance in `src/exo/worker/runner/runner.py`
    - Replace the current `initialize_mlx(bound_instance)` call in the `is_pytorch_ipex` branch with `init_process_group()` from `distributed.py`
    - Derive `ProcessGroupConfig` from `bound_instance`: rank from `shard_metadata.device_rank`, world_size from shard count, MASTER_ADDR from rank 0's ethernet IP in `hosts_by_node`, MASTER_PORT from `instance.ephemeral_port`
    - On success: transition to `RunnerConnected`
    - On failure: transition to `RunnerFailed` with descriptive error including rank, world_size, "gloo"
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2a.1, 2a.4_

  - [x] 10.2 Implement LoadModel handler for PyTorchIPEXRingInstance
    - Use `detect_gpus()` to determine local GPU device type and index
    - Use HuggingFace transformers to load model architecture
    - Selectively load only layers in `[start_layer, end_layer)` range from `shard_metadata` onto the detected GPU device
    - On success: transition to `RunnerLoaded`
    - On OOM: transition to `RunnerFailed` with required vs available memory in error message
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 10.3 Write property test for selective layer loading
    - **Property 5: Selective layer loading respects shard range**
    - Use Hypothesis to generate random (start_layer, end_layer, n_layers) tuples where `0 <= start_layer < end_layer <= n_layers`
    - Mock model loading, verify exactly `end_layer - start_layer` layers are loaded
    - Verify no layer outside `[start_layer, end_layer)` is present
    - **Validates: Requirements 4.1**

  - [x] 10.4 Implement StartWarmup handler for PyTorchIPEXRingInstance
    - Execute a forward pass through assigned layers using a dummy input tensor
    - Use CPU tensor staging: `send_activation()` to next rank, `recv_activation()` from previous rank
    - On success: transition to `RunnerReady`
    - On failure: transition to `RunnerFailed`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 10.5 Implement TextGeneration handler for PyTorchIPEXRingInstance
    - Rank 0: tokenize input prompt, run forward pass on assigned layers, `send_activation()` to rank 1
    - Middle ranks: `recv_activation()` from previous rank, run forward pass, `send_activation()` to next rank
    - Last rank: `recv_activation()`, run forward pass, decode output tokens, emit `ChunkGenerated` events
    - All ranks: use CPU tensor staging for activation passing
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7_

  - [x] 10.6 Implement Shutdown handler for PyTorchIPEXRingInstance
    - Call `destroy_process_group()` with 5-second timeout
    - Clear GPU caches: `torch.cuda.empty_cache()` or `torch.xpu.empty_cache()` based on device type
    - Delete model references to free GPU memory
    - On timeout: log warning, proceed with local cleanup
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 10.7 Write unit tests for runner task dispatch
    - Test: ConnectToGroup success → `RunnerConnected` status emitted
    - Test: ConnectToGroup failure → `RunnerFailed` with descriptive message
    - Test: LoadModel success → `RunnerLoaded` status emitted
    - Test: LoadModel OOM → `RunnerFailed` with memory info
    - Test: StartWarmup success → `RunnerReady` status emitted
    - Test: Shutdown → `destroy_process_group` called, GPU caches cleared
    - Test: Backend dispatch: `PyTorchIPEXRingInstance` → PyTorch path, `MlxRingInstance` → MLX path
    - Test: Import isolation: PyTorch runner doesn't import MLX
    - _Requirements: 1.4, 4.4, 5.3, 5.4, 7.1, 7.2, 7.3, 7.4, 8.1_

- [x] 11. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. NixOS service configuration for distributed inference
  - [x] 12.1 Update NixOS module to configure MASTER_ADDR and MASTER_PORT environment variables
    - Add environment variable configuration to the exo systemd service unit for distributed mode
    - Configure ephemeral port range in firewall rules for Gloo TCP communication
    - Ensure PyTorch with XPU support is available on Intel nodes and CUDA support on gremlin-1
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 12.2 Add GPU availability verification on service startup
    - On service start: detect GPU type, log memory architecture (shared vs discrete) and available memory
    - Verify GPU drivers are functional before joining cluster
    - _Requirements: 9.4_

  - [x] 12.3 Write NixOS configuration smoke tests
    - Test: systemd service unit has MASTER_ADDR/MASTER_PORT environment variables
    - Test: firewall rules include ephemeral port range
    - Test: PyTorch with XPU support is importable on Intel nodes
    - Test: no IPEX imports remain in codebase (`grep -r "intel_extension_for_pytorch"`)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 11.1_

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each major component
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases
- The implementation language is Python throughout, matching the existing codebase and design document
- All `torch.xpu` usage targets native PyTorch 2.11+ — no IPEX dependency
- The Gloo backend is used exclusively for `torch.distributed` — no NCCL or XCCL for inter-node communication

# Implementation Plan: Upstream Rebase XPU

## Overview

This plan implements a unified PyTorch inference engine supporting both CUDA and XPU device backends, pipeline parallelism across the 4-node gremlin cluster, and end-to-end validation. All code is additive to upstream — new files in `src/exo/worker/engines/pytorch/` and `nix/` with minimal registration in the engine factory.

## Tasks

- [x] 1. Upstream sync and branch setup
  - [x] 1.1 Create a new branch from upstream exo-explore/exo `main` at current HEAD
    - Fetch upstream remote and create branch preserving full commit history
    - Verify no modifications to existing MLX engine files, networking code, or discovery/election code
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 1.2 Verify upstream test suite passes on the new branch
    - Run `uv run pytest` and confirm existing tests pass without modification
    - _Requirements: 1.6_
  - [x] 1.3 Create the `src/exo/worker/engines/pytorch/` package directory structure
    - Create `__init__.py`, `engine.py`, `device_detector.py`, `gpu_validator.py`, `sampling.py`
    - Create `pipeline/` sub-package: `__init__.py`, `coordinator.py`, `stage.py`, `activation_pass.py`
    - Create `distributed/` sub-package: `__init__.py`, `communicator.py`, `transport.py`, `cpu_staging.py`
    - Create `model/` sub-package: `__init__.py`, `loader.py`, `shard_loader.py`, `kv_cache.py`
    - Create `tests/` sub-package: `__init__.py`
    - _Requirements: 2.7_

- [x] 2. Implement device detection and GPU validation
  - [x] 2.1 Implement `device_detector.py` — GPU detection and selection
    - Define `DetectedDevice` frozen dataclass with `device_type`, `device_index`, `device_name`, `memory_bytes`, `memory_architecture`
    - Implement `detect_devices()` that queries `torch.cuda` and `torch.xpu` availability
    - Implement `select_primary_device()` that prefers CUDA over XPU, raises fatal error if no GPU
    - Log selected device type and name at startup
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_
  - [x] 2.2 Write property test: Device detection selects correctly and never falls back to CPU
    - **Property 1: Device detection selects correctly and never falls back to CPU**
    - **Validates: Requirements 3.1, 3.2, 3.4, 3.5**
  - [x] 2.3 Write property test: Detected device report contains all required fields
    - **Property 2: Detected device report contains all required fields**
    - **Validates: Requirements 3.6**
  - [x] 2.4 Implement `gpu_validator.py` — runtime GPU assertions
    - Define `GpuValidator` class with `expected_device` and `debug_mode` parameters
    - Implement `assert_on_device()` that raises AssertionError with tensor name and expected device if tensor is on CPU
    - Implement `validate_model_parameters()` that checks all model parameters are on expected device
    - Implement `validate_forward_pass()` that validates input and output tensors
    - Support debug mode (every forward pass) vs production mode (first forward pass only)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [x] 2.5 Write property test: GPU validator rejects CPU tensors with descriptive errors
    - **Property 3: GPU validator rejects CPU tensors with descriptive errors**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [x] 3. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement pipeline parallelism
  - [x] 4.1 Implement `pipeline/stage.py` — stage assignment logic
    - Define `StageAssignment` frozen dataclass with `rank`, `start_layer`, `end_layer`, `has_embedding`, `has_lm_head`
    - Implement `compute_stage_assignments(total_layers, world_size)` that divides layers evenly (earlier stages get extra if uneven)
    - Rank 0 gets `has_embedding=True`, last rank gets `has_lm_head=True`
    - Support world_size of 2, 3, or 4
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6, 5.8_
  - [x] 4.2 Write property test: Pipeline stage assignment produces valid partitioning
    - **Property 4: Pipeline stage assignment produces valid partitioning**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 5.8**
  - [x] 4.3 Implement `pipeline/activation_pass.py` — inter-stage activation transfer
    - Implement `send_activation()` that sends hidden state tensor to next pipeline stage
    - Implement `recv_activation()` that receives hidden state tensor from previous stage
    - Include request_id and sequence_position metadata with each transfer
    - _Requirements: 5.4, 5.7_
  - [x] 4.4 Implement `pipeline/coordinator.py` — pipeline orchestration
    - Define `PipelineConfig` frozen dataclass with `world_size`, `rank`, `total_layers`, `master_addr`, `master_port`, `transport`
    - Implement `PipelineCoordinator` class with `get_stage_assignment()`, `forward_pipeline()`, `generate_token()`
    - `forward_pipeline()` executes local layers then sends/receives activations to/from adjacent stages
    - `generate_token()` coordinates full token generation across all stages, sampling on last node
    - Report failure with node identity and pipeline position if a node becomes unreachable
    - _Requirements: 5.4, 5.7, 5.8, 5.9_

- [x] 5. Implement distributed communication layer
  - [x] 5.1 Implement `distributed/communicator.py` — process group management
    - Define `CommConfig` frozen dataclass with `rank`, `world_size`, `master_addr`, `master_port`, `backend`, `timeout_seconds`, `transport`
    - Implement `Communicator` class with `initialize()`, `send_tensor()`, `recv_tensor()`, `destroy()`
    - `initialize()` sets up PyTorch distributed process group with Gloo or NCCL backend
    - `send_tensor()` and `recv_tensor()` handle CPU-staged transfers for Gloo
    - Configurable timeout (default 30s) with descriptive error on timeout (source rank, dest rank, operation type)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.8, 6.9_
  - [x] 5.2 Write property test: Distributed backend selection follows device-type rules
    - **Property 5: Distributed backend selection follows device-type rules**
    - **Validates: Requirements 6.1, 6.2, 6.3**
  - [x] 5.3 Implement `distributed/cpu_staging.py` — CPU-staged tensor transfer
    - Implement `stage_to_cpu(tensor)` that moves a GPU tensor to CPU for Gloo send
    - Implement `unstage_from_cpu(tensor, target_device)` that moves a CPU tensor back to GPU after Gloo recv
    - Handle mixed CUDA↔XPU transfers through CPU intermediate
    - _Requirements: 6.7_
  - [x] 5.4 Write property test: CPU staging round-trip preserves tensor data
    - **Property 6: CPU staging round-trip preserves tensor data**
    - **Validates: Requirements 6.7**
  - [x] 5.5 Implement `distributed/transport.py` — transport layer detection and configuration
    - Define `TransportInfo` and `TransportConfig` frozen dataclasses
    - Implement `detect_transport(preferred)` that detects available transport, falls back to ethernet if unavailable
    - Implement `select_backend_for_devices(device_types)` that returns "nccl" only if ALL devices are CUDA, "gloo" otherwise
    - Set `GLOO_SOCKET_IFNAME` based on detected transport interface
    - Report active transport type and bandwidth at initialization
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_
  - [x] 5.6 Write property test: Transport fallback defaults to Ethernet
    - **Property 7: Transport fallback defaults to Ethernet**
    - **Validates: Requirements 7.9**

- [x] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement model loading and engine
  - [x] 7.1 Implement `model/loader.py` — HuggingFace safetensors model loading
    - Load model weights from HuggingFace safetensors format
    - Accept model specifications in upstream's `BoundInstance` format
    - Report model loading progress through the Builder's load generator
    - Return descriptive error if model exceeds available GPU memory
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.6_
  - [x] 7.2 Implement `model/shard_loader.py` — pipeline-stage-aware partial loading
    - Load only the layers assigned to the local pipeline stage based on `StageAssignment`
    - Load embedding layer only on rank 0, LM head only on last rank
    - Move loaded parameters to the target GPU device
    - Validate loaded layer count matches stage assignment
    - _Requirements: 10.4_
  - [x] 7.3 Write property test: Partial model loading respects stage assignment
    - **Property 8: Partial model loading respects stage assignment**
    - **Validates: Requirements 10.4**
  - [x] 7.4 Implement `model/kv_cache.py` — KV cache management
    - Allocate KV cache tensors on the target GPU device
    - Support dynamic sequence length growth
    - _Requirements: 2.6, 2.8_
  - [x] 7.5 Implement `sampling.py` — token sampling
    - Implement temperature, top-k, and top-p sampling on GPU tensors
    - _Requirements: 2.6_
  - [x] 7.6 Implement `engine.py` — UnifiedPyTorchEngine
    - Implement `InferenceBackend` protocol from `src/exo/worker/engines/base.py`
    - Implement `encode()`, `decode()`, `infer_tensor()`, `sample()`, `load_checkpoint()`
    - Accept `device_type` parameter ("cuda" or "xpu") and `pipeline_config`
    - Use device-agnostic APIs (`tensor.to(device)`) for all tensor operations
    - Integrate `GpuValidator` for runtime assertions on forward passes
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 8. Register engine with upstream's selection system
  - [x] 8.1 Add engine registration in `src/exo/worker/engines/factory.py`
    - Add `elif backend_name == "pytorch"` branch that imports `UnifiedPyTorchEngine` and `select_primary_device`
    - Pass detected device type and index to engine constructor
    - Wrap in try/except ImportError for graceful degradation when PyTorch unavailable
    - Ensure MLX engine selection on macOS/Apple Silicon is not affected
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [x] 8.2 Update `pyproject.toml` with any new dependencies
    - Add `psutil` if not already present (for transport detection)
    - No other dependency changes — PyTorch is managed by Nix
    - _Requirements: 1.7_

- [x] 9. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. NixOS packaging updates
  - [x] 10.1 Create or update `nix/intel-oneapi-runtime.nix`
    - Package libsycl, libur, and libumf from Intel oneAPI runtime
    - Bundle libumf.so.1 from PyPI umf wheel
    - Include hwloc in buildInputs for libhwloc.so.15 resolution
    - Ensure same glibc version as the exo Python process
    - _Requirements: 12.1, 12.5_
  - [x] 10.2 Create or update `nix/pytorch-xpu.nix`
    - Fetch PyTorch 2.11+ XPU wheel from download.pytorch.org/whl/xpu/
    - Do NOT use CUDA wheels (+cu128)
    - _Requirements: 12.2_
  - [x] 10.3 Create or update `nix/distributed-inference.nix`
    - Configure systemd service for distributed inference
    - Set environment variables: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
    - Configure LD_LIBRARY_PATH for Level Zero driver discovery
    - Support NVIDIA CUDA runtime on gremlin-1 alongside Intel XPU runtime
    - _Requirements: 12.3, 12.4, 12.6, 12.7, 12.8_

- [x] 11. End-to-end integration test
  - [x] 11.1 Implement `tests/test_e2e_pipeline.py` — full cluster validation
    - Write pytest test marked with `@pytest.mark.slow` and `@pytest.mark.e2e`
    - Initialize pipeline across 10.1.1.12–15 (all 4 gremlin nodes)
    - Load Qwen3.5:4B sharded across 4 pipeline stages
    - Use XPU on gremlin-2/3/4, CUDA or XPU on gremlin-1
    - Submit "hello world" prompt and assert ≥10 tokens generated
    - Verify all 4 nodes participated by checking pipeline stage execution
    - Assert no CPU tensor operations during generation (GPU validation)
    - Assert completion within 120 seconds
    - Report which node failed and error reason if any stage fails to initialize
    - Runnable with `uv run pytest` from any node that can reach all 4 gremlin IPs
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9_

- [x] 12. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (Hypothesis library)
- The MikroTik LACP switch configuration is documented in the design but is NOT a coding task — it is executed via SSH by an operator
- All new code resides in `src/exo/worker/engines/pytorch/` and `nix/` — no modifications to existing upstream files except `factory.py` registration and `pyproject.toml` dependencies
- NixOS packaging uses the exo flake's pkgsExo for GPU runtime libraries to ensure glibc version consistency

# Requirements Document

## Introduction

This feature adds CUDA and XPU as new inference engine backends to the upstream exo project, implements pipeline parallelism that shards models across all 4 gremlin nodes, validates that inference runs on Intel Arc GPUs (not CPU), supports multiple network transports (RDMA, Ethernet, LACP-bonded Ethernet), and includes MikroTik switch configuration for LACP bonding. The implementation is purely additive — existing MLX engine, networking, and discovery code remain untouched. The target validation is an end-to-end "hello world" test running Qwen3.5:4B across all 4 nodes via pipeline parallelism on XPU.

## Glossary

- **Upstream**: The exo-explore/exo repository on the `main` branch at github.com/exo-explore/exo
- **Unified_Engine**: The PyTorch inference backend supporting both CUDA (`torch.cuda`) and XPU (`torch.xpu`) device types
- **Pipeline_Parallelism**: A model sharding strategy where consecutive transformer layers are assigned to different nodes, with activations passed between nodes sequentially
- **Pipeline_Stage**: A subset of consecutive model layers assigned to a single node in the pipeline
- **Gremlin_Cluster**: The 4-node cluster (gremlin-1 through gremlin-4) at 10.1.1.12–15 running NixOS
- **XPU_Device**: An Intel Arc GPU accessed via `torch.xpu`, specifically the Meteor Lake-P integrated Arc Graphics
- **CUDA_Device**: An NVIDIA GPU accessed via `torch.cuda`, specifically the RTX 4070 Ti SUPER on gremlin-1
- **Device_Backend**: The PyTorch device abstraction (either `torch.xpu` or `torch.cuda`) used for tensor operations
- **Gloo_Backend**: PyTorch's distributed communication backend that supports XPU tensors (NCCL does not support Intel GPUs)
- **NCCL_Backend**: PyTorch's distributed communication backend optimized for NVIDIA GPUs
- **RDMA_Transport**: Remote Direct Memory Access over Thunderbolt 4 for low-latency inter-node communication
- **LACP_Transport**: Link Aggregation Control Protocol bonding 2×2.5Gbps Ethernet links per node via the MikroTik switch
- **Ethernet_Transport**: Plain single-link Ethernet without bonding
- **MikroTik_Switch**: The aggregate switch at 10.1.1.253 (ssh://admin@10.1.1.253) providing LACP bonding and 10GBE uplink
- **Level_Zero**: Intel's low-level GPU programming API used by the XPU runtime stack
- **Engine_Interface**: The abstract base class at `src/exo/worker/engines/base.py` defining engine methods
- **Builder_Interface**: The abstract base class defining engine construction methods
- **NixOS_Packaging**: The Nix derivations packaging the Intel XPU and CUDA runtime stacks
- **Qwen3_5_4B**: The Qwen3.5 4-billion parameter model used as the validation target

## Requirements

### Requirement 1: Upstream Sync via Fresh Branch (Additive Only)

**User Story:** As a developer, I want to create a clean branch from upstream's HEAD and add new engine backends without modifying existing code, so that the changes can be contributed upstream as a clean addition.

#### Acceptance Criteria

1. THE Rebase_Process SHALL create a new branch from upstream exo-explore/exo `main` at its current HEAD
2. THE Rebase_Process SHALL preserve the full upstream commit history on the new branch
3. THE Rebase_Process SHALL not modify any existing MLX engine files in `src/exo/worker/engines/`
4. THE Rebase_Process SHALL not modify any existing networking code in `rust/networking/`
5. THE Rebase_Process SHALL not modify any existing discovery or election code
6. WHEN the new branch is created, THE Rebase_Process SHALL verify that upstream's existing test suite passes without modification
7. THE Rebase_Process SHALL add new files only — no deletions or modifications of upstream files except `pyproject.toml` for dependency declarations

### Requirement 2: Unified PyTorch Engine Supporting CUDA and XPU

**User Story:** As a developer, I want a single PyTorch engine that supports both CUDA and XPU device backends, so that gremlin-1 can use its NVIDIA GPU while gremlin-2/3/4 use their Intel Arc GPUs through the same code path.

#### Acceptance Criteria

1. THE Unified_Engine SHALL implement the Engine_Interface abstract base class from upstream
2. THE Unified_Engine SHALL implement the Builder_Interface abstract base class from upstream
3. THE Unified_Engine SHALL accept a `device_type` parameter of either `"xpu"` or `"cuda"` at initialization
4. WHEN `device_type` is `"xpu"`, THE Unified_Engine SHALL execute all tensor operations on XPU_Device via `torch.xpu`
5. WHEN `device_type` is `"cuda"`, THE Unified_Engine SHALL execute all tensor operations on CUDA_Device via `torch.cuda`
6. THE Unified_Engine SHALL share model loading, KV cache management, and token generation logic between both device backends
7. THE Unified_Engine SHALL reside in `src/exo/worker/engines/pytorch/` as a new directory without modifying existing engine directories
8. THE Unified_Engine SHALL use PyTorch's device-agnostic APIs (`tensor.to(device)`, `torch.zeros(..., device=device)`) for all tensor allocation

### Requirement 3: GPU Device Detection and Selection

**User Story:** As a developer, I want automatic detection of available GPUs so that each node selects the correct device backend without manual configuration.

#### Acceptance Criteria

1. WHEN `torch.cuda.is_available()` returns True and an NVIDIA GPU is detected, THE Device_Detector SHALL report `"cuda"` as an available backend
2. WHEN `torch.xpu.is_available()` returns True and an Intel Arc GPU is detected, THE Device_Detector SHALL report `"xpu"` as an available backend
3. IF neither `torch.cuda.is_available()` nor `torch.xpu.is_available()` returns True, THEN THE Device_Detector SHALL raise a fatal error and refuse to start inference
4. THE Device_Detector SHALL never fall back to CPU inference — CPU is not an acceptable device
5. WHEN both CUDA and XPU devices are available on the same node, THE Device_Detector SHALL prefer CUDA for the primary inference device
6. THE Device_Detector SHALL report device memory capacity, device name, and device count for the selected backend
7. THE Device_Detector SHALL log the selected device type and device name at startup for operator visibility

### Requirement 4: GPU Validation Assertions

**User Story:** As a developer, I want runtime assertions that verify inference is actually executing on the GPU, so that silent CPU fallback is impossible.

#### Acceptance Criteria

1. WHEN the Unified_Engine loads a model, THE Unified_Engine SHALL assert that all model parameter tensors reside on the selected GPU device
2. WHEN the Unified_Engine performs a forward pass, THE Unified_Engine SHALL assert that input tensors are on the selected GPU device before execution
3. WHEN the Unified_Engine produces output tokens, THE Unified_Engine SHALL assert that output tensors reside on the selected GPU device
4. IF any tensor is found on CPU during inference, THEN THE Unified_Engine SHALL raise an assertion error with a message identifying the offending tensor and its expected device
5. THE Unified_Engine SHALL perform device validation checks on every forward pass in debug mode and on the first forward pass in production mode
6. WHEN running on XPU_Device, THE Unified_Engine SHALL verify that `torch.xpu.is_available()` returns True before any inference operation
7. WHEN running on CUDA_Device, THE Unified_Engine SHALL verify that `torch.cuda.is_available()` returns True before any inference operation

### Requirement 5: Pipeline Parallelism Implementation

**User Story:** As a developer, I want pipeline parallelism that shards Qwen3.5:4B model layers across all 4 gremlin nodes, so that the model runs distributed across the cluster with each node handling a subset of layers.

#### Acceptance Criteria

1. THE Pipeline_Parallelism SHALL divide a transformer model's layers into N consecutive Pipeline_Stages where N equals the number of participating nodes
2. THE Pipeline_Parallelism SHALL assign each Pipeline_Stage to exactly one node in the cluster
3. WHEN Qwen3_5_4B is loaded across 4 nodes, THE Pipeline_Parallelism SHALL assign approximately equal numbers of layers to each node
4. THE Pipeline_Parallelism SHALL pass intermediate activations (hidden states) from one Pipeline_Stage to the next Pipeline_Stage on the subsequent node
5. THE Pipeline_Parallelism SHALL execute the embedding layer and first Pipeline_Stage on the first node in the pipeline
6. THE Pipeline_Parallelism SHALL execute the final Pipeline_Stage and language model head on the last node in the pipeline
7. WHEN a generation request arrives, THE Pipeline_Parallelism SHALL coordinate token generation across all Pipeline_Stages sequentially
8. THE Pipeline_Parallelism SHALL support variable numbers of nodes (2, 3, or 4) for the pipeline without code changes
9. IF a node in the pipeline becomes unreachable, THEN THE Pipeline_Parallelism SHALL report the failure with the node identity and pipeline position

### Requirement 6: Distributed Communication Layer

**User Story:** As a developer, I want a distributed communication layer that uses Gloo for XPU nodes and NCCL for CUDA nodes, so that inter-node tensor transfer works correctly for both device types.

#### Acceptance Criteria

1. WHEN all participating nodes use XPU_Device, THE Distributed_Communication SHALL use Gloo_Backend for collective operations
2. WHEN all participating nodes use CUDA_Device, THE Distributed_Communication SHALL use NCCL_Backend for collective operations
3. WHEN the cluster contains a mix of CUDA and XPU nodes, THE Distributed_Communication SHALL use Gloo_Backend as the common denominator
4. THE Distributed_Communication SHALL support `send()` and `recv()` operations for passing activation tensors between Pipeline_Stages
5. THE Distributed_Communication SHALL initialize a PyTorch distributed process group with one rank per participating node
6. THE Distributed_Communication SHALL use the node's IP address and a configured port for the rendezvous endpoint
7. WHEN transferring tensors between a CUDA node and an XPU node, THE Distributed_Communication SHALL move tensors through CPU memory as an intermediate step
8. THE Distributed_Communication SHALL support configurable timeout values for send/recv operations with a default of 30 seconds
9. IF a distributed operation times out, THEN THE Distributed_Communication SHALL raise a descriptive error identifying the source rank, destination rank, and operation type

### Requirement 7: Transport Layer Support (RDMA, Ethernet, LACP)

**User Story:** As a developer, I want support for multiple network transports so that the cluster can use RDMA over Thunderbolt 4 for lowest latency, LACP-bonded Ethernet for bandwidth, or plain Ethernet as a fallback.

#### Acceptance Criteria

1. THE Transport_Layer SHALL support RDMA over Thunderbolt 4 connections between adjacent nodes
2. THE Transport_Layer SHALL support plain Ethernet (single 2.5Gbps link) for inter-node communication
3. THE Transport_Layer SHALL support LACP-bonded Ethernet (2×2.5Gbps per node, layer3+4 hash) via the MikroTik_Switch
4. THE Transport_Layer SHALL allow transport selection via a configuration parameter at startup
5. WHEN RDMA_Transport is selected, THE Distributed_Communication SHALL configure Gloo to use the RDMA interface for tensor transfers
6. WHEN LACP_Transport is selected, THE Distributed_Communication SHALL bind to the bonded interface IP address
7. WHEN Ethernet_Transport is selected, THE Distributed_Communication SHALL bind to the primary Ethernet interface IP address
8. THE Transport_Layer SHALL report the active transport type and measured bandwidth at initialization
9. IF the selected transport is unavailable, THEN THE Transport_Layer SHALL fall back to Ethernet_Transport and log a warning

### Requirement 8: MikroTik LACP Switch Configuration

**User Story:** As a developer, I want copy-pasteable MikroTik CLI commands for configuring LACP on the switch, so that another agent or operator can configure the switch via SSH at admin@10.1.1.253.

#### Acceptance Criteria

1. THE MikroTik_Configuration SHALL provide CLI commands that configure LACP bonding interfaces for all 4 gremlin nodes
2. THE MikroTik_Configuration SHALL configure 2 ports per node as an LACP bond group (8 ports total)
3. THE MikroTik_Configuration SHALL set the LACP transmit hash policy to layer3+4 for optimal traffic distribution
4. THE MikroTik_Configuration SHALL assign the bonded interfaces to the correct VLAN or bridge for the 10.1.1.0/24 subnet
5. THE MikroTik_Configuration SHALL be executable via SSH at `ssh://admin@10.1.1.253` without interactive prompts
6. THE MikroTik_Configuration SHALL include verification commands that confirm bond status and link aggregation is active
7. THE MikroTik_Configuration SHALL be a standalone block in the design document that can be extracted and executed independently

### Requirement 9: Engine Registration with Upstream's Selection System

**User Story:** As a developer, I want the Unified_Engine to register with upstream's engine selection mechanism so that it is discovered and used alongside the existing MLX engine.

#### Acceptance Criteria

1. THE Unified_Engine SHALL register itself in upstream's engine selection mechanism as a new engine option
2. WHEN a node has Intel Arc GPUs or NVIDIA GPUs available, THE Unified_Engine SHALL be selectable as the active engine
3. THE Unified_Engine SHALL not interfere with MLX engine selection on macOS/Apple Silicon nodes
4. THE Unified_Engine SHALL report its supported device types to the engine selection system
5. WHEN the Unified_Engine is selected, THE Engine_Selection SHALL pass the detected device type to the engine's Builder

### Requirement 10: Model Compatibility with Upstream's Model System

**User Story:** As a developer, I want the Unified_Engine to work with upstream's model download and task system so that models are loaded using the same HuggingFace safetensors format.

#### Acceptance Criteria

1. THE Unified_Engine SHALL accept model specifications in the format used by upstream's `BoundInstance` type
2. THE Unified_Engine SHALL load HuggingFace safetensors model files downloaded by upstream's model system
3. THE Unified_Engine SHALL support Qwen3_5_4B as the primary validation model
4. WHEN loading a model for Pipeline_Parallelism, THE Unified_Engine SHALL load only the layers assigned to the local Pipeline_Stage
5. THE Unified_Engine SHALL report model loading progress through the Builder's load generator
6. IF a model requires more GPU memory than available on the selected device, THEN THE Unified_Engine SHALL return a descriptive error identifying the memory requirement and available capacity

### Requirement 11: End-to-End Validation Test

**User Story:** As a developer, I want an end-to-end test that runs Qwen3.5:4B across all 4 gremlin nodes via pipeline parallelism on XPU and produces a text response, proving the full distributed inference pipeline works.

#### Acceptance Criteria

1. THE End_to_End_Test SHALL load Qwen3_5_4B sharded across all 4 gremlin nodes using Pipeline_Parallelism
2. THE End_to_End_Test SHALL use XPU_Device on gremlin-2, gremlin-3, and gremlin-4
3. THE End_to_End_Test SHALL use either CUDA_Device or XPU_Device on gremlin-1 (both are acceptable for the test)
4. THE End_to_End_Test SHALL submit a "hello world" prompt and receive a generated text response of at least 10 tokens
5. THE End_to_End_Test SHALL verify that all 4 nodes participated in the generation by checking pipeline stage execution on each node
6. THE End_to_End_Test SHALL complete within 120 seconds for the hello world prompt
7. THE End_to_End_Test SHALL assert that no tensor operations occurred on CPU during the generation (GPU validation)
8. THE End_to_End_Test SHALL be runnable as a pytest test with `uv run pytest` from any node that can reach all 4 gremlin IPs
9. IF any node fails to initialize its Pipeline_Stage, THEN THE End_to_End_Test SHALL report which node failed and the error reason

### Requirement 12: NixOS Packaging Preservation

**User Story:** As a developer, I want to keep the NixOS packaging for Intel oneAPI runtime, PyTorch XPU wheels, and the distributed inference module so that the cluster deploys with `nixos-rebuild switch`.

#### Acceptance Criteria

1. THE NixOS_Packaging SHALL include the `intel-oneapi-runtime.nix` derivation providing libsycl, libur, and libumf
2. THE NixOS_Packaging SHALL include the `pytorch-xpu.nix` derivation fetching PyTorch 2.11+ XPU wheels
3. THE NixOS_Packaging SHALL include the `distributed-inference.nix` NixOS module for systemd service configuration
4. THE NixOS_Packaging SHALL configure `LD_LIBRARY_PATH` so that Level_Zero discovers `libze_intel_gpu.so.1`
5. THE NixOS_Packaging SHALL ensure all GPU runtime libraries use the same glibc version as the exo Python process
6. THE NixOS_Packaging SHALL be located in the `nix/` directory without modifying upstream's existing files
7. WHEN deployed to the Gremlin_Cluster, THE NixOS_Packaging SHALL configure the distributed process group environment variables (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK)
8. THE NixOS_Packaging SHALL support NVIDIA CUDA runtime on gremlin-1 in addition to the Intel XPU runtime on all nodes


# Requirements Document

## Introduction

This feature enables distributed model inference across 4 NixOS gremlin nodes (gremlin-1 through gremlin-4) connected via a switched ethernet fabric, with each node utilizing its GPU. The nodes are connected via LACP-bonded 2.5Gbps ethernet links to an aggregation switch, providing full-mesh any-to-any connectivity between all nodes. The upstream exo codebase currently implements distributed inference exclusively through MLX (Apple Silicon) primitives. This feature extends the architecture to support PyTorch-based distributed communication over ethernet using `torch.distributed`, enabling pipeline-parallel model sharding across heterogeneous Linux GPU nodes.

**Cluster hardware:**
- **gremlin-1** (10.1.1.12): NVIDIA discrete GPU with dedicated VRAM
- **gremlin-2** (10.1.1.13): Intel integrated GPU (Core Ultra / Arc integrated) — shares system RAM, no discrete VRAM
- **gremlin-3** (10.1.1.14): Intel integrated GPU (Core Ultra / Arc integrated) — shares system RAM, no discrete VRAM
- **gremlin-4** (10.1.1.15): Intel integrated GPU (Core Ultra / Arc integrated) — shares system RAM, no discrete VRAM

**Critical backend finding:** XCCL is validated ONLY on Intel Data Center GPU Max Series and does NOT work on Intel Arc client GPUs (discrete or integrated). For this heterogeneous cluster mixing NVIDIA and Intel Arc integrated GPUs, **Gloo is the required `torch.distributed` backend** for all inter-node communication. A single process group can only have one backend, and Gloo is the only backend that works across both NVIDIA CUDA and Intel XPU devices. All activation passing uses CPU tensor staging: tensors must be moved to CPU before `dist.send()` and back to GPU after `dist.recv()`.

**Critical version note:** Intel Extension for PyTorch (IPEX) has been discontinued after version 2.8, with all XPU features upstreamed into PyTorch itself. Starting with PyTorch 2.5, Intel GPU support (torch.xpu) is native. PyTorch 2.11 was released March 30, 2026 (2723 commits from 432 contributors) and provides native `torch.xpu` for Intel Arc client GPUs (both discrete and integrated). This build targets the latest packages and does NOT depend on IPEX — it uses native PyTorch XPU support instead.

**NPU note:** Intel NPU is not relevant to this feature. There is no native `torch.npu` in upstream PyTorch; Intel NPU requires a separate `intel_npu_acceleration_library` package and is not used on the gremlin nodes.

### Target Package Versions

All packages SHALL be sourced from nixpkgs-unstable or directly from upstream repositories. The following are the target versions:

| Package | Version | Source | Notes |
|---------|---------|--------|-------|
| PyTorch | 2.11.0 | nixpkgs-unstable or pytorch.org | Released 2026-03-30. Native XPU support, CUDA 12.6/12.8/13.0 (13.0 default on PyPI). XPU wheels: `pip3 install torch --index-url https://download.pytorch.org/whl/xpu` |
| HuggingFace Transformers | 5.7.x (latest v5) | nixpkgs-unstable or PyPI | Major v5 rewrite with modular architecture |
| NVIDIA NCCL | 2.28+ | nixpkgs-unstable or NVIDIA | For NVIDIA-only sub-group communication (optional optimization) |
| Python | 3.12 | nixpkgs-unstable | Compatible with all above packages |
| NVIDIA CUDA Toolkit | 12.6+ | nixpkgs-unstable | 12.6 for broadest GPU compat |

**IPEX is NOT a dependency.** PyTorch 2.11 provides native `torch.xpu` device support for Intel Arc client GPUs (discrete and integrated). The old IPEX-based code in this repository should be replaced with native PyTorch XPU calls.

**XCCL is NOT used.** XCCL is only validated on Intel Data Center GPU Max Series, not on Intel Arc client GPUs. The distributed backend for this cluster is Gloo.

The existing `PyTorchIPEXRingInstance` type and placement infrastructure already exist in the codebase but lack the actual distributed communication layer. The `ConnectToGroup` task handler only calls `initialize_mlx()`, and the info gatherer is macOS-focused with no Linux GPU detection. This feature fills those gaps.

Note: Although the upstream exo codebase uses "ring topology" terminology for its instance types (e.g., PyTorchIPEXRingInstance), the actual gremlin network is a switched fabric where all nodes can communicate directly with each other through the aggregation switch. The `torch.distributed` rendezvous model (MASTER_ADDR/MASTER_PORT) is well-suited to this topology since it uses a star pattern for connection establishment, not physical ring wiring.

## Glossary

- **Runner**: A subprocess (multiprocessing.Process) managed by RunnerSupervisor that handles model loading, warmup, and inference for a single shard on a single node
- **Instance**: A distributed model deployment across one or more nodes, containing shard assignments mapping runners to nodes (e.g., MlxRingInstance, PyTorchIPEXRingInstance)
- **Shard**: A partition of a model assigned to a single node; for pipeline parallelism this is a contiguous range of layers [start_layer, end_layer)
- **Pipeline_Parallelism**: A sharding strategy where model layers are split sequentially across nodes; each node processes its layer range and passes activations to the next node in the pipeline
- **Process_Group**: A `torch.distributed` communication group initialized with the Gloo backend that coordinates tensor operations (send, recv, broadcast) across nodes via the switched ethernet fabric; all send/recv operations use CPU tensors
- **Rank**: A node's integer position (0 to world_size-1) within the distributed process group, derived from device_rank in PipelineShardMetadata
- **World_Size**: The total number of nodes participating in a distributed instance
- **Gloo**: The `torch.distributed` backend used for all inter-node communication in this heterogeneous cluster; Gloo operates over TCP and requires tensors to be on CPU for send/recv operations; it is the only backend that works across both NVIDIA CUDA and Intel XPU devices in a single process group
- **Backend_Communicator**: The transport layer used by `torch.distributed` for inter-node communication; for this cluster, Gloo is the primary backend for all 4 ranks; optionally, NCCL sub-groups may be created for NVIDIA-only collective operations
- **CPU_Tensor_Staging**: The pattern of moving tensors from GPU to CPU before `dist.send()` and from CPU back to GPU after `dist.recv()`; required because Gloo operates on CPU tensors only; on NVIDIA nodes this crosses PCIe, on Intel integrated GPU nodes this is nearly free due to shared memory architecture
- **Shared_Memory_GPU**: An Intel integrated GPU (e.g., Intel Arc integrated in Core Ultra processors) that shares system RAM with the CPU rather than having discrete VRAM; GPU↔CPU tensor copies are essentially free (same physical memory), but model weights, activations, KV cache, and OS all compete for the same RAM pool
- **Discrete_GPU**: A GPU with its own dedicated VRAM separate from system RAM (e.g., NVIDIA GPUs on gremlin-1); GPU↔CPU copies cross the PCIe bus
- **Switched_Fabric**: The network topology where all gremlin nodes connect via LACP-bonded 2.5Gbps ethernet to an aggregation switch, providing full-mesh any-to-any connectivity; `torch.distributed` uses a rendezvous pattern (MASTER_ADDR/MASTER_PORT) over this fabric rather than point-to-point ring wiring
- **Activation_Tensor**: The intermediate tensor output from one pipeline stage that must be sent to the next node in the pipeline for continued inference; approximately 8 KiB per token at hidden_size=4096 with fp16, manageable at 2.5Gbps link speed
- **Info_Gatherer**: The node information collection system (info_gatherer.py) that reports hardware capabilities, memory, GPU memory architecture (shared vs discrete), and network interfaces to the master
- **Gremlin_Node**: One of the 4 NixOS machines (gremlin-1 at 10.1.1.12, gremlin-2 at 10.1.1.13, gremlin-3 at 10.1.1.14, gremlin-4 at 10.1.1.15) connected via LACP-bonded 2.5Gbps ethernet to an aggregation switch
- **Master**: The elected coordinator node that runs placement algorithms and broadcasts cluster state via event sourcing
- **Worker**: A node component that receives instance assignments, spawns runners, and executes the plan loop
- **ConnectToGroup**: A task dispatched by the plan loop that triggers distributed backend initialization before model loading

## Requirements

### Requirement 1: PyTorch Distributed Initialization via Gloo

**User Story:** As a cluster operator, I want each gremlin node to initialize a `torch.distributed` process group using the Gloo backend when assigned to a PyTorchIPEXRingInstance, so that nodes with heterogeneous GPUs (NVIDIA + Intel Arc integrated) can communicate tensors over ethernet during distributed inference.

#### Acceptance Criteria

1. WHEN a Runner receives a ConnectToGroup task for a PyTorchIPEXRingInstance, THE Runner SHALL initialize a `torch.distributed` process group using `backend="gloo"`, with the rank and world_size from the bound shard metadata
2. THE Runner SHALL use the Gloo backend for all nodes regardless of GPU type, because XCCL is not supported on Intel Arc client GPUs and a single process group requires a single backend that works across both NVIDIA CUDA and Intel XPU devices
3. WHEN initializing the process group, THE Runner SHALL derive the MASTER_ADDR and MASTER_PORT from the hosts_by_node data in the PyTorchIPEXRingInstance (rank 0 node's ethernet IP and the ephemeral_port), using the env:// init_method for rendezvous through the aggregation switch
4. WHEN the process group is successfully initialized, THE Runner SHALL transition to RunnerConnected status
5. IF process group initialization fails, THEN THE Runner SHALL transition to RunnerFailed status with a descriptive error message including the rank, world_size, and backend ("gloo") that was attempted
6. WHEN initializing the process group, THE Runner SHALL set a configurable timeout (default 120 seconds) to prevent indefinite hangs when a peer node is unreachable

### Requirement 2: Pipeline-Parallel Activation Passing with CPU Tensor Staging

**User Story:** As a cluster operator, I want model activations to be passed between gremlin nodes in pipeline order over ethernet using CPU tensor staging, so that each node processes its assigned layer range and forwards results to the next node regardless of GPU vendor.

#### Acceptance Criteria

1. WHEN a node completes inference on its assigned layer range and is not the last pipeline stage, THE Runner SHALL move the output activation tensor to CPU using `.to("cpu")`, then send the CPU tensor to the next rank using `torch.distributed.send`
2. WHEN a node is not the first pipeline stage, THE Runner SHALL allocate a CPU tensor buffer, receive the input activation tensor from the previous rank using `torch.distributed.recv` on the CPU tensor, then move the received tensor to the local GPU device (`.to("cuda")` for NVIDIA, `.to("xpu")` for Intel) before starting inference on its layer range
3. THE Activation_Tensor passed between nodes SHALL preserve dtype and shape as defined by the model's hidden_size and sequence length
4. WHEN the first pipeline stage (rank 0) receives a TextGeneration task, THE Runner SHALL tokenize the input prompt and run inference on its layer range before sending activations (via CPU staging) to rank 1
5. WHEN the last pipeline stage (rank world_size-1) completes inference, THE Runner SHALL decode the output tokens and emit ChunkGenerated events
6. IF an activation send or receive operation times out, THEN THE Runner SHALL transition to RunnerFailed status with an error message identifying the source rank, destination rank, and tensor shape
7. WHEN running on a Shared_Memory_GPU node (gremlin-2/3/4), THE Runner SHALL perform the `.to("cpu")` call before send and `.to("xpu")` call after recv for API consistency, noting that these copies are nearly free due to shared memory architecture

### Requirement 2a: Switched Fabric Communication Model via Gloo over TCP

**User Story:** As a cluster operator, I want `torch.distributed` with the Gloo backend to leverage the full-mesh connectivity of the aggregation switch over TCP, so that any rank can communicate directly with any other rank without relay hops.

#### Acceptance Criteria

1. WHEN initializing `torch.distributed`, THE Runner SHALL use the Gloo backend with TCP-based rendezvous (env:// init_method with MASTER_ADDR/MASTER_PORT) which establishes direct TCP connections between all ranks through the switched fabric
2. THE communication pattern SHALL use point-to-point `torch.distributed.send`/`recv` on CPU tensors between adjacent pipeline stages (rank N sends to rank N+1), with the switch fabric providing the physical connectivity
3. THE Runner SHALL NOT assume or require physical ring wiring between nodes; all inter-node communication SHALL route through the aggregation switch via Gloo's TCP transport
4. THE Runner SHALL NOT use NCCL or XCCL as the process group backend for inter-node activation passing; Gloo is the only backend that supports the heterogeneous NVIDIA + Intel Arc integrated GPU configuration in a single process group

### Requirement 3: Linux GPU Detection with Shared-Memory Architecture Awareness

**User Story:** As a cluster operator, I want each gremlin node to report its GPU type, memory architecture (shared vs discrete), and available memory to the master, so that the placement algorithm can make informed shard assignments accounting for shared-memory constraints.

#### Acceptance Criteria

1. WHEN running on Linux, THE Info_Gatherer SHALL detect NVIDIA GPUs by querying `torch.cuda` device properties (name, total memory, available memory) and report them as Discrete_GPU type
2. WHEN running on Linux, THE Info_Gatherer SHALL detect Intel Arc GPUs by querying native `torch.xpu` device properties (name, total memory) — no IPEX dependency required with PyTorch 2.11+
3. WHEN an Intel XPU device is detected, THE Info_Gatherer SHALL determine whether the GPU is a Shared_Memory_GPU (integrated, sharing system RAM) or a Discrete_GPU by checking device properties, and report the memory architecture type
4. WHEN a Shared_Memory_GPU is detected, THE Info_Gatherer SHALL report total system RAM (via psutil) as the shared memory pool, along with current system RAM usage, so the placement algorithm can account for OS and other processes competing for the same memory
5. WHEN no GPU is detected on Linux, THE Info_Gatherer SHALL report CPU-only capability with system RAM from psutil
6. THE Info_Gatherer SHALL report GPU memory and memory architecture type as part of the MemoryUsage structure so that the placement algorithm can use this information for shard allocation decisions
7. WHEN running on Linux, THE Info_Gatherer SHALL report ethernet network interfaces with their IP addresses and interface type "ethernet"
8. IF GPU detection fails due to missing drivers or libraries, THEN THE Info_Gatherer SHALL log a warning and fall back to CPU-only reporting without crashing the node

### Requirement 4: PyTorch Model Loading for Distributed Shards

**User Story:** As a cluster operator, I want each gremlin node to load only its assigned layer range from a HuggingFace model, so that GPU memory is used efficiently across the cluster.

#### Acceptance Criteria

1. WHEN a Runner receives a LoadModel task for a PyTorchIPEXRingInstance, THE Runner SHALL load only the layers in the range [start_layer, end_layer) as defined by the PipelineShardMetadata
2. WHEN loading model layers, THE Runner SHALL place the model weights on the detected GPU device (CUDA for NVIDIA, XPU for Intel Arc) or fall back to CPU
3. WHEN loading model layers, THE Runner SHALL use the HuggingFace transformers library to load the model architecture and selectively load layer weights matching the shard range
4. WHEN model loading completes successfully, THE Runner SHALL transition to RunnerLoaded status
5. IF model loading fails due to insufficient GPU memory, THEN THE Runner SHALL transition to RunnerFailed status with an error message reporting required memory versus available memory
6. THE Runner SHALL support loading models defined in the existing model_cards.toml format without requiring model-card changes specific to PyTorch

### Requirement 5: Distributed Warmup Coordination

**User Story:** As a cluster operator, I want all gremlin nodes to complete a coordinated warmup pass before accepting inference requests, so that the first real request does not incur cold-start latency.

#### Acceptance Criteria

1. WHEN all runners in a PyTorchIPEXRingInstance reach RunnerLoaded status, THE plan loop SHALL dispatch StartWarmup tasks following the existing rank ordering (non-zero ranks first, rank 0 last)
2. WHEN a Runner receives a StartWarmup task, THE Runner SHALL execute a forward pass through its assigned layers using a dummy input tensor, sending and receiving activations through the process group using CPU tensor staging (GPU→CPU before send, CPU→GPU after recv)
3. WHEN warmup completes successfully on a Runner, THE Runner SHALL transition to RunnerReady status
4. IF warmup fails on any Runner, THEN THE Runner SHALL transition to RunnerFailed status, which SHALL trigger shutdown of all runners in the instance via the existing plan loop logic

### Requirement 6: Switched Ethernet Host Resolution for torch.distributed

**User Story:** As a cluster operator, I want the placement algorithm to generate correct host mappings for PyTorchIPEXRingInstance using the gremlin nodes' ethernet IP addresses, so that `torch.distributed` can establish connections through the aggregation switch.

#### Acceptance Criteria

1. WHEN creating a PyTorchIPEXRingInstance, THE Placement module SHALL generate hosts_by_node using ethernet IP addresses from the topology graph (the 10.1.1.x addresses on the LACP-bonded interfaces)
2. THE Placement module SHALL designate rank 0's IP as the MASTER_ADDR for `torch.distributed.init_process_group` rendezvous; all other ranks connect to this address through the switch fabric
3. WHEN generating host lists, THE Placement module SHALL prioritize ethernet connections over other interface types for PyTorchIPEXRingInstance
4. THE hosts_by_node mapping SHALL contain entries for all nodes in the selected cycle, with each node's entry containing the ethernet IP addresses reachable through the aggregation switch
5. IF a selected node does not have an ethernet IP address reported in the topology, THEN THE Placement module SHALL raise a ValueError identifying the node that lacks ethernet connectivity

### Requirement 7: Backend-Agnostic Runner Dispatch

**User Story:** As a developer, I want the runner to cleanly dispatch to the correct backend (MLX or PyTorch) based on instance type, so that adding new backends does not require modifying the core task loop.

#### Acceptance Criteria

1. WHEN a Runner is created for a PyTorchIPEXRingInstance, THE Runner SHALL use PyTorch-based code paths for ConnectToGroup, LoadModel, StartWarmup, and TextGeneration tasks
2. WHEN a Runner is created for an MlxRingInstance or MlxJacclInstance, THE Runner SHALL use MLX-based code paths (existing behavior preserved)
3. THE Runner SHALL determine the backend type once at startup from the bound instance type and use that determination consistently for all subsequent task handling
4. THE Runner SHALL not import MLX modules when running a PyTorch backend, and SHALL not import PyTorch modules when running an MLX backend

### Requirement 8: Graceful Shutdown of Distributed Process Group

**User Story:** As a cluster operator, I want distributed process groups to be cleanly torn down when an instance is deleted or a runner fails, so that network ports and GPU memory are released.

#### Acceptance Criteria

1. WHEN a Runner receives a Shutdown task for a PyTorchIPEXRingInstance, THE Runner SHALL call `torch.distributed.destroy_process_group()` before releasing GPU resources
2. WHEN a Runner transitions to RunnerFailed status during distributed operation, THE Runner SHALL attempt to destroy the process group with a short timeout (5 seconds) to avoid blocking on unreachable peers
3. WHEN shutting down, THE Runner SHALL release GPU memory by clearing CUDA or XPU caches and deleting model references
4. IF process group destruction hangs beyond the timeout, THEN THE Runner SHALL log a warning and proceed with local resource cleanup

### Requirement 9: NixOS Service Configuration for Distributed Inference

**User Story:** As a cluster operator, I want the exo systemd service on each gremlin node to be configured with the necessary environment variables and network permissions for `torch.distributed`, so that distributed inference works without manual setup.

#### Acceptance Criteria

1. THE NixOS module for the exo service SHALL configure environment variables MASTER_ADDR and MASTER_PORT when distributed mode is enabled
2. THE NixOS module SHALL open the configured ephemeral port range in the firewall for inter-node `torch.distributed` Gloo TCP communication
3. THE NixOS module SHALL ensure PyTorch (with XPU support on Intel nodes, with CUDA support on gremlin-1) and appropriate GPU drivers are available in the service's runtime environment
4. WHEN the exo service starts on a gremlin node, THE service SHALL verify GPU availability and log the detected GPU type, memory architecture (shared vs discrete), and available memory before joining the cluster

### Requirement 10: Error Recovery for Node Failures

**User Story:** As a cluster operator, I want the cluster to detect and recover from individual gremlin node failures during distributed inference, so that a single node crash does not leave the cluster in a broken state indefinitely.

#### Acceptance Criteria

1. WHEN a Runner in a distributed instance transitions to RunnerFailed, THE plan loop SHALL dispatch Shutdown tasks to all other runners in the same instance (existing behavior via _kill_runner)
2. WHEN a `torch.distributed` operation raises a communication error (e.g., peer disconnected, Gloo TCP connection lost), THE Runner SHALL transition to RunnerFailed with an error message identifying the failed peer rank
3. WHEN all runners in a failed instance have shut down, THE Master SHALL be able to re-place the instance on available nodes through the normal placement flow
4. IF a node becomes unreachable during model loading or warmup, THEN THE Runner SHALL detect the failure within the configured timeout and transition to RunnerFailed rather than hanging indefinitely

### Requirement 11: Remove IPEX Dependency and Use Native PyTorch XPU

**User Story:** As a developer, I want the codebase to use native PyTorch XPU support instead of the deprecated Intel Extension for PyTorch (IPEX), so that the project stays on supported, upstream packages.

#### Acceptance Criteria

1. THE codebase SHALL NOT import or depend on `intel_extension_for_pytorch` (IPEX) — all Intel GPU operations SHALL use native `torch.xpu` APIs available in PyTorch 2.11+
2. THE Runner SHALL use `torch.xpu.is_available()` and `torch.xpu.device_count()` from native PyTorch for Intel GPU detection, without requiring IPEX to be installed
3. THE distributed backend for all nodes (including Intel XPU nodes) SHALL use Gloo (`torch.distributed.init_process_group(backend='gloo')`), not XCCL — XCCL is validated only on Intel Data Center GPU Max Series and does not work on Intel Arc client GPUs (discrete or integrated)
4. THE NixOS package configuration SHALL include PyTorch 2.11+ with XPU support enabled, and SHALL NOT include IPEX or XCCL-specific dependencies
5. THE existing IPEX-specific code in `src/exo/worker/engines/pytorch_ipex/` SHALL be refactored to remove all `import intel_extension_for_pytorch` statements and replace them with native PyTorch equivalents
6. WHEN PyTorch is built or installed on gremlin nodes, THE build SHALL include XPU backend support (SYCL/Level Zero) for Intel Arc GPU acceleration via native `torch.xpu`

### Requirement 12: Memory Budgeting for Shared-Memory GPU Nodes

**User Story:** As a cluster operator, I want the placement algorithm to account for the shared-memory architecture of Intel integrated GPUs on gremlin-2/3/4, so that shard assignments do not over-commit system RAM and cause out-of-memory failures.

#### Acceptance Criteria

1. WHEN placing shards on a node with a Shared_Memory_GPU, THE Placement module SHALL treat GPU memory and system memory as the same pool, because Intel integrated GPUs share system RAM with the CPU
2. WHEN calculating available memory for shard placement on a Shared_Memory_GPU node, THE Placement module SHALL subtract a configurable OS and system overhead reservation (default 2 GiB) from total system RAM before allocating memory to model weights, activations, and KV cache
3. WHEN placing shards on a node with a Discrete_GPU (e.g., gremlin-1 with NVIDIA), THE Placement module SHALL use dedicated GPU VRAM for shard memory calculations, independent of system RAM
4. THE Placement module SHALL use the memory architecture type (shared vs discrete) reported by the Info_Gatherer to select the correct memory budgeting strategy for each node
5. IF a shard assignment would exceed the available memory budget on a Shared_Memory_GPU node (accounting for OS overhead, model weights, activation buffers, and KV cache), THEN THE Placement module SHALL reject the placement and log a warning identifying the memory shortfall
6. THE Placement module SHALL expose the memory budget calculation in placement logs so that operators can diagnose why a particular shard assignment was accepted or rejected

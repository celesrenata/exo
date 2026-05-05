# Requirements Document

## Introduction

This feature implements tensor parallelism for the PyTorch XPU backend on the gremlin cluster, complementing the existing pipeline parallelism implementation. The cluster has 4 NixOS nodes (gremlin-1 through gremlin-4), each with Intel Arc Graphics Meteor Lake-P iGPUs and ~94 GiB shared memory. Nodes will be interconnected via Thunderbolt 4 cables (40 Gbps per port, 4 root ports per node), providing high-bandwidth, low-latency communication suitable for the frequent all-reduce operations that tensor parallelism requires.

**Why tensor parallelism?** Pipeline parallelism (current implementation) processes tokens sequentially through the pipeline — only one node computes at a time during decode. Tensor parallelism splits weight matrices across all nodes so every node computes simultaneously on the same token, then combines results via all-reduce. This dramatically reduces per-token latency during decode at the cost of higher interconnect bandwidth requirements. Thunderbolt 4 at 40 Gbps (vs. 2.5 Gbps ethernet) makes this feasible.

**Interconnect:** Thunderbolt 4 presents as network interfaces via the `thunderbolt-net` kernel module. Each TB4 port provides 40 Gbps bidirectional bandwidth. With 4 ports per node and 4 nodes, a full mesh topology is achievable (each node directly connected to all 3 others), or a ring/daisy-chain for simpler cabling. The Gloo backend's TCP transport runs over these TB4 network interfaces, gaining the bandwidth advantage without requiring custom RDMA drivers.

**Tensor parallel strategy for transformer layers:**
- Attention QKV projection: Column-parallel (split attention heads across ranks)
- Attention output projection: Row-parallel (each rank has partial output, all-reduce to combine)
- MLP up-projection (gate): Column-parallel (split intermediate dimension across ranks)
- MLP down-projection: Row-parallel (each rank has partial result, all-reduce to combine)

This results in 2 all-reduce operations per transformer layer. For Qwen3.5-4B with 32 layers, that is 64 all-reduces per generated token.

**Coexistence with pipeline parallelism:** The system supports hybrid parallelism where tensor parallelism is used within a group of TB4-connected nodes, and pipeline parallelism is used across groups (if the cluster grows). For the initial 4-node TB4 cluster, pure tensor parallelism (all 4 nodes in one tensor-parallel group) is the primary target.

## Glossary

- **Tensor_Parallelism**: A model parallelism strategy where weight matrices within each layer are split across multiple nodes; all nodes compute simultaneously on the same input and synchronize via collective operations (all-reduce) after each parallel computation
- **Pipeline_Parallelism**: The existing sharding strategy where model layers are split sequentially across nodes; each node processes its layer range and passes activations to the next node (sequential computation)
- **All_Reduce**: A collective communication operation where each node contributes a tensor and all nodes receive the element-wise sum (or other reduction) of all contributions; used after column-parallel and row-parallel linear layers to combine partial results
- **Column_Parallel_Linear**: A linear layer where the weight matrix is split along the output dimension (columns) across ranks; each rank computes a slice of the output; used for QKV projections and MLP up-projections
- **Row_Parallel_Linear**: A linear layer where the weight matrix is split along the input dimension (rows) across ranks; each rank computes with a slice of the input and produces a partial output that must be all-reduced; used for attention output projections and MLP down-projections
- **Tensor_Parallel_Group**: The set of ranks participating in tensor parallelism for a given model; all ranks in the group process the same token simultaneously and synchronize via all-reduce
- **Thunderbolt_4**: The high-bandwidth interconnect (40 Gbps bidirectional per port) used for inter-node communication; presents as network interfaces via the `thunderbolt-net` kernel module on Linux
- **TB4_Mesh**: A network topology where each node has a direct Thunderbolt 4 connection to every other node in the tensor-parallel group; with 4 nodes and 4 TB4 ports each, a full mesh uses 3 ports per node
- **TB4_Ring**: A network topology where nodes are connected in a ring via Thunderbolt 4 cables; simpler cabling (1 port per neighbor) but higher latency for all-reduce (must traverse multiple hops)
- **Weight_Shard**: The portion of a weight matrix assigned to a specific rank in tensor parallelism; for a weight matrix of shape (out_features, in_features) split across N ranks column-parallel, each rank holds (out_features/N, in_features)
- **Hybrid_Parallelism**: Combining tensor parallelism (within a TB4-connected group) with pipeline parallelism (across groups or for layers that cannot be tensor-parallelized); allows scaling beyond a single TB4 group
- **Gloo_All_Reduce**: The all-reduce collective operation implemented by the Gloo backend over TCP; runs over TB4 network interfaces to achieve 40 Gbps throughput
- **Hidden_Size**: The dimension of the transformer's hidden state vector; determines the size of weight matrices and all-reduce payloads (e.g., 2560 for Qwen3.5-4B, 3584 for Qwen3.5-4B's intermediate MLP)
- **Attention_Heads**: The number of attention heads in the transformer; must be evenly divisible by the tensor-parallel world size for head-parallel splitting
- **TransformerShard**: The existing model wrapper that executes a contiguous range of transformer layers; tensor parallelism modifies this to execute ALL layers but with sharded weights within each layer
- **TensorParallelShard**: A new model wrapper that holds sharded weights for all layers and performs all-reduce synchronization after each parallel computation within a layer
- **Model_Card**: A TOML configuration file in `resources/inference_model_cards/` that describes a model's architecture; the `supports_tensor` field indicates whether the model's dimensions are compatible with tensor-parallel splitting

## Requirements

### Requirement 1: Thunderbolt 4 Network Interface Configuration

**User Story:** As a cluster operator, I want each gremlin node to configure Thunderbolt 4 ports as network interfaces with static IP addresses, so that the Gloo backend can use TB4 for high-bandwidth inter-node communication.

#### Acceptance Criteria

1. WHEN a Thunderbolt 4 cable is connected between two gremlin nodes, THE NixOS module SHALL load the `thunderbolt-net` kernel module and create a network interface for each active TB4 port
2. THE NixOS module SHALL assign static IP addresses to TB4 network interfaces on a dedicated subnet (separate from the existing 10.1.1.x ethernet subnet), so that TB4 traffic is isolated from management traffic
3. WHEN multiple TB4 ports are connected on a single node, THE NixOS module SHALL configure each port as a separate network interface with a unique IP address on the TB4 subnet
4. THE NixOS module SHALL configure the firewall to allow Gloo TCP traffic (ports 49152–65535) on TB4 network interfaces
5. IF a Thunderbolt 4 cable is disconnected, THEN THE NixOS module SHALL log the link-down event and the affected TB4 network interface SHALL become unavailable without crashing the node or other TB4 interfaces
6. THE NixOS module SHALL expose a configuration option for the TB4 subnet prefix and per-node IP assignments

### Requirement 2: TB4 Topology Discovery

**User Story:** As a cluster operator, I want the system to discover the Thunderbolt 4 topology (which nodes are directly connected via TB4), so that the tensor-parallel group can be formed using only TB4-connected nodes.

#### Acceptance Criteria

1. WHEN the exo service starts on a gremlin node, THE Topology_Discoverer SHALL enumerate all active TB4 network interfaces and their peer connections by probing configured TB4 subnet addresses
2. THE Topology_Discoverer SHALL determine the TB4 topology type (mesh, ring, or partial) by testing reachability between all node pairs on the TB4 subnet
3. WHEN a full mesh is detected (all nodes directly connected to all others via TB4), THE Topology_Discoverer SHALL report the topology as "mesh" with the set of directly-connected node pairs
4. WHEN a ring topology is detected (each node connected to exactly 2 neighbors via TB4), THE Topology_Discoverer SHALL report the topology as "ring" with the ordered ring sequence
5. THE Topology_Discoverer SHALL report the measured bandwidth between directly-connected TB4 peers using a short bandwidth probe (configurable, default disabled in production)
6. IF fewer than 2 nodes are reachable on the TB4 subnet, THEN THE Topology_Discoverer SHALL report that tensor parallelism is unavailable and the system SHALL fall back to pipeline parallelism over ethernet

### Requirement 3: Weight Matrix Sharding for Tensor Parallelism

**User Story:** As a cluster operator, I want transformer weight matrices to be split across tensor-parallel ranks, so that each node holds only its portion of the weights and computes its share of the output.

#### Acceptance Criteria

1. WHEN loading a model for tensor parallelism, THE TensorParallelShard SHALL split each attention QKV projection weight matrix along the output dimension (column-parallel), assigning each rank attention_heads/world_size heads
2. WHEN loading a model for tensor parallelism, THE TensorParallelShard SHALL split each attention output projection weight matrix along the input dimension (row-parallel), so each rank contributes a partial result that is all-reduced
3. WHEN loading a model for tensor parallelism, THE TensorParallelShard SHALL split each MLP up-projection (gate) weight matrix along the output dimension (column-parallel), assigning each rank intermediate_size/world_size columns
4. WHEN loading a model for tensor parallelism, THE TensorParallelShard SHALL split each MLP down-projection weight matrix along the input dimension (row-parallel), so each rank contributes a partial result that is all-reduced
5. THE TensorParallelShard SHALL load only its rank's portion of each weight matrix into GPU memory, reducing per-node memory usage by approximately 1/world_size for parallelized layers
6. IF the model's attention_heads count is not evenly divisible by the tensor-parallel world_size, THEN THE TensorParallelShard SHALL raise a ValueError identifying the incompatible dimensions

### Requirement 4: All-Reduce Synchronization via Gloo over TB4

**User Story:** As a cluster operator, I want all-reduce operations to run over the Thunderbolt 4 network interfaces using the Gloo backend, so that tensor-parallel ranks synchronize their partial results with high bandwidth and low latency.

#### Acceptance Criteria

1. WHEN initializing the tensor-parallel process group, THE Distributed_Communicator SHALL create a Gloo process group that binds to TB4 network interfaces (using GLOO_SOCKET_IFNAME set to the TB4 interface name)
2. WHEN a Row_Parallel_Linear layer completes its computation, THE TensorParallelShard SHALL invoke `torch.distributed.all_reduce` on the partial output tensor using the tensor-parallel process group
3. THE all-reduce operation SHALL use sum reduction (each rank's partial output is summed element-wise to produce the complete output on all ranks)
4. WHEN the TB4 topology is a mesh, THE Gloo_All_Reduce SHALL utilize direct connections between all ranks for optimal all-reduce performance
5. IF an all-reduce operation times out (configurable, default 30 seconds), THEN THE TensorParallelShard SHALL raise a RuntimeError identifying the layer index and tensor shape that failed to synchronize
6. THE Distributed_Communicator SHALL support creating separate process groups for tensor parallelism (over TB4) and pipeline parallelism (over ethernet), enabling hybrid parallelism configurations

### Requirement 5: Tensor-Parallel Forward Pass

**User Story:** As a cluster operator, I want each transformer layer to execute with tensor-parallel weight sharding and all-reduce synchronization, so that all nodes compute simultaneously and produce correct outputs equivalent to single-node execution.

#### Acceptance Criteria

1. WHEN executing a transformer layer's attention block, THE TensorParallelShard SHALL compute QKV projections using the rank's column-parallel weight shard, producing partial attention outputs for the rank's assigned heads
2. WHEN the attention output projection completes on each rank, THE TensorParallelShard SHALL perform an all-reduce (sum) across all tensor-parallel ranks to combine the partial outputs into the full hidden state
3. WHEN executing a transformer layer's MLP block, THE TensorParallelShard SHALL compute the up-projection (gate) using the rank's column-parallel weight shard, apply the activation function, then compute the down-projection using the rank's row-parallel weight shard
4. WHEN the MLP down-projection completes on each rank, THE TensorParallelShard SHALL perform an all-reduce (sum) across all tensor-parallel ranks to combine the partial outputs into the full hidden state
5. THE TensorParallelShard SHALL execute embedding lookup and final layer norm on all ranks redundantly (these layers are not sharded), so that each rank has the full input and can produce logits independently
6. WHEN the final lm_head projection is tensor-parallelized, THE TensorParallelShard SHALL split the vocabulary dimension across ranks and use all-gather to reconstruct the full logits on rank 0 for token sampling

### Requirement 6: Tensor-Parallel Autoregressive Generation

**User Story:** As a user, I want tensor-parallel inference to generate tokens with the same autoregressive loop as pipeline parallelism but with lower per-token latency, so that I receive faster responses.

#### Acceptance Criteria

1. WHEN generation begins with tensor parallelism, ALL ranks SHALL process the full tokenized prompt simultaneously through all layers (prefill phase), with all-reduce synchronization at each layer
2. WHEN the prefill phase completes, ALL ranks SHALL have identical logits (due to all-reduce synchronization), and rank 0 SHALL sample the next token
3. WHEN rank 0 samples a token, THE Generation_Pipeline SHALL broadcast the token ID to all other ranks so all ranks process the same token in the next decode iteration
4. DURING the decode phase, ALL ranks SHALL process the single new token simultaneously through all layers with all-reduce synchronization, producing the next token's logits
5. THE Generation_Pipeline SHALL maintain a KV cache on each rank containing only the key-value pairs for that rank's assigned attention heads (head-parallel KV cache)
6. WHEN generation terminates (EOS token or max_tokens), rank 0 SHALL broadcast a termination signal to all other ranks

### Requirement 7: Model Card Tensor Parallelism Compatibility

**User Story:** As a developer, I want model cards to indicate whether a model supports tensor parallelism on the XPU backend, so that the placement algorithm can select the correct parallelism strategy.

#### Acceptance Criteria

1. THE Model_Card format SHALL use the existing `supports_tensor` boolean field to indicate tensor parallelism compatibility for both MLX and PyTorch XPU backends
2. WHEN `supports_tensor` is true for a model, THE Placement module SHALL verify that the model's `attention_heads` count (derivable from hidden_size and head_dim) is evenly divisible by the requested tensor-parallel world_size before creating a tensor-parallel instance
3. THE Model_Card for Qwen3.5-4B SHALL be updated to set `supports_tensor = true` once tensor parallelism is implemented and validated, since its 32 attention heads are divisible by 4 (the cluster size)
4. THE Model_Card for Qwen3.5-2B SHALL be updated to set `supports_tensor = true` once validated, since its 20 attention heads are divisible by 4
5. IF a model's `supports_tensor` is false and tensor parallelism is requested, THEN THE Placement module SHALL fall back to pipeline parallelism and log the reason

### Requirement 8: Hybrid Parallelism (Tensor + Pipeline)

**User Story:** As a cluster operator, I want the option to combine tensor parallelism within TB4-connected node groups with pipeline parallelism across groups, so that the system can scale beyond a single TB4 interconnect group.

#### Acceptance Criteria

1. WHEN the cluster has a single TB4-connected group of N nodes, THE Placement module SHALL default to pure tensor parallelism across all N nodes (no pipeline stages)
2. WHERE hybrid parallelism is configured, THE Placement module SHALL assign tensor-parallel groups based on TB4 connectivity (nodes directly connected via TB4 form a tensor-parallel group) and pipeline stages across groups
3. WHEN hybrid parallelism is active, THE Generation_Pipeline SHALL perform all-reduce operations within each tensor-parallel group (over TB4) and send/recv activation passing between pipeline stages (over ethernet or TB4)
4. THE Placement module SHALL validate that each tensor-parallel group has the same number of nodes (symmetric tensor parallelism) before creating a hybrid instance
5. IF the TB4 topology does not support the requested tensor-parallel group size, THEN THE Placement module SHALL reject the configuration and suggest valid group sizes based on the discovered topology

### Requirement 9: Tensor-Parallel Process Group Initialization

**User Story:** As a cluster operator, I want the tensor-parallel process group to initialize over Thunderbolt 4 interfaces with correct rank assignments, so that all-reduce operations use the high-bandwidth TB4 links.

#### Acceptance Criteria

1. WHEN a Runner receives a ConnectToGroup task for a tensor-parallel instance, THE Runner SHALL initialize a Gloo process group with GLOO_SOCKET_IFNAME set to the TB4 network interface name
2. THE Runner SHALL derive its tensor-parallel rank from the instance's rank assignment, with rank 0 designated as the node whose TB4 IP is used as MASTER_ADDR
3. WHEN the tensor-parallel process group is initialized, THE Runner SHALL verify connectivity by performing a test all-reduce (sum of rank IDs) and validating the result equals the expected sum
4. IF the TB4 process group initialization fails, THEN THE Runner SHALL attempt fallback to ethernet-based tensor parallelism (lower bandwidth) and log a warning about degraded performance
5. THE Runner SHALL set a configurable initialization timeout (default 60 seconds) for the TB4 process group, shorter than the ethernet timeout since TB4 connections are local and should establish quickly
6. WHEN both tensor-parallel (TB4) and pipeline-parallel (ethernet) process groups are needed for hybrid parallelism, THE Runner SHALL initialize them as separate named process groups

### Requirement 10: Performance Monitoring and Comparison

**User Story:** As a cluster operator, I want to monitor tensor-parallel performance metrics (all-reduce latency, tokens per second, TB4 bandwidth utilization), so that I can compare tensor parallelism against pipeline parallelism and identify bottlenecks.

#### Acceptance Criteria

1. THE Generation_Pipeline SHALL measure and log the wall-clock time for each all-reduce operation during generation, reporting the mean and p99 latency per layer
2. THE Generation_Pipeline SHALL measure and report tokens-per-second for both prefill and decode phases, enabling direct comparison with pipeline-parallel performance
3. THE Generation_Pipeline SHALL calculate and log the effective TB4 bandwidth utilization per all-reduce (bytes transferred / time / theoretical 40 Gbps) to identify whether the interconnect is the bottleneck
4. WHEN generation completes, THE Generation_Pipeline SHALL include a performance summary in the GenerationStats comparing actual throughput against theoretical maximum given the interconnect bandwidth
5. IF all-reduce latency exceeds a configurable threshold (default 10ms for decode-phase all-reduces), THEN THE Generation_Pipeline SHALL log a warning identifying the slow layer and suggesting potential causes (TB4 congestion, asymmetric topology)

### Requirement 11: Graceful Degradation on TB4 Link Failure

**User Story:** As a cluster operator, I want the system to detect Thunderbolt 4 link failures and gracefully degrade to pipeline parallelism over ethernet, so that inference continues (at reduced performance) rather than failing completely.

#### Acceptance Criteria

1. WHEN a TB4 all-reduce operation fails due to a communication error, THE TensorParallelShard SHALL detect the failure and report it to the Generation_Pipeline
2. WHEN a TB4 link failure is detected during generation, THE Generation_Pipeline SHALL terminate the current generation with an error and transition the instance to a failed state
3. WHEN a tensor-parallel instance fails due to TB4 link loss, THE Master SHALL be able to re-place the model using pipeline parallelism over ethernet as a fallback strategy
4. THE Topology_Discoverer SHALL periodically (configurable, default every 60 seconds) verify TB4 link health by checking interface status, and report link-down events to the cluster state
5. IF a TB4 link is restored after a failure, THEN THE Topology_Discoverer SHALL report the link-up event, enabling the Master to consider tensor parallelism for future instance placements

### Requirement 12: NixOS Module for Thunderbolt 4 Networking

**User Story:** As a cluster operator, I want a NixOS module that configures Thunderbolt 4 networking (kernel modules, interfaces, IP addresses, firewall rules) declaratively, so that TB4 interconnect setup is reproducible and version-controlled.

#### Acceptance Criteria

1. THE NixOS module SHALL load the `thunderbolt-net` and `thunderbolt` kernel modules at boot to enable TB4 networking
2. THE NixOS module SHALL accept a configuration specifying the TB4 subnet, per-node IP assignments, and expected peer connections
3. THE NixOS module SHALL configure systemd-networkd (or equivalent) to assign static IPs to TB4 network interfaces as they appear, using udev rules to match Thunderbolt network devices
4. THE NixOS module SHALL configure the NixOS firewall to allow Gloo TCP traffic (ports 49152–65535) and ICMP on TB4 interfaces
5. THE NixOS module SHALL authorize Thunderbolt devices for networking (TB4 security levels) so that connections are established automatically without manual approval
6. THE NixOS module SHALL expose a systemd service that waits for TB4 interfaces to come up and logs their status, providing a dependency target for the exo service


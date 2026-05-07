# Design Document: Upstream Rebase XPU

## Overview

This design adds a unified PyTorch inference engine to the upstream exo project that supports both CUDA and XPU device backends, implements pipeline parallelism across the 4-node gremlin cluster, and validates that all inference runs on GPU (never CPU). The implementation is purely additive — all new code lives in new files/directories with no modifications to existing upstream code except minimal registration in the engine factory.

The key architectural insight is that PyTorch's device-agnostic APIs (`tensor.to(device)`, `torch.zeros(..., device=device)`) allow a single engine implementation to target both Intel Arc iGPUs (via `torch.xpu`) and NVIDIA GPUs (via `torch.cuda`). The Gloo distributed backend serves as the common denominator for inter-node communication since it supports both device types (NCCL only supports CUDA).

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        exo Cluster (4 nodes)                            │
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │
│  │  gremlin-1  │   │  gremlin-2  │   │  gremlin-3  │   │  gremlin-4 │ │
│  │  10.1.1.12  │   │  10.1.1.13  │   │  10.1.1.14  │   │  10.1.1.15 │ │
│  │             │   │             │   │             │   │            │ │
│  │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌────────┐ │ │
│  │ │Stage 0  │ │   │ │Stage 1  │ │   │ │Stage 2  │ │   │ │Stage 3 │ │ │
│  │ │Embed +  │─┼──▶│ │Layers   │─┼──▶│ │Layers   │─┼──▶│ │Layers +│ │ │
│  │ │Layers   │ │   │ │7-13     │ │   │ │14-20    │ │   │ │LM Head │ │ │
│  │ │0-6      │ │   │ │         │ │   │ │         │ │   │ │21-27   │ │ │
│  │ └─────────┘ │   │ └─────────┘ │   │ └─────────┘ │   │ └────────┘ │ │
│  │             │   │             │   │             │   │            │ │
│  │ XPU/CUDA   │   │ XPU         │   │ XPU         │   │ XPU        │ │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └─────┬──────┘ │
│         │                  │                  │                │        │
│         └──────────────────┴──────────────────┴────────────────┘        │
│                    Gloo over LACP-bonded Ethernet                        │
│                    (2×2.5G per node via MikroTik)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Single Token Generation)

```
Prompt → gremlin-1 (encode + embed + layers 0-6)
       → send hidden_state to gremlin-2
       → gremlin-2 (layers 7-13)
       → send hidden_state to gremlin-3
       → gremlin-3 (layers 14-20)
       → send hidden_state to gremlin-4
       → gremlin-4 (layers 21-27 + LM head + sample)
       → token back to gremlin-1 (for next iteration or output)
```

## Architecture

### Component Diagram

```
src/exo/worker/engines/pytorch/
├── __init__.py              # Package exports
├── engine.py                # UnifiedPyTorchEngine (InferenceBackend impl)
├── device_detector.py       # GPU detection and selection
├── gpu_validator.py         # Runtime GPU assertions
├── pipeline/
│   ├── __init__.py
│   ├── coordinator.py       # Pipeline parallelism orchestration
│   ├── stage.py             # Single pipeline stage (layer subset)
│   └── activation_pass.py   # Inter-stage activation transfer
├── distributed/
│   ├── __init__.py
│   ├── communicator.py      # Gloo/NCCL process group management
│   ├── transport.py         # Transport layer (RDMA, Ethernet, LACP)
│   └── cpu_staging.py       # CPU-staged tensor transfer for Gloo
├── model/
│   ├── __init__.py
│   ├── loader.py            # HuggingFace safetensors model loading
│   ├── shard_loader.py      # Pipeline-stage-aware partial model loading
│   └── kv_cache.py          # KV cache management
├── sampling.py              # Token sampling (temperature, top-k, top-p)
└── tests/
    ├── __init__.py
    ├── test_device_detector.py
    ├── test_gpu_validator.py
    ├── test_pipeline_stage_assignment.py
    ├── test_cpu_staging.py
    └── test_e2e_pipeline.py  # End-to-end cluster test
```

### Design Principles

1. **Additive only** — No modifications to existing files except `factory.py` (add one `elif` branch) and `pyproject.toml` (add `psutil` dependency if not present).
2. **Device-agnostic** — All tensor operations use `tensor.to(device)` pattern. The `device` string is either `"xpu:0"` or `"cuda:0"`.
3. **Fail-fast on CPU** — Runtime assertions verify tensors are on GPU. No silent CPU fallback.
4. **CPU-staged communication** — Gloo requires CPU tensors for send/recv. On Intel iGPU (shared memory), CPU↔GPU copy is nearly free.
5. **Single process group** — One Gloo process group for all nodes. NCCL is only used when ALL nodes are CUDA.

## Components and Interfaces

### 1. UnifiedPyTorchEngine (`engine.py`)

Implements `InferenceBackend` from `src/exo/worker/engines/base.py`.

```python
@final
class UnifiedPyTorchEngine(InferenceBackend):
    """Unified PyTorch engine supporting CUDA and XPU devices.
    
    Implements the InferenceBackend protocol for both NVIDIA and Intel GPUs
    using PyTorch's device-agnostic APIs.
    """
    
    def __init__(
        self,
        device_type: Literal["cuda", "xpu"],
        device_index: int = 0,
        pipeline_config: PipelineConfig | None = None,
    ) -> None: ...
    
    async def encode(self, shard_metadata: ShardMetadata, prompt: str) -> np.ndarray: ...
    async def decode(self, shard_metadata: ShardMetadata, tokens: np.ndarray) -> str: ...
    async def infer_tensor(
        self, request_id: str, shard_metadata: ShardMetadata,
        input_data: np.ndarray, inference_state: dict | None = None,
    ) -> tuple[np.ndarray, dict | None]: ...
    async def sample(self, logits: np.ndarray) -> np.ndarray: ...
    async def load_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None: ...
```

### 2. DeviceDetector (`device_detector.py`)

Detects available GPUs and selects the appropriate device backend.

```python
@dataclass(frozen=True)
class DetectedDevice:
    device_type: Literal["cuda", "xpu"]
    device_index: int
    device_name: str
    memory_bytes: int
    memory_architecture: Literal["shared", "discrete"]

def detect_devices() -> list[DetectedDevice]: ...
def select_primary_device() -> DetectedDevice: ...
    """Select primary device. Prefers CUDA over XPU. Raises if no GPU found."""
```

### 3. GpuValidator (`gpu_validator.py`)

Runtime assertions that tensors are on the expected GPU device.

```python
@final
class GpuValidator:
    """Validates that all tensors reside on the expected GPU device."""
    
    def __init__(self, expected_device: str, debug_mode: bool = False) -> None: ...
    
    def assert_on_device(self, tensor: torch.Tensor, name: str) -> None:
        """Assert tensor is on expected device. Raises AssertionError if on CPU."""
    
    def validate_model_parameters(self, model: torch.nn.Module) -> None:
        """Assert all model parameters are on expected device."""
    
    def validate_forward_pass(
        self, inputs: dict[str, torch.Tensor], outputs: torch.Tensor
    ) -> None:
        """Validate input and output tensors are on device."""
```

### 4. PipelineCoordinator (`pipeline/coordinator.py`)

Orchestrates pipeline-parallel inference across N nodes.

```python
@dataclass(frozen=True)
class PipelineConfig:
    world_size: int          # Number of nodes (2, 3, or 4)
    rank: int                # This node's rank (0-indexed)
    total_layers: int        # Total transformer layers in model
    master_addr: str         # Rank 0 IP for rendezvous
    master_port: int         # Rendezvous port
    transport: Literal["rdma", "ethernet", "lacp"]

@final
class PipelineCoordinator:
    """Coordinates pipeline-parallel inference across nodes."""
    
    def __init__(self, config: PipelineConfig, engine: UnifiedPyTorchEngine) -> None: ...
    
    def get_stage_assignment(self) -> tuple[int, int]:
        """Return (start_layer, end_layer) for this node's pipeline stage."""
    
    async def forward_pipeline(
        self, input_tensor: torch.Tensor, request_id: str
    ) -> torch.Tensor | None:
        """Execute one forward pass through the pipeline.
        Returns output only on the last stage (rank == world_size - 1)."""
    
    async def generate_token(self, prompt_tokens: np.ndarray) -> np.ndarray:
        """Full token generation: forward through all stages, sample on last node."""
```

### 5. Communicator (`distributed/communicator.py`)

Manages the PyTorch distributed process group.

```python
@dataclass(frozen=True)
class CommConfig:
    rank: int
    world_size: int
    master_addr: str
    master_port: int
    backend: Literal["gloo", "nccl"]
    timeout_seconds: int = 30
    transport: Literal["rdma", "ethernet", "lacp"] = "ethernet"

@final
class Communicator:
    """Manages distributed process group and tensor communication."""
    
    def __init__(self, config: CommConfig) -> None: ...
    
    def initialize(self) -> None:
        """Initialize process group. Sets GLOO_SOCKET_IFNAME based on transport."""
    
    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """Send tensor to destination rank (CPU-staged for Gloo)."""
    
    def recv_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype, src_rank: int,
        target_device: str
    ) -> torch.Tensor:
        """Receive tensor from source rank and move to target device."""
    
    def destroy(self) -> None:
        """Destroy process group."""
```

### 6. TransportLayer (`distributed/transport.py`)

Detects and configures network transport.

```python
@dataclass(frozen=True)
class TransportInfo:
    transport_type: Literal["rdma", "ethernet", "lacp"]
    interface_name: str
    bind_address: str
    bandwidth_gbps: float

def detect_transport(preferred: Literal["rdma", "ethernet", "lacp"]) -> TransportInfo:
    """Detect available transport. Falls back to ethernet if preferred unavailable."""

def select_backend_for_devices(
    device_types: list[Literal["cuda", "xpu"]]
) -> Literal["gloo", "nccl"]:
    """Select distributed backend. NCCL only if ALL devices are CUDA."""
```

### 7. Engine Registration

Minimal change to `src/exo/worker/engines/factory.py`:

```python
elif backend_name == "pytorch":
    try:
        from exo.worker.engines.pytorch.engine import UnifiedPyTorchEngine
        from exo.worker.engines.pytorch.device_detector import select_primary_device
        
        device = select_primary_device()
        return UnifiedPyTorchEngine(
            device_type=device.device_type,
            device_index=device.device_index,
        )
    except ImportError as e:
        raise BackendNotAvailableError(
            "pytorch", f"Unified PyTorch backend not available: {e}"
        ) from e
```

## Data Models

### Pipeline Stage Assignment

```python
@dataclass(frozen=True)
class StageAssignment:
    """Assignment of model layers to a pipeline stage."""
    rank: int
    start_layer: int        # Inclusive
    end_layer: int          # Exclusive
    has_embedding: bool     # True for rank 0
    has_lm_head: bool       # True for last rank
    
def compute_stage_assignments(
    total_layers: int, world_size: int
) -> list[StageAssignment]:
    """Divide layers evenly across stages.
    
    Layers are divided as evenly as possible. If total_layers is not
    evenly divisible by world_size, earlier stages get one extra layer.
    
    Example: 28 layers across 4 nodes → [7, 7, 7, 7]
    Example: 30 layers across 4 nodes → [8, 8, 7, 7]
    """
```

### Distributed Communication Types

```python
@dataclass(frozen=True)
class ActivationMessage:
    """Activation tensor passed between pipeline stages."""
    request_id: str
    sequence_position: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str  # "float16", "bfloat16", "float32"

@dataclass(frozen=True)
class PipelineStatus:
    """Status of a pipeline stage on a node."""
    rank: int
    node_ip: str
    device_type: Literal["cuda", "xpu"]
    device_name: str
    layers_loaded: tuple[int, int]  # (start, end)
    is_ready: bool
    error: str | None = None
```

### Transport Configuration

```python
@dataclass(frozen=True)
class TransportConfig:
    """Network transport configuration for distributed communication."""
    transport_type: Literal["rdma", "ethernet", "lacp"]
    interface_name: str          # e.g., "bond0", "enp2s0", "thunderbolt0"
    bind_address: str            # IP address to bind to
    mtu: int = 1500              # MTU size
    lacp_hash_policy: str = "layer3+4"  # For LACP transport
```

## MikroTik LACP Switch Configuration (Standalone Task)

> **EXTRACTABLE — can be given to another Kiro instance to execute via SSH**

This section contains complete RouterOS CLI commands for configuring LACP bonding on the MikroTik switch at `ssh://admin@10.1.1.253`. The commands configure 2 ports per gremlin node (8 ports total) as LACP bond groups with layer3+4 transmit hash policy.

### Prerequisites

- SSH access: `ssh admin@10.1.1.253`
- RouterOS version: 7.21.3
- Switch has 10GBE uplink and sufficient ports for 8× 2.5G connections
- Password: **ask the user**

### Port Assignment Plan

| Node | Bond Name | Port 1 | Port 2 | IP (node side) |
|------|-----------|--------|--------|----------------|
| gremlin-1 | bond-gremlin1 | ether1 | ether2 | 10.1.1.12 |
| gremlin-2 | bond-gremlin2 | ether3 | ether4 | 10.1.1.13 |
| gremlin-3 | bond-gremlin3 | ether5 | ether6 | 10.1.1.14 |
| gremlin-4 | bond-gremlin4 | ether7 | ether8 | 10.1.1.15 |

### RouterOS CLI Commands

```routeros
# ============================================================
# MikroTik LACP Configuration for Gremlin Cluster
# Execute via: ssh admin@10.1.1.253
# RouterOS 7.21.3
# ============================================================

# --- Create LACP bonding interfaces ---

/interface bonding
add name=bond-gremlin1 slaves=ether1,ether2 mode=802.3ad \
    transmit-hash-policy=layer-3-and-4 lacp-rate=fast \
    link-monitoring=mii mii-interval=100ms

add name=bond-gremlin2 slaves=ether3,ether4 mode=802.3ad \
    transmit-hash-policy=layer-3-and-4 lacp-rate=fast \
    link-monitoring=mii mii-interval=100ms

add name=bond-gremlin3 slaves=ether5,ether6 mode=802.3ad \
    transmit-hash-policy=layer-3-and-4 lacp-rate=fast \
    link-monitoring=mii mii-interval=100ms

add name=bond-gremlin4 slaves=ether7,ether8 mode=802.3ad \
    transmit-hash-policy=layer-3-and-4 lacp-rate=fast \
    link-monitoring=mii mii-interval=100ms

# --- Add bond interfaces to the bridge ---
# (Assumes a bridge named "bridge" already exists for the 10.1.1.0/24 network)

/interface bridge port
add bridge=bridge interface=bond-gremlin1
add bridge=bridge interface=bond-gremlin2
add bridge=bridge interface=bond-gremlin3
add bridge=bridge interface=bond-gremlin4

# --- Remove individual ports from bridge if previously added ---
# (Skip if ports were not previously bridged individually)

/interface bridge port
remove [find interface=ether1]
remove [find interface=ether2]
remove [find interface=ether3]
remove [find interface=ether4]
remove [find interface=ether5]
remove [find interface=ether6]
remove [find interface=ether7]
remove [find interface=ether8]
```

### Verification Commands

```routeros
# --- Verify bond status ---

/interface bonding print detail
# Expected: All 4 bonds show "R" (running), mode=802.3ad

/interface bonding monitor bond-gremlin1
# Expected: mode=802.3ad, active-ports=ether1,ether2, lacp-partner-system-id present

/interface bonding monitor bond-gremlin2
/interface bonding monitor bond-gremlin3
/interface bonding monitor bond-gremlin4

# --- Verify bridge membership ---

/interface bridge port print where interface~"bond-gremlin"
# Expected: All 4 bond interfaces listed under "bridge"

# --- Verify connectivity ---

/ping 10.1.1.12 count=3
/ping 10.1.1.13 count=3
/ping 10.1.1.14 count=3
/ping 10.1.1.15 count=3

# --- Check LACP partner info (confirms nodes are negotiating) ---

/interface bonding monitor bond-gremlin1 once
# Look for: lacp-system-id, lacp-partner-system-id (non-empty = negotiated)
```

### Node-Side Configuration (NixOS)

Each gremlin node needs a matching LACP bond configuration in its NixOS config:

```nix
# In each gremlin's /etc/nixos/configuration.nix
networking.bonds.bond0 = {
  interfaces = [ "enp2s0" "enp3s0" ];  # Adjust interface names per node
  driverOptions = {
    mode = "802.3ad";
    xmit_hash_policy = "layer3+4";
    lacp_rate = "fast";
    miimon = "100";
  };
};

networking.interfaces.bond0.ipv4.addresses = [{
  address = "10.1.1.12";  # Adjust per node: .12, .13, .14, .15
  prefixLength = 24;
}];
```

### Rollback Commands

```routeros
# If something goes wrong, remove bonds and re-add individual ports:

/interface bonding remove [find name~"bond-gremlin"]
/interface bridge port
add bridge=bridge interface=ether1
add bridge=bridge interface=ether2
add bridge=bridge interface=ether3
add bridge=bridge interface=ether4
add bridge=bridge interface=ether5
add bridge=bridge interface=ether6
add bridge=bridge interface=ether7
add bridge=bridge interface=ether8
```

## Error Handling

### Device Errors

| Error Condition | Behavior |
|----------------|----------|
| No GPU detected (neither CUDA nor XPU) | Fatal error at startup. Refuse to initialize engine. |
| GPU available at init but disappears | Fatal error on next forward pass. Pipeline reports node failure. |
| Tensor found on CPU during inference | `AssertionError` with tensor name and expected device. |
| `torch.xpu.is_available()` returns False | Fatal error. Log Level Zero debugging hints. |
| OOM on GPU | Return descriptive error with memory requirement vs available. |

### Pipeline Errors

| Error Condition | Behavior |
|----------------|----------|
| Node unreachable during activation send | Timeout after 30s. Report failed rank and pipeline position. |
| Process group init fails | Fatal error with master_addr, master_port, and rank info. |
| Activation shape mismatch on recv | RuntimeError with expected vs received shape. |
| Node fails to load its pipeline stage | Report which node, which layers, and the error reason. |

### Transport Errors

| Error Condition | Behavior |
|----------------|----------|
| RDMA transport unavailable | Fall back to Ethernet. Log warning. |
| LACP bond not detected | Fall back to Ethernet. Log warning. |
| Network interface not found | Fatal error with interface name and available interfaces. |

## Testing Strategy

### Unit Tests (example-based)

- Device detection with mocked `torch.cuda`/`torch.xpu` availability
- Pipeline stage assignment for various layer counts and world sizes
- GPU validator assertions (tensor on correct device vs wrong device)
- Transport detection with mocked network interfaces
- Backend selection logic (CUDA-only → NCCL, mixed → Gloo, XPU-only → Gloo)

### Property-Based Tests (Hypothesis)

Property-based testing is appropriate for this feature because:
- Pipeline stage assignment is a pure function with clear invariants
- CPU staging is a round-trip operation (stage then unstage = identity)
- Device detection has universal properties (exactly one primary device selected)
- Backend selection has deterministic rules based on device type combinations

Library: **Hypothesis** (already in use in this project — see `.hypothesis/` directory)
Configuration: Minimum 100 iterations per property test.

### Integration Tests

- End-to-end test across 4 gremlin nodes (Qwen3.5:4B, hello world prompt)
- Process group initialization across 2+ nodes
- Activation tensor send/recv between adjacent pipeline stages
- Model loading with pipeline-stage-aware partial loading

### End-to-End Validation Test

The E2E test (`test_e2e_pipeline.py`) runs as a pytest test:

```python
@pytest.mark.slow
@pytest.mark.e2e
async def test_qwen35_4b_pipeline_4_nodes():
    """Validate Qwen3.5:4B runs across all 4 gremlin nodes via pipeline parallelism."""
    # 1. Initialize pipeline across 10.1.1.12-15
    # 2. Load Qwen3.5:4B sharded across 4 stages
    # 3. Submit "hello world" prompt
    # 4. Assert ≥10 tokens generated
    # 5. Assert all 4 nodes participated (check pipeline stage execution)
    # 6. Assert no CPU tensors during generation (GPU validation)
    # 7. Assert completion within 120 seconds
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Device detection selects correctly and never falls back to CPU

*For any* combination of `(cuda_available: bool, xpu_available: bool)` where at least one is True, the device detector SHALL select "cuda" when CUDA is available (preferring it over XPU), select "xpu" when only XPU is available, and SHALL never select "cpu" as the primary device. When neither is available, it SHALL raise a fatal error.

**Validates: Requirements 3.1, 3.2, 3.4, 3.5**

### Property 2: Detected device report contains all required fields

*For any* detected GPU device (whether CUDA or XPU), the detection result SHALL include a non-empty device name, a non-negative memory capacity in bytes, a valid device index, and a memory architecture classification of either "shared" or "discrete".

**Validates: Requirements 3.6**

### Property 3: GPU validator rejects CPU tensors with descriptive errors

*For any* tensor and any expected device string (e.g., "xpu:0", "cuda:0"), if the tensor resides on CPU, the GPU validator SHALL raise an AssertionError whose message contains both the tensor's name and the expected device string. If the tensor resides on the expected device, the validator SHALL not raise.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

### Property 4: Pipeline stage assignment produces valid partitioning

*For any* `total_layers > 0` and `world_size` in `{2, 3, 4}` where `world_size <= total_layers`, the stage assignment function SHALL produce exactly `world_size` stages such that: (a) the stages cover all layers exactly once with no gaps or overlaps, (b) the layer count per stage differs by at most 1 between any two stages, (c) rank 0 has `has_embedding=True`, (d) rank `world_size-1` has `has_lm_head=True`, and (e) each rank maps to exactly one contiguous range of layers.

**Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 5.8**

### Property 5: Distributed backend selection follows device-type rules

*For any* non-empty list of device types (each being "cuda" or "xpu"), the backend selection function SHALL return "nccl" if and only if ALL device types are "cuda", and SHALL return "gloo" otherwise (i.e., when any device is "xpu" or the list is mixed).

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 6: CPU staging round-trip preserves tensor data

*For any* tensor with arbitrary shape, dtype (float16, bfloat16, float32), and values, staging to CPU via `stage_to_cpu()` and then unstaging back to the original device via `unstage_from_cpu()` SHALL produce a tensor that is element-wise equal to the original, with the same shape and dtype.

**Validates: Requirements 6.7**

### Property 7: Transport fallback defaults to Ethernet

*For any* transport type in `{"rdma", "lacp", "ethernet"}`, if the selected transport is detected as unavailable (interface not found, bond not configured, RDMA not present), the transport layer SHALL fall back to `"ethernet"` and the resulting `TransportInfo.transport_type` SHALL be `"ethernet"`.

**Validates: Requirements 7.9**

### Property 8: Partial model loading respects stage assignment

*For any* valid stage assignment `(start_layer, end_layer)` and any model with `total_layers` layers, the shard loader SHALL load exactly the layers in `[start_layer, end_layer)` and no others. The number of loaded layer modules SHALL equal `end_layer - start_layer`.

**Validates: Requirements 10.4**

# Intel IPEX Integration Design Document

## Overview

This design document outlines the integration of Intel Extension for PyTorch (IPEX) into the EXO distributed AI inference system. The integration will add Intel GPU acceleration as a first-class engine alongside existing MLX, CUDA, and CPU engines, providing optimized inference for Intel Arc GPUs and Intel Data Center GPUs.

## Architecture

### Engine System Integration

The IPEX engine will integrate into EXO's existing pluggable engine architecture by implementing the same interfaces used by MLX and Torch engines:

```
EXO Engine Architecture
├── engine_init.py (dispatch logic)
├── engine_utils.py (detection & selection)
├── mlx/ (Apple Silicon)
├── torch/ (CPU/CUDA fallback)
└── ipex/ (NEW - Intel GPU acceleration)
    ├── __init__.py (protocols & wrappers)
    ├── utils_ipex.py (initialization)
    └── generator/
        └── generate.py (inference logic)
```

### Hardware Detection Flow

```mermaid
graph TD
    A[System Startup] --> B[detect_available_engines()]
    B --> C{Intel GPU Present?}
    C -->|Yes| D[Check Intel GPU Driver]
    C -->|No| E[Skip IPEX]
    D --> F{IPEX Available?}
    F -->|Yes| G[Add 'ipex' to available engines]
    F -->|No| H[Log warning, fallback to torch]
    G --> I[Engine Selection Logic]
    H --> I
    E --> I
    I --> J[select_best_engine()]
    J --> K{Priority Order}
    K --> L[MLX > IPEX > CUDA > Torch > CPU]
```

## Components and Interfaces

### 1. Engine Detection and Selection (`engine_utils.py`)

**Modifications Required:**
- Add `"ipex"` to `EngineType` literal type
- Implement `detect_intel_gpu()` function
- Add IPEX detection to `detect_available_engines()`
- Update engine priority in `select_best_engine()`

**Intel GPU Detection Logic:**
```python
def detect_intel_gpu() -> bool:
    """Detect Intel GPU and IPEX availability."""
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        # Check for Intel GPU devices
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            if device_count > 0:
                return True
    except ImportError:
        pass
    return False
```

### 2. Engine Initialization (`engine_init.py`)

**Modifications Required:**
- Add IPEX case to `initialize_engine()`
- Add IPEX case to `warmup_engine()`
- Add IPEX case to `generate_with_engine()`

**Integration Pattern:**
```python
elif engine_type == "ipex":
    from exo.worker.engines.ipex.utils_ipex import initialize_ipex
    return initialize_ipex(bound_instance)
```

### 3. IPEX Engine Implementation

#### 3.1 Protocol Definitions (`ipex/__init__.py`)

```python
class IPEXModel(Protocol):
    """Protocol for IPEX-optimized models."""
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any: ...
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> torch.Tensor: ...

class IPEXTokenizerWrapper:
    """Wrapper for tokenizers with IPEX optimizations."""
    # Same interface as torch/TokenizerWrapper
```

#### 3.2 Model Initialization (`ipex/utils_ipex.py`)

```python
def initialize_ipex(bound_instance: BoundInstance) -> tuple[IPEXModel, IPEXTokenizerWrapper, Callable]:
    """Initialize IPEX model, tokenizer, and sampler."""
    
    # Load model with IPEX optimizations
    model = load_ipex_model(bound_instance.model_id)
    tokenizer = IPEXTokenizerWrapper(load_tokenizer(bound_instance.model_id))
    sampler = create_ipex_sampler()
    
    return model, tokenizer, sampler

def load_ipex_model(model_id: str) -> IPEXModel:
    """Load and optimize model for Intel GPU."""
    import intel_extension_for_pytorch as ipex
    import torch
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Move to Intel GPU
    device = torch.device("xpu")
    model = model.to(device)
    
    # Apply IPEX optimizations
    model = ipex.optimize(model, dtype=torch.float16)
    
    return model
```

#### 3.3 Inference Generation (`ipex/generator/generate.py`)

```python
def ipex_generate(
    model: IPEXModel,
    tokenizer: IPEXTokenizerWrapper,
    sampler: Callable,
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """Generate text using IPEX-optimized inference."""
    
    # Implementation mirrors torch_generate but with Intel GPU optimizations
    # - Use torch.xpu device context
    # - Apply IPEX-specific memory management
    # - Utilize Intel GPU tensor cores
```

### 4. Dashboard Integration

#### 4.1 Instance Type Support (`dashboard/src/routes/+page.svelte`)

**Modifications Required:**
- Add `'IpexRing'` to `InstanceMeta` type
- Add Intel IPEX button to instance type selection
- Update `matchesSelectedRuntime()` logic
- Add Intel GPU compatibility checking

**UI Changes:**
```typescript
type InstanceMeta = 'MlxRing' | 'MlxIbv' | 'MlxJaccl' | 'CpuRing' | 'CudaRing' | 'IpexRing';

// Add Intel IPEX button
<button 
    onclick={() => selectedInstanceType = 'IpexRing'}
    class="flex items-center gap-2 py-2 px-4 text-sm font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType === 'IpexRing' ? 'bg-transparent text-exo-yellow border-exo-yellow' : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
>
    <span class="w-4 h-4 rounded-full border-2 flex items-center justify-center {selectedInstanceType === 'IpexRing' ? 'border-exo-yellow' : 'border-exo-medium-gray'}">
        {#if selectedInstanceType === 'IpexRing'}
            <span class="w-2 h-2 rounded-full bg-exo-yellow"></span>
        {/if}
    </span>
    Intel IPEX
</button>
```

#### 4.2 System Information Display

**Engine Information Integration:**
- Add `ipex_available` field to `NodeInfo` interface
- Display Intel GPU information in topology nodes
- Show IPEX engine status in debug information

### 5. System Information Gathering

#### 5.1 Intel GPU Detection (`system_info.py`)

```python
async def get_intel_gpu_info() -> dict[str, Any]:
    """Get Intel GPU information."""
    info = {
        "intel_gpu_available": False,
        "intel_gpu_count": 0,
        "intel_gpu_memory": 0,
        "ipex_version": None,
    }
    
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            info["intel_gpu_available"] = True
            info["intel_gpu_count"] = torch.xpu.device_count()
            info["ipex_version"] = ipex.__version__
            
            # Get memory info for first device
            if info["intel_gpu_count"] > 0:
                props = torch.xpu.get_device_properties(0)
                info["intel_gpu_memory"] = props.total_memory
                
    except ImportError:
        pass
    
    return info
```

## Data Models

### Engine Information Schema

```python
@dataclass
class EngineInfo:
    available_engines: list[str]
    selected_engine: str | None
    mlx_available: bool
    torch_available: bool
    cpu_available: bool
    ipex_available: bool  # NEW
    intel_gpu_count: int  # NEW
    intel_gpu_memory: int  # NEW
```

### Instance Metadata

```python
# Add to existing instance types
class IpexRingInstance:
    shard_assignments: ShardAssignments
    intel_gpu_devices: list[int]
    ipex_optimization_level: str
```

## Error Handling

### Intel GPU Error Scenarios

1. **Driver Not Available**
   - Fallback to CPU/Torch engine
   - Log clear error message about Intel GPU driver requirements

2. **IPEX Import Failure**
   - Graceful degradation to torch engine
   - Provide installation instructions in logs

3. **GPU Memory Exhaustion**
   - Implement memory management similar to CUDA engine
   - Support model offloading and gradient checkpointing

4. **Device Communication Errors**
   - Retry logic for transient Intel GPU errors
   - Automatic fallback to CPU inference

### Error Logging Strategy

```python
class IPEXEngineError(Exception):
    """Base exception for IPEX engine errors."""
    pass

class IPEXDriverError(IPEXEngineError):
    """Intel GPU driver not available or incompatible."""
    pass

class IPEXMemoryError(IPEXEngineError):
    """Intel GPU memory allocation failed."""
    pass
```

## Testing Strategy

### Unit Tests

1. **Engine Detection Tests**
   - Mock Intel GPU availability
   - Test engine selection priority
   - Validate fallback behavior

2. **Model Loading Tests**
   - Test IPEX model optimization
   - Validate tokenizer wrapper functionality
   - Test memory allocation patterns

3. **Inference Tests**
   - Compare output consistency with torch engine
   - Test streaming generation
   - Validate performance metrics

### Integration Tests

1. **Multi-Engine Cluster Tests**
   - Mixed IPEX/MLX/CUDA clusters
   - Cross-engine model sharding
   - Load balancing validation

2. **Dashboard Integration Tests**
   - UI element rendering with Intel GPU
   - Instance creation with IPEX engine
   - Monitoring and metrics display

### Performance Benchmarks

1. **Inference Speed Comparison**
   - IPEX vs Torch CPU performance
   - Memory usage optimization
   - Throughput measurements

2. **Distributed Inference Tests**
   - Multi-GPU Intel setups
   - Cross-device communication latency
   - Scaling efficiency metrics

## NixOS Integration

### Package Dependencies

```nix
# Add to flake.nix propagatedBuildInputs for intel variant
intel-extension-for-pytorch = python.pkgs.buildPythonPackage {
  # IPEX package definition
};

# Intel GPU driver dependencies
intel-gpu-drivers = [
  intel-media-driver
  intel-compute-runtime
  level-zero
];
```

### Hardware Detection in Flake

```nix
# Conditional package selection based on Intel GPU presence
mkExoPackage = { pkgs, system, accelerator ? "cpu" }:
  let
    hasIntelGPU = # Intel GPU detection logic
    selectedAccelerator = if hasIntelGPU then "ipex" else accelerator;
  in
  # Package build logic
```

This design provides a comprehensive integration of Intel IPEX support that maintains consistency with the existing EXO architecture while adding Intel GPU acceleration capabilities across all system components.
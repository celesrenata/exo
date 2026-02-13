# Design Document: Intel Arc GPU Memory Allocation Fix

## Overview

This design addresses the OpenCL `CL_INVALID_VALUE` error occurring on Intel Arc GPUs when allocating large buffers (>4GB) during tinygrad model inference. The solution involves diagnostic instrumentation to identify problematic allocations and a buffer splitting mechanism to work around the hardware limitation.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     exo Application                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         tinygrad Backend (patched)                     │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  OpenCL Runtime (runtime/ops_opencl.py)          │ │ │
│  │  │  ┌────────────────────────────────────────────┐  │ │ │
│  │  │  │  Diagnostic Logger                         │  │ │ │
│  │  │  │  - Track allocations                       │  │ │ │
│  │  │  │  - Log large buffers                       │  │ │ │
│  │  │  └────────────────────────────────────────────┘  │ │ │
│  │  │  ┌────────────────────────────────────────────┐  │ │ │
│  │  │  │  Buffer Allocator                          │  │ │ │
│  │  │  │  - Check allocation size                   │  │ │ │
│  │  │  │  - Split if > 3.5GB                        │  │ │ │
│  │  │  │  - Manage sub-buffers                      │  │ │ │
│  │  │  └────────────────────────────────────────────┘  │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Intel Arc GPU  │
                  │  (4GB limit)    │
                  └─────────────────┘
```

### Patch Application Flow

```
NixOS Build
    │
    ├─> Fetch tinygrad dependency
    │
    ├─> Apply patch: patches/tinygrad-intel-arc-4gb-fix.patch
    │   │
    │   ├─> Modify runtime/ops_opencl.py
    │   │   ├─> Add diagnostic logging
    │   │   └─> Add buffer size checking
    │   │
    │   └─> Modify device initialization
    │       └─> Detect Intel Arc GPU
    │
    ├─> Build exo with patched tinygrad
    │
    └─> Deploy to gremlin-1
```

## Components and Interfaces

### 1. Diagnostic Logger

**Location**: `tinygrad/runtime/ops_opencl.py` (patched)

**Purpose**: Instrument buffer allocations to identify large allocations

**Interface**:
```python
def _log_allocation(size: int, context: str = "") -> None:
    """Log buffer allocation for diagnostic purposes."""
    size_mb = size / (1024 * 1024)
    size_gb = size / (1024 * 1024 * 1024)
    
    if size_mb > 100:
        print(f"[INTEL ARC DEBUG] Allocating buffer: {size} bytes ({size_mb:.2f} MB)", 
              flush=True)
    
    if size_gb > 4.0:
        print(f"[INTEL ARC DEBUG] WARNING: Buffer >4GB: {size_gb:.2f} GB", 
              flush=True)
        print(f"[INTEL ARC DEBUG] Context: {context}", flush=True)
```

**Key Features**:
- Logs allocations >100MB to identify patterns
- Warns on allocations >4GB
- Uses `flush=True` to ensure logs reach journalctl
- Includes context information when available

### 2. Buffer Size Checker

**Location**: `tinygrad/runtime/ops_opencl.py` (patched)

**Purpose**: Intercept buffer allocations and check size before OpenCL call

**Interface**:
```python
def _check_buffer_size(size: int, device_name: str) -> tuple[bool, str]:
    """
    Check if buffer size is safe for the device.
    
    Returns:
        (is_safe, message)
    """
    MAX_BUFFER_SIZE = 3.5 * 1024 * 1024 * 1024  # 3.5GB
    
    if "Intel" in device_name and "Arc" in device_name:
        if size > MAX_BUFFER_SIZE:
            return False, f"Buffer size {size} exceeds Intel Arc limit"
    
    return True, ""
```

### 3. Buffer Allocator (Phase 2)

**Location**: `tinygrad/runtime/ops_opencl.py` (patched)

**Purpose**: Split large buffers into multiple smaller allocations

**Interface**:
```python
class SplitBuffer:
    """Manages a logical buffer split across multiple physical buffers."""
    
    def __init__(self, total_size: int, chunk_size: int):
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.chunks: list[cl.Buffer] = []
    
    def allocate(self, ctx: cl.Context, flags: int) -> None:
        """Allocate all chunks."""
        num_chunks = (self.total_size + self.chunk_size - 1) // self.chunk_size
        for i in range(num_chunks):
            size = min(self.chunk_size, self.total_size - i * self.chunk_size)
            self.chunks.append(cl.Buffer(ctx, flags, size))
    
    def copy_to_device(self, queue: cl.CommandQueue, data: bytes) -> None:
        """Copy data to split buffers."""
        offset = 0
        for chunk in self.chunks:
            chunk_size = chunk.size
            cl.enqueue_copy(queue, chunk, data[offset:offset + chunk_size])
            offset += chunk_size
    
    def copy_from_device(self, queue: cl.CommandQueue) -> bytes:
        """Copy data from split buffers."""
        result = bytearray()
        for chunk in self.chunks:
            chunk_data = bytearray(chunk.size)
            cl.enqueue_copy(queue, chunk_data, chunk)
            result.extend(chunk_data)
        return bytes(result)
```

### 4. Device Detection

**Location**: `tinygrad/runtime/ops_opencl.py` (patched)

**Purpose**: Detect Intel Arc GPU and enable workarounds

**Interface**:
```python
def _is_intel_arc(device: cl.Device) -> bool:
    """Check if device is Intel Arc GPU."""
    device_name = device.name
    vendor = device.vendor
    
    return ("Intel" in vendor and 
            ("Arc" in device_name or "A770" in device_name or "A750" in device_name))

def _get_max_buffer_size(device: cl.Device) -> int:
    """Get maximum safe buffer size for device."""
    if _is_intel_arc(device):
        return int(3.5 * 1024 * 1024 * 1024)  # 3.5GB for Intel Arc
    else:
        return device.max_mem_alloc_size  # Use device reported limit
```

## Data Models

### Allocation Record

```python
@dataclass
class AllocationRecord:
    """Record of a buffer allocation for diagnostics."""
    size: int
    timestamp: float
    context: str
    device_name: str
    is_split: bool
    num_chunks: int = 1
```

### Device Capabilities

```python
@dataclass
class DeviceCapabilities:
    """GPU device capabilities relevant to memory allocation."""
    name: str
    vendor: str
    max_mem_alloc_size: int
    max_buffer_size_safe: int  # Conservative limit
    requires_split: bool
    is_intel_arc: bool
```

## Error Handling

### Allocation Failure Handling

```python
def allocate_buffer_safe(ctx: cl.Context, size: int, device: cl.Device) -> cl.Buffer:
    """
    Allocate buffer with error handling and fallback.
    
    Raises:
        RuntimeError: If allocation fails even after retry with smaller size
    """
    try:
        # Try direct allocation first
        return cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size)
    except cl.Error as e:
        if e.code == -30:  # CL_INVALID_VALUE
            # Log the failure
            print(f"[INTEL ARC DEBUG] Allocation failed: {size} bytes", flush=True)
            print(f"[INTEL ARC DEBUG] Error: {e}", flush=True)
            
            # If Intel Arc, try split allocation
            if _is_intel_arc(device):
                print(f"[INTEL ARC DEBUG] Attempting split allocation", flush=True)
                return allocate_split_buffer(ctx, size, device)
            else:
                raise RuntimeError(f"Buffer allocation failed: {e}")
        else:
            raise
```

### Diagnostic Error Reporting

```python
def report_allocation_failure(size: int, device: cl.Device, error: Exception) -> None:
    """Generate detailed error report for allocation failure."""
    print(f"[INTEL ARC DEBUG] ===== ALLOCATION FAILURE =====", flush=True)
    print(f"[INTEL ARC DEBUG] Requested size: {size} bytes ({size/(1024**3):.2f} GB)", flush=True)
    print(f"[INTEL ARC DEBUG] Device: {device.name}", flush=True)
    print(f"[INTEL ARC DEBUG] Max alloc size: {device.max_mem_alloc_size}", flush=True)
    print(f"[INTEL ARC DEBUG] Error: {error}", flush=True)
    print(f"[INTEL ARC DEBUG] ==============================", flush=True)
```

## Testing Strategy

### Phase 1: Diagnostic Testing

**Objective**: Confirm diagnostic logging works and identify problematic allocations

**Test Steps**:
1. Deploy patched tinygrad to gremlin-1
2. Trigger model loading (Llama-3.2-3B-Instruct)
3. Monitor logs for `[INTEL ARC DEBUG]` messages
4. Identify allocations >100MB
5. Identify any allocations >4GB
6. Document allocation patterns

**Success Criteria**:
- Diagnostic logs appear in journalctl
- All large allocations are logged with sizes
- Allocation causing CL_INVALID_VALUE is identified

### Phase 2: Fix Testing

**Objective**: Verify buffer splitting resolves the issue

**Test Steps**:
1. Implement buffer splitting for identified large allocations
2. Deploy updated patch to gremlin-1
3. Trigger model loading
4. Verify no CL_INVALID_VALUE errors
5. Verify inference completes successfully
6. Compare output with expected results

**Success Criteria**:
- Model loads without errors
- Inference generates tokens
- Output matches expected results
- Performance is within 90% of baseline

### Phase 3: Integration Testing

**Objective**: Verify fix works across different models and scenarios

**Test Cases**:
1. Llama-3.2-3B-Instruct (primary test case)
2. Llama-3.1-8B-Instruct (larger model)
3. Multiple concurrent inferences
4. Long context sequences (stress test)

**Success Criteria**:
- All test cases pass without CL_INVALID_VALUE
- Performance meets requirements
- Memory usage is reasonable

## Deployment Strategy

### Patch Management

**Patch File**: `patches/tinygrad-intel-arc-4gb-fix.patch`

**Application**: Via NixOS flake.nix

```nix
tinygrad = prev.python3Packages.tinygrad.overrideAttrs (old: {
  patches = (old.patches or []) ++ [
    ./patches/tinygrad-intel-arc-4gb-fix.patch
  ];
});
```

### Verification Steps

1. **Build Verification**:
   ```bash
   nix build .#exo
   ```

2. **Patch Application Check**:
   ```bash
   # Check if patch applied
   nix-store -q --tree $(nix-build -A exo) | grep tinygrad
   ```

3. **Runtime Verification**:
   ```bash
   # Check logs for diagnostic output
   ssh root@10.1.1.12 "journalctl -u exo | grep 'INTEL ARC DEBUG'"
   ```

### Rollback Plan

If the patch causes issues:

1. **Immediate**: Revert to previous commit
   ```bash
   git revert HEAD
   git push
   bash force_update_gremlin1.sh
   ```

2. **Disable Patch**: Comment out patch in flake.nix
   ```nix
   # patches = (old.patches or []) ++ [
   #   ./patches/tinygrad-intel-arc-4gb-fix.patch
   # ];
   ```

3. **Fallback**: Use CPU backend temporarily
   ```bash
   # Set environment variable
   export TINYGRAD_BACKEND=CPU
   ```

## Performance Considerations

### Expected Overhead

**Buffer Splitting Overhead**:
- Memory copy operations: ~5-10% overhead
- Multiple kernel launches: ~2-5% overhead
- Total expected: ~10-15% performance impact

**Mitigation Strategies**:
1. Only split buffers that exceed limit
2. Use asynchronous copies where possible
3. Batch operations on split buffers
4. Cache split buffer metadata

### Memory Usage

**Additional Memory**:
- Split buffer metadata: ~1KB per split buffer
- Temporary copy buffers: None (in-place operations)
- Total overhead: <1MB for typical models

## Security Considerations

### Buffer Overflow Prevention

- Validate buffer sizes before allocation
- Check array bounds when splitting/merging
- Verify chunk sizes sum to total size

### Resource Limits

- Enforce maximum number of split chunks (e.g., 16)
- Limit total memory allocation per model
- Monitor and log excessive allocation attempts

## Future Enhancements

### Potential Improvements

1. **Automatic Tuning**: Dynamically adjust chunk size based on device capabilities
2. **Caching**: Cache split buffer configurations for repeated allocations
3. **Compression**: Compress inactive buffers to reduce memory pressure
4. **Unified Memory**: Explore Intel's unified memory features for Arc GPUs
5. **Upstream Contribution**: Submit patch to tinygrad upstream

### Monitoring and Metrics

1. **Allocation Statistics**: Track allocation sizes and patterns
2. **Performance Metrics**: Monitor inference speed with/without splitting
3. **Error Rates**: Track CL_INVALID_VALUE occurrences
4. **Memory Pressure**: Monitor GPU memory usage over time

## References

- OpenCL Specification 3.0: Buffer allocation limits
- Intel Arc GPU Documentation: Memory architecture
- tinygrad Documentation: Runtime and device management
- exo Architecture: Backend integration

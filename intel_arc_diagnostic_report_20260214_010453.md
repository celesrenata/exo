# Intel Arc GPU Diagnostic Report - Task 3

## Test Information

- **Date**: Sat Feb 14 01:04:53 AM PST 2026
- **Target**: 10.1.1.12
- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Instance**: TinygradRing (28edb645-7a7d-468f-8252-718c90741b69)
- **Log File**: intel_arc_tinygrad_20260214_010334.log

## Diagnostic Logs

### Intel Arc Debug Messages

```
Feb 14 01:03:52 gremlin-1 exo[1117701]: [INTEL ARC DEBUG] ===== DEVICE INFORMATION =====
Feb 14 01:03:52 gremlin-1 exo[1117701]: [INTEL ARC DEBUG] Device: NVIDIA GeForce RTX 4070 Ti SUPER
Feb 14 01:03:52 gremlin-1 exo[1117701]: [INTEL ARC DEBUG] Driver: 580.126.09
Feb 14 01:03:52 gremlin-1 exo[1117701]: [INTEL ARC DEBUG] ==============================
Feb 14 01:03:52 gremlin-1 exo[1117701]: [INTEL ARC DEBUG] Allocation #2: 1576009728 bytes (1503.00 MB / 1.468 GB)
Feb 14 01:03:52 gremlin-1 exo[1117701]: [INTEL ARC DEBUG]   Context: /nix/store/qk6in5b1ixjpx0djis1s1zrr10j0cwry-python3.13-tinygrad-0.11.0/lib/python3.13/site-packages/tinygrad/device.py:226 in alloc
```

### OpenCL Errors

```
Feb 14 01:03:53 gremlin-1 exo[1117701]: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE
Feb 14 01:03:53 gremlin-1 exo[1117701]: Inference failed at token 0: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE
Feb 14 01:03:53 gremlin-1 exo[1116617]:     │    │   │     └ ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_...
Feb 14 01:03:53 gremlin-1 exo[1116617]:           │            │       │            └ ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE')
Feb 14 01:03:53 gremlin-1 exo[1116617]:           │            └ ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE')
Feb 14 01:03:53 gremlin-1 exo[1116617]:           └ ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE')
Feb 14 01:03:53 gremlin-1 exo[1116617]:                      └ 'Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE'
Feb 14 01:03:53 gremlin-1 exo[1116617]: ValueError: Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE
Feb 14 01:03:53 gremlin-1 exo[1116617]:     │    │   │     └ ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_...
Feb 14 01:03:53 gremlin-1 exo[1116617]:           │            │       │            └ ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE')
Feb 14 01:03:53 gremlin-1 exo[1116617]:           │            └ ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE')
Feb 14 01:03:53 gremlin-1 exo[1116617]:           └ ValueError('Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE')
Feb 14 01:03:53 gremlin-1 exo[1116617]:                      └ 'Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE'
Feb 14 01:03:53 gremlin-1 exo[1116617]: ValueError: Error during inference: Inference failed: OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE
```

## Analysis

### Requirements Coverage

- **Requirement 1.1**: Log allocations >100MB - 1 allocations logged
- **Requirement 1.2**: Log buffer sizes - 1 size logs
- **Requirement 1.3**: Warn on >4GB - 0
0 warnings
- **Requirement 2.1**: Capture call stack - 1 context captures
- **Requirement 2.2**: Include tensor shape - 0
0 shape logs
- **Requirement 2.3**: Include data type - 0
0 dtype logs

### Findings

- ✓ No allocations >4GB detected
- Found 1 allocations >100MB

### Next Steps

- [ ] Review allocation patterns
- [ ] Identify problematic allocations
- [ ] Proceed to Task 4: Analyze diagnostic results

## Raw Logs

Full logs available in: `intel_arc_tinygrad_20260214_010334.log`

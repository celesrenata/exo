# Architecture Update: Alignment with exo's Distributed System

## Summary

Updated the PyTorch+IPEX backend specification to align with exo's actual distributed architecture. The original design incorrectly assumed a custom ring topology with manual activation forwarding, which doesn't match how exo actually works.

## Key Changes

### 1. Requirements (requirements.md)

**Changed**: Requirement 5 - Distributed Multi-Node Inference

**Before**: 
- Custom ring topology
- Manual activation forwarding
- Custom node failure detection

**After**:
- Integration with exo's Master/Worker pattern
- Use exo's event sourcing
- Support PyTorchIPEXRingInstance
- Leverage exo's existing coordination

### 2. Design (design.md)

**Removed**: Section 6 - Distributed Coordinator
- No longer need custom DistributedCoordinator class
- No manual ring topology management
- No custom activation forwarding

**Added**: Section 6 - Integration with exo Architecture
- Runner.py integration details
- Master/Worker coordination explanation
- Event sourcing integration
- PyTorchIPEXRingInstance support

**Updated**: Architecture diagrams
- Added event bus and libp2p communication
- Showed Master/Worker pattern
- Clarified runner process role

### 3. Tasks (tasks.md)

**Changed**: Task 7 - From "Distributed Coordinator" to "Runner Integration"

**Before**:
- 7.1: Create DistributedCoordinator class
- 7.2: Implement activation forwarding
- 7.3: Add node failure handling
- 7.4: Implement load balancing

**After**:
- 7.1: Implement model loading in runner.py
- 7.2: Implement generation loop in runner.py
- 7.3: Support distributed coordination via exo
- 7.4: Handle warmup and cleanup

**Updated**: Task 8 - Factory and Testing
- Removed "Update model registry" (not needed)
- Added "Update bootstrap and configuration"
- Clarified factory integration

## Why These Changes?

### Original Design Was Based on Misunderstanding

The original design assumed exo used a simple ring topology where:
1. Each node manually forwards activations to the next node
2. Custom coordinator manages the ring
3. Manual serialization/deserialization of tensors
4. Custom failure detection and recovery

### Actual exo Architecture

exo actually uses a sophisticated distributed system:
1. **Master/Worker Pattern**: Master coordinates, workers execute
2. **Event Sourcing**: Immutable events for all state changes
3. **libp2p**: Peer-to-peer networking with pub/sub
4. **Runner Processes**: Workers spawn runners that load models
5. **Shard Assignment**: Master assigns model shards to workers
6. **No Manual Forwarding**: Coordination happens at Master level

### How Other Backends Work

Looking at Tinygrad and MLX backends:
- They integrate into `runner.py` with backend detection
- They use exo's existing coordination
- They don't implement custom coordinators
- They follow the same pattern for model loading and generation

## Implementation Impact

### What Stays the Same

Tasks 1-6 remain unchanged:
- ✅ Environment setup
- ✅ Device Manager
- ✅ Model Loader
- ✅ KV Cache Manager
- ✅ PyTorchIPEXBackend
- ✅ Token Generator

These components are still needed and correctly designed.

### What Changes

Task 7 becomes runner.py integration:
- Implement model loading in runner.py (following Tinygrad pattern)
- Implement generation loop in runner.py
- Support PyTorchIPEXRingInstance
- Integrate with exo's event system

Task 8 focuses on factory and testing:
- Add backend to factory.py
- Update bootstrap.py
- Test with exo's coordination

## Next Steps

1. **Complete Task 7**: Implement runner.py integration
   - Add model loading logic for pytorch_ipex backend
   - Add generation loop with streaming
   - Emit proper events (BackendInitialized, ChunkGenerated)
   - Handle cleanup

2. **Complete Task 8**: Factory and testing
   - Add PyTorchIPEXRingInstance to factory
   - Test with single node
   - Test with multi-node cluster

3. **Validation**: End-to-end testing
   - Verify integration with exo's Master/Worker
   - Test distributed inference
   - Validate API compatibility

## References

- `src/exo/worker/runner/runner.py`: Main runner implementation
- `src/exo/shared/apply.py`: Event sourcing implementation
- `src/exo/routing/topics.py`: Pub/sub topics
- `src/exo/shared/types/worker/instances.py`: Instance types including PyTorchIPEXRingInstance

## Conclusion

The updated specification now correctly reflects exo's architecture. The PyTorch+IPEX backend will integrate seamlessly with exo's existing distributed coordination, following the same pattern as Tinygrad and MLX backends.

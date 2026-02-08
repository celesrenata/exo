# Implementation Plan

## Overview

This implementation plan follows the exo-cuda reference (https://github.com/Scottcjn/exo-cuda) to restore tinygrad backend integration, then adapts the GPU runtime from CUDA to Intel Arc (Level Zero/OpenCL). The plan is organized into discrete coding tasks that build incrementally.

**Reference**: Study exo-cuda repository for tinygrad integration patterns before implementing each task.

## Tasks

- [x] 1. Study exo-cuda reference implementation
  - Clone and review https://github.com/Scottcjn/exo-cuda
  - Understand runner integration pattern for TinygradRingInstance
  - Document device detection approach
  - Note model loading and generation patterns
  - Identify multi-node communication setup
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement Intel Arc device detection
  - [x] 2.1 Create device_config.py with GPU detection
    - Implement `detect_intel_gpu()` function
    - Check for Level Zero runtime availability
    - Check for OpenCL runtime availability as fallback
    - Return device info (name, memory, runtime)
    - _Requirements: 2.1, 2.2, 2.5_
  
  - [x] 2.2 Add runtime availability checks
    - Implement `check_level_zero_available()` function
    - Implement `check_opencl_available()` function
    - Log which runtime will be used
    - _Requirements: 2.1, 2.3_
  
  - [x] 2.3 Integrate detection with bootstrap
    - Call device detection in bootstrap.py
    - Set TINYGRAD_BACKEND based on detection results
    - Set runtime-specific environment variables
    - Log device selection decision
    - _Requirements: 2.5, 3.4_

- [x] 3. Implement tinygrad model loader (following exo-cuda pattern)
  - [x] 3.1 Create model_loader.py structure
    - Define `load_tinygrad_model()` function signature
    - Implement model file download/caching
    - Add weight loading with tinygrad Tensor
    - _Requirements: 1.3, 5.2_
  
  - [x] 3.2 Add model architecture creation
    - Implement model architecture builders for common model families
    - Support loading specific layers based on shard_metadata
    - Handle pipeline sharding (start_layer, end_layer)
    - _Requirements: 1.3, 5.4_
  
  - [x] 3.3 Add tokenizer loading
    - Load tokenizer from HuggingFace format
    - Ensure compatibility with exo's tokenizer interface
    - _Requirements: 1.3_

- [x] 4. Implement tinygrad text generation (following exo-cuda pattern)
  - [x] 4.1 Create generator.py with generation loop
    - Define `tinygrad_generate()` function
    - Implement forward pass with tinygrad Tensor
    - Add token sampling (temperature, top-p)
    - Yield TokenChunk events
    - _Requirements: 1.4, 5.3_
  
  - [x] 4.2 Add KV cache support
    - Implement KV cache for efficient generation
    - Handle cache updates during generation
    - _Requirements: 1.4_
  
  - [x] 4.3 Add streaming response handling
    - Emit ChunkGenerated events
    - Handle tool calls if model supports them
    - Implement error chunk emission
    - _Requirements: 5.3_

- [x] 5. Integrate tinygrad backend into runner (following exo-cuda pattern)
  - [x] 5.1 Update runner.py main() function
    - Add TinygradRingInstance detection
    - Lazy-load tinygrad modules when detected
    - Import tinygrad backend components
    - _Requirements: 1.1, 1.2, 5.1_
  
  - [x] 5.2 Implement LoadModel task handling
    - Call load_tinygrad_model() on LoadModel task
    - Emit RunnerStatusUpdated events (Loading → Loaded)
    - Handle model loading errors
    - _Requirements: 1.3, 5.2_
  
  - [x] 5.3 Implement TextGeneration task handling
    - Call tinygrad_generate() on TextGeneration task
    - Forward ChunkGenerated events to event_sender
    - Update runner status (Ready → Running → Ready)
    - _Requirements: 1.4, 5.3_
  
  - [x] 5.4 Implement Shutdown task handling
    - Clean up tinygrad resources
    - Release GPU memory
    - Emit RunnerShutdown status
    - _Requirements: 5.5_

- [x] 6. Add NixOS configuration for Intel Arc support
  - [x] 6.1 Create NixOS module for tinygrad backend
    - Add option to enable tinygrad backend
    - Add option to enable Intel Arc GPU support
    - Define package dependencies
    - _Requirements: 6.1, 6.2_
  
  - [x] 6.2 Add Level Zero runtime packages
    - Include level-zero loader and drivers
    - Configure udev rules if needed
    - _Requirements: 6.2_
  
  - [x] 6.3 Add OpenCL runtime packages
    - Include Intel OpenCL runtime
    - Add as fallback when Level Zero unavailable
    - _Requirements: 6.3_
  
  - [x] 6.4 Set environment variables
    - Configure TINYGRAD_BACKEND=GPU
    - Set runtime-specific variables
    - _Requirements: 6.4_

- [ ] 7. Implement multi-node validation (following exo-cuda pattern)
  - [ ] 7.1 Test TinygradRingInstance creation
    - Verify placement creates TinygradRingInstance correctly
    - Check shard assignments across nodes
    - _Requirements: 4.1, 4.2_
  
  - [ ] 7.2 Test ring communication
    - Verify activations flow between tinygrad runners
    - Check that ring protocol matches MLX behavior
    - _Requirements: 4.3_
  
  - [ ] 7.3 Test distributed inference
    - Launch model across multiple nodes
    - Generate text and verify correctness
    - Measure latency and throughput
    - _Requirements: 4.4_

- [ ] 8. Add error handling and logging
  - [ ] 8.1 Add GPU initialization error handling
    - Catch GPU init failures
    - Emit BackendFailed event with clear message
    - Fail task explicitly (no silent fallback)
    - _Requirements: 7.1_
  
  - [ ] 8.2 Add model loading error handling
    - Catch model loading failures
    - Emit RunnerFailed status
    - Log error with context
    - _Requirements: 7.1_
  
  - [ ] 8.3 Add generation error handling
    - Catch runtime errors during generation
    - Emit ErrorChunk to client
    - Log error details
    - _Requirements: 7.1_
  
  - [ ] 8.4 Add structured logging
    - Log device selection decisions
    - Log runtime being used (Level Zero/OpenCL/CPU)
    - Log model loading progress
    - Log generation metrics
    - _Requirements: 9.3, 9.4_

- [ ] 9. Write tests for tinygrad backend
  - [ ]* 9.1 Add device detection unit tests
    - Test Level Zero detection
    - Test OpenCL fallback
    - Test CPU fallback when no GPU
    - _Requirements: 8.1_
  
  - [ ]* 9.2 Add model loading unit tests
    - Test loading small model on CPU
    - Test shard metadata handling
    - Test error cases
    - _Requirements: 8.1_
  
  - [ ]* 9.3 Add generation unit tests
    - Test token generation on CPU
    - Test sampling parameters
    - Test streaming output format
    - _Requirements: 8.1_
  
  - [ ]* 9.4 Add integration test for single-node GPU
    - Launch TinygradRingInstance on Intel Arc node
    - Load model to GPU
    - Generate text
    - Verify GPU was used (not CPU)
    - _Requirements: 8.2_
  
  - [ ]* 9.5 Add integration test for multi-node
    - Launch across 2+ nodes
    - Distribute model shards
    - Generate text
    - Verify correct output
    - _Requirements: 8.4_

- [ ] 10. Update documentation
  - [ ]* 10.1 Document exo-cuda reference usage
    - Explain how exo-cuda was used as reference
    - Document key patterns adapted
    - Note differences between CUDA and Intel Arc
    - _Requirements: 9.2_
  
  - [ ]* 10.2 Write setup guide for Intel Arc
    - Document NixOS configuration steps
    - Explain Level Zero vs OpenCL selection
    - Provide troubleshooting tips
    - _Requirements: 9.1_
  
  - [ ]* 10.3 Document tinygrad backend architecture
    - Explain component structure
    - Document data flow
    - Describe error handling strategy
    - _Requirements: 9.3_
  
  - [ ]* 10.4 Update dashboard documentation
    - Document TinygradRingInstance display
    - Explain backend selection in UI
    - _Requirements: 9.4_

## Implementation Notes

### Key Principles

1. **Follow exo-cuda**: Use exo-cuda as the blueprint for all tinygrad integration patterns
2. **Lazy Loading**: Only import tinygrad modules when TinygradRingInstance is detected
3. **Fail Fast**: Don't silently fall back to CPU; fail with clear error messages
4. **No IPEX**: Tinygrad is the execution engine; do not use IPEX or PyTorch
5. **Runtime Adaptation**: Adapt CUDA runtime calls to Level Zero/OpenCL equivalents

### Testing Strategy

- Unit tests run on CPU (no GPU required)
- Integration tests require Intel Arc GPU hardware
- Multi-node tests require 2+ nodes with tinygrad support
- CI pipeline runs unit tests; integration tests run manually

### Dependencies

- tinygrad (Python package)
- Level Zero runtime (system package)
- Intel OpenCL runtime (system package, fallback)
- HuggingFace transformers (for tokenizers)
- exo-cuda repository (reference only, not a dependency)

## Current Status

### Completed
- ✅ Dashboard support for TinygradRingInstance
- ✅ Placement logic for TinygradRingInstance
- ✅ Runner lazy-loading of tinygrad modules
- ✅ Bootstrap detection of TinygradRingInstance

### In Progress
- ⏳ Tinygrad inference loop implementation

### Not Started
- ❌ Intel Arc device detection
- ❌ Model loading with tinygrad
- ❌ Text generation with tinygrad
- ❌ Multi-node validation
- ❌ NixOS configuration
- ❌ Tests
- ❌ Documentation

## Next Steps

1. Study exo-cuda repository thoroughly
2. Implement Intel Arc device detection (Task 2)
3. Implement model loader following exo-cuda pattern (Task 3)
4. Implement generator following exo-cuda pattern (Task 4)
5. Complete runner integration (Task 5)

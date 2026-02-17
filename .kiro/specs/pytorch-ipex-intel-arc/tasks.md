# Implementation Tasks: Intel Arc GPU Support via PyTorch + IPEX

## Overview

This document outlines the implementation tasks for integrating Intel Arc GPU support into exo using PyTorch and Intel Extension for PyTorch (IPEX). Tasks are organized by priority and dependencies.

## Task List

- [x] 1. Set up development environment and dependencies
  - Set up NixOS development environment with PyTorch and IPEX
  - Verify Intel Arc GPU detection and basic PyTorch operations
  - Create test harness for validating GPU operations
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 1.1 Configure NixOS packages
  - Add PyTorch 2.0+ to flake.nix dependencies
  - Add Intel Extension for PyTorch (IPEX) 2.0+
  - Add intel-compute-runtime and level-zero drivers
  - Add HuggingFace transformers and safetensors libraries
  - _Requirements: 7.1_

- [x] 1.2 Create basic GPU detection script
  - Write Python script to detect Intel Arc via torch.xpu
  - Test XPU device enumeration and properties
  - Verify IPEX import and initialization
  - Log device capabilities (memory, compute units)
  - _Requirements: 1.1_

- [x] 1.3 Validate IPEX functionality
  - Create simple tensor operations on XPU device
  - Test IPEX optimization on dummy model
  - Benchmark basic operations (matmul, softmax)
  - Verify bfloat16 support
  - _Requirements: 8.1_

- [x] 2. Implement Device Manager component
  - Create device detection and selection logic
  - Implement fallback mechanism for device failures
  - Add device memory monitoring
  - Provide device abstraction layer
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2.1 Create DeviceManager class
  - Implement detect_devices() method
  - Implement select_device() with priority logic
  - Add get_device_memory() for memory queries
  - Add is_device_available() for health checks
  - _Requirements: 1.1_

- [x] 2.2 Implement Intel Arc detection
  - Use torch.xpu.is_available() for detection
  - Enumerate XPU devices with torch.xpu.device_count()
  - Query device properties with torch.xpu.get_device_properties()
  - Select device with most free memory
  - _Requirements: 1.2_

- [x] 2.3 Implement fallback logic
  - Check for NVIDIA GPU if Intel Arc unavailable
  - Fall back to CPU if no GPU available
  - Log all fallback decisions at INFO level
  - Provide clear error messages for device issues
  - _Requirements: 1.4, 10.1_

- [x] 2.4 Add device monitoring
  - Track GPU memory usage per device
  - Monitor device temperature and utilization
  - Implement health check for device availability
  - Log device statistics at DEBUG level
  - _Requirements: 9.2_

- [x] 3. Implement Model Loader component
  - Create model loading from HuggingFace
  - Apply IPEX optimizations to loaded models
  - Support model sharding for distributed inference
  - Validate model compatibility
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3.1 Create ModelLoader class
  - Implement load_model() async method
  - Support loading from HuggingFace hub
  - Support loading from local cache
  - Handle safetensors and PyTorch formats
  - _Requirements: 2.1_

- [x] 3.2 Implement IPEX optimization
  - Apply ipex.optimize() to loaded models
  - Configure bfloat16 precision
  - Enable weights prepacking
  - Test optimization impact on performance
  - _Requirements: 2.2, 8.1_

- [x] 3.3 Add model sharding support
  - Extract layer ranges based on Shard metadata
  - Create TransformerShard wrapper class
  - Validate shard boundaries
  - Test with multi-node setup
  - _Requirements: 5.2_

- [x] 3.4 Implement model validation
  - Check model architecture compatibility
  - Verify required model components exist
  - Validate tensor shapes and dtypes
  - Provide detailed error messages for incompatibilities
  - _Requirements: 2.4_

- [x] 4. Implement KV Cache Manager
  - Create cache data structures
  - Implement LRU eviction policy
  - Add memory monitoring and limits
  - Provide cache statistics
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Create KVCacheManager class
  - Define KVCache dataclass structure
  - Implement get_cache() method
  - Implement create_cache() method
  - Implement update_cache() method
  - Implement evict_cache() method
  - _Requirements: 4.1_

- [x] 4.2 Implement LRU eviction
  - Track last access time per cache entry
  - Sort caches by access time
  - Evict oldest when memory threshold exceeded
  - Never evict active request caches
  - _Requirements: 4.2_

- [x] 4.3 Add memory monitoring
  - Track total cache memory usage
  - Set threshold at 80% of available GPU memory
  - Trigger eviction when threshold exceeded
  - Log eviction events at INFO level
  - _Requirements: 4.2, 9.2_

- [x] 4.4 Implement cache statistics
  - Track cache hit/miss rates
  - Monitor cache memory usage
  - Count active caches
  - Provide get_stats() method
  - _Requirements: 4.5, 9.2_

- [x] 5. Implement PyTorchInferenceEngine
  - Create main inference engine class
  - Implement model lifecycle management
  - Add async inference execution
  - Integrate all components
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 5.1 Create PyTorchInferenceEngine class
  - Implement InferenceEngine protocol
  - Add __init__ with shard_downloader
  - Initialize device_manager, model_loader, cache_manager
  - Set up logging
  - _Requirements: 3.1_

- [x] 5.2 Implement ensure_shard() method
  - Check if shard already loaded
  - Download model if needed
  - Load and optimize model
  - Cache model instance
  - _Requirements: 2.1, 2.2_

- [x] 5.3 Implement infer_tensor() method
  - Accept request_id, shard, input_data, inference_state
  - Ensure correct shard is loaded
  - Get or create KV cache for request
  - Execute forward pass with cache
  - Return output and updated state
  - _Requirements: 3.2, 4.1_

- [x] 5.4 Implement sample() method
  - Accept logits, temperature, top_p parameters
  - Apply temperature scaling
  - Implement top-p (nucleus) sampling
  - Return sampled token
  - _Requirements: 3.3_

- [x] 5.5 Add error handling
  - Catch and handle device errors
  - Catch and handle model errors
  - Catch and handle inference errors
  - Provide meaningful error messages
  - _Requirements: 10.1, 10.2, 10.4_

- [x] 6. Implement Token Generator
  - Create sampling logic
  - Support temperature, top-p, top-k
  - Handle special tokens
  - Optimize for performance
  - _Requirements: 3.3_

- [x] 6.1 Create TokenGenerator class
  - Implement sample() method
  - Support temperature parameter
  - Support top_p parameter
  - Support top_k parameter
  - _Requirements: 3.3_

- [x] 6.2 Implement sampling algorithms
  - Apply temperature scaling to logits
  - Implement top-k filtering
  - Implement top-p (nucleus) filtering
  - Sample from filtered distribution
  - _Requirements: 3.3_

- [x] 6.3 Handle special tokens
  - Detect EOS (end of sequence) token
  - Handle PAD tokens appropriately
  - Support custom stop sequences
  - Return token with metadataallow it
  - _Requirements: 6.3_

- [x] 7. Integrate with exo runner and architecture
  - Implement model loading in runner.py for pytorch_ipex backend
  - Implement generation loop in runner.py for pytorch_ipex backend
  - Support PyTorchIPEXRingInstance for distributed inference
  - Test with exo's Master/Worker coordination
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7.1 Implement model loading in runner.py
  - Add model loading logic for pytorch_ipex backend type
  - Use ModelLoader to load and optimize model
  - Initialize DeviceManager for device selection
  - Emit BackendInitialized event with device info
  - _Requirements: 5.1, 2.1, 2.2_

- [x] 7.2 Implement generation loop in runner.py
  - Add text generation logic for pytorch_ipex backend
  - Use PyTorchIPEXBackend for inference
  - Handle streaming token generation
  - Support temperature, top_p, top_k parameters
  - _Requirements: 5.2, 3.1, 3.2, 3.3_

- [x] 7.3 Support distributed coordination
  - Ensure PyTorchIPEXRingInstance is properly handled
  - Integrate with exo's shard assignment system
  - Test multi-node inference with Master/Worker pattern
  - Verify event sourcing integration
  - _Requirements: 5.3, 5.4_

- [x] 7.4 Handle warmup and cleanup
  - Implement warmup logic for pytorch_ipex backend
  - Add proper resource cleanup on shutdown
  - Test model loading timeout handling
  - Verify memory is released properly
  - _Requirements: 5.5, 10.1_

- [x] 8. Add backend to factory and test integration
  - Add PyTorch+IPEX backend to engine factory
  - Update instance type handling
  - Configure NixOS module
  - Test with existing exo components
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 8.1 Update inference engine factory
  - Add PyTorchIPEXRingInstance to factory.py
  - Ensure proper backend selection logic
  - Test instance type detection
  - Verify lazy loading works correctly
  - _Requirements: 6.1_

- [x] 8.2 Update bootstrap and configuration
  - Add PyTorchIPEXRingInstance handling in bootstrap.py
  - Configure environment variables for IPEX
  - Set up device detection at startup
  - Test configuration loading
  - _Requirements: 2.1, 7.1_

- [x] 8.3 Create NixOS module
  - Define module options for PyTorch backend
  - Add systemd service configuration
  - Set required environment variables
  - Provide example configuration
  - _Requirements: 7.4, 7.5_

- [x] 8.4 Test API compatibility
  - Verify OpenAI chat completions format
  - Test streaming responses
  - Test non-streaming responses
  - Validate error response format
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 9. Implement monitoring and logging
  - Add structured logging
  - Expose performance metrics
  - Integrate with systemd journal
  - Create health check endpoints
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 9.1 Configure structured logging
  - Use loguru for structured logs
  - Log in JSON format
  - Set appropriate log levels
  - Include context in all log messages
  - _Requirements: 9.1, 9.4_

- [x] 9.2 Add performance metrics
  - Track inference latency per request
  - Monitor GPU utilization
  - Track memory usage
  - Count requests per second
  - _Requirements: 9.2, 9.3_

- [x] 9.3 Integrate with systemd journal
  - Configure journal logging
  - Add service metadata to logs
  - Test log retrieval with journalctl
  - Verify log rotation
  - _Requirements: 9.5_

- [x] 9.4 Create health check endpoints
  - Add /health endpoint to API
  - Check device availability
  - Check model loading status
  - Return detailed health status
  - _Requirements: 10.5_

- [ ] 10. Build PyTorch+IPEX from source with XPU support in Nix
  - Create Nix derivation for PyTorch with XPU support
  - Create Nix derivation for IPEX with XPU support
  - Configure oneAPI dependencies
  - Verify XPU functionality after build
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

- [x] 10.1 Create PyTorch XPU Nix derivation
  - Fetch PyTorch source from GitHub with submodules
  - Configure CMake with USE_XPU=ON flag
  - Add oneAPI dependencies (dpcpp-compiler, mkl)
  - Add Intel GPU runtime dependencies (compute-runtime, level-zero)
  - Set up proper build environment variables
  - _Requirements: 11.1, 11.3_

- [x] 10.2 Create IPEX XPU Nix derivation
  - Fetch IPEX source from GitHub with submodules
  - Link against PyTorch XPU build
  - Configure CMake with USE_XPU=ON flag
  - Add oneAPI dependencies
  - Ensure proper dependency ordering in Nix
  - _Requirements: 11.2, 11.3_

- [x] 10.3 Configure oneAPI dependencies in Nix
  - Add oneapi-dpcpp-compiler to buildInputs
  - Add oneapi-mkl for optimized math operations
  - Add intel-compute-runtime for GPU runtime
  - Add level-zero for low-level GPU access
  - Set up proper library paths and environment
  - _Requirements: 11.3_

- [ ] 10.4 Create build verification script
  - Test torch.xpu.is_available() after build
  - Verify torch.xpu.device_count() returns devices
  - Test basic tensor operations on XPU
  - Verify IPEX import and optimization
  - Document build success criteria
  - _Requirements: 11.4_

- [ ] 10.5 Update flake.nix with new derivations
  - Add pytorch-xpu override to python packages
  - Add intel-extension-for-pytorch-xpu override
  - Update exo package to use XPU-enabled builds
  - Pin versions in flake.lock for reproducibility
  - Test full flake build
  - _Requirements: 11.5, 11.7_

- [ ] 10.6 Handle build failures and debugging
  - Document common build errors
  - Add troubleshooting steps to documentation
  - Create fallback strategies for build issues
  - Test build on clean NixOS system
  - Verify no impure dependencies
  - _Requirements: 11.6, 11.7_

- [ ] 10.7 Test XPU functionality end-to-end
  - Run detect_intel_arc.py with built packages
  - Test model loading with XPU-enabled PyTorch
  - Verify IPEX optimizations work correctly
  - Benchmark against CPU to confirm GPU acceleration
  - Document any XPU-specific issues
  - _Requirements: 11.4, 11.5_

- [ ] 11. Testing and validation
  - Write unit tests for all components
  - Create integration tests
  - Perform performance benchmarking
  - Validate on target hardware
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 11.1 Write unit tests
  - Test DeviceManager device selection
  - Test ModelLoader model loading
  - Test KVCacheManager cache operations
  - Test TokenGenerator sampling
  - Achieve >80% code coverage
  - _Requirements: 8.1_

- [ ] 11.2 Create integration tests
  - Test end-to-end inference pipeline via runner.py
  - Test multi-node distributed inference with exo Master/Worker
  - Test error handling and recovery
  - Test API compatibility with OpenAI format
  - _Requirements: 8.2_

- [ ] 11.3 Perform benchmarking
  - Measure inference latency
  - Measure throughput (tokens/sec)
  - Profile memory usage
  - Compare with baseline performance
  - _Requirements: 8.3_

- [ ] 11.4 Validate on Intel Arc hardware
  - Test on actual Intel Arc GPU
  - Verify IPEX optimizations work
  - Test with various model sizes
  - Document any hardware-specific issues
  - _Requirements: 1.1, 2.2, 8.1_

- [ ] 12. Documentation and deployment
  - Write user documentation
  - Create deployment guide
  - Document troubleshooting steps
  - Prepare release notes
  - _Requirements: All_

- [ ] 12.1 Write user documentation
  - Document installation steps
  - Explain configuration options
  - Provide usage examples
  - Include API reference
  - _Requirements: 7.4, 7.5_

- [ ] 12.2 Create deployment guide
  - Document NixOS deployment
  - Explain multi-node setup
  - Provide configuration templates
  - Include troubleshooting section
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 12.3 Document known issues
  - List hardware compatibility issues
  - Document performance limitations
  - Explain workarounds
  - Provide links to upstream issues
  - _Requirements: All_

- [ ] 12.4 Prepare release notes
  - Summarize new features
  - List breaking changes
  - Document migration path from tinygrad
  - Include performance benchmarks
  - _Requirements: All_

## Task Dependencies

```mermaid
graph TD
    T1[1. Setup Environment] --> T2[2. Device Manager]
    T1 --> T3[3. Model Loader]
    T2 --> T5[5. Inference Engine]
    T3 --> T5
    T4[4. KV Cache] --> T5
    T6[6. Token Generator] --> T5
    T5 --> T7[7. Runner Integration]
    T7 --> T8[8. Factory & Testing]
    T8 --> T9[9. Monitoring]
    T9 --> T10[10. Build from Source]
    T10 --> T11[11. Testing]
    T11 --> T12[12. Documentation]
```

## Priority Levels

### P0 (Critical - Must Have)
- Tasks 1, 2, 3, 4, 5, 6, 8, 10, 11

### P1 (High - Should Have)
- Tasks 7, 9

### P2 (Medium - Nice to Have)
- Task 12

## Estimated Timeline

- Phase 1 (Setup & Core Components): 2-3 weeks
  - Tasks 1-6

- Phase 2 (Integration & Distribution): 1-2 weeks
  - Tasks 7-8

- Phase 3 (Monitoring & Build): 2-3 weeks
  - Tasks 9-10

- Phase 4 (Testing & Polish): 1-2 weeks
  - Tasks 11-12

Total: 6-10 weeks

## Success Criteria

- [ ] PyTorch and IPEX built from source with XPU support in Nix
- [ ] torch.xpu.is_available() returns True after build
- [ ] Intel Arc GPU successfully detected and selected
- [ ] Llama-3.2-3B model loads and runs on Intel Arc via runner.py
- [ ] Inference achieves >15 tokens/sec on Intel Arc
- [ ] Multi-node distributed inference works with exo's Master/Worker coordination
- [ ] Backend integrates properly with PyTorchIPEXRingInstance
- [ ] API maintains OpenAI compatibility
- [ ] All tests pass with >80% coverage
- [ ] System runs stably on NixOS with no impure dependencies
- [ ] Documentation is complete and accurate

## Notes

- Task 10 is critical: Building PyTorch+IPEX from source with XPU support is the "NixOS way"
- Avoid pip installations outside Nix store - everything must be in Nix derivations
- PyTorch build requires USE_XPU=1 flag and oneAPI dependencies
- IPEX build must link against XPU-enabled PyTorch
- Focus on runner.py integration first - this is where the backend connects to exo
- exo handles distributed coordination via Master/Worker pattern - no custom coordinator needed
- Use small models (1B-3B) for initial testing
- Benchmark against CPU baseline to validate GPU acceleration
- Test fallback mechanisms thoroughly
- Follow the pattern established by Tinygrad backend in runner.py
- Document all Intel Arc-specific quirks and workarounds
- Expect Task 10 to take significant time - building PyTorch from source is complex

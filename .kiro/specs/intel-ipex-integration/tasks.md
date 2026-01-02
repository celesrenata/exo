# Implementation Plan

- [x] 1. Set up IPEX engine infrastructure
  - Create the basic IPEX engine directory structure and core files
  - Implement Intel GPU detection and availability checking
  - Add IPEX to the engine type system and selection logic
  - _Requirements: 1.1, 1.2, 1.4, 3.1, 3.2_

- [x] 1.1 Create IPEX engine directory structure
  - Create `src/exo/worker/engines/ipex/` directory
  - Create `__init__.py` with IPEXModel protocol and TokenizerWrapper
  - Create `utils_ipex.py` for model initialization functions
  - Create `generator/generate.py` for inference implementation
  - _Requirements: 3.1, 3.2_

- [x] 1.2 Implement Intel GPU detection
  - Add Intel GPU detection function to `engine_utils.py`
  - Check for Intel Extension for PyTorch availability
  - Verify Intel GPU driver and XPU device accessibility
  - Add comprehensive error handling for detection failures
  - _Requirements: 1.1, 1.2, 1.4_

- [x] 1.3 Update engine type system
  - Add "ipex" to EngineType literal in `engine_utils.py`
  - Update `detect_available_engines()` to include IPEX detection
  - Modify `select_best_engine()` priority order: MLX > IPEX > CUDA > Torch > CPU
  - Add IPEX compatibility checking for model selection
  - _Requirements: 1.1, 1.4, 3.1_

- [x] 1.4 Integrate IPEX into engine initialization
  - Add IPEX case to `initialize_engine()` in `engine_init.py`
  - Add IPEX case to `warmup_engine()` function
  - Add IPEX case to `generate_with_engine()` function
  - Ensure consistent interface with existing engines
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2. Implement IPEX model loading and optimization
  - Create IPEX model initialization with Intel GPU optimizations
  - Implement tokenizer wrapper for IPEX compatibility
  - Add Intel GPU memory management and device handling
  - _Requirements: 4.1, 4.2, 4.3, 6.1, 6.2, 6.4_

- [x] 2.1 Implement IPEX model initialization
  - Create `initialize_ipex()` function in `utils_ipex.py`
  - Load HuggingFace models with IPEX optimizations
  - Move models to Intel XPU device with proper error handling
  - Apply IPEX optimization passes for performance
  - _Requirements: 4.1, 4.2, 6.1, 6.4_

- [x] 2.2 Create IPEX tokenizer wrapper
  - Implement `IPEXTokenizerWrapper` class matching MLX interface
  - Add encode/decode methods with Intel GPU tensor handling
  - Implement chat template application for IPEX models
  - Ensure compatibility with existing tokenizer patterns
  - _Requirements: 4.1, 4.2, 3.3_

- [x] 2.3 Add Intel GPU memory management
  - Implement Intel GPU memory allocation and cleanup
  - Add memory usage monitoring and reporting
  - Handle GPU memory exhaustion with graceful fallback
  - Optimize memory usage for large model inference
  - _Requirements: 6.4, 8.4, 1.4_

- [x] 2.4 Implement quantization support
  - Add support for 4-bit and 8-bit quantized models
  - Implement mixed precision inference (fp16, bf16)
  - Optimize quantization for Intel GPU architecture
  - Maintain compatibility with existing quantization formats
  - _Requirements: 4.3, 6.3_

- [x] 3. Implement IPEX inference generation
  - Create streaming text generation for IPEX models
  - Implement performance optimizations for Intel GPU
  - Add distributed inference support for multi-GPU setups
  - _Requirements: 4.4, 6.1, 6.2, 6.3, 7.1, 7.3_

- [x] 3.1 Create IPEX text generation
  - Implement `ipex_generate()` function in `generator/generate.py`
  - Add streaming token generation with Intel GPU optimization
  - Implement sampling strategies compatible with IPEX
  - Ensure output format matches existing engines
  - _Requirements: 4.4, 6.2, 3.3_

- [x] 3.2 Add performance optimizations
  - Utilize Intel GPU tensor cores for matrix operations
  - Implement memory-efficient attention mechanisms
  - Add Intel-specific kernel optimizations where available
  - Optimize batch processing for Intel GPU architecture
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 3.3 Implement distributed inference support
  - Add multi-Intel GPU model sharding support
  - Implement cross-device communication for distributed inference
  - Add coordination with other engine types in heterogeneous clusters
  - Ensure compatibility with existing placement algorithms
  - _Requirements: 7.1, 7.3, 7.4_

- [ ]* 3.4 Add warmup and benchmarking
  - Implement `warmup_inference()` function for IPEX
  - Add performance metric collection and reporting
  - Create benchmarking utilities for Intel GPU performance
  - Add comparison metrics with other engines
  - _Requirements: 6.5, 8.4_

- [x] 4. Update dashboard UI for Intel IPEX support
  - Add Intel IPEX as instance type option in dashboard
  - Update UI components to display Intel GPU information
  - Add Intel GPU monitoring and metrics display
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 4.1 Add IPEX instance type to dashboard
  - Add 'IpexRing' to InstanceMeta type in `+page.svelte`
  - Create Intel IPEX button in instance type selection UI
  - Update `matchesSelectedRuntime()` logic for IPEX
  - Add Intel GPU compatibility checking for model selection
  - _Requirements: 2.1, 2.4_

- [x] 4.2 Update model compatibility logic
  - Modify `isModelCompatible()` to handle Intel GPU models
  - Add Intel GPU tag support for model filtering
  - Update model sorting and display logic for IPEX
  - Ensure proper model selection based on Intel GPU capabilities
  - _Requirements: 2.4, 2.5_

- [x] 4.3 Add Intel GPU information display
  - Update topology node display to show Intel GPU info
  - Add Intel GPU memory and utilization metrics
  - Display IPEX engine status in debug information
  - Show Intel GPU temperature and power information where available
  - _Requirements: 2.2, 2.3_

- [x] 4.4 Update instance status and monitoring
  - Add Intel GPU status tracking for running instances
  - Display IPEX-specific performance metrics
  - Update instance type display logic for Intel GPU instances
  - Add Intel GPU error state handling in UI
  - _Requirements: 2.2, 2.3, 8.4_

- [x] 5. Enhance system information gathering for Intel GPU
  - Add Intel GPU detection to system info collection
  - Implement Intel GPU metrics gathering and reporting
  - Update node profile information to include Intel GPU data
  - _Requirements: 1.1, 1.3, 2.2, 8.2, 8.3_

- [x] 5.1 Add Intel GPU detection to system info
  - Create `get_intel_gpu_info()` function in `system_info.py`
  - Detect Intel GPU devices and driver availability
  - Report Intel GPU memory, compute capabilities, and device count
  - Add IPEX version and compatibility information
  - _Requirements: 1.1, 1.3, 8.2_

- [x] 5.2 Update node profile schema
  - Add Intel GPU fields to NodeInfo interface
  - Include `ipex_available`, `intel_gpu_count`, `intel_gpu_memory` fields
  - Update topology data transformation for Intel GPU info
  - Ensure backward compatibility with existing node profiles
  - _Requirements: 2.2, 8.3_

- [x] 5.3 Implement Intel GPU monitoring
  - Add Intel GPU utilization and memory usage monitoring
  - Implement temperature and power monitoring where available
  - Create periodic Intel GPU status updates
  - Add Intel GPU performance metrics collection
  - _Requirements: 2.2, 2.3, 8.4_

- [x] 6. Update NixOS flake for Intel IPEX support
  - Add Intel GPU dependencies to flake packages
  - Create exo-intel package variant with IPEX support
  - Update NixOS module for Intel GPU configuration
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6.1 Add Intel GPU dependencies to flake
  - Add Intel Extension for PyTorch to Python dependencies
  - Include Intel GPU drivers (intel-media-driver, intel-compute-runtime)
  - Add Level Zero runtime and development libraries
  - Ensure proper version compatibility and dependency resolution
  - _Requirements: 5.2, 5.4_

- [x] 6.2 Create exo-intel package variant
  - Add exo-intel to package outputs in flake.nix
  - Configure Intel GPU-specific build optimizations
  - Include all required Intel GPU runtime dependencies
  - Test package build and installation on Intel GPU systems
  - _Requirements: 5.1, 5.2, 5.5_

- [x] 6.3 Update NixOS service module
  - Add Intel GPU detection to NixOS module configuration
  - Update service environment variables for Intel GPU support
  - Configure proper device permissions for Intel GPU access
  - Add Intel GPU-specific service configuration options
  - _Requirements: 5.3, 5.4_

- [x] 6.4 Add hardware-based package selection
  - Implement Intel GPU detection in flake evaluation
  - Add automatic selection of exo-intel when Intel GPU present
  - Update overlay to include Intel GPU package variants
  - Ensure graceful fallback when Intel GPU unavailable
  - _Requirements: 5.1, 5.5_

- [x] 7. Implement comprehensive error handling and logging
  - Add Intel GPU-specific error handling throughout the system
  - Implement comprehensive logging for debugging and monitoring
  - Create fallback mechanisms for Intel GPU failures
  - _Requirements: 1.4, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 7.1 Create IPEX-specific error classes
  - Define IPEXEngineError, IPEXDriverError, IPEXMemoryError classes
  - Add Intel GPU-specific error handling in engine initialization
  - Implement graceful fallback to CPU/Torch on Intel GPU failures
  - Add clear error messages for common Intel GPU issues
  - _Requirements: 1.4, 8.1, 8.2_

- [x] 7.2 Add comprehensive IPEX logging
  - Implement detailed logging for Intel GPU detection and initialization
  - Add performance metrics logging for IPEX inference
  - Log Intel GPU driver and runtime version information
  - Integrate with existing EXO logging infrastructure
  - _Requirements: 8.3, 8.4, 8.5_

- [x] 7.3 Implement Intel GPU health monitoring
  - Add Intel GPU device health checking and reporting
  - Implement automatic recovery from transient Intel GPU errors
  - Add Intel GPU memory leak detection and cleanup
  - Create Intel GPU performance degradation alerts
  - _Requirements: 8.4, 8.5_

- [ ]* 7.4 Add debugging and diagnostic tools
  - Create Intel GPU diagnostic utilities for troubleshooting
  - Add IPEX engine performance profiling tools
  - Implement Intel GPU memory usage analysis
  - Create Intel GPU compatibility checking utilities
  - _Requirements: 8.1, 8.3, 8.5_

- [x] 8. Create comprehensive testing suite
  - Implement unit tests for IPEX engine components
  - Create integration tests for multi-engine scenarios
  - Add performance benchmarking and validation tests
  - _Requirements: All requirements validation_

- [x] 8.1 Implement IPEX engine unit tests
  - Test Intel GPU detection and engine selection logic
  - Test IPEX model loading and initialization
  - Test tokenizer wrapper functionality and compatibility
  - Test error handling and fallback mechanisms
  - _Requirements: 1.1, 1.2, 3.1, 3.2, 4.1_

- [x] 8.2 Create IPEX inference tests
  - Test IPEX text generation and streaming output
  - Validate output consistency with other engines
  - Test quantized model inference with IPEX
  - Test distributed inference across multiple Intel GPUs
  - _Requirements: 4.2, 4.3, 4.4, 7.1, 7.3_

- [x] 8.3 Add dashboard integration tests
  - Test Intel IPEX UI elements and interactions
  - Test Intel GPU information display and updates
  - Test instance creation and management with IPEX
  - Test Intel GPU monitoring and metrics display
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 8.4 Create performance benchmarking tests
  - Benchmark IPEX vs CPU/Torch inference performance
  - Test Intel GPU memory usage and optimization
  - Measure inference throughput and latency with IPEX
  - Compare distributed inference performance across engines
  - _Requirements: 6.1, 6.2, 6.3, 6.5, 7.1_

- [ ]* 8.5 Add NixOS integration tests
  - Test exo-intel package build and installation
  - Test Intel GPU driver integration and detection
  - Test NixOS service configuration with Intel GPU
  - Validate hardware-based package selection logic
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
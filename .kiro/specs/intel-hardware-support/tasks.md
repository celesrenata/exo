# Implementation Plan

- [x] 1. Create backend abstraction layer
  - Create `src/exo/worker/engines/base.py` with `InferenceBackend` protocol defining initialize, load_model, warmup, generate, and cleanup methods
  - Create `src/exo/worker/engines/factory.py` with backend factory function and BackendType literal
  - Add `BackendConfig` model to `src/exo/shared/types/backend_config.py` with backend_type, device, runtime, and fallback options
  - _Requirements: 1.1, 1.4_

- [ ] 2. Implement tinygrad backend foundation
  - [x] 2.1 Create tinygrad backend structure
    - Create directory `src/exo/worker/engines/tinygrad/` with __init__.py
    - Create `backend.py` with TinygradBackend class implementing InferenceBackend protocol
    - Create `device_config.py` with DeviceCapabilities dataclass and detect_capabilities function
    - _Requirements: 1.1, 1.5_
  
  - [x] 2.2 Implement device detection and configuration
    - Implement `_configure_device` method to detect CPU/GPU availability
    - Add device capability detection (memory, compute units)
    - Implement fallback logic from GPU to CPU when GPU unavailable
    - _Requirements: 1.5, 4.2_
  
  - [x] 2.3 Implement model loading for tinygrad
    - Create `model_loader.py` with functions to load HuggingFace weights
    - Implement weight conversion from safetensors to tinygrad format
    - Add model initialization on configured device
    - _Requirements: 1.2_
  
  - [x] 2.4 Implement text generation
    - Create `generator.py` with text generation logic using tinygrad
    - Implement token-by-token generation yielding GenerationResponse objects
    - Add proper error handling and logging
    - _Requirements: 1.2, 1.3_
  
  - [ ]* 2.5 Write unit tests for tinygrad backend
    - Create `src/exo/worker/engines/tinygrad/tests/test_backend.py`
    - Test backend initialization, device detection, and fallback behavior
    - Test model loading and generation on CPU
    - _Requirements: 7.1_

- [x] 3. Integrate tinygrad backend with runner
  - [x] 3.1 Refactor runner.py to use backend abstraction
    - Modify `src/exo/worker/runner/runner.py` to detect backend type from configuration
    - Replace direct MLX calls with backend interface calls
    - Maintain existing MLX code path for backward compatibility
    - _Requirements: 1.1, 4.1_
  
  - [x] 3.2 Add backend selection logic
    - Implement backend detection based on shard metadata or environment variables
    - Add feature flags (EXO_TINYGRAD_ENABLED) to control backend selection
    - Implement fallback chain (tinygrad → MLX) with proper error handling
    - _Requirements: 4.1, 4.2_
  
  - [x] 3.3 Add backend status events
    - Add BackendInitialized and BackendFailed events to `src/exo/shared/types/events.py`
    - Emit events when backend initializes or fails
    - Include device information in events for observability
    - _Requirements: 8.3_



- [x] 4. Add Intel Arc iGPU support
  - [x] 4.1 Implement runtime detection
    - Create `src/exo/worker/engines/tinygrad/intel_arc.py` with detect_intel_arc function
    - Implement check_level_zero_available to verify Level Zero runtime
    - Implement check_opencl_available to verify OpenCL runtime
    - Add select_runtime function to choose best available runtime
    - _Requirements: 2.1, 2.2, 2.4_
  
  - [x] 4.2 Configure tinygrad for Intel Arc
    - Implement GPU configuration with Level Zero as primary runtime
    - Add OpenCL fallback configuration when Level Zero unavailable
    - Set appropriate environment variables for tinygrad GPU backends
    - Add logging to indicate which runtime is being used
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 4.3 Add GPU performance monitoring
    - Create `src/exo/worker/engines/tinygrad/metrics.py` with GPUMetrics dataclass
    - Implement collect_gpu_metrics function to query device utilization
    - Add memory usage tracking during inference
    - Emit metrics as events for dashboard visibility
    - _Requirements: 2.5, 8.3_
  
  - [ ]* 4.4 Write Intel Arc integration tests
    - Create `src/exo/worker/engines/tinygrad/tests/test_intel_arc.py`
    - Test Intel Arc detection on hardware with iGPU
    - Test Level Zero and OpenCL runtime selection
    - Verify GPU inference performance vs CPU baseline
    - _Requirements: 7.2_

- [x] 5. Create NixOS configuration module
  - [x] 5.1 Add Intel hardware support to flake
    - Create nixosModules.exo-intel in `flake.nix` with enable option
    - Add arc.enable and arc.runtime options for Intel Arc configuration
    - Add npu.enable and npu.servicePort options for NPU support
    - _Requirements: 3.1, 3.2_
  
  - [x] 5.2 Configure hardware packages
    - Add intel-compute-runtime package for OpenCL support
    - Add level-zero package for Level Zero runtime
    - Configure hardware.graphics.extraPackages with Intel drivers
    - _Requirements: 3.2, 3.3_
  
  - [x] 5.3 Add tinygrad to Python environment
    - Update `python/parts.nix` to include tinygrad package
    - Add pyopencl as dependency for OpenCL support
    - Configure build flags for Intel backend support
    - _Requirements: 3.2_
  
  - [x] 5.4 Test NixOS configuration on hardware
    - Deploy configuration to gremlin-1 node with Intel Core Ultra 9 185H
    - Verify Level Zero and OpenCL runtimes are available
    - Test exo startup with Intel Arc enabled
    - _Requirements: 3.4_

- [x] 6. Add observability and documentation
  - [x] 6.1 Implement structured logging
    - Add structured log messages for backend initialization with device details
    - Log runtime selection decisions (Level Zero vs OpenCL)
    - Log fallback events when GPU unavailable
    - _Requirements: 8.3_
  
  - [x] 6.2 Add metrics collection
    - Create BackendMetricsCollector class in `src/exo/worker/engines/metrics.py`
    - Record inference metrics (tokens/sec, memory usage, GPU utilization)
    - Emit InferenceMetrics events to cluster state
    - _Requirements: 8.3, 8.5_
  
  - [x] 6.3 Update dashboard for hardware visibility
    - Add BackendInfo and NodeHardware types to `dashboard/src/lib/types/hardware.ts`
    - Create NodeHardwareStatus.svelte component to display backend info
    - Show active backend, device type, runtime, and utilization
    - _Requirements: 8.5_
  
  - [x] 6.4 Write setup documentation
    - Document tinygrad backend installation and configuration
    - Add Intel Arc setup guide with Level Zero and OpenCL instructions
    - Document NixOS module usage with example configurations
    - Add troubleshooting section for common issues
    - _Requirements: 8.1, 8.2_



- [x] 7. Intel NPU discovery and assessment (Optional - Phase 3)
  - [x] 7.1 Implement NPU hardware discovery
    - Create `src/exo/worker/engines/npu/discovery.py` with NPUCapabilities dataclass
    - Implement discover_npu function to check for /dev/accel/accel0
    - Check for intel_vpu and ivpu kernel modules
    - Determine OpenVINO availability on NixOS
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 7.2 Create NPU capability report
    - Document which device nodes are present on Core Ultra 9 185H
    - List loaded kernel modules and driver versions
    - Identify supported model types (vision, audio, embeddings)
    - Test OpenVINO NPU execution with simple model
    - _Requirements: 5.4, 5.5_
  
  - [x] 7.3 Implement NPU smoke test
    - Create standalone script to test NPU via OpenVINO
    - Load a known-supported model (e.g., MobileNet for vision)
    - Execute inference and verify NPU is actually used (not CPU fallback)
    - Measure latency and compare to CPU baseline
    - _Requirements: 5.4, 5.5_

- [-] 8. Implement NPU sidecar service (Optional - Phase 3)
  - [x] 8.1 Create NPU inference service
    - Create `src/exo/worker/engines/npu/service.py` with NPUInferenceService class
    - Initialize OpenVINO core and configure for NPU device
    - Implement model loading and caching
    - Add inference execution method
    - _Requirements: 6.1, 6.2_
  
  - [x] 8.2 Add service communication protocol
    - Define gRPC or REST API for NPU service
    - Implement request/response handling
    - Add error handling and timeout logic
    - _Requirements: 6.1_
  
  - [x] 8.3 Implement workload routing
    - Create should_use_npu function to determine NPU-suitable tasks
    - Route embedding and vision tasks to NPU service
    - Keep LLM decode on GPU/CPU
    - Add fallback when NPU unavailable
    - _Requirements: 6.1, 6.4_
  
  - [x] 8.4 Add NPU service to NixOS configuration
    - Create systemd service definition for exo-npu service
    - Configure service isolation (PrivateNetwork, ProtectSystem)
    - Add resource limits (MemoryMax, CPUQuota)
    - Add kernel module loading for intel_vpu
    - _Requirements: 3.1, 6.4_
  
  - [ ] 8.5 Test NPU integration
    - Deploy NPU service on Core Ultra hardware
    - Test embedding generation via NPU
    - Measure performance vs CPU/GPU
    - Verify service stability over 24 hours
    - _Requirements: 6.5_
  
  - [x] 8.6 Document NPU capabilities and limitations
    - Document which workloads benefit from NPU
    - List unsupported operations and models
    - Add performance comparison data
    - Mark feature as experimental
    - _Requirements: 8.4_

- [ ] 9. Single-node validation on gremlin-1
  - [-] 9.1 Build exo with Intel hardware support
    - Build flake successfully on gremlin-1
    - Verify all dependencies resolve correctly
    - Ensure tinygrad backend compiles
    - _Requirements: 1.1, 2.1_
  
  - [x] 9.2 Start exo service
    - Start exo with tinygrad backend enabled
    - Verify service starts without errors
    - Check logs for proper initialization
    - _Requirements: 2.1, 3.1_
  
  - [x] 9.3 Verify web service endpoint
    - Confirm API is accessible at http://10.1.1.12:52415
    - Test health endpoint responds correctly
    - Verify OpenAI-compatible API is available
    - _Requirements: 7.3_
  
  - [x] 9.4 Validate Intel GPU detection
    - Verify Intel Arc iGPU is detected
    - Confirm Level Zero runtime is available
    - Check OpenCL fallback if Level Zero fails
    - Validate device appears in metrics
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 9.5 Validate Intel NPU detection
    - Verify Intel NPU hardware is detected
    - Confirm kernel modules are loaded (intel_vpu)
    - Check OpenVINO can access NPU device
    - Validate NPU appears in capability report
    - _Requirements: 6.1, 6.2_
  
  - [x] 9.6 Download and load tiny model
    - Download a small test model (e.g., TinyLlama-1.1B)
    - Verify model downloads successfully
    - Load model into tinygrad backend
    - Confirm model is ready for inference
    - _Requirements: 2.2, 4.1_
  
  - [x] 9.7 Run inference on tiny model
    - Execute test inference request
    - Verify tokens are generated correctly
    - Measure basic performance (tokens/sec)
    - Confirm GPU is being used (not CPU fallback)
    - _Requirements: 2.3, 4.2, 4.3_

- [ ] 10. Git environment and multi-node deployment
  - [ ] 10.1 Prepare git repository for deployment
    - Ensure all changes are committed to git
    - Verify flake.nix is complete and tested
    - Tag release version for deployment
    - Push to remote repository
    - _Requirements: 1.1_
  
  - [ ] 10.2 Update gremlin-1 to use git flake
    - Remove local drive mapping from gremlin-1 config
    - Update to import flake from git repository
    - Rebuild and verify functionality unchanged
    - Document the git flake URL for other nodes
    - _Requirements: 1.1, 3.1_
  
  - [ ] 10.3 Deploy to gremlin-2, 3, 4
    - Update configurations for gremlin-2, 3, 4
    - Import flake from git repository
    - Rebuild all nodes
    - Verify all nodes start successfully
    - _Requirements: 1.1, 3.1_
  
  - [ ] 10.4 Test cluster formation
    - Start exo on all gremlin nodes
    - Verify nodes discover each other
    - Confirm cluster forms correctly
    - Check node status in dashboard
    - _Requirements: 7.3, 8.1_
  
  - [ ] 10.5 Validate cluster stability
    - Run cluster for extended period (1+ hours)
    - Monitor for crashes or disconnections
    - Verify model sharding works across nodes
    - Test inference requests across cluster
    - Confirm stable behavior under load
    - _Requirements: 7.5, 8.1_


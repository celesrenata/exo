# Requirements Document

## Introduction

This document specifies requirements for enabling Intel hardware acceleration in the exo distributed AI inference system using the tinygrad backend. The implementation will use https://github.com/Scottcjn/exo-cuda as the reference for restoring tinygrad backend wiring (backend selection, device capability detection, multi-node validation), then adapt the GPU runtime path from CUDA to Intel GPU (OpenCL or Level Zero) for Intel Arc on Core Ultra 185H processors.

## Glossary

- **exo**: The distributed AI inference system that connects multiple devices into a cluster
- **exo-cuda**: Reference implementation showing tinygrad backend integration with CUDA GPUs
- **tinygrad**: A lightweight deep learning framework that can target multiple hardware backends
- **Intel Arc iGPU**: Intel's integrated graphics processor with AI acceleration capabilities (Xe-LPG architecture)
- **Core Ultra 185H**: Intel's mobile processor with integrated Arc graphics and NPU
- **Level Zero**: Intel's low-level API for GPU programming and compute acceleration
- **OpenCL**: Open Computing Language, a cross-platform API for heterogeneous computing
- **Backend**: An execution engine in exo that handles model inference on specific hardware
- **Worker**: The exo component that handles inference tasks and manages runner processes
- **Runner**: A process that executes model inference using a specific backend
- **TinygradRingInstance**: Instance type for tinygrad-based distributed inference
- **NixOS**: The Linux distribution used for declarative system configuration

## Requirements

### Requirement 1: Tinygrad Backend Wiring (from exo-cuda reference)

**User Story:** As an exo developer, I want to restore tinygrad backend integration using the exo-cuda reference implementation so that exo can leverage tinygrad's multi-platform capabilities.

#### Acceptance Criteria

1. WHEN the tinygrad backend is initialized, THE exo Worker SHALL create a tinygrad-based runner process following the pattern from exo-cuda
2. WHEN a TinygradRingInstance is created, THE Runner SHALL import and initialize tinygrad modules only when needed (lazy loading)
3. WHEN the tinygrad runner receives a LoadModel task, THE Runner SHALL load model weights using tinygrad operations
4. WHEN the tinygrad runner receives a TextGeneration task, THE Runner SHALL execute forward passes and generate tokens using tinygrad
5. THE tinygrad Backend SHALL support CPU execution as the baseline target for all model types

### Requirement 2: Device Capability Detection (adapted from exo-cuda)

**User Story:** As an exo cluster operator, I want automatic detection of Intel Arc GPU capabilities so that the system can determine if GPU acceleration is available.

#### Acceptance Criteria

1. WHEN exo starts on a node with Intel Arc iGPU, THE Device Detection SHALL identify the GPU using tinygrad's device enumeration
2. THE Device Detection SHALL report GPU memory capacity, compute units, and driver version to the cluster state
3. WHEN Level Zero runtime is available, THE Detection SHALL verify Level Zero device accessibility
4. IF Level Zero is unavailable, THEN THE Detection SHALL check for OpenCL runtime availability
5. THE Device Detection SHALL log which runtime (Level Zero, OpenCL, or CPU fallback) will be used

### Requirement 3: Intel Arc GPU Runtime Configuration

**User Story:** As an exo user with Intel Arc graphics, I want tinygrad to use the integrated GPU for inference so that I can achieve better performance than CPU-only execution.

#### Acceptance Criteria

1. WHEN TINYGRAD_BACKEND environment variable is set to "GPU", THE tinygrad Backend SHALL attempt GPU execution
2. THE tinygrad Backend SHALL configure Level Zero as the primary GPU runtime when available
3. IF Level Zero fails, THEN THE tinygrad Backend SHALL fall back to OpenCL for GPU execution
4. WHEN GPU execution is active, THE Runner SHALL log which specific device and runtime is being used
5. WHILE executing on Intel Arc iGPU, THE Backend SHALL monitor for GPU errors and report failures clearly

### Requirement 4: Multi-Node Validation (from exo-cuda reference)

**User Story:** As an exo cluster operator, I want to validate that tinygrad backend works correctly in multi-node distributed inference scenarios.

#### Acceptance Criteria

1. WHEN multiple nodes with TinygradRingInstance are connected, THE Cluster SHALL establish ring communication following the exo-cuda pattern
2. THE Multi-Node Setup SHALL verify that model shards can be distributed across tinygrad runners
3. WHEN distributed inference executes, THE System SHALL validate that activations flow correctly between nodes
4. THE Validation Tests SHALL confirm that tinygrad ring communication matches MLX ring behavior
5. WHEN a node fails, THE Cluster SHALL detect the failure and handle it gracefully without data corruption

### Requirement 5: Runner Integration (adapted from exo-cuda)

**User Story:** As an exo developer, I want the tinygrad runner to integrate cleanly with exo's task system so that it behaves consistently with other backends.

#### Acceptance Criteria

1. WHEN a TinygradRingInstance is detected in runner bootstrap, THE Runner SHALL import tinygrad modules and initialize the backend
2. THE tinygrad Runner SHALL implement the same task handling loop as MLX runner (LoadModel, TextGeneration, Shutdown, etc.)
3. WHEN generating text, THE tinygrad Runner SHALL emit TokenChunk events compatible with exo's streaming API
4. THE tinygrad Runner SHALL support the same shard metadata types (PipelineShardMetadata, TensorShardMetadata) as MLX
5. WHEN the runner shuts down, THE Cleanup SHALL properly release tinygrad resources and GPU memory

### Requirement 6: NixOS Configuration Management

**User Story:** As a system administrator, I want declarative NixOS configuration for Intel hardware support so that I can easily enable tinygrad backend with Intel Arc acceleration.

#### Acceptance Criteria

1. THE NixOS Configuration SHALL provide options for enabling tinygrad backend with Intel Arc GPU support
2. WHEN Intel Arc support is enabled, THE NixOS Configuration SHALL ensure Level Zero drivers and runtime libraries are installed
3. WHERE OpenCL fallback is needed, THE NixOS Configuration SHALL include Intel OpenCL runtime packages
4. THE NixOS Configuration SHALL set appropriate environment variables (TINYGRAD_BACKEND=GPU) for GPU execution
5. WHEN configuration changes are applied, THE NixOS System SHALL rebuild without breaking existing exo functionality

### Requirement 7: Backend Selection and Fallback

**User Story:** As an exo user, I want automatic backend selection with graceful fallbacks so that inference works reliably even when preferred hardware is unavailable.

#### Acceptance Criteria

1. WHEN exo starts, THE Placement Logic SHALL support creating TinygradRingInstance for nodes with tinygrad capability
2. IF tinygrad GPU initialization fails, THEN THE Runner SHALL log the error clearly and fail the task (no silent CPU fallback)
3. THE Dashboard SHALL display "Tinygrad Ring" as an instance type option when tinygrad is available
4. WHEN a TinygradRingInstance is launched, THE System SHALL validate that tinygrad dependencies are available before starting
5. THE Backend Selection SHALL NOT use IPEX or PyTorch; tinygrad is the primary execution engine

### Requirement 8: Testing and Validation

**User Story:** As an exo contributor, I want comprehensive tests for tinygrad backend so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE tinygrad Backend SHALL include unit tests that verify model loading and inference execution on CPU
2. WHERE Intel Arc iGPU is available, THE Integration Tests SHALL verify GPU execution with Level Zero or OpenCL
3. THE Test Suite SHALL validate that TinygradRingInstance creation and placement work correctly
4. THE Multi-Node Tests SHALL verify distributed inference across multiple tinygrad runners
5. THE CI Pipeline SHALL run tinygrad backend tests and report failures before allowing code merge

### Requirement 9: Documentation and Observability

**User Story:** As an exo user, I want clear documentation and runtime visibility into tinygrad backend behavior so that I can understand and troubleshoot issues.

#### Acceptance Criteria

1. THE exo Documentation SHALL include setup instructions for enabling tinygrad backend on NixOS systems with Intel Arc
2. THE Documentation SHALL reference exo-cuda as the implementation guide for tinygrad integration patterns
3. THE exo Logging System SHALL emit structured logs indicating tinygrad backend status and GPU device selection
4. THE Dashboard SHALL display "Tinygrad Ring" instances with their hardware configuration (CPU/GPU, device name)
5. THE Documentation SHALL explicitly state that IPEX is NOT used; tinygrad is the execution engine

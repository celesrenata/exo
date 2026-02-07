# Requirements Document

## Introduction

This document specifies requirements for enabling Intel hardware acceleration in the exo distributed AI inference system. The feature encompasses three phases: restoring tinygrad backend support for generic CPU/GPU execution, enabling Intel Arc integrated GPU acceleration, and optionally exploring Intel NPU (Neural Processing Unit) capabilities on Core Ultra processors.

## Glossary

- **exo**: The distributed AI inference system that connects multiple devices into a cluster
- **tinygrad**: A lightweight deep learning framework that can target multiple hardware backends
- **Intel Arc iGPU**: Intel's integrated graphics processor with AI acceleration capabilities
- **Intel NPU**: Neural Processing Unit, a dedicated AI accelerator in Intel Core Ultra processors (also known as iVPU)
- **Level Zero**: Intel's low-level API for GPU programming and compute acceleration
- **OpenCL**: Open Computing Language, a cross-platform API for heterogeneous computing
- **OpenVINO**: Intel's toolkit for optimizing and deploying AI inference on Intel hardware
- **Backend**: An execution engine in exo that handles model inference on specific hardware
- **Worker**: The exo component that handles inference tasks and manages runner processes
- **Runner**: A process that executes model inference using a specific backend
- **NixOS**: The Linux distribution used for declarative system configuration

## Requirements

### Requirement 1: Tinygrad Backend Foundation

**User Story:** As an exo developer, I want a properly integrated tinygrad backend so that exo can leverage tinygrad's multi-platform capabilities for inference workloads.

#### Acceptance Criteria

1. WHEN the tinygrad backend is initialized, THE exo Worker SHALL create a tinygrad-based runner process that can execute inference requests
2. WHEN a model inference request is received, THE tinygrad Backend SHALL load the model weights and execute forward passes using tinygrad operations
3. WHEN the tinygrad backend encounters an error, THE exo Worker SHALL log the error with sufficient context and gracefully handle the failure without crashing the node
4. WHERE tinygrad backend is selected, THE exo Configuration SHALL validate that required tinygrad dependencies are available before starting inference
5. THE tinygrad Backend SHALL support CPU execution as the baseline target for all model types

### Requirement 2: Intel Arc iGPU Acceleration

**User Story:** As an exo cluster operator with Intel Arc graphics, I want to utilize the integrated GPU for inference so that I can achieve better performance than CPU-only execution.

#### Acceptance Criteria

1. WHEN Level Zero runtime is available, THE tinygrad Backend SHALL configure tinygrad to use Level Zero as the primary GPU execution target
2. IF Level Zero is unavailable, THEN THE tinygrad Backend SHALL fall back to OpenCL for GPU execution
3. WHEN GPU execution is active, THE exo System SHALL log which runtime (Level Zero or OpenCL) is being used for transparency
4. THE tinygrad Backend SHALL detect Intel Arc iGPU availability during initialization and report the device capabilities to the cluster state
5. WHILE executing on Intel Arc iGPU, THE tinygrad Backend SHALL monitor for GPU errors and fall back to CPU execution if GPU becomes unavailable

### Requirement 3: NixOS Configuration Management

**User Story:** As a system administrator, I want declarative NixOS configuration for Intel hardware support so that I can easily enable or disable hardware acceleration features across my cluster.

#### Acceptance Criteria

1. THE NixOS Configuration SHALL provide toggleable options for enabling tinygrad backend, Intel Arc support, and Intel NPU support
2. WHEN Intel Arc support is enabled, THE NixOS Configuration SHALL ensure Level Zero drivers and runtime libraries are installed
3. WHERE OpenCL fallback is needed, THE NixOS Configuration SHALL include Intel OpenCL runtime packages
4. THE NixOS Configuration SHALL follow the structure from the reference repository (celesrenata/nixos-k3s-configs) for consistency
5. WHEN configuration changes are applied, THE NixOS System SHALL rebuild without breaking existing exo functionality

### Requirement 4: Backend Selection and Fallback

**User Story:** As an exo user, I want automatic backend selection with graceful fallbacks so that inference works reliably even when preferred hardware is unavailable.

#### Acceptance Criteria

1. WHEN exo starts, THE Worker SHALL detect available hardware accelerators and select the most capable backend
2. IF the selected backend fails to initialize, THEN THE Worker SHALL attempt fallback backends in order of preference (GPU → CPU)
3. THE exo API SHALL expose backend status information so that clients can query which hardware is being used
4. WHEN a backend becomes unavailable during operation, THE Worker SHALL log the failure and attempt to reinitialize or fall back to an alternative backend
5. THE Backend Selection Logic SHALL prioritize Intel Arc iGPU over CPU when both are available and functional

### Requirement 5: Intel NPU Discovery and Capability Assessment

**User Story:** As an exo developer, I want to understand Intel NPU capabilities and limitations so that I can determine if and how to integrate NPU acceleration.

#### Acceptance Criteria

1. THE NPU Discovery Tool SHALL check for NPU device nodes (/dev/accel/accel0 or similar) and report their presence
2. WHEN NPU hardware is detected, THE Discovery Tool SHALL identify loaded kernel modules (intel_vpu, ivpu) and log driver information
3. THE Discovery Tool SHALL determine which software stack (OpenVINO or other) is available for NPU execution on NixOS
4. THE Capability Report SHALL document which model classes (vision, audio, embeddings, transformers) can realistically execute on the NPU
5. THE Discovery Tool SHALL verify that NPU execution does not silently fall back to CPU without logging

### Requirement 6: NPU Integration (Optional)

**User Story:** As an exo cluster operator with Intel Core Ultra processors, I want to optionally utilize the NPU for suitable workloads so that I can offload specific inference tasks and improve power efficiency.

#### Acceptance Criteria

1. WHERE NPU support is enabled, THE exo Worker SHALL provide a mechanism to route suitable tasks (embeddings, vision, audio) to NPU execution
2. WHEN NPU execution is attempted, THE NPU Backend SHALL log which device is actually used (NPU vs CPU fallback) for transparency
3. IF a model or operation is unsupported on NPU, THEN THE NPU Backend SHALL provide clear error messages and fall back to CPU or GPU
4. THE NPU Integration SHALL be modular and optional, ensuring that disabling NPU support does not affect CPU or GPU execution paths
5. WHILE NPU is executing inference, THE exo System SHALL measure and report latency metrics for performance comparison

### Requirement 7: Testing and Validation

**User Story:** As an exo contributor, I want comprehensive tests for Intel hardware backends so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE tinygrad Backend SHALL include unit tests that verify model loading, inference execution, and error handling on CPU
2. WHERE Intel Arc iGPU is available, THE Integration Tests SHALL verify GPU execution and fallback behavior
3. THE Test Suite SHALL validate that backend selection logic correctly prioritizes and falls back between available hardware
4. WHEN NPU support is implemented, THE NPU Tests SHALL verify device targeting and measure inference correctness against CPU baseline
5. THE CI Pipeline SHALL run all backend tests and report failures before allowing code merge

### Requirement 8: Documentation and Observability

**User Story:** As an exo user, I want clear documentation and runtime visibility into hardware acceleration so that I can understand and troubleshoot backend behavior.

#### Acceptance Criteria

1. THE exo Documentation SHALL include setup instructions for enabling tinygrad backend on NixOS systems
2. WHEN Intel Arc support is configured, THE Documentation SHALL provide troubleshooting steps for Level Zero and OpenCL issues
3. THE exo Logging System SHALL emit structured logs indicating which backend and hardware device is active for each inference request
4. WHERE NPU support is available, THE Documentation SHALL clearly mark it as experimental and list supported workload types
5. THE exo Dashboard SHALL display active backend information and hardware utilization metrics for each node in the cluster

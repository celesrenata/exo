# Requirements Document: Intel Arc GPU Support via PyTorch + IPEX

## Introduction

This document specifies the requirements for integrating Intel Arc GPU support into the exo distributed AI inference system using PyTorch and Intel Extension for PyTorch (IPEX). This approach replaces the previous tinygrad-based attempt which encountered insurmountable device selection issues.

## Glossary

- **IPEX**: Intel Extension for PyTorch - Intel's optimization library for PyTorch on Intel hardware
- **XPU**: Intel's cross-architecture GPU abstraction (covers Arc, Flex, Max GPUs)
- **exo**: Distributed AI inference system supporting multiple backends
- **Ring Topology**: Distributed computing pattern where nodes form a logical ring
- **KV Cache**: Key-Value cache for transformer model inference optimization
- **HuggingFace**: Model repository and library ecosystem
- **NixOS**: Immutable Linux distribution with declarative configuration

## Requirements

### Requirement 1: Device Detection and Selection

**User Story:** As a system administrator, I want the system to automatically detect and utilize Intel Arc GPUs so that I can leverage available hardware without manual configuration.

#### Acceptance Criteria

1. WHEN the system starts, THE System SHALL enumerate all available compute devices
2. WHEN Intel Arc GPU is detected, THE System SHALL prioritize it for inference workloads
3. WHEN multiple Intel Arc GPUs are present, THE System SHALL select the device with most available memory
4. IF Intel Arc GPU is not available, THEN THE System SHALL fall back to NVIDIA GPU or CPU
5. THE System SHALL log device selection decisions at INFO level

### Requirement 2: Model Loading and Optimization

**User Story:** As an AI researcher, I want to load HuggingFace Llama models on Intel Arc GPUs so that I can run inference with Intel-optimized performance.

#### Acceptance Criteria

1. THE System SHALL load HuggingFace Llama models in safetensors or PyTorch format
2. WHEN loading a model, THE System SHALL apply IPEX optimizations automatically
3. THE System SHALL support models from 1B to 70B parameters
4. THE System SHALL validate model compatibility before loading
5. IF model loading fails, THEN THE System SHALL provide detailed error messages

### Requirement 3: Asynchronous Inference Execution

**User Story:** As a developer, I want inference requests to execute asynchronously so that the system can handle multiple concurrent requests efficiently.

#### Acceptance Criteria

1. THE System SHALL execute inference requests asynchronously using Python asyncio
2. THE System SHALL maintain separate execution contexts for concurrent requests
3. THE System SHALL limit concurrent inference tasks based on available GPU memory
4. WHEN inference completes, THE System SHALL return results via async callback
5. THE System SHALL handle inference errors without blocking other requests

### Requirement 4: KV Cache Management

**User Story:** As a system operator, I want efficient memory management for transformer inference so that the system can handle longer conversations without running out of memory.

#### Acceptance Criteria

1. THE System SHALL maintain per-request KV cache for transformer layers
2. THE System SHALL automatically evict least-recently-used cache entries when memory is constrained
3. THE System SHALL support cache sizes up to 8192 tokens per request
4. WHEN a request completes, THE System SHALL release associated cache memory
5. THE System SHALL log cache statistics at DEBUG level

### Requirement 5: Integration with exo Distributed Architecture

**User Story:** As a cluster administrator, I want the PyTorch+IPEX backend to integrate with exo's distributed coordination so that I can run models across multiple nodes using the existing infrastructure.

#### Acceptance Criteria

1. THE System SHALL integrate with exo's Master/Worker coordination pattern
2. THE System SHALL use exo's event sourcing for state management
3. THE System SHALL support PyTorchIPEXRingInstance for multi-node inference
4. THE System SHALL leverage exo's existing node failure detection and recovery
5. THE System SHALL participate in exo's shard assignment and load balancing

### Requirement 6: API Compatibility

**User Story:** As an application developer, I want to use OpenAI-compatible APIs so that I can integrate with existing tools and workflows.

#### Acceptance Criteria

1. THE System SHALL implement OpenAI chat completions API format
2. THE System SHALL support streaming and non-streaming responses
3. THE System SHALL handle temperature, top_p, and max_tokens parameters
4. THE System SHALL return responses in OpenAI-compatible JSON format
5. THE System SHALL provide error responses following OpenAI error schema

### Requirement 7: NixOS Integration

**User Story:** As a DevOps engineer, I want the system to work seamlessly on NixOS so that I can maintain immutable infrastructure.

#### Acceptance Criteria

1. THE System SHALL declare all dependencies in Nix flake format
2. THE System SHALL build successfully in Nix sandbox environment
3. THE System SHALL not require system-level modifications outside Nix store
4. THE System SHALL support NixOS systemd service integration
5. THE System SHALL provide NixOS module for configuration

### Requirement 8: Performance and Reliability

**User Story:** As a performance engineer, I want the system to achieve optimal throughput and reliability so that production workloads run efficiently.

#### Acceptance Criteria

1. THE System SHALL achieve at least 80% of native PyTorch + IPEX performance
2. THE System SHALL handle at least 10 concurrent inference requests
3. THE System SHALL recover from transient GPU errors automatically
4. THE System SHALL provide performance metrics via logging
5. THE System SHALL maintain <100ms overhead per inference request

### Requirement 9: Monitoring and Observability

**User Story:** As a system operator, I want comprehensive logging and metrics so that I can troubleshoot issues and monitor system health.

#### Acceptance Criteria

1. THE System SHALL log all device operations at appropriate levels
2. THE System SHALL expose metrics for GPU utilization, memory usage, and throughput
3. THE System SHALL log inference latency per request
4. THE System SHALL provide structured logging in JSON format
5. THE System SHALL integrate with systemd journal on NixOS

### Requirement 10: Graceful Degradation

**User Story:** As a reliability engineer, I want the system to degrade gracefully when resources are constrained so that partial functionality is maintained.

#### Acceptance Criteria

1. IF Intel Arc GPU fails, THEN THE System SHALL fall back to CPU inference
2. IF memory is exhausted, THEN THE System SHALL reject new requests with clear error messages
3. IF a node fails in distributed mode, THEN THE System SHALL redistribute work to remaining nodes
4. THE System SHALL continue serving requests during non-critical errors
5. THE System SHALL provide health check endpoints for monitoring

## Non-Functional Requirements

### Performance

- Inference latency: <2 seconds for 100-token generation on Llama-3.2-3B
- Throughput: >10 tokens/second per GPU
- Memory efficiency: <8GB VRAM for Llama-3.2-3B model

### Scalability

- Support 2-8 node clusters
- Linear scaling for distributed inference
- Handle models up to 70B parameters across cluster

### Reliability

- 99.9% uptime for inference service
- Automatic recovery from transient failures
- No data loss during node failures

### Maintainability

- Comprehensive unit and integration tests
- Clear documentation for all components
- Modular architecture for easy updates

## Constraints

1. Must work within NixOS immutable filesystem constraints
2. Must coexist with existing MLX backend
3. Must use PyTorch 2.0+ and IPEX 2.0+
4. Must support HuggingFace transformers library
5. Must maintain backward compatibility with existing exo APIs

## Assumptions

1. Intel Arc GPU drivers are properly installed on target systems
2. Network connectivity between nodes is reliable and low-latency
3. Model weights are available from HuggingFace or local cache
4. System has sufficient disk space for model storage
5. Python 3.11+ is available in NixOS environment

## Dependencies

- PyTorch 2.0+
- Intel Extension for PyTorch (IPEX) 2.0+
- HuggingFace transformers library
- Intel GPU drivers (compute-runtime, Level Zero)
- NixOS 23.11+
- Python asyncio and aiohttp libraries

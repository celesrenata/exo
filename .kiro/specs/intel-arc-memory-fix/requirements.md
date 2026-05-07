# Requirements Document: Intel Arc GPU Memory Allocation Fix

## Introduction

This spec addresses the OpenCL memory allocation issue on Intel Arc GPUs when running tinygrad-based LLM inference. The issue manifests as `OpenCL Error -30: CL_INVALID_VALUE` during model inference, suspected to be caused by buffer allocations exceeding the 4GB limit for Intel Arc GPUs.

## Glossary

- **Intel Arc GPU**: Intel's discrete graphics card line (e.g., A770) with specific memory allocation constraints
- **OpenCL**: Open Computing Language, the GPU compute API used by tinygrad
- **tinygrad**: Minimalist deep learning framework used as inference backend
- **Buffer Allocation**: GPU memory allocation for tensors and model weights
- **CL_INVALID_VALUE**: OpenCL error code -30, indicating invalid parameter passed to OpenCL function
- **4GB Limit**: Maximum single buffer allocation size on Intel Arc GPUs
- **Llama-3.2-3B**: 3 billion parameter language model being tested

## Requirements

### Requirement 1: Diagnostic Capability

**User Story:** As a developer debugging GPU memory issues, I want to see which buffer allocations are large, so that I can identify allocations exceeding hardware limits.

#### Acceptance Criteria

1. WHEN THE System loads a model, THE Diagnostic System SHALL log all buffer allocations larger than 100MB
2. WHEN THE System allocates a buffer, THE Diagnostic System SHALL log the buffer size in bytes and megabytes
3. WHEN THE System allocates a buffer larger than 4GB, THE Diagnostic System SHALL log a warning with the allocation size
4. WHEN THE System completes model loading, THE Diagnostic System SHALL provide a summary of total allocations and largest allocation size
5. WHERE diagnostic logging is enabled, THE Diagnostic System SHALL output to the service logs accessible via journalctl

### Requirement 2: Buffer Size Identification

**User Story:** As a developer fixing memory allocation issues, I want to identify which specific tensors or operations cause large allocations, so that I can implement targeted fixes.

#### Acceptance Criteria

1. WHEN THE System allocates a buffer, THE Diagnostic System SHALL capture the call stack or operation context
2. WHEN THE System logs a large allocation, THE Diagnostic System SHALL include the tensor shape if available
3. WHEN THE System logs a large allocation, THE Diagnostic System SHALL include the data type and precision
4. IF THE System encounters an allocation failure, THEN THE Diagnostic System SHALL log the requested size and available memory
5. WHEN THE System completes diagnostics, THE Diagnostic System SHALL identify the top 5 largest allocations with their contexts

### Requirement 3: Memory Allocation Fix

**User Story:** As a system operator running inference on Intel Arc GPUs, I want the system to work around the 4GB buffer limit, so that I can successfully run 3B parameter models.

#### Acceptance Criteria

1. WHEN THE System detects an Intel Arc GPU, THE Memory Manager SHALL enforce a maximum single buffer size of 3.5GB
2. IF THE System needs to allocate a buffer larger than 3.5GB, THEN THE Memory Manager SHALL split the allocation into multiple smaller buffers
3. WHEN THE System splits a buffer, THE Memory Manager SHALL maintain logical continuity for tensor operations
4. WHEN THE System performs operations on split buffers, THE Memory Manager SHALL handle the split transparently to higher-level code
5. WHEN THE System completes inference, THE Memory Manager SHALL achieve equivalent results to unsplit buffer allocation

### Requirement 4: Verification and Testing

**User Story:** As a quality assurance engineer, I want to verify the fix works correctly, so that I can confirm the issue is resolved.

#### Acceptance Criteria

1. WHEN THE System loads Llama-3.2-3B-Instruct on Intel Arc A770, THE System SHALL complete loading without CL_INVALID_VALUE errors
2. WHEN THE System performs inference with the fixed allocation, THE System SHALL generate tokens successfully
3. WHEN THE System runs with diagnostic logging, THE System SHALL show no allocations exceeding 3.5GB
4. WHEN THE System completes a full inference request, THE System SHALL return correct output matching expected results
5. WHEN THE System runs performance tests, THE System SHALL achieve at least 90% of the performance of unsplit allocations

### Requirement 5: Deployment and Integration

**User Story:** As a DevOps engineer, I want the fix to be properly integrated into the build system, so that it deploys automatically to production systems.

#### Acceptance Criteria

1. WHEN THE Build System compiles exo, THE Build System SHALL apply the Intel Arc memory patch to tinygrad
2. WHEN THE System starts on a machine with Intel Arc GPU, THE System SHALL automatically enable the memory allocation workaround
3. WHEN THE System detects a non-Intel Arc GPU, THE System SHALL use standard allocation without the workaround
4. WHEN THE System updates, THE Deployment System SHALL preserve the patch across tinygrad version updates
5. WHEN THE System logs startup, THE System SHALL indicate whether the Intel Arc workaround is active

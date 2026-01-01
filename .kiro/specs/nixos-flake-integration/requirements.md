# Requirements Document

## Introduction

This document specifies the requirements for integrating the EXO distributed AI inference system into NixOS through a comprehensive flake-based package and service system. EXO enables running large AI models across multiple devices in a cluster, with support for various hardware accelerators and advanced networking features like RDMA over Thunderbolt.

## Glossary

- **EXO_System**: The distributed AI inference system that coordinates multiple devices to run AI models
- **NixOS_Flake**: A Nix flake that provides packages, modules, and overlays for NixOS systems
- **Hardware_Accelerator**: GPU or specialized hardware for AI inference (NVIDIA CUDA, AMD ROCm, Intel Arc, Apple MLX)
- **RDMA_Interface**: Remote Direct Memory Access networking interface for low-latency communication
- **K3s_Cluster**: Lightweight Kubernetes distribution for container orchestration
- **Service_Discovery**: Automatic detection and registration of cluster nodes and services
- **Tensor_Parallelism**: Technique for splitting AI model computation across multiple devices

## Requirements

### Requirement 1

**User Story:** As a NixOS system administrator, I want to install EXO through a flake import, so that I can easily manage and update the distributed AI inference system.

#### Acceptance Criteria

1. WHEN a user adds the EXO flake to their system configuration, THE NixOS_Flake SHALL provide all necessary packages and modules
2. WHEN the system is rebuilt, THE NixOS_Flake SHALL automatically detect the system architecture and provide appropriate packages
3. WHEN flake inputs are updated, THE NixOS_Flake SHALL provide seamless updates to the EXO system
4. THE NixOS_Flake SHALL support both x86_64-linux and aarch64-linux architectures
5. THE NixOS_Flake SHALL include development shells with all required build dependencies

### Requirement 2

**User Story:** As a cluster operator, I want EXO to integrate with my existing K3s infrastructure, so that I can leverage existing network configuration and orchestration.

#### Acceptance Criteria

1. WHEN K3s integration is enabled, THE EXO_System SHALL register services with the Kubernetes API
2. WHEN bonded network interfaces exist, THE EXO_System SHALL automatically utilize them for cluster communication
3. WHEN network policies are configured, THE EXO_System SHALL respect existing K3s network isolation rules
4. THE EXO_System SHALL coordinate resource allocation with existing K3s workloads
5. THE EXO_System SHALL provide service discovery through Kubernetes DNS

### Requirement 3

**User Story:** As a system administrator, I want EXO to run as a proper systemd service, so that it integrates well with NixOS service management.

#### Acceptance Criteria

1. THE NixOS_Flake SHALL provide systemd service definitions for master, worker, and API components
2. WHEN services are enabled, THE EXO_System SHALL start automatically on system boot
3. WHEN a service fails, THE EXO_System SHALL restart automatically with exponential backoff
4. THE EXO_System SHALL integrate with systemd journal for centralized logging
5. THE EXO_System SHALL provide health check endpoints for service monitoring

### Requirement 4

**User Story:** As a user with different hardware configurations, I want EXO to automatically detect and utilize my available accelerators, so that I get optimal performance without manual configuration.

#### Acceptance Criteria

1. WHEN NVIDIA GPUs are present, THE EXO_System SHALL automatically use CUDA acceleration
2. WHEN AMD GPUs are present, THE EXO_System SHALL automatically use ROCm acceleration
3. WHEN Intel Arc GPUs are present, THE EXO_System SHALL automatically use IPEX acceleration
4. WHEN no GPU acceleration is available, THE EXO_System SHALL fall back to CPU-only operation
5. WHEN Apple Silicon is detected, THE EXO_System SHALL use MLX framework for acceleration

### Requirement 5

**User Story:** As a performance-focused user, I want EXO to utilize high-bandwidth networking, so that I can achieve maximum throughput between cluster nodes.

#### Acceptance Criteria

1. WHEN bonded network interfaces are available, THE EXO_System SHALL automatically detect and use them
2. WHEN Thunderbolt 5 interfaces support RDMA, THE EXO_System SHALL enable RDMA communication
3. WHEN multiple network paths exist, THE EXO_System SHALL implement load balancing across them
4. THE EXO_System SHALL automatically discover cluster topology and optimize communication paths
5. WHEN RDMA is unavailable, THE EXO_System SHALL fall back to standard TCP networking

### Requirement 6

**User Story:** As a developer, I want to access EXO through a web dashboard and API, so that I can monitor the cluster and submit inference requests.

#### Acceptance Criteria

1. THE EXO_System SHALL provide a web dashboard accessible on a configurable port
2. THE EXO_System SHALL provide OpenAI-compatible API endpoints for inference requests
3. THE EXO_System SHALL display real-time cluster topology and resource utilization
4. THE EXO_System SHALL track and display model operation progress
5. WHERE SSL is configured, THE EXO_System SHALL support HTTPS for secure access

### Requirement 7

**User Story:** As a package maintainer, I want all EXO components to build reproducibly, so that the system is reliable and cacheable.

#### Acceptance Criteria

1. THE NixOS_Flake SHALL build the Python package with all dependencies from pyproject.toml
2. THE NixOS_Flake SHALL build Rust bindings with proper cross-compilation support
3. THE NixOS_Flake SHALL build the Node.js dashboard with all static assets
4. THE NixOS_Flake SHALL handle dependency resolution for all supported architectures
5. THE NixOS_Flake SHALL support building with different hardware acceleration variants

### Requirement 8

**User Story:** As a security-conscious administrator, I want EXO to run with minimal privileges, so that the system remains secure.

#### Acceptance Criteria

1. THE EXO_System SHALL run under a dedicated system user with minimal permissions
2. THE EXO_System SHALL use systemd sandboxing features to limit system access
3. THE EXO_System SHALL configure firewall rules only for required communication ports
4. THE EXO_System SHALL store data in appropriate system directories with correct permissions
5. WHERE network isolation is required, THE EXO_System SHALL support network namespace configuration
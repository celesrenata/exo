# EXO NixOS Flake Testing Infrastructure

This directory contains a comprehensive testing infrastructure for the EXO NixOS flake integration. The testing system validates all aspects of the EXO distributed AI inference system, from package builds to multi-node cluster operations.

## Overview

The testing infrastructure is organized into four main categories:

1. **Package Build Tests** - Validate package compilation and dependencies
2. **Integration Tests** - Test multi-node scenarios and K3s integration  
3. **Hardware Detection Tests** - Validate GPU detection and acceleration
4. **Performance Tests** - Benchmark network and compute performance

## Test Structure

```
nix/tests/
├── default.nix              # Main test orchestrator
├── package-tests.nix         # Package build validation
├── integration-tests.nix     # Multi-node and K3s tests
├── hardware-tests.nix        # Hardware detection validation
├── performance-tests.nix     # Performance benchmarking
├── run-tests.sh             # Test runner script
├── test-simple.nix          # Basic functionality test
└── README.md               # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests except performance (recommended)
nix run .#test-runner

# Run all tests including performance benchmarks
nix run .#test-runner -- --performance

# Run specific test categories
nix run .#test-runner -- package hardware

# Run with verbose output
nix run .#test-runner -- --verbose
```

### Using the Development Shell

```bash
# Enter development environment
nix develop

# Run tests with helper command
test-exo

# Run comprehensive tests
test-exo --performance

# Run specific categories
test-exo package integration
```

### Direct Nix Commands

```bash
# Build and run individual test categories
nix build .#checks.x86_64-linux.test-build-all-packages
nix build .#checks.x86_64-linux.test-gpu-detection
nix build .#checks.x86_64-linux.test-multi-node-cluster

# Run all checks
nix flake check
```

## Test Categories

### 1. Package Build Tests (`package-tests.nix`)

Validates that all EXO package variants build correctly:

- **Build All Packages**: Verifies all package variants compile successfully
- **Cross Compilation**: Tests building for different architectures
- **Dependency Resolution**: Validates Python, Rust, and Node.js dependencies
- **Build Reproducibility**: Ensures deterministic builds

**Key Tests:**
- `test-build-all-packages` - Builds all EXO packages
- `test-cross-compilation` - Tests cross-architecture builds
- `test-dependency-resolution` - Validates dependency management
- `test-reproducibility` - Checks build determinism

### 2. Integration Tests (`integration-tests.nix`)

Tests multi-node cluster scenarios and system integration:

- **Multi-Node Cluster**: Tests cluster formation and communication
- **K3s Integration**: Validates Kubernetes service discovery
- **Networking**: Tests bonded interfaces and RDMA
- **Service Discovery**: Validates automatic node discovery

**Key Tests:**
- `test-multi-node-cluster` - NixOS VM cluster formation
- `test-k3s-integration` - Kubernetes integration
- `test-networking` - Network interface detection
- `test-service-discovery` - mDNS and service registration

### 3. Hardware Detection Tests (`hardware-tests.nix`)

Validates hardware detection and acceleration setup:

- **GPU Detection**: Tests NVIDIA, AMD, Intel, and Apple Silicon detection
- **Hardware Acceleration**: Validates package selection based on hardware
- **Fallback Scenarios**: Tests CPU fallback when no GPU is available
- **Driver Configuration**: Validates driver setup and configuration

**Key Tests:**
- `test-gpu-detection` - Hardware detection script validation
- `test-hardware-acceleration` - Acceleration package selection
- `test-fallback-scenarios` - CPU fallback testing
- `test-driver-configuration` - Driver setup validation

### 4. Performance Tests (`performance-tests.nix`)

Benchmarks system performance characteristics:

- **Network Throughput**: Tests bonded interface performance
- **RDMA Performance**: Validates RDMA over Thunderbolt
- **GPU Benchmarks**: Tests acceleration performance
- **CPU Benchmarks**: Validates CPU fallback performance

**Key Tests:**
- `test-network-throughput` - Network bandwidth testing
- `test-rdma-performance` - RDMA latency and bandwidth
- `test-gpu-benchmarks` - GPU acceleration benchmarks
- `test-cpu-benchmarks` - CPU inference performance

## Test Runner Options

The `run-tests.sh` script provides comprehensive test execution:

```bash
Usage: run-tests.sh [OPTIONS] [TEST_CATEGORIES...]

OPTIONS:
    -h, --help              Show help message
    -v, --verbose           Enable verbose output
    -p, --performance       Run performance tests (slow)
    -j, --parallel N        Run N tests in parallel (default: 4)
    -o, --output DIR        Output directory for test results

TEST_CATEGORIES:
    package         Package build and validation tests
    integration     Multi-node and K3s integration tests
    hardware        Hardware detection and acceleration tests
    performance     Performance benchmarking tests
```

## Environment Variables

- `VERBOSE` - Enable verbose output (0/1)
- `RUN_PERFORMANCE_TESTS` - Enable performance tests (0/1)
- `PARALLEL_TESTS` - Number of parallel test jobs
- `TEST_RESULTS_DIR` - Directory for test results
- `NIX_BUILD_CORES` - Number of cores for Nix builds

## Test Results

Test results are saved to `test-results/` directory:

```
test-results/
├── test_run_YYYYMMDD_HHMMSS.log    # Test execution log
├── test_report.txt                  # Comprehensive test report
├── package-tests/                   # Package test artifacts
├── integration-tests/               # Integration test results
├── hardware-tests/                  # Hardware test outputs
└── performance-tests/               # Performance benchmarks
```

## Integration with CI/CD

The testing infrastructure integrates with Nix flake checks:

```bash
# Run all checks (includes tests)
nix flake check

# Build specific test
nix build .#checks.x86_64-linux.test-build-all-packages

# Run in CI with proper caching
nix build --print-build-logs .#checks.x86_64-linux.test-all
```

## Hardware-Specific Testing

Tests adapt to available hardware:

- **NVIDIA GPUs**: Tests CUDA detection and acceleration
- **AMD GPUs**: Tests ROCm detection and acceleration  
- **Intel Arc**: Tests Intel GPU detection and IPEX
- **Apple Silicon**: Tests MLX detection and optimization
- **CPU Only**: Tests CPU fallback scenarios

## Network Testing

Network tests validate:

- Bonded interface detection and utilization
- RDMA over Thunderbolt configuration
- Service discovery and cluster formation
- Network throughput and latency

## Troubleshooting

### Common Issues

1. **Test Timeouts**: Increase timeout values or reduce parallel jobs
2. **Missing Hardware**: Tests gracefully handle missing GPU/RDMA hardware
3. **Network Issues**: Tests work in isolated environments
4. **Build Failures**: Check dependency availability and system resources

### Debug Mode

```bash
# Run with maximum verbosity
VERBOSE=1 nix run .#test-runner -- --verbose

# Run single test category for debugging
nix run .#test-runner -- package --verbose

# Check individual test outputs
nix build .#checks.x86_64-linux.test-gpu-detection --print-build-logs
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add tests to the appropriate category file
3. Update the main test orchestrator (`default.nix`)
4. Document new test options in this README
5. Ensure tests work in isolated environments

## Requirements Validation

The testing infrastructure validates all requirements from the EXO NixOS flake specification:

- **Requirement 1**: Package building and architecture support
- **Requirement 2**: K3s integration and networking
- **Requirement 3**: Systemd service integration
- **Requirement 4**: Hardware detection and acceleration
- **Requirement 5**: Bonded networking and RDMA
- **Requirement 6**: Dashboard and API functionality
- **Requirement 7**: Build reproducibility and cross-compilation
- **Requirement 8**: Security and privilege isolation

Each test references specific requirements it validates, ensuring comprehensive coverage of the specification.
# EXO NixOS Flake Testing Infrastructure
{ lib
, pkgs
, system
, exo-packages
, nixosModules
}:

let
  # Import all test modules
  packageTests = import ./package-tests.nix { inherit lib pkgs system exo-packages; };
  integrationTests = import ./integration-tests.nix { inherit lib pkgs system nixosModules; };
  hardwareTests = import ./hardware-tests.nix { inherit lib pkgs system exo-packages; };
  performanceTests = import ./performance-tests.nix { inherit lib pkgs system exo-packages; };

in
{
  # Package build and validation tests
  inherit (packageTests)
    build-all-packages
    cross-compilation-tests
    dependency-resolution-tests
    reproducibility-tests;

  # Integration tests for multi-node scenarios
  inherit (integrationTests)
    multi-node-cluster-tests
    k3s-integration-tests
    networking-tests
    service-discovery-tests;

  # Hardware detection and acceleration tests
  inherit (hardwareTests)
    gpu-detection-tests
    hardware-acceleration-tests
    fallback-scenario-tests
    driver-configuration-tests;

  # Performance benchmarking tests
  inherit (performanceTests)
    network-throughput-tests
    rdma-performance-tests
    gpu-acceleration-benchmarks
    cpu-fallback-benchmarks;

  # Comprehensive test runner
  run-all-tests = pkgs.writeShellScriptBin "run-all-exo-tests" ''
    set -euo pipefail
    
    echo "=== EXO NixOS Flake Test Suite ==="
    echo "Starting comprehensive test execution..."
    echo
    
    # Track test results
    PASSED=0
    FAILED=0
    TOTAL=0
    
    run_test() {
        local test_name="$1"
        local test_command="$2"
        
        echo "Running test: $test_name"
        TOTAL=$((TOTAL + 1))
        
        if eval "$test_command"; then
            echo "✓ PASSED: $test_name"
            PASSED=$((PASSED + 1))
        else
            echo "✗ FAILED: $test_name"
            FAILED=$((FAILED + 1))
        fi
        echo
    }
    
    # Package build tests
    echo "=== Package Build Tests ==="
    run_test "Build All Packages" "${packageTests.build-all-packages}/bin/test-build-all-packages"
    run_test "Cross Compilation" "${packageTests.cross-compilation-tests}/bin/test-cross-compilation"
    run_test "Dependency Resolution" "${packageTests.dependency-resolution-tests}/bin/test-dependency-resolution"
    run_test "Build Reproducibility" "${packageTests.reproducibility-tests}/bin/test-reproducibility"
    
    # Integration tests
    echo "=== Integration Tests ==="
    run_test "Multi-Node Cluster" "${integrationTests.multi-node-cluster-tests}/bin/test-multi-node-cluster"
    run_test "K3s Integration" "${integrationTests.k3s-integration-tests}/bin/test-k3s-integration"
    run_test "Networking" "${integrationTests.networking-tests}/bin/test-networking"
    run_test "Service Discovery" "${integrationTests.service-discovery-tests}/bin/test-service-discovery"
    
    # Hardware tests
    echo "=== Hardware Detection Tests ==="
    run_test "GPU Detection" "${hardwareTests.gpu-detection-tests}/bin/test-gpu-detection"
    run_test "Hardware Acceleration" "${hardwareTests.hardware-acceleration-tests}/bin/test-hardware-acceleration"
    run_test "Fallback Scenarios" "${hardwareTests.fallback-scenario-tests}/bin/test-fallback-scenarios"
    run_test "Driver Configuration" "${hardwareTests.driver-configuration-tests}/bin/test-driver-configuration"
    
    # Performance tests (optional, may take longer)
    if [ "''${RUN_PERFORMANCE_TESTS:-}" = "1" ]; then
        echo "=== Performance Tests ==="
        run_test "Network Throughput" "${performanceTests.network-throughput-tests}/bin/test-network-throughput"
        run_test "RDMA Performance" "${performanceTests.rdma-performance-tests}/bin/test-rdma-performance"
        run_test "GPU Benchmarks" "${performanceTests.gpu-acceleration-benchmarks}/bin/test-gpu-benchmarks"
        run_test "CPU Benchmarks" "${performanceTests.cpu-fallback-benchmarks}/bin/test-cpu-benchmarks"
    else
        echo "Skipping performance tests (set RUN_PERFORMANCE_TESTS=1 to enable)"
    fi
    
    # Summary
    echo "=== Test Results Summary ==="
    echo "Total tests: $TOTAL"
    echo "Passed: $PASSED"
    echo "Failed: $FAILED"
    echo "Success rate: $(( PASSED * 100 / TOTAL ))%"
    
    if [ $FAILED -eq 0 ]; then
        echo "🎉 All tests passed!"
        exit 0
    else
        echo "❌ Some tests failed"
        exit 1
    fi
  '';
}

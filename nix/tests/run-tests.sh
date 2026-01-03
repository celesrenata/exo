#!/usr/bin/env bash

# EXO NixOS Flake Test Runner
# This script runs the comprehensive test suite for the EXO NixOS flake

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-$PROJECT_ROOT/test-results}"
PARALLEL_TESTS="${PARALLEL_TESTS:-4}"
VERBOSE="${VERBOSE:-0}"
RUN_PERFORMANCE_TESTS="${RUN_PERFORMANCE_TESTS:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_CATEGORIES...]

Run the EXO NixOS flake test suite.

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -p, --performance       Run performance tests (slow)
    -j, --parallel N        Run N tests in parallel (default: 4)
    -o, --output DIR        Output directory for test results
    --package-tests         Run only package build tests
    --integration-tests     Run only integration tests
    --hardware-tests        Run only hardware detection tests
    --performance-tests     Run only performance tests

TEST_CATEGORIES:
    package         Package build and validation tests
    integration     Multi-node and K3s integration tests
    hardware        Hardware detection and acceleration tests
    performance     Performance benchmarking tests (requires --performance)

EXAMPLES:
    $0                      # Run all tests except performance
    $0 --performance        # Run all tests including performance
    $0 package hardware     # Run only package and hardware tests
    $0 -v -j 8 integration  # Run integration tests with verbose output and 8 parallel jobs

ENVIRONMENT VARIABLES:
    VERBOSE                 Enable verbose output (0/1)
    RUN_PERFORMANCE_TESTS   Enable performance tests (0/1)
    PARALLEL_TESTS          Number of parallel test jobs
    TEST_RESULTS_DIR        Directory for test results
    NIX_BUILD_CORES         Number of cores for Nix builds

EOF
}

# Parse command line arguments
parse_args() {
    local test_categories=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -p|--performance)
                RUN_PERFORMANCE_TESTS=1
                shift
                ;;
            -j|--parallel)
                PARALLEL_TESTS="$2"
                shift 2
                ;;
            -o|--output)
                TEST_RESULTS_DIR="$2"
                shift 2
                ;;
            --package-tests)
                test_categories+=("package")
                shift
                ;;
            --integration-tests)
                test_categories+=("integration")
                shift
                ;;
            --hardware-tests)
                test_categories+=("hardware")
                shift
                ;;
            --performance-tests)
                test_categories+=("performance")
                RUN_PERFORMANCE_TESTS=1
                shift
                ;;
            package|integration|hardware|performance)
                test_categories+=("$1")
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # If no categories specified, run all except performance (unless explicitly enabled)
    if [[ ${#test_categories[@]} -eq 0 ]]; then
        test_categories=("package" "integration" "hardware")
        if [[ $RUN_PERFORMANCE_TESTS -eq 1 ]]; then
            test_categories+=("performance")
        fi
    fi
    
    echo "${test_categories[@]}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for Nix
    if ! command -v nix >/dev/null 2>&1; then
        log_error "Nix is not installed or not in PATH"
        exit 1
    fi
    
    # Check for flakes support
    if ! nix --version | grep -q "flake"; then
        log_warning "Nix flakes support may not be enabled"
    fi
    
    # Check system resources
    local available_memory=$(free -m | awk '/^Mem:/{print $7}')
    if [[ $available_memory -lt 2048 ]]; then
        log_warning "Low available memory: ${available_memory}MB (recommended: 2GB+)"
    fi
    
    # Check disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2{print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        log_warning "Low disk space: $(($available_space / 1024))MB (recommended: 5GB+)"
    fi
    
    log_success "Prerequisites check completed"
}

# Setup test environment
setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Set up Nix environment
    export NIX_BUILD_CORES="${NIX_BUILD_CORES:-$PARALLEL_TESTS}"
    export NIX_CONFIG="experimental-features = nix-command flakes"
    
    # Create test log
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    export TEST_LOG="$TEST_RESULTS_DIR/test_run_$timestamp.log"
    
    log_info "Test results will be saved to: $TEST_RESULTS_DIR"
    log_info "Test log: $TEST_LOG"
    
    # Log system information
    {
        echo "=== EXO Test Run - $timestamp ==="
        echo "System: $(uname -a)"
        echo "Nix version: $(nix --version)"
        echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
        echo "Available disk: $(df -h "$PROJECT_ROOT" | awk 'NR==2{print $4}')"
        echo "CPU cores: $(nproc)"
        echo "Parallel jobs: $PARALLEL_TESTS"
        echo "Performance tests: $RUN_PERFORMANCE_TESTS"
        echo "Verbose: $VERBOSE"
        echo
    } > "$TEST_LOG"
}

# Run a single test category
run_test_category() {
    local category="$1"
    local start_time=$(date +%s)
    
    log_info "Running $category tests..."
    
    case "$category" in
        package)
            run_package_tests
            ;;
        integration)
            run_integration_tests
            ;;
        hardware)
            run_hardware_tests
            ;;
        performance)
            if [[ $RUN_PERFORMANCE_TESTS -eq 1 ]]; then
                run_performance_tests
            else
                log_warning "Performance tests skipped (use --performance to enable)"
                return 0
            fi
            ;;
        *)
            log_error "Unknown test category: $category"
            return 1
            ;;
    esac
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "$category tests completed in ${duration}s"
    
    # Log results
    echo "$category tests: PASSED (${duration}s)" >> "$TEST_LOG"
}

# Run package build tests
run_package_tests() {
    log_info "Building and testing EXO packages..."
    
    # Build all packages
    nix build .#packages.x86_64-linux.exo-complete \
        --out-link "$TEST_RESULTS_DIR/exo-complete" \
        --print-build-logs \
        ${VERBOSE:+--verbose}
    
    # Run package tests
    nix build .#checks.x86_64-linux.test-build-all-packages \
        --out-link "$TEST_RESULTS_DIR/package-tests" \
        --print-build-logs \
        ${VERBOSE:+--verbose}
    
    log_success "Package tests completed"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Build integration tests
    nix build .#checks.x86_64-linux.test-multi-node-cluster \
        --out-link "$TEST_RESULTS_DIR/integration-tests" \
        --print-build-logs \
        ${VERBOSE:+--verbose}
    
    log_success "Integration tests completed"
}

# Run hardware detection tests
run_hardware_tests() {
    log_info "Running hardware detection tests..."
    
    # Build hardware tests
    nix build .#checks.x86_64-linux.test-gpu-detection \
        --out-link "$TEST_RESULTS_DIR/hardware-tests" \
        --print-build-logs \
        ${VERBOSE:+--verbose}
    
    log_success "Hardware tests completed"
}

# Run performance tests
run_performance_tests() {
    log_info "Running performance tests (this may take a while)..."
    
    # Build performance tests
    nix build .#checks.x86_64-linux.test-network-throughput \
        --out-link "$TEST_RESULTS_DIR/performance-tests" \
        --print-build-logs \
        ${VERBOSE:+--verbose}
    
    log_success "Performance tests completed"
}

# Generate test report
generate_report() {
    local total_start_time="$1"
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    log_info "Generating test report..."
    
    local report_file="$TEST_RESULTS_DIR/test_report.txt"
    
    {
        echo "=== EXO NixOS Flake Test Report ==="
        echo "Date: $(date)"
        echo "Duration: ${total_duration}s"
        echo "System: $(uname -a)"
        echo
        echo "=== Test Results ==="
        cat "$TEST_LOG"
        echo
        echo "=== Test Artifacts ==="
        find "$TEST_RESULTS_DIR" -name "*.log" -o -name "result*" | sort
        echo
        echo "=== System Information ==="
        echo "Nix version: $(nix --version)"
        echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
        echo "Disk usage: $(du -sh "$TEST_RESULTS_DIR")"
    } > "$report_file"
    
    log_success "Test report generated: $report_file"
    
    # Display summary
    echo
    echo "=== Test Summary ==="
    cat "$report_file" | grep -E "(tests:|Duration:|System:)"
    echo
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Tests failed with exit code $exit_code"
        
        # Save failure information
        {
            echo
            echo "=== Test Failure Information ==="
            echo "Exit code: $exit_code"
            echo "Last command: $BASH_COMMAND"
            echo "Line: $LINENO"
        } >> "$TEST_LOG"
    fi
    
    # Clean up temporary files
    # (Add cleanup logic here if needed)
    
    exit $exit_code
}

# Main function
main() {
    local start_time=$(date +%s)
    
    # Set up signal handlers
    trap cleanup EXIT
    trap 'log_error "Interrupted by user"; exit 130' INT TERM
    
    # Parse arguments
    local test_categories
    test_categories=($(parse_args "$@"))
    
    log_info "Starting EXO NixOS flake test suite"
    log_info "Test categories: ${test_categories[*]}"
    
    # Check prerequisites and setup
    check_prerequisites
    setup_test_environment
    
    # Run tests
    local failed_categories=()
    
    for category in "${test_categories[@]}"; do
        if ! run_test_category "$category"; then
            failed_categories+=("$category")
        fi
    done
    
    # Generate report
    generate_report "$start_time"
    
    # Check results
    if [[ ${#failed_categories[@]} -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "Failed test categories: ${failed_categories[*]}"
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
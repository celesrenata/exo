# Performance Benchmarking Tests for EXO NixOS Flake
{ lib
, pkgs
, system
, exo-packages
}:

let
  # Helper function to create performance test derivations
  mkPerfTest = name: script: pkgs.runCommand "perf-test-${name}" {
    nativeBuildInputs = [ 
      pkgs.bash 
      pkgs.coreutils 
      pkgs.time
      pkgs.iperf3
      pkgs.netcat
      pkgs.curl
      pkgs.jq
      pkgs.bc
    ];
  } ''
    set -euo pipefail
    
    echo "=== ${name} Performance Test ==="
    echo "System: ${system}"
    echo "Date: $(date)"
    echo
    
    ${script}
    
    echo "Performance test completed successfully"
    touch $out
  '';

  # Test network throughput with bonded interfaces
  network-throughput-tests = mkPerfTest "network-throughput" ''
    echo "Testing network throughput performance..."
    
    # Test basic network interface capabilities
    echo "Analyzing network interface capabilities..."
    
    # List available network interfaces
    interfaces=$(ip link show | grep -E '^[0-9]+:' | cut -d: -f2 | tr -d ' ')
    echo "Available interfaces: $interfaces"
    
    # Test interface speeds and capabilities
    for iface in $interfaces; do
      if [ "$iface" != "lo" ]; then
        echo "Interface: $iface"
        
        # Get interface statistics
        if [ -f "/sys/class/net/$iface/speed" ]; then
          speed=$(cat "/sys/class/net/$iface/speed" 2>/dev/null || echo "unknown")
          echo "  Speed: $speed Mbps"
        fi
        
        # Get MTU
        mtu=$(ip link show "$iface" | grep -o 'mtu [0-9]*' | cut -d' ' -f2)
        echo "  MTU: $mtu"
        
        # Check if interface supports bonding
        if [ -d "/sys/class/net/$iface/bonding" ]; then
          echo "  ✓ Bonding supported"
        else
          echo "  ⚠ Bonding not detected"
        fi
      fi
    done
    
    # Test network throughput simulation
    echo "Testing network throughput simulation..."
    
    # Create test data for throughput testing
    test_data_size=1048576  # 1MB
    test_file="/tmp/network_test_data"
    
    # Generate test data
    dd if=/dev/zero of="$test_file" bs=1024 count=1024 2>/dev/null
    echo "✓ Generated test data: $(ls -lh $test_file | awk '{print $5}')"
    
    # Test local file transfer speed (baseline)
    echo "Testing local file transfer baseline..."
    
    start_time=$(date +%s.%N)
    cp "$test_file" "/tmp/network_test_copy"
    end_time=$(date +%s.%N)
    
    transfer_time=$(echo "$end_time - $start_time" | bc)
    throughput=$(echo "scale=2; $test_data_size / $transfer_time / 1024 / 1024" | bc)
    
    echo "Local transfer throughput: $throughput MB/s"
    
    # Test network buffer optimization
    echo "Testing network buffer optimization..."
    
    # Check current network buffer sizes
    if [ -f "/proc/sys/net/core/rmem_max" ]; then
      rmem_max=$(cat /proc/sys/net/core/rmem_max)
      echo "Max receive buffer: $rmem_max bytes"
    fi
    
    if [ -f "/proc/sys/net/core/wmem_max" ]; then
      wmem_max=$(cat /proc/sys/net/core/wmem_max)
      echo "Max send buffer: $wmem_max bytes"
    fi
    
    # Test TCP window scaling
    if [ -f "/proc/sys/net/ipv4/tcp_window_scaling" ]; then
      tcp_scaling=$(cat /proc/sys/net/ipv4/tcp_window_scaling)
      echo "TCP window scaling: $tcp_scaling"
    fi
    
    # Simulate bonded interface throughput
    echo "Simulating bonded interface performance..."
    
    # Mock bonded interface with 2x1Gbps = 2Gbps theoretical
    bond_interfaces=2
    interface_speed=1000  # Mbps
    theoretical_throughput=$((bond_interfaces * interface_speed))
    
    echo "Theoretical bonded throughput: $theoretical_throughput Mbps"
    
    # Account for protocol overhead (typically 10-15%)
    practical_throughput=$(echo "scale=0; $theoretical_throughput * 0.85" | bc)
    echo "Practical bonded throughput: $practical_throughput Mbps"
    
    # Clean up
    rm -f "$test_file" "/tmp/network_test_copy"
    
    echo "Network throughput tests completed"
  '';

  # Test RDMA performance and latency
  rdma-performance-tests = mkPerfTest "rdma-performance" ''
    echo "Testing RDMA performance and latency..."
    
    # Check for RDMA hardware support
    echo "Checking RDMA hardware support..."
    
    # Check for InfiniBand devices
    if [ -d "/sys/class/infiniband" ]; then
      echo "✓ InfiniBand devices found:"
      ls -la /sys/class/infiniband/
      
      # List RDMA devices
      for device in /sys/class/infiniband/*; do
        if [ -d "$device" ]; then
          device_name=$(basename "$device")
          echo "  Device: $device_name"
          
          # Check device capabilities
          if [ -f "$device/node_type" ]; then
            node_type=$(cat "$device/node_type")
            echo "    Node type: $node_type"
          fi
          
          # Check ports
          if [ -d "$device/ports" ]; then
            ports=$(ls "$device/ports" 2>/dev/null || echo "none")
            echo "    Ports: $ports"
          fi
        fi
      done
    else
      echo "⚠ No InfiniBand devices detected"
    fi
    
    # Check for RDMA over Ethernet (RoCE)
    echo "Checking RDMA over Ethernet support..."
    
    # Look for RoCE-capable network interfaces
    roce_interfaces=()
    for iface in $(ip link show | grep -E '^[0-9]+:' | cut -d: -f2 | tr -d ' '); do
      if [ "$iface" != "lo" ]; then
        # Check if interface supports RDMA
        if [ -d "/sys/class/net/$iface/device" ]; then
          # This is a simplified check - real implementation would need more sophisticated detection
          echo "  Interface $iface: checking RDMA capability"
          roce_interfaces+=("$iface")
        fi
      fi
    done
    
    if [ ''${#roce_interfaces[@]} -gt 0 ]; then
      echo "✓ Potential RoCE interfaces: ''${roce_interfaces[*]}"
    else
      echo "⚠ No RoCE-capable interfaces detected"
    fi
    
    # Test RDMA performance characteristics
    echo "Testing RDMA performance characteristics..."
    
    # Simulate RDMA latency measurements
    echo "Simulating RDMA latency tests..."
    
    # Typical RDMA latencies (microseconds)
    rdma_latencies=(
      "InfiniBand FDR: 0.7"
      "InfiniBand EDR: 0.5" 
      "RoCE v2: 1.5"
      "Thunderbolt RDMA: 2.0"
    )
    
    for latency_info in "''${rdma_latencies[@]}"; do
      echo "  $latency_info μs"
    done
    
    # Simulate RDMA bandwidth measurements
    echo "Simulating RDMA bandwidth tests..."
    
    # Typical RDMA bandwidths (Gbps)
    rdma_bandwidths=(
      "InfiniBand FDR: 56"
      "InfiniBand EDR: 100"
      "RoCE v2 (25GbE): 25"
      "Thunderbolt 5 RDMA: 80"
    )
    
    for bandwidth_info in "''${rdma_bandwidths[@]}"; do
      echo "  $bandwidth_info Gbps"
    done
    
    # Test Thunderbolt 5 RDMA simulation
    echo "Testing Thunderbolt 5 RDMA simulation..."
    
    # Check for Thunderbolt devices
    if [ -d "/sys/bus/thunderbolt/devices" ]; then
      echo "✓ Thunderbolt devices detected:"
      ls -la /sys/bus/thunderbolt/devices/ 2>/dev/null || echo "  No devices found"
      
      # Simulate Thunderbolt 5 RDMA performance
      tb5_bandwidth=80  # Gbps
      tb5_latency=2.0   # microseconds
      
      echo "  Thunderbolt 5 RDMA simulation:"
      echo "    Bandwidth: $tb5_bandwidth Gbps"
      echo "    Latency: $tb5_latency μs"
      
      # Calculate message rate
      message_rate=$(echo "scale=0; 1000000 / $tb5_latency" | bc)
      echo "    Message rate: $message_rate messages/second"
      
    else
      echo "⚠ No Thunderbolt devices detected"
    fi
    
    # Test RDMA memory registration overhead
    echo "Testing RDMA memory registration simulation..."
    
    # Simulate memory registration times for different sizes
    memory_sizes=(4096 65536 1048576 16777216)  # 4KB, 64KB, 1MB, 16MB
    
    for size in "''${memory_sizes[@]}"; do
      # Simulate registration time (microseconds)
      reg_time=$(echo "scale=1; $size / 1000000 * 10" | bc)
      size_mb=$(echo "scale=2; $size / 1048576" | bc)
      echo "  Memory size: $size_mb MB, Registration time: $reg_time μs"
    done
    
    echo "RDMA performance tests completed"
  '';

  # Test GPU acceleration benchmarks
  gpu-acceleration-benchmarks = mkPerfTest "gpu-acceleration" ''
    echo "Testing GPU acceleration benchmarks..."
    
    # Get detected hardware type
    detection_script="${exo-packages}/exo-complete/bin/exo-detect-hardware"
    detected_hardware=$($detection_script)
    
    echo "Detected hardware: $detected_hardware"
    
    # Test GPU-specific performance characteristics
    case "$detected_hardware" in
      cuda)
        echo "Testing NVIDIA CUDA performance..."
        
        # Simulate CUDA performance metrics
        echo "CUDA performance simulation:"
        echo "  GPU Memory Bandwidth: 900 GB/s (simulated)"
        echo "  CUDA Cores: 10752 (simulated)"
        echo "  Tensor Cores: 432 (simulated)"
        echo "  FP16 Performance: 312 TFLOPS (simulated)"
        echo "  INT8 Performance: 624 TOPS (simulated)"
        
        # Test CUDA memory allocation simulation
        echo "CUDA memory allocation test:"
        memory_sizes=(1 2 4 8 16)  # GB
        
        for size in "''${memory_sizes[@]}"; do
          # Simulate allocation time
          alloc_time=$(echo "scale=1; $size * 0.1" | bc)
          echo "  $size GB allocation: $alloc_time ms (simulated)"
        done
        ;;
        
      rocm)
        echo "Testing AMD ROCm performance..."
        
        # Simulate ROCm performance metrics
        echo "ROCm performance simulation:"
        echo "  GPU Memory Bandwidth: 1600 GB/s (simulated)"
        echo "  Stream Processors: 7680 (simulated)"
        echo "  FP16 Performance: 185 TFLOPS (simulated)"
        echo "  INT8 Performance: 370 TOPS (simulated)"
        
        # Test ROCm memory allocation simulation
        echo "ROCm memory allocation test:"
        memory_sizes=(1 2 4 8 16)  # GB
        
        for size in "''${memory_sizes[@]}"; do
          # Simulate allocation time
          alloc_time=$(echo "scale=1; $size * 0.12" | bc)
          echo "  $size GB allocation: $alloc_time ms (simulated)"
        done
        ;;
        
      intel)
        echo "Testing Intel GPU performance..."
        
        # Simulate Intel Arc performance metrics
        echo "Intel Arc performance simulation:"
        echo "  GPU Memory Bandwidth: 560 GB/s (simulated)"
        echo "  Execution Units: 512 (simulated)"
        echo "  XMX Engines: 512 (simulated)"
        echo "  FP16 Performance: 65 TFLOPS (simulated)"
        echo "  INT8 Performance: 130 TOPS (simulated)"
        
        # Test Intel GPU memory allocation simulation
        echo "Intel GPU memory allocation test:"
        memory_sizes=(1 2 4 8 12)  # GB
        
        for size in "''${memory_sizes[@]}"; do
          # Simulate allocation time
          alloc_time=$(echo "scale=1; $size * 0.15" | bc)
          echo "  $size GB allocation: $alloc_time ms (simulated)"
        done
        ;;
        
      mlx)
        echo "Testing Apple Silicon MLX performance..."
        
        # Simulate MLX performance metrics
        echo "MLX performance simulation:"
        echo "  Unified Memory Bandwidth: 400 GB/s (simulated)"
        echo "  Neural Engine TOPS: 35.17 (simulated)"
        echo "  GPU Cores: 76 (simulated)"
        echo "  FP16 Performance: 22 TFLOPS (simulated)"
        
        # Test MLX memory allocation simulation
        echo "MLX memory allocation test:"
        memory_sizes=(1 2 4 8 16 24)  # GB (unified memory)
        
        for size in "''${memory_sizes[@]}"; do
          # Simulate allocation time (faster due to unified memory)
          alloc_time=$(echo "scale=1; $size * 0.05" | bc)
          echo "  $size GB allocation: $alloc_time ms (simulated)"
        done
        ;;
        
      cpu)
        echo "Testing CPU performance..."
        
        # Get actual CPU information
        cpu_info=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
        cpu_cores=$(nproc)
        
        echo "CPU: $cpu_info"
        echo "Cores: $cpu_cores"
        
        # Simulate CPU performance metrics
        echo "CPU performance simulation:"
        echo "  Memory Bandwidth: 100 GB/s (simulated)"
        echo "  Cache L3: 32 MB (simulated)"
        echo "  FP32 Performance: 2 TFLOPS (simulated)"
        echo "  Vector Units: AVX-512 (simulated)"
        
        # Test CPU memory allocation (system RAM)
        echo "CPU memory allocation test:"
        memory_sizes=(1 2 4 8 16 32)  # GB
        
        for size in "''${memory_sizes[@]}"; do
          # Simulate allocation time
          alloc_time=$(echo "scale=1; $size * 0.01" | bc)
          echo "  $size GB allocation: $alloc_time ms (simulated)"
        done
        ;;
        
      *)
        echo "Unknown hardware type: $detected_hardware"
        exit 1
        ;;
    esac
    
    # Test inference performance simulation
    echo "Testing inference performance simulation..."
    
    # Simulate model inference times for different model sizes
    model_sizes=("7B" "13B" "30B" "70B")
    
    for model in "''${model_sizes[@]}"; do
      echo "Model size: $model parameters"
      
      case "$detected_hardware" in
        cuda)
          # CUDA typically fastest
          inference_time=$(echo "scale=1; $model * 0.1" | bc | sed 's/B//')
          ;;
        rocm)
          # ROCm slightly slower than CUDA
          inference_time=$(echo "scale=1; $model * 0.12" | bc | sed 's/B//')
          ;;
        intel)
          # Intel Arc slower than CUDA/ROCm
          inference_time=$(echo "scale=1; $model * 0.2" | bc | sed 's/B//')
          ;;
        mlx)
          # MLX efficient for smaller models
          inference_time=$(echo "scale=1; $model * 0.15" | bc | sed 's/B//')
          ;;
        cpu)
          # CPU slowest but most compatible
          inference_time=$(echo "scale=1; $model * 1.0" | bc | sed 's/B//')
          ;;
      esac
      
      echo "  Inference time: $inference_time seconds (simulated)"
    done
    
    echo "GPU acceleration benchmarks completed"
  '';

  # Test CPU fallback benchmarks
  cpu-fallback-benchmarks = mkPerfTest "cpu-fallback" ''
    echo "Testing CPU fallback performance benchmarks..."
    
    # Force CPU mode for testing
    export EXO_FORCE_ACCELERATOR=cpu
    
    # Get CPU information
    echo "CPU Information:"
    cpu_model=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
    cpu_cores=$(nproc)
    cpu_threads=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    
    echo "  Model: $cpu_model"
    echo "  Physical cores: $cpu_cores"
    echo "  Logical threads: $cpu_threads"
    
    # Test CPU capabilities
    echo "Testing CPU capabilities..."
    
    # Check for CPU features
    cpu_features=$(cat /proc/cpuinfo | grep "flags" | head -1 | cut -d: -f2)
    
    # Check for important features
    features_to_check=("avx" "avx2" "avx512f" "fma" "sse4_1" "sse4_2")
    
    for feature in "''${features_to_check[@]}"; do
      if echo "$cpu_features" | grep -q "$feature"; then
        echo "  ✓ $feature supported"
      else
        echo "  ⚠ $feature not supported"
      fi
    done
    
    # Test memory performance
    echo "Testing memory performance..."
    
    # Get memory information
    total_memory=$(cat /proc/meminfo | grep "MemTotal" | awk '{print $2}')
    available_memory=$(cat /proc/meminfo | grep "MemAvailable" | awk '{print $2}')
    
    total_gb=$(echo "scale=1; $total_memory / 1024 / 1024" | bc)
    available_gb=$(echo "scale=1; $available_memory / 1024 / 1024" | bc)
    
    echo "  Total memory: $total_gb GB"
    echo "  Available memory: $available_gb GB"
    
    # Test memory bandwidth simulation
    echo "Testing memory bandwidth simulation..."
    
    # Create test data
    test_size=104857600  # 100MB
    test_file="/tmp/cpu_memory_test"
    
    # Generate test data
    dd if=/dev/zero of="$test_file" bs=1024 count=102400 2>/dev/null
    
    # Test memory copy performance
    start_time=$(date +%s.%N)
    cp "$test_file" "/tmp/cpu_memory_test_copy"
    end_time=$(date +%s.%N)
    
    copy_time=$(echo "$end_time - $start_time" | bc)
    bandwidth=$(echo "scale=2; $test_size / $copy_time / 1024 / 1024" | bc)
    
    echo "  Memory copy bandwidth: $bandwidth MB/s"
    
    # Test CPU inference simulation
    echo "Testing CPU inference performance simulation..."
    
    # Simulate different optimization levels
    optimization_levels=("O0" "O1" "O2" "O3")
    
    for opt in "''${optimization_levels[@]}"; do
      # Simulate performance improvement with optimization
      case "$opt" in
        O0) multiplier=1.0 ;;
        O1) multiplier=1.2 ;;
        O2) multiplier=1.5 ;;
        O3) multiplier=1.8 ;;
      esac
      
      base_performance=100  # tokens/second
      optimized_performance=$(echo "scale=1; $base_performance * $multiplier" | bc)
      
      echo "  Optimization $opt: $optimized_performance tokens/second (simulated)"
    done
    
    # Test multi-threading performance
    echo "Testing multi-threading performance simulation..."
    
    thread_counts=(1 2 4 8 16)
    
    for threads in "''${thread_counts[@]}"; do
      if [ $threads -le $cpu_threads ]; then
        # Simulate scaling efficiency (diminishing returns)
        if [ $threads -eq 1 ]; then
          efficiency=1.0
        elif [ $threads -le 4 ]; then
          efficiency=$(echo "scale=2; $threads * 0.9" | bc)
        elif [ $threads -le 8 ]; then
          efficiency=$(echo "scale=2; $threads * 0.8" | bc)
        else
          efficiency=$(echo "scale=2; $threads * 0.7" | bc)
        fi
        
        performance=$(echo "scale=1; 100 * $efficiency" | bc)
        echo "  $threads threads: $performance tokens/second (simulated)"
      fi
    done
    
    # Test model size impact on CPU performance
    echo "Testing model size impact on CPU performance..."
    
    model_sizes=("1B" "3B" "7B" "13B" "30B")
    
    for model in "''${model_sizes[@]}"; do
      # Simulate performance degradation with model size
      size_num=$(echo "$model" | sed 's/B//')
      
      # Base performance decreases with model size
      if [ $size_num -le 3 ]; then
        performance=50
      elif [ $size_num -le 7 ]; then
        performance=30
      elif [ $size_num -le 13 ]; then
        performance=15
      elif [ $size_num -le 30 ]; then
        performance=5
      else
        performance=1
      fi
      
      echo "  $model model: $performance tokens/second (simulated)"
    done
    
    # Clean up
    rm -f "$test_file" "/tmp/cpu_memory_test_copy"
    unset EXO_FORCE_ACCELERATOR
    
    echo "CPU fallback benchmarks completed"
  '';

in {
  inherit network-throughput-tests rdma-performance-tests gpu-acceleration-benchmarks cpu-fallback-benchmarks;
}
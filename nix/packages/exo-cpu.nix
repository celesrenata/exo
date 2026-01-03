{ lib
, stdenv
, exo-python
, makeWrapper
}:

stdenv.mkDerivation rec {
  pname = "exo-cpu";
  version = "0.3.0";

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [ exo-python ];

  installPhase = ''
    mkdir -p $out/bin $out/share/exo

    # Create CPU detection and optimization script
    cat > $out/share/exo/cpu-detect << 'EOF'
    #!/bin/bash
    
    # Detect CPU capabilities and optimize settings
    detect_cpu_features() {
        echo "=== CPU Detection and Optimization ==="
        
        # Get CPU information
        local cpu_info=""
        if [ -f "/proc/cpuinfo" ]; then
            cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
            echo "CPU: $cpu_info"
            
            # Check for CPU features
            local cpu_flags=$(grep "flags" /proc/cpuinfo | head -1 | cut -d: -f2)
            
            # Check for AVX support
            if echo "$cpu_flags" | grep -q "avx2"; then
                echo "AVX2 support: Available"
                export EXO_CPU_FEATURES="avx2"
            elif echo "$cpu_flags" | grep -q "avx"; then
                echo "AVX support: Available"
                export EXO_CPU_FEATURES="avx"
            else
                echo "AVX support: Not available"
                export EXO_CPU_FEATURES="basic"
            fi
            
            # Check for other useful features
            if echo "$cpu_flags" | grep -q "fma"; then
                echo "FMA support: Available"
                export EXO_CPU_FMA="1"
            fi
            
            if echo "$cpu_flags" | grep -q "f16c"; then
                echo "F16C support: Available"
                export EXO_CPU_F16C="1"
            fi
            
        elif command -v sysctl > /dev/null 2>&1; then
            # macOS detection
            cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
            echo "CPU: $cpu_info"
            
            # Check for Apple Silicon
            if echo "$cpu_info" | grep -q "Apple"; then
                echo "Apple Silicon detected - consider using MLX variant"
                export EXO_CPU_FEATURES="apple_silicon"
            fi
        fi
        
        # Get CPU core count
        local cpu_cores=""
        if command -v nproc > /dev/null 2>&1; then
            cpu_cores=$(nproc)
        elif [ -f "/proc/cpuinfo" ]; then
            cpu_cores=$(grep -c "^processor" /proc/cpuinfo)
        elif command -v sysctl > /dev/null 2>&1; then
            cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
        fi
        
        if [ -n "$cpu_cores" ] && [ "$cpu_cores" != "unknown" ]; then
            echo "CPU Cores: $cpu_cores"
            export EXO_CPU_CORES="$cpu_cores"
            
            # Set optimal thread count (usually cores - 1 for system)
            local optimal_threads=$((cpu_cores > 1 ? cpu_cores - 1 : 1))
            export OMP_NUM_THREADS="$optimal_threads"
            export MKL_NUM_THREADS="$optimal_threads"
            export OPENBLAS_NUM_THREADS="$optimal_threads"
            echo "Optimal thread count: $optimal_threads"
        fi
        
        # Memory detection
        local memory_gb=""
        if [ -f "/proc/meminfo" ]; then
            memory_gb=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
        elif command -v sysctl > /dev/null 2>&1; then
            memory_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
        fi
        
        if [ -n "$memory_gb" ]; then
            echo "System Memory: $memory_gb GB"
            export EXO_SYSTEM_MEMORY_GB="$memory_gb"
            
            # Set memory limits based on available RAM
            if [ "$memory_gb" -lt 4 ]; then
                echo "Warning: Low memory system detected. Performance may be limited."
                export EXO_MEMORY_LIMIT="conservative"
            elif [ "$memory_gb" -lt 8 ]; then
                export EXO_MEMORY_LIMIT="moderate"
            else
                export EXO_MEMORY_LIMIT="normal"
            fi
        fi
        
        echo "CPU-only inference configured"
        return 0
    }
    
    # Main detection
    detect_cpu_features
    EOF
    chmod +x $out/share/exo/cpu-detect
    
    # Create wrapper scripts for CPU-only execution with optimization
    makeWrapper ${exo-python}/bin/exo $out/bin/exo \
      --set EXO_HARDWARE_ACCELERATOR "cpu" \
      --set MLX_DISABLE "1" \
      --set CUDA_VISIBLE_DEVICES "" \
      --set HIP_VISIBLE_DEVICES "" \
      --set SYCL_DEVICE_FILTER "host" \
      --run '$out/share/exo/cpu-detect'
    
    makeWrapper ${exo-python}/bin/exo-master $out/bin/exo-master \
      --set EXO_HARDWARE_ACCELERATOR "cpu" \
      --set MLX_DISABLE "1" \
      --set CUDA_VISIBLE_DEVICES "" \
      --set HIP_VISIBLE_DEVICES "" \
      --set SYCL_DEVICE_FILTER "host" \
      --run '$out/share/exo/cpu-detect'
    
    makeWrapper ${exo-python}/bin/exo-worker $out/bin/exo-worker \
      --set EXO_HARDWARE_ACCELERATOR "cpu" \
      --set MLX_DISABLE "1" \
      --set CUDA_VISIBLE_DEVICES "" \
      --set HIP_VISIBLE_DEVICES "" \
      --set SYCL_DEVICE_FILTER "host" \
      --run '$out/share/exo/cpu-detect'

    # Create CPU info script for debugging
    cat > $out/bin/exo-cpu-info << 'EOF'
    #!/bin/bash
    echo "=== EXO CPU Information ==="
    echo "Package: ${exo-python}"
    echo
    echo "=== System Information ==="
    echo "Platform: $(uname -s) $(uname -m)"
    echo "Kernel: $(uname -r)"
    echo
    echo "=== CPU Information ==="
    if [ -f "/proc/cpuinfo" ]; then
        echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "CPU Cores: $(grep -c "^processor" /proc/cpuinfo)"
        echo "CPU Flags: $(grep "flags" /proc/cpuinfo | head -1 | cut -d: -f2 | tr ' ' '\n' | grep -E "(avx|avx2|fma|f16c)" | tr '\n' ' ')"
    elif command -v sysctl > /dev/null 2>&1; then
        echo "CPU Model: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")"
        echo "CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")"
    fi
    echo
    echo "=== Memory Information ==="
    if [ -f "/proc/meminfo" ]; then
        echo "Total Memory: $(awk '/MemTotal/ {print int($2/1024/1024) " GB"}' /proc/meminfo)"
        echo "Available Memory: $(awk '/MemAvailable/ {print int($2/1024/1024) " GB"}' /proc/meminfo)"
    elif command -v sysctl > /dev/null 2>&1; then
        echo "Total Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024) " GB"}')"
    fi
    echo
    echo "=== Thread Configuration ==="
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-not set}"
    echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-not set}"
    echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS:-not set}"
    echo
    echo "=== Python Environment ==="
    python3 -c "
    import sys
    print('Python Version:', sys.version.split()[0])
    
    try:
        import numpy as np
        print('NumPy Version:', np.__version__)
        print('NumPy BLAS Info:', np.show_config())
    except ImportError:
        print('NumPy: Not available')
    
    try:
        import torch
        print('PyTorch Version:', torch.__version__)
        print('PyTorch CPU Threads:', torch.get_num_threads())
    except ImportError:
        print('PyTorch: Not available')
    "
    EOF
    chmod +x $out/bin/exo-cpu-info
  '';

  meta = with lib; {
    description = "EXO distributed AI inference - CPU only with optimizations";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.all;
    
    # Additional metadata
    longDescription = ''
      EXO CPU variant provides optimized CPU-only inference for systems
      without GPU acceleration. Includes automatic CPU feature detection,
      thread optimization, and memory management. Serves as the fallback
      option when no GPU acceleration is available.
    '';
  };
}
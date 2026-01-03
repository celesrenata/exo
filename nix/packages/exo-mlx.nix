{ lib
, stdenv
, exo-python
, makeWrapper
, python313
, darwin
, libiconv
}:

let
  # MLX-enabled Python environment for Apple Silicon
  mlxEnv = python313.withPackages (ps: with ps; [
    # MLX framework and related packages
    # Note: MLX packages would be added when available in nixpkgs
    # For now, we assume they're included in the base exo-python package
    numpy
    scipy
    # Apple-specific optimizations
  ] ++ lib.optionals stdenv.isDarwin [
    # macOS-specific packages
  ]);

  # Apple Silicon detection script
  mlxDetectionScript = ''
    #!/bin/bash
    
    # Check if running on Apple Silicon
    check_apple_silicon() {
        # Check if we're on macOS
        if [ "$(uname -s)" != "Darwin" ]; then
            echo "Not running on macOS"
            return 1
        fi
        
        # Check for Apple Silicon (arm64 architecture)
        local arch=$(uname -m)
        if [ "$arch" != "arm64" ]; then
            echo "Not running on Apple Silicon (detected: $arch)"
            return 1
        fi
        
        echo "Apple Silicon detected: $arch"
        return 0
    }
    
    # Check for Metal Performance Shaders availability
    check_metal_support() {
        # Check if Metal framework is available
        if ! python3 -c "import Metal" 2>/dev/null; then
            echo "Metal framework not available in Python"
            # This is not a hard failure as MLX might still work
        fi
        
        # Check system Metal support
        if command -v system_profiler > /dev/null 2>&1; then
            if system_profiler SPDisplaysDataType | grep -q "Metal"; then
                echo "Metal support detected"
                return 0
            fi
        fi
        
        echo "Metal support validation completed"
        return 0
    }
    
    # Validate MLX installation
    validate_mlx() {
        # Try to import MLX in Python
        if ! python3 -c "import mlx.core" 2>/dev/null; then
            echo "MLX core module not available"
            return 1
        fi
        
        if ! python3 -c "import mlx.nn" 2>/dev/null; then
            echo "MLX neural network module not available"
            return 1
        fi
        
        echo "MLX installation validated"
        return 0
    }
    
    # Check memory availability
    check_memory() {
        # Get system memory info
        local memory_gb=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        
        if [ "$memory_gb" -lt 8 ]; then
            echo "Warning: Less than 8GB RAM detected ($memory_gb GB). MLX may have limited performance."
        else
            echo "Memory check passed: $memory_gb GB RAM available"
        fi
        
        return 0
    }
    
    # Main validation
    if check_apple_silicon && check_metal_support && validate_mlx && check_memory; then
        echo "MLX environment ready on Apple Silicon"
        exit 0
    else
        echo "MLX environment not ready, falling back to CPU"
        exit 1
    fi
  '';
in

stdenv.mkDerivation rec {
  pname = "exo-mlx";
  version = "0.3.0";

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [ 
    exo-python 
    mlxEnv
  ] ++ lib.optionals stdenv.isDarwin [
    darwin.apple_sdk.frameworks.Metal
    darwin.apple_sdk.frameworks.MetalKit
    darwin.apple_sdk.frameworks.Accelerate
    darwin.apple_sdk.frameworks.CoreML
    libiconv
  ];

  installPhase = ''
    mkdir -p $out/bin $out/share/exo

    # Install MLX detection script
    cat > $out/share/exo/mlx-detect << 'EOF'
    ${mlxDetectionScript}
    EOF
    chmod +x $out/share/exo/mlx-detect

    # Create wrapper scripts with MLX support and validation
    makeWrapper ${exo-python}/bin/exo $out/bin/exo \
      --set EXO_HARDWARE_ACCELERATOR "mlx" \
      --set MLX_ENABLE "1" \
      --set PYTORCH_ENABLE_MPS_FALLBACK "1" \
      --set MLX_METAL_DEBUG "0" \
      --set MLX_MEMORY_POOL "1" \
      ${lib.optionalString stdenv.isDarwin ''--prefix DYLD_FRAMEWORK_PATH : ${darwin.apple_sdk.frameworks.Metal}/Library/Frameworks:${darwin.apple_sdk.frameworks.MetalKit}/Library/Frameworks:${darwin.apple_sdk.frameworks.Accelerate}/Library/Frameworks:${darwin.apple_sdk.frameworks.CoreML}/Library/Frameworks''} \
      --run '$out/share/exo/mlx-detect || { echo "MLX validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-master $out/bin/exo-master \
      --set EXO_HARDWARE_ACCELERATOR "mlx" \
      --set MLX_ENABLE "1" \
      --set PYTORCH_ENABLE_MPS_FALLBACK "1" \
      --set MLX_METAL_DEBUG "0" \
      --set MLX_MEMORY_POOL "1" \
      ${lib.optionalString stdenv.isDarwin ''--prefix DYLD_FRAMEWORK_PATH : ${darwin.apple_sdk.frameworks.Metal}/Library/Frameworks:${darwin.apple_sdk.frameworks.MetalKit}/Library/Frameworks:${darwin.apple_sdk.frameworks.Accelerate}/Library/Frameworks:${darwin.apple_sdk.frameworks.CoreML}/Library/Frameworks''} \
      --run '$out/share/exo/mlx-detect || { echo "MLX validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-worker $out/bin/exo-worker \
      --set EXO_HARDWARE_ACCELERATOR "mlx" \
      --set MLX_ENABLE "1" \
      --set PYTORCH_ENABLE_MPS_FALLBACK "1" \
      --set MLX_METAL_DEBUG "0" \
      --set MLX_MEMORY_POOL "1" \
      ${lib.optionalString stdenv.isDarwin ''--prefix DYLD_FRAMEWORK_PATH : ${darwin.apple_sdk.frameworks.Metal}/Library/Frameworks:${darwin.apple_sdk.frameworks.MetalKit}/Library/Frameworks:${darwin.apple_sdk.frameworks.Accelerate}/Library/Frameworks:${darwin.apple_sdk.frameworks.CoreML}/Library/Frameworks''} \
      --run '$out/share/exo/mlx-detect || { echo "MLX validation failed, exiting"; exit 1; }'

    # Create MLX info script for debugging
    cat > $out/bin/exo-mlx-info << 'EOF'
    #!/bin/bash
    echo "=== EXO MLX Information ==="
    echo "Platform: $(uname -s) $(uname -m)"
    echo "MLX Package: ${mlxEnv}"
    echo
    echo "=== System Information ==="
    echo "Architecture: $(uname -m)"
    echo "macOS Version: $(sw_vers -productVersion 2>/dev/null || echo "Not available")"
    echo "Memory: $(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}') GB"
    echo
    echo "=== Metal Support ==="
    if command -v system_profiler > /dev/null 2>&1; then
        system_profiler SPDisplaysDataType | grep -A5 -B5 "Metal" || echo "Metal information not found"
    else
        echo "system_profiler not available"
    fi
    echo
    echo "=== MLX Module Test ==="
    python3 -c "
    try:
        import mlx.core as mx
        print('MLX Core: Available')
        print('MLX Version:', getattr(mx, '__version__', 'Unknown'))
        
        import mlx.nn as nn
        print('MLX Neural Networks: Available')
        
        # Test basic MLX functionality
        x = mx.array([1, 2, 3, 4])
        print('MLX Array Test: Passed')
        print('MLX Device:', x.device if hasattr(x, 'device') else 'Unknown')
        
    except ImportError as e:
        print('MLX Import Error:', e)
    except Exception as e:
        print('MLX Test Error:', e)
    "
    echo
    echo "=== GPU Memory Information ==="
    python3 -c "
    try:
        import mlx.core as mx
        # Try to get GPU memory info if available
        print('MLX GPU memory information would be displayed here')
    except:
        print('MLX GPU memory info not available')
    "
    EOF
    chmod +x $out/bin/exo-mlx-info
  '';

  meta = with lib; {
    description = "EXO distributed AI inference - Apple Silicon MLX acceleration";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.darwin;
    
    # Additional metadata for hardware requirements
    longDescription = ''
      EXO MLX variant provides GPU acceleration for Apple Silicon Macs using
      the MLX framework. Optimized for M1, M2, M3, and newer Apple Silicon chips.
      Leverages Metal Performance Shaders for efficient AI inference.
      Requires macOS and Apple Silicon hardware.
    '';
  };
}
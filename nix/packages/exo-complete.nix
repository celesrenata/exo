{ lib
, stdenv
, makeWrapper
, exo-cpu
, exo-cuda
, exo-rocm
, exo-intel
, exo-mlx ? null
, exo-dashboard
, pciutils
, glxinfo
, kmod
, coreutils
}:

stdenv.mkDerivation rec {
  pname = "exo-complete";
  version = "0.3.0";

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [
    exo-cpu
    exo-cuda
    exo-rocm
    exo-intel
    exo-dashboard
    pciutils
    glxinfo
    kmod
    coreutils
  ] ++ lib.optionals (exo-mlx != null) [ exo-mlx ];

  installPhase = ''
    mkdir -p $out/bin $out/share/exo

    # Copy dashboard
    cp -r ${exo-dashboard}/share/exo/dashboard $out/share/exo/

    # Create comprehensive hardware detection script
    cat > $out/bin/exo-detect-hardware << 'EOF'
    #!/bin/bash

    # Comprehensive hardware detection for EXO
    
    # Logging function
    log() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
    }
    
    # Hardware detection functions
    detect_nvidia_gpu() {
        log "Checking for NVIDIA GPU..."
        
        # Check PCI devices
        if ! ${pciutils}/bin/lspci | grep -i nvidia > /dev/null 2>&1; then
            log "No NVIDIA GPU found in PCI devices"
            return 1
        fi
        
        # Check driver
        if ! ${kmod}/bin/lsmod | grep -q nvidia; then
            log "NVIDIA driver not loaded"
            return 1
        fi
        
        # Check nvidia-smi availability
        if ! command -v nvidia-smi > /dev/null 2>&1; then
            log "nvidia-smi not available"
            return 1
        fi
        
        # Test nvidia-smi functionality
        if ! nvidia-smi > /dev/null 2>&1; then
            log "nvidia-smi failed to execute"
            return 1
        fi
        
        log "NVIDIA GPU detected and functional"
        return 0
    }
    
    detect_amd_gpu() {
        log "Checking for AMD GPU..."
        
        # Check PCI devices
        if ! ${pciutils}/bin/lspci | grep -i amd | grep -i vga > /dev/null 2>&1; then
            log "No AMD GPU found in PCI devices"
            return 1
        fi
        
        # Check driver
        if ! ${kmod}/bin/lsmod | grep -q amdgpu; then
            log "AMD GPU driver not loaded"
            return 1
        fi
        
        # Check device files
        if [ ! -d "/dev/dri" ] || [ ! -e "/dev/kfd" ]; then
            log "AMD GPU device files not found"
            return 1
        fi
        
        log "AMD GPU detected and functional"
        return 0
    }
    
    detect_intel_gpu() {
        log "Checking for Intel GPU..."
        
        # Check for Intel Arc or Xe GPUs
        local intel_gpu_found=false
        
        # Look for Intel Arc discrete GPUs
        if ${pciutils}/bin/lspci | grep -i intel | grep -E "(arc|xe|dg)" > /dev/null 2>&1; then
            intel_gpu_found=true
            log "Intel Arc discrete GPU detected"
        fi
        
        # Look for Intel integrated GPUs with Xe architecture
        if ${pciutils}/bin/lspci | grep -i intel | grep -i vga | grep -E "(xe|iris)" > /dev/null 2>&1; then
            intel_gpu_found=true
            log "Intel Xe integrated GPU detected"
        fi
        
        if [ "$intel_gpu_found" = false ]; then
            log "No Intel Arc/Xe GPU found"
            return 1
        fi
        
        # Check driver
        if ! ${kmod}/bin/lsmod | grep -E "(i915|xe)" > /dev/null 2>&1; then
            log "Intel GPU driver not loaded"
            return 1
        fi
        
        # Check DRM devices
        if [ ! -d "/sys/class/drm" ] || ! ls /dev/dri/render* 2>/dev/null | head -1 > /dev/null; then
            log "Intel GPU render nodes not found"
            return 1
        fi
        
        log "Intel GPU detected and functional"
        return 0
    }
    
    detect_apple_silicon() {
        log "Checking for Apple Silicon..."
        
        # Check if we're on macOS
        if [ "$(uname -s)" != "Darwin" ]; then
            log "Not running on macOS"
            return 1
        fi
        
        # Check for Apple Silicon (arm64 architecture)
        local arch=$(uname -m)
        if [ "$arch" != "arm64" ]; then
            log "Not running on Apple Silicon (detected: $arch)"
            return 1
        fi
        
        # Check for MLX availability
        if ! python3 -c "import mlx.core" 2>/dev/null; then
            log "MLX not available"
            return 1
        fi
        
        log "Apple Silicon with MLX detected"
        return 0
    }
    
    # Priority-based GPU detection
    detect_gpu() {
        # Priority order: CUDA > ROCm > Intel > MLX > CPU
        
        # Check for NVIDIA CUDA (highest priority for performance)
        if detect_nvidia_gpu; then
            echo "cuda"
            return 0
        fi
        
        # Check for AMD ROCm
        if detect_amd_gpu; then
            echo "rocm"
            return 0
        fi
        
        # Check for Intel Arc/Xe
        if detect_intel_gpu; then
            echo "intel"
            return 0
        fi
        
        # Check for Apple Silicon MLX (macOS only)
        if detect_apple_silicon; then
            echo "mlx"
            return 0
        fi
        
        # Default to CPU
        log "No GPU acceleration detected, using CPU"
        echo "cpu"
        return 0
    }
    
    # Environment variable override support
    check_override() {
        if [ -n "$EXO_FORCE_ACCELERATOR" ]; then
            case "$EXO_FORCE_ACCELERATOR" in
                cuda|rocm|intel|mlx|cpu)
                    log "Using forced accelerator: $EXO_FORCE_ACCELERATOR"
                    echo "$EXO_FORCE_ACCELERATOR"
                    return 0
                    ;;
                *)
                    log "Invalid EXO_FORCE_ACCELERATOR value: $EXO_FORCE_ACCELERATOR"
                    ;;
            esac
        fi
        return 1
    }
    
    # Main detection logic
    main() {
        log "Starting EXO hardware detection..."
        
        # Check for environment override first
        if GPU_TYPE=$(check_override); then
            echo "$GPU_TYPE"
            return 0
        fi
        
        # Perform automatic detection
        GPU_TYPE=$(detect_gpu)
        log "Detected GPU type: $GPU_TYPE"
        echo "$GPU_TYPE"
        return 0
    }
    
    # Execute main function
    main "$@"
    EOF
    chmod +x $out/bin/exo-detect-hardware

    # Create main wrapper that uses hardware detection
    cat > $out/bin/exo << 'EOF'
    #!/bin/bash
    
    # Get hardware type
    GPU_TYPE=$($out/bin/exo-detect-hardware)
    
    case $GPU_TYPE in
        cuda)
            exec ${exo-cuda}/bin/exo "$@"
            ;;
        rocm)
            exec ${exo-rocm}/bin/exo "$@"
            ;;
        intel)
            exec ${exo-intel}/bin/exo "$@"
            ;;
        mlx)
            ${lib.optionalString (exo-mlx != null) ''exec ${exo-mlx}/bin/exo "$@"''}
            ${lib.optionalString (exo-mlx == null) ''
            echo "MLX support not available in this build, falling back to CPU"
            exec ${exo-cpu}/bin/exo "$@"
            ''}
            ;;
        *)
            exec ${exo-cpu}/bin/exo "$@"
            ;;
    esac
    EOF
    chmod +x $out/bin/exo

    # Create master wrapper with hardware detection
    cat > $out/bin/exo-master << 'EOF'
    #!/bin/bash
    
    # Get hardware type
    GPU_TYPE=$($out/bin/exo-detect-hardware)
    
    case $GPU_TYPE in
        cuda)
            exec ${exo-cuda}/bin/exo-master "$@"
            ;;
        rocm)
            exec ${exo-rocm}/bin/exo-master "$@"
            ;;
        intel)
            exec ${exo-intel}/bin/exo-master "$@"
            ;;
        mlx)
            ${lib.optionalString (exo-mlx != null) ''exec ${exo-mlx}/bin/exo-master "$@"''}
            ${lib.optionalString (exo-mlx == null) ''
            echo "MLX support not available in this build, falling back to CPU"
            exec ${exo-cpu}/bin/exo-master "$@"
            ''}
            ;;
        *)
            exec ${exo-cpu}/bin/exo-master "$@"
            ;;
    esac
    EOF
    chmod +x $out/bin/exo-master

    # Create worker wrapper with hardware detection
    cat > $out/bin/exo-worker << 'EOF'
    #!/bin/bash
    
    # Get hardware type
    GPU_TYPE=$($out/bin/exo-detect-hardware)
    
    case $GPU_TYPE in
        cuda)
            exec ${exo-cuda}/bin/exo-worker "$@"
            ;;
        rocm)
            exec ${exo-rocm}/bin/exo-worker "$@"
            ;;
        intel)
            exec ${exo-intel}/bin/exo-worker "$@"
            ;;
        mlx)
            ${lib.optionalString (exo-mlx != null) ''exec ${exo-mlx}/bin/exo-worker "$@"''}
            ${lib.optionalString (exo-mlx == null) ''
            echo "MLX support not available in this build, falling back to CPU"
            exec ${exo-cpu}/bin/exo-worker "$@"
            ''}
            ;;
        *)
            exec ${exo-cpu}/bin/exo-worker "$@"
            ;;
    esac
    EOF
    chmod +x $out/bin/exo-worker

    # Create dashboard wrapper
    ln -s ${exo-dashboard}/bin/exo-dashboard $out/bin/exo-dashboard

    # Create comprehensive system info script
    cat > $out/bin/exo-system-info << 'EOF'
    #!/bin/bash
    echo "=== EXO System Information ==="
    echo "Version: ${version}"
    echo "Platform: $(uname -s) $(uname -m)"
    echo "Date: $(date)"
    echo
    
    # Hardware detection
    echo "=== Hardware Detection ==="
    GPU_TYPE=$($out/bin/exo-detect-hardware)
    echo "Selected accelerator: $GPU_TYPE"
    echo
    
    # Show available packages
    echo "=== Available Packages ==="
    echo "CPU: ${exo-cpu}"
    echo "CUDA: ${exo-cuda}"
    echo "ROCm: ${exo-rocm}"
    echo "Intel: ${exo-intel}"
    ${lib.optionalString (exo-mlx != null) ''echo "MLX: ${exo-mlx}"''}
    echo "Dashboard: ${exo-dashboard}"
    echo
    
    # Call specific info script based on detected hardware
    case $GPU_TYPE in
        cuda)
            if [ -x "${exo-cuda}/bin/exo-cuda-info" ]; then
                ${exo-cuda}/bin/exo-cuda-info
            fi
            ;;
        rocm)
            if [ -x "${exo-rocm}/bin/exo-rocm-info" ]; then
                ${exo-rocm}/bin/exo-rocm-info
            fi
            ;;
        intel)
            if [ -x "${exo-intel}/bin/exo-intel-info" ]; then
                ${exo-intel}/bin/exo-intel-info
            fi
            ;;
        mlx)
            ${lib.optionalString (exo-mlx != null) ''
            if [ -x "${exo-mlx}/bin/exo-mlx-info" ]; then
                ${exo-mlx}/bin/exo-mlx-info
            fi
            ''}
            ;;
        *)
            if [ -x "${exo-cpu}/bin/exo-cpu-info" ]; then
                ${exo-cpu}/bin/exo-cpu-info
            fi
            ;;
    esac
    EOF
    chmod +x $out/bin/exo-system-info

    # Create hardware override helper
    cat > $out/bin/exo-set-accelerator << 'EOF'
    #!/bin/bash
    
    usage() {
        echo "Usage: $0 <accelerator>"
        echo "Available accelerators: cuda, rocm, intel, mlx, cpu, auto"
        echo
        echo "Examples:"
        echo "  $0 cuda     # Force CUDA acceleration"
        echo "  $0 cpu      # Force CPU-only mode"
        echo "  $0 auto     # Use automatic detection (default)"
        echo
        echo "This sets the EXO_FORCE_ACCELERATOR environment variable."
        echo "Add 'export EXO_FORCE_ACCELERATOR=<type>' to your shell profile for persistence."
    }
    
    if [ $# -ne 1 ]; then
        usage
        exit 1
    fi
    
    case "$1" in
        cuda|rocm|intel|mlx|cpu)
            echo "Setting EXO accelerator to: $1"
            echo "export EXO_FORCE_ACCELERATOR=$1"
            echo
            echo "Run the above command to set the accelerator for your current session."
            echo "Add it to your ~/.bashrc or ~/.zshrc for persistence."
            ;;
        auto)
            echo "Enabling automatic hardware detection"
            echo "unset EXO_FORCE_ACCELERATOR"
            echo
            echo "Run the above command to enable automatic detection."
            ;;
        *)
            echo "Error: Invalid accelerator '$1'"
            usage
            exit 1
            ;;
    esac
    EOF
    chmod +x $out/bin/exo-set-accelerator
  '';

  meta = with lib; {
    description = "EXO distributed AI inference - Complete package with automatic hardware detection";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.unix;

    longDescription = ''
      EXO complete package provides automatic hardware detection and optimal
      acceleration selection. Supports NVIDIA CUDA, AMD ROCm, Intel Arc GPUs,
      Apple Silicon MLX, and CPU fallback. Includes comprehensive system
      information and debugging tools.
      
      Hardware detection priority: CUDA > ROCm > Intel > MLX > CPU
      
      Use EXO_FORCE_ACCELERATOR environment variable to override detection.
    '';
  };
}

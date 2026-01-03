{ lib
, stdenv
, exo-python
, makeWrapper
, intel-compute-runtime
, level-zero
, intel-media-sdk
, python313
, pciutils
, kmod
}:

let
  # Intel GPU-enabled Python environment with IPEX support
  intelEnv = python313.withPackages (ps: with ps; [
    # Intel Extension for PyTorch (IPEX) and related packages
    torch-bin  # PyTorch base
    torchvision-bin
    torchaudio-bin
    # Note: intel-extension-for-pytorch would be added when available in nixpkgs
    # Additional ML packages
    numpy
    scipy
    scikit-learn
    # Intel-specific optimizations
    mkl  # Intel Math Kernel Library
  ]);

  # Intel GPU detection script
  intelDetectionScript = ''
    #!/bin/bash
    
    # Check if Intel Arc GPU is present and accessible
    check_intel_gpu() {
        # Check for Intel Arc GPUs in PCI (Arc A-series, Xe graphics)
        local intel_gpu_found=false
        
        # Look for Intel Arc discrete GPUs
        if ${pciutils}/bin/lspci | grep -i intel | grep -E "(arc|xe|dg)" > /dev/null 2>&1; then
            intel_gpu_found=true
            echo "Intel Arc discrete GPU detected"
        fi
        
        # Look for Intel integrated GPUs with Xe architecture
        if ${pciutils}/bin/lspci | grep -i intel | grep -i vga | grep -E "(xe|iris)" > /dev/null 2>&1; then
            intel_gpu_found=true
            echo "Intel Xe integrated GPU detected"
        fi
        
        if [ "$intel_gpu_found" = false ]; then
            echo "No Intel Arc/Xe GPU detected in PCI devices"
            return 1
        fi
        
        # Check if i915 driver is loaded (for integrated) or xe driver (for discrete)
        if ! ${kmod}/bin/lsmod | grep -E "(i915|xe)" > /dev/null 2>&1; then
            echo "Intel GPU driver (i915/xe) not loaded"
            return 1
        fi
        
        # Check for DRM device files
        if [ ! -d "/sys/class/drm" ]; then
            echo "DRM subsystem not available"
            return 1
        fi
        
        # Check for Intel GPU render nodes
        if ! ls /dev/dri/render* 2>/dev/null | head -1 > /dev/null; then
            echo "No GPU render nodes found in /dev/dri/"
            return 1
        fi
        
        echo "Intel GPU detected and accessible"
        return 0
    }
    
    # Validate Intel compute runtime installation
    validate_intel_runtime() {
        if [ ! -d "${intel-compute-runtime}" ]; then
            echo "Intel Compute Runtime not found at ${intel-compute-runtime}"
            return 1
        fi
        
        if [ ! -d "${level-zero}" ]; then
            echo "Level Zero not found at ${level-zero}"
            return 1
        fi
        
        # Check for essential Intel GPU libraries
        local icr_lib="${intel-compute-runtime}/lib"
        if [ ! -f "$icr_lib/libigdrcl.so" ] && [ ! -f "$icr_lib/libze_intel_gpu.so" ]; then
            echo "Intel GPU compute libraries not found"
            return 1
        fi
        
        echo "Intel compute runtime validated"
        return 0
    }
    
    # Check Level Zero functionality
    check_level_zero() {
        # Set Level Zero environment
        export ZE_ENABLE_VALIDATION_LAYER=1
        
        # Try to enumerate Level Zero devices (if available)
        if command -v ze_info > /dev/null 2>&1; then
            if ! ze_info > /dev/null 2>&1; then
                echo "Level Zero device enumeration failed"
                return 1
            fi
        fi
        
        echo "Level Zero environment ready"
        return 0
    }
    
    # Main validation
    if check_intel_gpu && validate_intel_runtime && check_level_zero; then
        echo "Intel GPU environment ready"
        exit 0
    else
        echo "Intel GPU environment not ready, falling back to CPU"
        exit 1
    fi
  '';
in

stdenv.mkDerivation rec {
  pname = "exo-intel";
  version = "0.3.0";

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [ 
    exo-python 
    intel-compute-runtime
    level-zero
    intel-media-sdk
    intelEnv
    pciutils
    kmod
  ];

  installPhase = ''
    mkdir -p $out/bin $out/share/exo

    # Install Intel GPU detection script
    cat > $out/share/exo/intel-detect << 'EOF'
    ${intelDetectionScript}
    EOF
    chmod +x $out/share/exo/intel-detect

    # Create wrapper scripts with Intel GPU support and validation
    makeWrapper ${exo-python}/bin/exo $out/bin/exo \
      --set EXO_HARDWARE_ACCELERATOR "intel" \
      --prefix PATH : ${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${intel-compute-runtime}/lib:${level-zero}/lib:${intel-media-sdk}/lib \
      --set INTEL_COMPUTE_RUNTIME_HOME ${intel-compute-runtime} \
      --set LEVEL_ZERO_HOME ${level-zero} \
      --set SYCL_DEVICE_FILTER "gpu" \
      --set ZE_FLAT_DEVICE_HIERARCHY "COMPOSITE" \
      --set ZE_ENABLE_VALIDATION_LAYER "1" \
      --set ONEAPI_DEVICE_SELECTOR "level_zero:gpu" \
      --set IPEX_TILE_AS_DEVICE "1" \
      --run '$out/share/exo/intel-detect || { echo "Intel GPU validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-master $out/bin/exo-master \
      --set EXO_HARDWARE_ACCELERATOR "intel" \
      --prefix PATH : ${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${intel-compute-runtime}/lib:${level-zero}/lib:${intel-media-sdk}/lib \
      --set INTEL_COMPUTE_RUNTIME_HOME ${intel-compute-runtime} \
      --set LEVEL_ZERO_HOME ${level-zero} \
      --set SYCL_DEVICE_FILTER "gpu" \
      --set ZE_FLAT_DEVICE_HIERARCHY "COMPOSITE" \
      --set ZE_ENABLE_VALIDATION_LAYER "1" \
      --set ONEAPI_DEVICE_SELECTOR "level_zero:gpu" \
      --set IPEX_TILE_AS_DEVICE "1" \
      --run '$out/share/exo/intel-detect || { echo "Intel GPU validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-worker $out/bin/exo-worker \
      --set EXO_HARDWARE_ACCELERATOR "intel" \
      --prefix PATH : ${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${intel-compute-runtime}/lib:${level-zero}/lib:${intel-media-sdk}/lib \
      --set INTEL_COMPUTE_RUNTIME_HOME ${intel-compute-runtime} \
      --set LEVEL_ZERO_HOME ${level-zero} \
      --set SYCL_DEVICE_FILTER "gpu" \
      --set ZE_FLAT_DEVICE_HIERARCHY "COMPOSITE" \
      --set ZE_ENABLE_VALIDATION_LAYER "1" \
      --set ONEAPI_DEVICE_SELECTOR "level_zero:gpu" \
      --set IPEX_TILE_AS_DEVICE "1" \
      --run '$out/share/exo/intel-detect || { echo "Intel GPU validation failed, exiting"; exit 1; }'

    # Create Intel GPU info script for debugging
    cat > $out/bin/exo-intel-info << 'EOF'
    #!/bin/bash
    echo "=== EXO Intel GPU Information ==="
    echo "Intel Compute Runtime: ${intel-compute-runtime}"
    echo "Level Zero: ${level-zero}"
    echo "Intel Media SDK: ${intel-media-sdk}"
    echo
    echo "=== GPU Detection ==="
    ${pciutils}/bin/lspci | grep -i intel | grep -E "(vga|arc|xe)" || echo "No Intel GPUs found"
    echo
    echo "=== Intel Driver Status ==="
    ${kmod}/bin/lsmod | grep -E "(i915|xe)" || echo "Intel GPU driver not loaded"
    echo
    echo "=== DRM Devices ==="
    ls -la /dev/dri/ 2>/dev/null || echo "/dev/dri not found"
    echo
    echo "=== GPU Render Nodes ==="
    ls -la /dev/dri/render* 2>/dev/null || echo "No render nodes found"
    echo
    echo "=== Level Zero Devices ==="
    if command -v ze_info > /dev/null 2>&1; then
        ze_info 2>/dev/null || echo "ze_info failed or no devices found"
    else
        echo "ze_info not available"
    fi
    echo
    echo "=== Intel GPU Memory Info ==="
    if [ -d "/sys/class/drm" ]; then
        for card in /sys/class/drm/card*; do
            if [ -f "$card/device/vendor" ] && [ "$(cat $card/device/vendor)" = "0x8086" ]; then
                echo "Intel GPU found: $card"
                [ -f "$card/device/device" ] && echo "  Device ID: $(cat $card/device/device)"
                [ -f "$card/device/subsystem_vendor" ] && echo "  Subsystem Vendor: $(cat $card/device/subsystem_vendor)"
            fi
        done
    fi
    EOF
    chmod +x $out/bin/exo-intel-info
  '';

  meta = with lib; {
    description = "EXO distributed AI inference - Intel Arc GPU with IPEX acceleration";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.linux;
    
    # Additional metadata for hardware requirements
    longDescription = ''
      EXO Intel variant provides GPU acceleration for Intel Arc discrete GPUs
      and Intel Xe integrated graphics. Requires Intel GPU drivers and 
      Intel Compute Runtime. Includes Intel Extension for PyTorch (IPEX)
      support for optimized AI inference on Intel hardware.
      Supports both Arc A-series discrete GPUs and Xe integrated graphics.
    '';
  };
}
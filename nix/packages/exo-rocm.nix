{ lib
, stdenv
, exo-python
, makeWrapper
, rocmPackages
, python313
, pciutils
, kmod
}:

let
  # ROCm-enabled Python environment
  rocmEnv = python313.withPackages (ps: with ps; [
    # ROCm-specific packages
    torch-bin  # PyTorch base (ROCm version would be preferred when available)
    torchvision-bin
    torchaudio-bin
    # Additional ML packages
    numpy
    scipy
    scikit-learn
  ]);

  # ROCm detection script
  rocmDetectionScript = ''
    #!/bin/bash
    
    # Check if AMD GPU is present and accessible
    check_amd_gpu() {
        # Check for AMD devices in PCI
        if ! ${pciutils}/bin/lspci | grep -i amd | grep -i vga > /dev/null 2>&1; then
            echo "No AMD GPU detected in PCI devices"
            return 1
        fi
        
        # Check if amdgpu driver is loaded
        if ! ${kmod}/bin/lsmod | grep -q amdgpu; then
            echo "AMD GPU driver (amdgpu) not loaded"
            return 1
        fi
        
        # Check for ROCm device files
        if [ ! -d "/dev/dri" ] || [ ! -e "/dev/kfd" ]; then
            echo "ROCm device files not found (/dev/dri or /dev/kfd missing)"
            return 1
        fi
        
        # Check if rocm-smi is available and working
        if command -v rocm-smi > /dev/null 2>&1; then
            if ! rocm-smi > /dev/null 2>&1; then
                echo "rocm-smi failed to execute"
                return 1
            fi
        else
            echo "rocm-smi not available, but continuing with basic AMD GPU support"
        fi
        
        echo "AMD GPU detected and accessible"
        return 0
    }
    
    # Validate ROCm installation
    validate_rocm() {
        if [ ! -d "${rocmPackages.rocm-runtime}" ]; then
            echo "ROCm runtime not found at ${rocmPackages.rocm-runtime}"
            return 1
        fi
        
        if [ ! -d "${rocmPackages.hip}" ]; then
            echo "HIP not found at ${rocmPackages.hip}"
            return 1
        fi
        
        # Check for ROCm libraries
        local rocm_lib_path="${rocmPackages.rocm-runtime}/lib:${rocmPackages.hip}/lib:${rocmPackages.rocblas}/lib"
        if [ ! -f "${rocmPackages.hip}/lib/libhip_hcc.so" ] && [ ! -f "${rocmPackages.hip}/lib/libamdhip64.so" ]; then
            echo "HIP libraries not found"
            return 1
        fi
        
        echo "ROCm installation validated"
        return 0
    }
    
    # Main validation
    if check_amd_gpu && validate_rocm; then
        echo "ROCm environment ready"
        exit 0
    else
        echo "ROCm environment not ready, falling back to CPU"
        exit 1
    fi
  '';
in

stdenv.mkDerivation rec {
  pname = "exo-rocm";
  version = "0.3.0";

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [ 
    exo-python 
    rocmPackages.rocm-runtime
    rocmPackages.hip
    rocmPackages.rocblas
    rocmPackages.rocsparse
    rocmPackages.rocfft
    rocmPackages.rocrand
    rocmPackages.miopen
    rocmEnv
    pciutils
    kmod
  ];

  installPhase = ''
    mkdir -p $out/bin $out/share/exo

    # Install ROCm detection script
    cat > $out/share/exo/rocm-detect << 'EOF'
    ${rocmDetectionScript}
    EOF
    chmod +x $out/share/exo/rocm-detect

    # Create wrapper scripts with ROCm support and validation
    makeWrapper ${exo-python}/bin/exo $out/bin/exo \
      --set EXO_HARDWARE_ACCELERATOR "rocm" \
      --prefix PATH : ${rocmPackages.rocm-runtime}/bin:${rocmPackages.hip}/bin:${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${rocmPackages.rocm-runtime}/lib:${rocmPackages.hip}/lib:${rocmPackages.rocblas}/lib:${rocmPackages.rocsparse}/lib:${rocmPackages.rocfft}/lib:${rocmPackages.rocrand}/lib:${rocmPackages.miopen}/lib \
      --set ROCM_HOME ${rocmPackages.rocm-runtime} \
      --set HIP_HOME ${rocmPackages.hip} \
      --set ROCBLAS_HOME ${rocmPackages.rocblas} \
      --set HIP_PLATFORM "amd" \
      --set HSA_OVERRIDE_GFX_VERSION "10.3.0" \
      --set PYTORCH_ROCM_ARCH "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100" \
      --run '$out/share/exo/rocm-detect || { echo "ROCm validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-master $out/bin/exo-master \
      --set EXO_HARDWARE_ACCELERATOR "rocm" \
      --prefix PATH : ${rocmPackages.rocm-runtime}/bin:${rocmPackages.hip}/bin:${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${rocmPackages.rocm-runtime}/lib:${rocmPackages.hip}/lib:${rocmPackages.rocblas}/lib:${rocmPackages.rocsparse}/lib:${rocmPackages.rocfft}/lib:${rocmPackages.rocrand}/lib:${rocmPackages.miopen}/lib \
      --set ROCM_HOME ${rocmPackages.rocm-runtime} \
      --set HIP_HOME ${rocmPackages.hip} \
      --set ROCBLAS_HOME ${rocmPackages.rocblas} \
      --set HIP_PLATFORM "amd" \
      --set HSA_OVERRIDE_GFX_VERSION "10.3.0" \
      --set PYTORCH_ROCM_ARCH "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100" \
      --run '$out/share/exo/rocm-detect || { echo "ROCm validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-worker $out/bin/exo-worker \
      --set EXO_HARDWARE_ACCELERATOR "rocm" \
      --prefix PATH : ${rocmPackages.rocm-runtime}/bin:${rocmPackages.hip}/bin:${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${rocmPackages.rocm-runtime}/lib:${rocmPackages.hip}/lib:${rocmPackages.rocblas}/lib:${rocmPackages.rocsparse}/lib:${rocmPackages.rocfft}/lib:${rocmPackages.rocrand}/lib:${rocmPackages.miopen}/lib \
      --set ROCM_HOME ${rocmPackages.rocm-runtime} \
      --set HIP_HOME ${rocmPackages.hip} \
      --set ROCBLAS_HOME ${rocmPackages.rocblas} \
      --set HIP_PLATFORM "amd" \
      --set HSA_OVERRIDE_GFX_VERSION "10.3.0" \
      --set PYTORCH_ROCM_ARCH "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100" \
      --run '$out/share/exo/rocm-detect || { echo "ROCm validation failed, exiting"; exit 1; }'

    # Create ROCm info script for debugging
    cat > $out/bin/exo-rocm-info << 'EOF'
    #!/bin/bash
    echo "=== EXO ROCm Information ==="
    echo "ROCm Runtime: ${rocmPackages.rocm-runtime}"
    echo "HIP: ${rocmPackages.hip}"
    echo "ROCBlas: ${rocmPackages.rocblas}"
    echo "ROCSparse: ${rocmPackages.rocsparse}"
    echo "ROCFFT: ${rocmPackages.rocfft}"
    echo "ROCRand: ${rocmPackages.rocrand}"
    echo "MIOpen: ${rocmPackages.miopen}"
    echo
    echo "=== GPU Detection ==="
    ${pciutils}/bin/lspci | grep -i amd | grep -i vga || echo "No AMD GPUs found"
    echo
    echo "=== AMD Driver Status ==="
    ${kmod}/bin/lsmod | grep amdgpu || echo "AMD GPU driver not loaded"
    echo
    echo "=== ROCm Device Files ==="
    ls -la /dev/dri/ 2>/dev/null || echo "/dev/dri not found"
    ls -la /dev/kfd 2>/dev/null || echo "/dev/kfd not found"
    echo
    echo "=== rocm-smi Output ==="
    if command -v rocm-smi > /dev/null 2>&1; then
        rocm-smi
    else
        echo "rocm-smi not available"
    fi
    EOF
    chmod +x $out/bin/exo-rocm-info
  '';

  meta = with lib; {
    description = "EXO distributed AI inference - AMD ROCm acceleration with GPU detection";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.linux;
    
    # Additional metadata for hardware requirements
    longDescription = ''
      EXO ROCm variant provides GPU acceleration for AMD graphics cards.
      Requires AMD GPU driver (amdgpu) and ROCm stack installation.
      Includes automatic GPU detection and validation.
      Supports RDNA and CDNA architectures.
    '';
  };
}
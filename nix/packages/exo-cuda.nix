{ lib
, stdenv
, exo-python
, makeWrapper
, cudaPackages
, python313
, pciutils
, linuxPackages
}:

let
  # CUDA-enabled Python environment with PyTorch and related packages
  cudaEnv = python313.withPackages (ps: with ps; [
    # CUDA-specific packages
    torch-bin  # PyTorch with CUDA support
    torchvision-bin
    torchaudio-bin
    # Additional ML packages that benefit from CUDA
    numpy
    scipy
    scikit-learn
  ]);

  # CUDA detection script
  cudaDetectionScript = ''
    #!/bin/bash
    
    # Check if NVIDIA GPU is present and accessible
    check_nvidia_gpu() {
        # Check for NVIDIA devices in PCI
        if ! ${pciutils}/bin/lspci | grep -i nvidia > /dev/null 2>&1; then
            echo "No NVIDIA GPU detected in PCI devices"
            return 1
        fi
        
        # Check if nvidia driver is loaded
        if ! lsmod | grep -q nvidia; then
            echo "NVIDIA driver not loaded"
            return 1
        fi
        
        # Check if nvidia-smi is available and working
        if ! command -v nvidia-smi > /dev/null 2>&1; then
            echo "nvidia-smi not available"
            return 1
        fi
        
        # Test nvidia-smi functionality
        if ! nvidia-smi > /dev/null 2>&1; then
            echo "nvidia-smi failed to execute"
            return 1
        fi
        
        echo "NVIDIA GPU detected and accessible"
        return 0
    }
    
    # Validate CUDA installation
    validate_cuda() {
        if [ ! -d "${cudaPackages.cudatoolkit}" ]; then
            echo "CUDA toolkit not found at ${cudaPackages.cudatoolkit}"
            return 1
        fi
        
        if [ ! -d "${cudaPackages.cudnn}" ]; then
            echo "cuDNN not found at ${cudaPackages.cudnn}"
            return 1
        fi
        
        echo "CUDA installation validated"
        return 0
    }
    
    # Main validation
    if check_nvidia_gpu && validate_cuda; then
        echo "CUDA environment ready"
        exit 0
    else
        echo "CUDA environment not ready, falling back to CPU"
        exit 1
    fi
  '';
in

stdenv.mkDerivation rec {
  pname = "exo-cuda";
  version = "0.3.0";

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [ 
    exo-python 
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.nccl
    cudaPackages.cutensor
    cudaEnv
    pciutils
  ];

  installPhase = ''
    mkdir -p $out/bin $out/share/exo

    # Install CUDA detection script
    cat > $out/share/exo/cuda-detect << 'EOF'
    ${cudaDetectionScript}
    EOF
    chmod +x $out/share/exo/cuda-detect

    # Create wrapper scripts with CUDA support and validation
    makeWrapper ${exo-python}/bin/exo $out/bin/exo \
      --set EXO_HARDWARE_ACCELERATOR "cuda" \
      --prefix PATH : ${cudaPackages.cudatoolkit}/bin:${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${cudaPackages.nccl}/lib:${cudaPackages.cutensor}/lib \
      --set CUDA_HOME ${cudaPackages.cudatoolkit} \
      --set CUDNN_HOME ${cudaPackages.cudnn} \
      --set NCCL_HOME ${cudaPackages.nccl} \
      --set CUTENSOR_HOME ${cudaPackages.cutensor} \
      --set CUDA_VISIBLE_DEVICES "all" \
      --set NVIDIA_VISIBLE_DEVICES "all" \
      --run '$out/share/exo/cuda-detect || { echo "CUDA validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-master $out/bin/exo-master \
      --set EXO_HARDWARE_ACCELERATOR "cuda" \
      --prefix PATH : ${cudaPackages.cudatoolkit}/bin:${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${cudaPackages.nccl}/lib:${cudaPackages.cutensor}/lib \
      --set CUDA_HOME ${cudaPackages.cudatoolkit} \
      --set CUDNN_HOME ${cudaPackages.cudnn} \
      --set NCCL_HOME ${cudaPackages.nccl} \
      --set CUTENSOR_HOME ${cudaPackages.cutensor} \
      --set CUDA_VISIBLE_DEVICES "all" \
      --set NVIDIA_VISIBLE_DEVICES "all" \
      --run '$out/share/exo/cuda-detect || { echo "CUDA validation failed, exiting"; exit 1; }'
    
    makeWrapper ${exo-python}/bin/exo-worker $out/bin/exo-worker \
      --set EXO_HARDWARE_ACCELERATOR "cuda" \
      --prefix PATH : ${cudaPackages.cudatoolkit}/bin:${pciutils}/bin \
      --prefix LD_LIBRARY_PATH : ${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${cudaPackages.nccl}/lib:${cudaPackages.cutensor}/lib \
      --set CUDA_HOME ${cudaPackages.cudatoolkit} \
      --set CUDNN_HOME ${cudaPackages.cudnn} \
      --set NCCL_HOME ${cudaPackages.nccl} \
      --set CUTENSOR_HOME ${cudaPackages.cutensor} \
      --set CUDA_VISIBLE_DEVICES "all" \
      --set NVIDIA_VISIBLE_DEVICES "all" \
      --run '$out/share/exo/cuda-detect || { echo "CUDA validation failed, exiting"; exit 1; }'

    # Create CUDA info script for debugging
    cat > $out/bin/exo-cuda-info << 'EOF'
    #!/bin/bash
    echo "=== EXO CUDA Information ==="
    echo "CUDA Toolkit: ${cudaPackages.cudatoolkit}"
    echo "cuDNN: ${cudaPackages.cudnn}"
    echo "NCCL: ${cudaPackages.nccl}"
    echo "cuTENSOR: ${cudaPackages.cutensor}"
    echo
    echo "=== GPU Detection ==="
    ${pciutils}/bin/lspci | grep -i nvidia || echo "No NVIDIA GPUs found"
    echo
    echo "=== NVIDIA Driver Status ==="
    lsmod | grep nvidia || echo "NVIDIA driver not loaded"
    echo
    echo "=== nvidia-smi Output ==="
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi
    else
        echo "nvidia-smi not available"
    fi
    EOF
    chmod +x $out/bin/exo-cuda-info
  '';

  meta = with lib; {
    description = "EXO distributed AI inference - NVIDIA CUDA acceleration with GPU detection";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.linux;
    
    # Additional metadata for hardware requirements
    longDescription = ''
      EXO CUDA variant provides GPU acceleration for NVIDIA graphics cards.
      Requires NVIDIA driver installation and compatible CUDA-capable GPU.
      Includes automatic GPU detection and validation.
    '';
  };
}
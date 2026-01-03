# Hardware Detection and Validation Tests for EXO NixOS Flake
{ lib
, pkgs
, system
, exo-packages
}:

let
  # Helper function to create test derivations
  mkTest = name: script: pkgs.runCommand "test-${name}" {
    nativeBuildInputs = [ 
      pkgs.bash 
      pkgs.coreutils 
      pkgs.pciutils 
      pkgs.kmod
      pkgs.util-linux
      pkgs.jq
    ];
  } ''
    set -euo pipefail
    
    echo "=== ${name} Hardware Test ==="
    echo "System: ${system}"
    echo "Date: $(date)"
    echo
    
    ${script}
    
    echo "Hardware test completed successfully"
    touch $out
  '';

  # Test GPU detection functionality
  gpu-detection-tests = mkTest "gpu-detection" ''
    echo "Testing GPU detection functionality..."
    
    # Test the hardware detection script
    detection_script="${exo-packages}/exo-complete/bin/exo-detect-hardware"
    
    if [ ! -f "$detection_script" ]; then
      echo "ERROR: Hardware detection script not found at $detection_script"
      exit 1
    fi
    
    echo "✓ Hardware detection script found"
    
    # Test script execution
    echo "Testing hardware detection script execution..."
    
    # Run the detection script
    detected_hardware=$($detection_script)
    echo "Detected hardware: $detected_hardware"
    
    # Validate output format
    case "$detected_hardware" in
      cuda|rocm|intel|mlx|cpu)
        echo "✓ Hardware detection returned valid accelerator type: $detected_hardware"
        ;;
      *)
        echo "ERROR: Invalid hardware detection result: $detected_hardware"
        exit 1
        ;;
    esac
    
    # Test NVIDIA GPU detection logic
    echo "Testing NVIDIA GPU detection logic..."
    
    # Mock NVIDIA GPU presence test
    if lspci 2>/dev/null | grep -i nvidia >/dev/null; then
      echo "✓ NVIDIA GPU detected in system"
      
      # Check for NVIDIA driver
      if lsmod 2>/dev/null | grep nvidia >/dev/null; then
        echo "✓ NVIDIA driver loaded"
      else
        echo "⚠ NVIDIA GPU present but driver not loaded"
      fi
    else
      echo "⚠ No NVIDIA GPU detected (expected in test environment)"
    fi
    
    # Test AMD GPU detection logic
    echo "Testing AMD GPU detection logic..."
    
    if lspci 2>/dev/null | grep -i amd | grep -i vga >/dev/null; then
      echo "✓ AMD GPU detected in system"
      
      # Check for AMD driver
      if lsmod 2>/dev/null | grep amdgpu >/dev/null; then
        echo "✓ AMD GPU driver loaded"
      else
        echo "⚠ AMD GPU present but driver not loaded"
      fi
    else
      echo "⚠ No AMD GPU detected (expected in test environment)"
    fi
    
    # Test Intel GPU detection logic
    echo "Testing Intel GPU detection logic..."
    
    if lspci 2>/dev/null | grep -i intel | grep -E "(arc|xe|iris)" >/dev/null; then
      echo "✓ Intel GPU detected in system"
      
      # Check for Intel driver
      if lsmod 2>/dev/null | grep -E "(i915|xe)" >/dev/null; then
        echo "✓ Intel GPU driver loaded"
      else
        echo "⚠ Intel GPU present but driver not loaded"
      fi
    else
      echo "⚠ No Intel Arc/Xe GPU detected (expected in test environment)"
    fi
    
    # Test Apple Silicon detection (if on macOS)
    echo "Testing Apple Silicon detection..."
    
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
      echo "✓ Running on Apple Silicon"
      
      # Test MLX availability (would need actual MLX package)
      echo "⚠ MLX availability test would require actual MLX installation"
    else
      echo "⚠ Not running on Apple Silicon (expected on ${system})"
    fi
    
    echo "GPU detection tests completed"
  '';

  # Test hardware acceleration setup
  hardware-acceleration-tests = mkTest "hardware-acceleration" ''
    echo "Testing hardware acceleration setup..."
    
    # Test package selection based on hardware
    echo "Testing hardware-specific package selection..."
    
    # Get detected hardware type
    detection_script="${exo-packages}/exo-complete/bin/exo-detect-hardware"
    detected_hardware=$($detection_script)
    
    echo "Detected hardware: $detected_hardware"
    
    # Test that appropriate package variant exists
    case "$detected_hardware" in
      cuda)
        if [ -d "${exo-packages}/exo-cuda" ]; then
          echo "✓ CUDA package variant available"
          
          # Test CUDA-specific executables
          if [ -f "${exo-packages}/exo-cuda/bin/exo" ]; then
            echo "✓ CUDA exo executable found"
          else
            echo "ERROR: CUDA exo executable not found"
            exit 1
          fi
        else
          echo "ERROR: CUDA package variant not found"
          exit 1
        fi
        ;;
      rocm)
        if [ -d "${exo-packages}/exo-rocm" ]; then
          echo "✓ ROCm package variant available"
          
          if [ -f "${exo-packages}/exo-rocm/bin/exo" ]; then
            echo "✓ ROCm exo executable found"
          else
            echo "ERROR: ROCm exo executable not found"
            exit 1
          fi
        else
          echo "ERROR: ROCm package variant not found"
          exit 1
        fi
        ;;
      intel)
        if [ -d "${exo-packages}/exo-intel" ]; then
          echo "✓ Intel package variant available"
          
          if [ -f "${exo-packages}/exo-intel/bin/exo" ]; then
            echo "✓ Intel exo executable found"
          else
            echo "ERROR: Intel exo executable not found"
            exit 1
          fi
        else
          echo "ERROR: Intel package variant not found"
          exit 1
        fi
        ;;
      mlx)
        # MLX package might not be available on all systems
        if [ -d "${exo-packages}/exo-mlx" ]; then
          echo "✓ MLX package variant available"
          
          if [ -f "${exo-packages}/exo-mlx/bin/exo" ]; then
            echo "✓ MLX exo executable found"
          else
            echo "ERROR: MLX exo executable not found"
            exit 1
          fi
        else
          echo "⚠ MLX package variant not available (may not be built for this system)"
        fi
        ;;
      cpu)
        if [ -d "${exo-packages}/exo-cpu" ]; then
          echo "✓ CPU package variant available"
          
          if [ -f "${exo-packages}/exo-cpu/bin/exo" ]; then
            echo "✓ CPU exo executable found"
          else
            echo "ERROR: CPU exo executable not found"
            exit 1
          fi
        else
          echo "ERROR: CPU package variant not found"
          exit 1
        fi
        ;;
      *)
        echo "ERROR: Unknown hardware type: $detected_hardware"
        exit 1
        ;;
    esac
    
    # Test environment variable override functionality
    echo "Testing hardware override functionality..."
    
    # Test forcing CPU mode
    export EXO_FORCE_ACCELERATOR=cpu
    forced_hardware=$($detection_script)
    
    if [ "$forced_hardware" = "cpu" ]; then
      echo "✓ Hardware override to CPU successful"
    else
      echo "ERROR: Hardware override failed, got: $forced_hardware"
      exit 1
    fi
    
    # Test forcing CUDA mode (should work even without hardware)
    export EXO_FORCE_ACCELERATOR=cuda
    forced_hardware=$($detection_script)
    
    if [ "$forced_hardware" = "cuda" ]; then
      echo "✓ Hardware override to CUDA successful"
    else
      echo "ERROR: Hardware override to CUDA failed, got: $forced_hardware"
      exit 1
    fi
    
    # Clean up environment
    unset EXO_FORCE_ACCELERATOR
    
    echo "Hardware acceleration tests completed"
  '';

  # Test fallback scenarios
  fallback-scenario-tests = mkTest "fallback-scenarios" ''
    echo "Testing hardware fallback scenarios..."
    
    # Test CPU fallback when no GPU is available
    echo "Testing CPU fallback scenario..."
    
    # Force CPU mode to test fallback
    export EXO_FORCE_ACCELERATOR=cpu
    
    detection_script="${exo-packages}/exo-complete/bin/exo-detect-hardware"
    fallback_hardware=$($detection_script)
    
    if [ "$fallback_hardware" = "cpu" ]; then
      echo "✓ CPU fallback working correctly"
    else
      echo "ERROR: CPU fallback failed, got: $fallback_hardware"
      exit 1
    fi
    
    # Test that CPU package works
    cpu_executable="${exo-packages}/exo-cpu/bin/exo"
    if [ -f "$cpu_executable" ]; then
      echo "✓ CPU executable available for fallback"
      
      # Test that it can be executed (with timeout to prevent hanging)
      if timeout 10s "$cpu_executable" --help >/dev/null 2>&1; then
        echo "✓ CPU executable runs successfully"
      else
        echo "⚠ CPU executable help command failed or timed out"
      fi
    else
      echo "ERROR: CPU executable not found for fallback"
      exit 1
    fi
    
    # Test invalid accelerator handling
    echo "Testing invalid accelerator handling..."
    
    export EXO_FORCE_ACCELERATOR=invalid_accelerator
    invalid_hardware=$($detection_script)
    
    # Should fall back to automatic detection
    case "$invalid_hardware" in
      cuda|rocm|intel|mlx|cpu)
        echo "✓ Invalid accelerator properly handled, fell back to: $invalid_hardware"
        ;;
      *)
        echo "ERROR: Invalid accelerator not handled properly: $invalid_hardware"
        exit 1
        ;;
    esac
    
    # Test missing driver scenarios
    echo "Testing missing driver scenarios..."
    
    # This would require more complex mocking in a real test environment
    # For now, just verify the detection logic handles missing drivers gracefully
    echo "⚠ Missing driver scenario testing requires hardware-specific setup"
    
    # Clean up environment
    unset EXO_FORCE_ACCELERATOR
    
    echo "Fallback scenario tests completed"
  '';

  # Test driver configuration
  driver-configuration-tests = mkTest "driver-configuration" ''
    echo "Testing driver configuration functionality..."
    
    # Test system information gathering
    echo "Testing system information gathering..."
    
    system_info_script="${exo-packages}/exo-complete/bin/exo-system-info"
    
    if [ ! -f "$system_info_script" ]; then
      echo "ERROR: System info script not found"
      exit 1
    fi
    
    echo "✓ System info script found"
    
    # Run system info script
    system_info_output=$($system_info_script)
    echo "System info output:"
    echo "$system_info_output"
    
    # Verify system info contains expected sections
    if echo "$system_info_output" | grep -q "EXO System Information"; then
      echo "✓ System info header found"
    else
      echo "ERROR: System info header not found"
      exit 1
    fi
    
    if echo "$system_info_output" | grep -q "Hardware Detection"; then
      echo "✓ Hardware detection section found"
    else
      echo "ERROR: Hardware detection section not found"
      exit 1
    fi
    
    if echo "$system_info_output" | grep -q "Available Packages"; then
      echo "✓ Available packages section found"
    else
      echo "ERROR: Available packages section not found"
      exit 1
    fi
    
    # Test accelerator selection helper
    echo "Testing accelerator selection helper..."
    
    accelerator_script="${exo-packages}/exo-complete/bin/exo-set-accelerator"
    
    if [ ! -f "$accelerator_script" ]; then
      echo "ERROR: Accelerator selection script not found"
      exit 1
    fi
    
    echo "✓ Accelerator selection script found"
    
    # Test help output
    help_output=$($accelerator_script 2>&1 || true)
    
    if echo "$help_output" | grep -q "Usage:"; then
      echo "✓ Accelerator script shows usage information"
    else
      echo "ERROR: Accelerator script usage information not found"
      exit 1
    fi
    
    # Test valid accelerator options
    valid_accelerators=(cuda rocm intel mlx cpu auto)
    
    for accel in "''${valid_accelerators[@]}"; do
      output=$($accelerator_script "$accel" 2>&1 || true)
      
      if echo "$output" | grep -q "export EXO_FORCE_ACCELERATOR\|unset EXO_FORCE_ACCELERATOR"; then
        echo "✓ Accelerator $accel handled correctly"
      else
        echo "ERROR: Accelerator $accel not handled correctly"
        exit 1
      fi
    done
    
    # Test invalid accelerator
    invalid_output=$($accelerator_script "invalid" 2>&1 || true)
    
    if echo "$invalid_output" | grep -q "Error: Invalid accelerator"; then
      echo "✓ Invalid accelerator properly rejected"
    else
      echo "ERROR: Invalid accelerator not properly rejected"
      exit 1
    fi
    
    echo "Driver configuration tests completed"
  '';

in {
  inherit gpu-detection-tests hardware-acceleration-tests fallback-scenario-tests driver-configuration-tests;
}
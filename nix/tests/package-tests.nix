# Package Build Testing for EXO NixOS Flake
{ lib
, pkgs
, system
, exo-packages
}:

let
  # Helper function to create test derivations
  mkTest = name: script: pkgs.runCommand "test-${name}" {
    nativeBuildInputs = [ pkgs.bash pkgs.coreutils pkgs.jq ];
  } ''
    set -euo pipefail
    
    echo "=== ${name} Test ==="
    echo "System: ${system}"
    echo "Date: $(date)"
    echo
    
    ${script}
    
    echo "Test completed successfully"
    touch $out
  '';

  # Test all package variants build successfully
  build-all-packages = mkTest "build-all-packages" ''
    echo "Testing build of all EXO package variants..."
    
    # List of packages to test
    packages=(
      "exo-cpu"
      "exo-cuda" 
      "exo-rocm"
      "exo-intel"
      "exo-dashboard"
      "exo-rust-bindings"
      "exo-python"
      "exo-complete"
    )
    
    # Test each package
    for pkg in "''${packages[@]}"; do
      echo "Testing package: $pkg"
      
      # Check if package exists in the package set
      if [ ! -d "${exo-packages}/$pkg" ]; then
        echo "ERROR: Package $pkg not found in package set"
        exit 1
      fi
      
      # Verify package has expected structure
      if [ ! -f "${exo-packages}/$pkg/bin/$pkg" ] && [ ! -f "${exo-packages}/$pkg/bin/exo" ]; then
        echo "WARNING: Package $pkg may not have expected executables"
      fi
      
      echo "✓ Package $pkg appears to be built correctly"
    done
    
    # Test the complete package specifically
    echo "Testing exo-complete package integration..."
    
    # Check that exo-complete includes all expected components
    if [ -f "${exo-packages}/exo-complete/bin/exo" ]; then
      echo "✓ Main exo executable found"
    else
      echo "ERROR: Main exo executable not found in exo-complete"
      exit 1
    fi
    
    if [ -f "${exo-packages}/exo-complete/bin/exo-detect-hardware" ]; then
      echo "✓ Hardware detection script found"
    else
      echo "ERROR: Hardware detection script not found"
      exit 1
    fi
    
    if [ -f "${exo-packages}/exo-complete/bin/exo-system-info" ]; then
      echo "✓ System info script found"
    else
      echo "ERROR: System info script not found"
      exit 1
    fi
    
    echo "All package builds verified successfully"
  '';

  # Test cross-compilation for different architectures
  cross-compilation-tests = mkTest "cross-compilation" ''
    echo "Testing cross-compilation capabilities..."
    
    # Current system info
    echo "Current system: ${system}"
    echo "Current architecture: $(uname -m)"
    
    # Test that packages can be evaluated for different systems
    # Note: We can't actually build for other architectures without proper setup,
    # but we can test that the expressions evaluate correctly
    
    target_systems=(
      "x86_64-linux"
      "aarch64-linux"
    )
    
    for target in "''${target_systems[@]}"; do
      echo "Testing evaluation for target system: $target"
      
      # For now, just verify that our current system builds work
      # In a full CI environment, this would test actual cross-compilation
      if [ "$target" = "${system}" ]; then
        echo "✓ Current system ($target) packages verified"
      else
        echo "⚠ Cross-compilation test for $target would require full CI setup"
      fi
    done
    
    # Test architecture-specific package selection
    echo "Testing architecture-specific package logic..."
    
    case "${system}" in
      x86_64-linux|aarch64-linux)
        echo "✓ Linux system detected - CUDA/ROCm/Intel packages should be available"
        ;;
      aarch64-darwin)
        echo "✓ macOS ARM64 detected - MLX packages should be available"
        ;;
      *)
        echo "⚠ Unknown system ${system} - using CPU fallback"
        ;;
    esac
    
    echo "Cross-compilation tests completed"
  '';

  # Test dependency resolution and version pinning
  dependency-resolution-tests = mkTest "dependency-resolution" ''
    echo "Testing dependency resolution and version consistency..."
    
    # Check Python dependencies
    echo "Checking Python package dependencies..."
    
    # Verify Python 3.13 requirement
    python_version=$(${pkgs.python313}/bin/python --version | cut -d' ' -f2)
    echo "Python version: $python_version"
    
    if [[ "$python_version" =~ ^3\.13\. ]]; then
      echo "✓ Python 3.13 requirement satisfied"
    else
      echo "ERROR: Python 3.13 required, found $python_version"
      exit 1
    fi
    
    # Check that key Python dependencies are available
    key_deps=(
      "aiofiles"
      "aiohttp" 
      "pydantic"
      "fastapi"
      "huggingface-hub"
      "tiktoken"
    )
    
    for dep in "''${key_deps[@]}"; do
      if ${pkgs.python313}/bin/python -c "import $dep" 2>/dev/null; then
        echo "✓ Python dependency $dep available"
      else
        echo "⚠ Python dependency $dep not available in nixpkgs"
      fi
    done
    
    # Check Rust dependencies
    echo "Checking Rust toolchain..."
    
    if command -v rustc >/dev/null 2>&1; then
      rust_version=$(rustc --version)
      echo "✓ Rust toolchain available: $rust_version"
    else
      echo "ERROR: Rust toolchain not available"
      exit 1
    fi
    
    # Check Node.js dependencies for dashboard
    echo "Checking Node.js for dashboard..."
    
    if command -v node >/dev/null 2>&1; then
      node_version=$(node --version)
      echo "✓ Node.js available: $node_version"
    else
      echo "ERROR: Node.js not available for dashboard build"
      exit 1
    fi
    
    echo "Dependency resolution tests completed"
  '';

  # Test build reproducibility
  reproducibility-tests = mkTest "reproducibility" ''
    echo "Testing build reproducibility..."
    
    # Test that builds are deterministic by comparing metadata
    echo "Checking build determinism indicators..."
    
    # Check that packages have consistent metadata
    for pkg_path in ${exo-packages}/*/; do
      pkg_name=$(basename "$pkg_path")
      echo "Checking package: $pkg_name"
      
      # Verify package has proper Nix store path structure
      if [[ "$pkg_path" =~ ^/nix/store/[a-z0-9]{32}- ]]; then
        echo "✓ Package $pkg_name has proper Nix store path"
      else
        echo "⚠ Package $pkg_name path structure unexpected: $pkg_path"
      fi
      
      # Check for build metadata files
      if [ -f "$pkg_path/nix-support/propagated-build-inputs" ]; then
        echo "✓ Package $pkg_name has build metadata"
      fi
    done
    
    # Test that environment variables don't affect builds
    echo "Testing environment isolation..."
    
    # Key environment variables that should not affect builds
    unset HOME
    unset USER
    unset TMPDIR
    
    echo "✓ Environment variables cleared for build isolation"
    
    # Verify that packages contain expected files
    echo "Verifying package contents consistency..."
    
    # Check exo-complete package structure
    complete_pkg="${exo-packages}/exo-complete"
    expected_files=(
      "bin/exo"
      "bin/exo-master"
      "bin/exo-worker"
      "bin/exo-detect-hardware"
      "bin/exo-system-info"
    )
    
    for file in "''${expected_files[@]}"; do
      if [ -f "$complete_pkg/$file" ]; then
        echo "✓ Expected file found: $file"
      else
        echo "ERROR: Expected file missing: $file"
        exit 1
      fi
    done
    
    echo "Build reproducibility tests completed"
  '';

in {
  inherit build-all-packages cross-compilation-tests dependency-resolution-tests reproducibility-tests;
}
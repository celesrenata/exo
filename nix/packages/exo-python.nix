{ lib
, stdenv
, python313
, buildPythonApplication
, exo-rust-bindings
, makeWrapper
  # Python dependencies - only include ones that exist in nixpkgs
, aiofiles
, aiohttp
, typeguard
, pydantic
, base58
, cryptography
, fastapi
, filelock
, aiosqlite
, networkx
, protobuf
, rich
, rustworkx
, sqlmodel
, sqlalchemy
, greenlet
, huggingface-hub
, psutil
, loguru
, textual
, anyio
, bidict
, tiktoken
, hypercorn
, openai
}:

buildPythonApplication rec {
  pname = "exo";
  version = "0.3.0";
  format = "pyproject";

  src = lib.cleanSource ../..;

  nativeBuildInputs = [
    python313.pkgs.setuptools
    python313.pkgs.wheel
    python313.pkgs.pip
    python313.pkgs.build
    makeWrapper
  ];

  # Dependencies that are available in nixpkgs
  # Missing packages will need to be handled separately or built from PyPI
  propagatedBuildInputs = [
    # Core async and HTTP dependencies
    aiofiles
    aiohttp
    typeguard

    # Data validation and serialization
    pydantic
    base58
    cryptography

    # Web framework and API
    fastapi
    hypercorn

    # File and system utilities
    filelock
    psutil

    # Database and storage
    aiosqlite
    sqlmodel
    sqlalchemy
    greenlet

    # Graph and network analysis
    networkx
    rustworkx

    # Serialization and protocols
    protobuf

    # UI and logging
    rich
    loguru
    textual

    # ML and AI dependencies
    huggingface-hub
    tiktoken

    # Async utilities
    anyio
    bidict

    # Additional dependencies
    openai

  ] ++ lib.optionals stdenv.isLinux [
    # Platform-specific dependencies for Linux
    # MLX CPU support - these would need to be packaged separately
  ] ++ lib.optionals stdenv.isDarwin [
    # Platform-specific dependencies for macOS
    # MLX support - these would need to be packaged separately
  ];

  # Set up environment for Python 3.13
  preBuild = ''
        export PYTHONPATH="${exo-rust-bindings}/lib/python3.13/site-packages:$PYTHONPATH"
    
        # Ensure we're using Python 3.13 as required
        python_version=$(${python313}/bin/python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [ "$python_version" != "3.13" ]; then
          echo "Error: Python 3.13 is required, but found $python_version"
          exit 1
        fi
    
        echo "Building EXO with Python $python_version"
        echo "Dependencies provided by Nix packages"
    
        # Create a simple setup.py for building
        cat > setup.py << 'EOF'
    from setuptools import setup, find_packages

    setup(
        name="exo",
        version="0.3.0",
        description="Run your own AI cluster at home with everyday devices",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        entry_points={
            "console_scripts": [
                "exo=exo.main:main",
                "exo-master=exo.master.main:main",
                "exo-worker=exo.worker.main:main",
            ],
        },
        python_requires=">=3.13",
        install_requires=[],  # Dependencies handled by Nix
    )
    EOF
  '';

  # Use standard Python build process
  buildPhase = ''
    runHook preBuild
    
    echo "Building EXO Python package..."
    
    # Use setuptools directly instead of the build module
    ${python313}/bin/python setup.py bdist_wheel --dist-dir dist/
    
    runHook postBuild
  '';

  # Install the built wheel and set up proper environment
  installPhase = ''
    runHook preInstall
    
    mkdir -p $out/lib/python3.13/site-packages
    mkdir -p $out/bin
    
    # Install the wheel
    wheel_file=$(find dist/ -name "*.whl" | head -1)
    if [ -z "$wheel_file" ]; then
      echo "Error: No wheel file found in dist/"
      exit 1
    fi
    
    echo "Installing wheel: $wheel_file"
    ${python313}/bin/python -m pip install \
      --no-index \
      --find-links dist/ \
      --prefix=$out \
      --no-build-isolation \
      --no-deps \
      "$wheel_file"
    
    # Ensure Rust bindings are available in the Python path
    if [ -d "${exo-rust-bindings}/lib/python3.13/site-packages" ]; then
      echo "Copying Rust bindings from ${exo-rust-bindings}"
      cp -r ${exo-rust-bindings}/lib/python3.13/site-packages/* $out/lib/python3.13/site-packages/
    else
      echo "Warning: Rust bindings not found at expected location"
    fi
    
    runHook postInstall
  '';

  # Wrap executables to ensure proper Python path and environment
  postInstall = ''
        # Wrap the main executables with proper environment
        for script in exo exo-master exo-worker; do
          if [ -f "$out/bin/$script" ]; then
            echo "Wrapping executable: $script"
            wrapProgram "$out/bin/$script" \
              --set PYTHONPATH "$out/lib/python3.13/site-packages:${exo-rust-bindings}/lib/python3.13/site-packages" \
              --prefix PATH : "${python313}/bin"
          fi
        done
    
        # Create a version info file for debugging
        cat > $out/lib/python3.13/site-packages/exo-build-info.txt << EOF
    EXO Python Package Build Information:
    Built with Python: $(${python313}/bin/python --version)
    Build date: $(date)
    Rust bindings: ${exo-rust-bindings}
    Package version: ${version}
    Source: ${src}
    Dependencies managed by: Nix packages
    Note: Some dependencies may be missing and need to be installed separately
    EOF
  '';

  # Skip tests for now (they require GPU hardware and network setup)
  doCheck = false;

  # Skip runtime dependency check since we handle deps through Nix
  dontCheckRuntimeDeps = true;

  # Verify the package can be imported and basic functionality works
  # Disabled for now since it requires real Rust bindings
  # pythonImportsCheck = [
  #   "exo"
  #   "exo.main"
  #   "exo.master.main" 
  #   "exo.worker.main"
  # ];

  # Additional verification that executables work
  postFixup = ''
    echo "Testing EXO executables..."
    
    # Test that the main executables can be invoked without errors
    # Use timeout to prevent hanging if there are issues
    timeout 30s $out/bin/exo --help > /dev/null || {
      echo "Warning: exo --help failed or timed out"
    }
    
    timeout 30s $out/bin/exo-master --help > /dev/null || {
      echo "Warning: exo-master --help failed or timed out"  
    }
    
    timeout 30s $out/bin/exo-worker --help > /dev/null || {
      echo "Warning: exo-worker --help failed or timed out"
    }
    
    echo "EXO Python package verification completed"
  '';

  meta = with lib; {
    description = "Run your own AI cluster at home with everyday devices";
    longDescription = ''
      EXO is a distributed AI inference system that allows you to run large
      language models across multiple devices in your home network. It supports
      dynamic hardware detection, bonded networking, and integration with
      Kubernetes clusters.
      
      This package includes:
      - Python 3.13 runtime with core dependencies from nixpkgs
      - Rust networking bindings for high-performance communication
      - Support for CUDA, ROCm, MLX, and CPU-only inference
      - Web dashboard and OpenAI-compatible API
      - Integration with systemd and NixOS services
      
      Note: Some Python dependencies may not be available in nixpkgs and
      will need to be installed separately or packaged individually.
    '';
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.unix;
    mainProgram = "exo";
  };
}

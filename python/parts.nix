{ inputs, ... }:
{
  perSystem =
    { config, self', pkgs, lib, system, ... }:
    let
      # Load workspace from uv.lock
      workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = inputs.self;
      };

      # Create overlay from workspace
      # Use wheels from PyPI for most packages; we override mlx with our pure Nix Metal build
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      # Override overlay to inject Nix-built components
      exoOverlay = final: prev: {
        # Replace workspace exo_pyo3_bindings with Nix-built wheel
        exo-pyo3-bindings = pkgs.stdenv.mkDerivation {
          pname = "exo-pyo3-bindings";
          version = "0.1.0";
          src = self'.packages.exo_pyo3_bindings;
          # Install from pre-built wheel
          nativeBuildInputs = [ final.pyprojectWheelHook ];
          dontStrip = true;
        };
      };

      python = pkgs.python313;

      # Overlay to provide build systems and custom packages
      buildSystemsOverlay = final: prev: {
        # Stub out MLX on Linux (not available)
      } // lib.optionalAttrs pkgs.stdenv.isLinux {
        mlx = pkgs.runCommand "mlx-stub" {} ''
          mkdir -p $out/lib/python3.13/site-packages
          touch $out/lib/python3.13/site-packages/mlx.py
        '';
        mlx-lm = pkgs.runCommand "mlx-lm-stub" {} ''
          mkdir -p $out/lib/python3.13/site-packages
          touch $out/lib/python3.13/site-packages/mlx_lm.py
        '';
      } // lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
        # Use our pure Nix-built MLX with Metal support (macOS only)
        mlx = self'.packages.mlx;
        # mlx-lm is a git dependency that needs setuptools (macOS only)
        mlx-lm = prev.mlx-lm.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
        });
      } // {

        # tinygrad with Intel backend support
        tinygrad = prev.tinygrad.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            final.setuptools
          ];
          propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ lib.optionals pkgs.stdenv.isLinux [
            # Add pyopencl for OpenCL support on Linux
            final.pyopencl
          ];
        });

        # pyopencl needs OpenCL headers and libraries
        pyopencl = prev.pyopencl.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ lib.optionals pkgs.stdenv.isLinux [
            pkgs.opencl-headers
          ];
          buildInputs = (old.buildInputs or [ ]) ++ lib.optionals pkgs.stdenv.isLinux [
            pkgs.ocl-icd
          ];
        });
      };

      pythonSet = (pkgs.callPackage inputs.pyproject-nix.build.packages {
        inherit python;
      }).overrideScope (
        lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.default
          overlay
          exoOverlay
          buildSystemsOverlay
        ]
      );
      exoVenv = pythonSet.mkVirtualEnv "exo-env" workspace.deps.default;

      # Virtual environment with dev dependencies for testing
      testVenv = pythonSet.mkVirtualEnv "exo-test-env" (
        workspace.deps.default // {
          exo = [ "dev" ]; # Include pytest, pytest-asyncio, pytest-env
        }
      );

      mkPythonScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ exoVenv ];
        runtimeEnv = {
          EXO_DASHBOARD_DIR = self'.packages.dashboard;
          EXO_RESOURCES_DIR = inputs.self + /resources;
        };
        text = ''exec python ${path} "$@"'';
      };

      benchVenv = pythonSet.mkVirtualEnv "exo-bench-env" {
        exo-bench = [ ];
      };

      mkBenchScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ benchVenv ];
        text = ''exec python ${path} "$@"'';
      };

      mkSimplePythonScript = name: path: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ pkgs.python313 ];
        text = ''exec python ${path} "$@"'';
      };

      exoPackage = pkgs.runCommand "exo"
        {
          nativeBuildInputs = [ pkgs.makeWrapper ];
        }
        ''
          mkdir -p $out/bin

          # Create wrapper script
          makeWrapper ${exoVenv}/bin/exo $out/bin/exo \
            --set EXO_DASHBOARD_DIR ${self'.packages.dashboard} \
            --set EXO_RESOURCES_DIR ${inputs.self + /resources} \
            ${lib.optionalString pkgs.stdenv.hostPlatform.isDarwin "--prefix PATH : ${pkgs.macmon}/bin"}
        '';
    in
    {
      # Python packages
      packages = {
        # exo package - use different build methods per platform
        exo = if pkgs.stdenv.isLinux then
          # On Linux: use buildPythonApplication with explicit deps (like main branch)
          python.pkgs.buildPythonApplication {
            pname = "exo";
            version = "0.3.0";
            format = "pyproject";
            src = inputs.self;
            
            # Patch pyproject.toml to use setuptools instead of uv_build
            postPatch = ''
              sed -i 's/requires = \["uv_build.*"\]/requires = ["setuptools>=61.0", "wheel"]/' pyproject.toml
              sed -i 's/build-backend = "uv_build"/build-backend = "setuptools.build_meta"/' pyproject.toml
            '';
            
            nativeBuildInputs = [ python.pkgs.setuptools python.pkgs.wheel python.pkgs.pip pkgs.makeWrapper ];
            
            propagatedBuildInputs = with python.pkgs; [
              aiofiles
              aiohttp
              pydantic
              fastapi
              filelock
              rustworkx
              huggingface-hub
              psutil
              loguru
              anyio
              tiktoken
              hypercorn
              httpx
              toml
              pillow
              safetensors
              transformers
              tinygrad
              numpy
            ];
            
            # Install Rust bindings after main package
            postInstall = ''
              echo "Installing Rust bindings..."
              for wheel in ${self'.packages.exo_pyo3_bindings}/*.whl; do
                if [ -f "$wheel" ]; then
                  echo "Extracting wheel: $wheel"
                  ${python.pkgs.pip}/bin/pip install --no-deps --no-build-isolation --target $out/lib/python3.13/site-packages "$wheel"
                  break
                fi
              done
            '';
            
            # Rust bindings will be installed in postInstall
            preBuild = ''
              echo "Building exo package..."
            '';
            
            # Skip tests and dependency checks
            doCheck = false;
            dontUsePythonCatchConflicts = true;
            dontUsePythonImportsCheck = true;
            
            # Override the runtime deps check hook to skip it
            pythonRuntimeDepsCheckHook = pkgs.writeShellScript "skip-runtime-deps-check" ''
              echo "Skipping Python runtime dependency checking for Nix build"
            '';
            
            # Set environment variables
            makeWrapperArgs = [
              "--set EXO_TINYGRAD_ENABLED true"
            ];
          }
        else
          # On macOS: use the uv2nix approach with MLX
          exoPackage;
          
        # Test environment for running pytest outside of Nix sandbox (needs GPU access)
        exo-test-env = testVenv;
        exo-bench = mkBenchScript "exo-bench" (inputs.self + /bench/exo_bench.py);
        exo-get-all-models-on-cluster = mkSimplePythonScript "exo-get-all-models-on-cluster" (inputs.self + /tests/get_all_models_on_cluster.py);
      };

      checks = {
        # Ruff linting (works on all platforms)
        lint = pkgs.runCommand "ruff-lint" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}
          touch $out
        '';
      };
    };
}

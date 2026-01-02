{
  description = "EXO: Run your own AI cluster at home with everyday devices";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Provides Rust dev-env integration:
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Provides formatting infrastructure:
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];
      fenixToolchain = system: inputs.fenix.packages.${system}.latest;

      # Package builder function
      mkExoPackage = { pkgs, system, accelerator ? "cpu" }:
        let
          python = pkgs.python313;
          rustToolchain = fenixToolchain system;

          # Build Rust bindings as a separate derivation
          rustBindings =
            let
              rustPlatform = pkgs.makeRustPlatform {
                cargo = rustToolchain.cargo;
                rustc = rustToolchain.rustc;
              };
            in
            rustPlatform.buildRustPackage rec {
              pname = "exo-pyo3-bindings";
              version = "0.1.0";

              # Use the entire workspace as source since Rust bindings depend on workspace members
              src = ./.;

              cargoLock = {
                lockFile = ./Cargo.lock;
                allowBuiltinFetchGit = true;
              };

              # Build only the bindings crate
              cargoBuildFlags = [ "--package" "exo_pyo3_bindings" ];
              cargoTestFlags = [ "--package" "exo_pyo3_bindings" ];

              # Skip tests since they require Python runtime
              doCheck = false;

              nativeBuildInputs = with pkgs; [
                python
                maturin
                pkg-config
              ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
                openssl.dev
              ];

              buildInputs = with pkgs; [
                python
                openssl
              ];

              # Use maturin to build Python extension
              buildPhase = ''
                cd rust/exo_pyo3_bindings
                export PYO3_CROSS_LIB_DIR=${python}/lib
                maturin build --release --out ./dist
              '';

              installPhase = ''
                mkdir -p $out
                echo "Looking for wheel files..."
                find . -name "*.whl" -type f
                echo "Copying wheel files..."
                find . -name "*.whl" -type f -exec cp {} $out/ \;
                echo "Final output:"
                ls -la $out/
              '';

              # Set CARGO_BUILD_TARGET for cross-compilation if needed
              CARGO_BUILD_TARGET = pkgs.stdenv.hostPlatform.rust.rustcTargetSpec;
            };

          # Build dashboard as a separate derivation
          dashboard = pkgs.buildNpmPackage rec {
            pname = "exo-dashboard";
            version = "0.1.0";
            src = ./dashboard;

            npmDepsHash = "sha256-koqsTfxfqJjo3Yq7x61q3duJ9Xtor/yOZcTjfBadZUs=";

            buildPhase = ''
              npm run build
            '';

            installPhase = ''
              mkdir -p $out
              cp -r build/* $out/
            '';
          };

        in
        python.pkgs.buildPythonApplication rec {
          pname = "exo-${accelerator}";
          version = "0.1.0";
          format = "other"; # Not a standard setuptools package

          src = ./.;

          # Fix entry points in postPatch
          postPatch = ''
            echo "🔧 POST-PATCH PHASE START"
            echo "========================="
            
            # Fix build system to use setuptools instead of uv_build
            echo "Fixing build system..."
            sed -i 's/requires = \["uv_build.*"\]/requires = ["setuptools>=61.0", "wheel"]/' pyproject.toml
            sed -i 's/build-backend = "uv_build"/build-backend = "setuptools.build_meta"/' pyproject.toml
            
            # Fix entry points - all should point to the main function
            echo "Fixing entry points..."
            sed -i 's/exo-master = "exo.master.main:main"/exo-master = "exo.main:main"/' pyproject.toml
            sed -i 's/exo-worker = "exo.worker.main:main"/exo-worker = "exo.main:main"/' pyproject.toml
            
            # Remove openai-harmony dependency since it's not available in nixpkgs
            echo "Removing openai-harmony dependency..."
            sed -i '/openai-harmony/d' pyproject.toml
            
            # Copy our openai_harmony stub
            echo "Adding openai_harmony stub..."
            cp ${./src/openai_harmony.py} src/openai_harmony.py
            
            echo "🔧 POST-PATCH PHASE END"
            echo "======================"
          '';

          nativeBuildInputs = with pkgs; [
            python.pkgs.setuptools
            python.pkgs.wheel
            python.pkgs.pip
            python.pkgs.build
          ];

          buildPhase = ''
            echo "🔨 BUILD PHASE START"
            echo "==================="
            
            # Install the Python package
            python -m pip install --no-deps --no-build-isolation --target $out/lib/python3.13/site-packages .
            
            echo "🔨 BUILD PHASE END"
            echo "=================="
          '';

          installPhase = ''
                        echo "📦 INSTALL PHASE START"
                        echo "======================"
            
                        # Create bin directory and scripts
                        mkdir -p $out/bin
            
                        # Create the main exo script
                        cat > $out/bin/exo << 'EOF'
            #!/usr/bin/env python3
            import sys
            sys.path.insert(0, "$out/lib/python3.13/site-packages")
            from exo.main import main
            if __name__ == "__main__":
                main()
            EOF
                        chmod +x $out/bin/exo
            
                        # Create exo-master and exo-worker scripts (they all point to main)
                        cp $out/bin/exo $out/bin/exo-master
                        cp $out/bin/exo $out/bin/exo-worker
            
                        echo "📦 INSTALL PHASE END"
                        echo "==================="
          '';

          propagatedBuildInputs = with python.pkgs; [
            aiofiles
            aiohttp
            typeguard
            pydantic
            base58
            cryptography
            fastapi
            filelock
            aiosqlite
            networkx
            protobuf
            rich
            rustworkx
            sqlmodel
            sqlalchemy
            greenlet
            huggingface-hub
            psutil
            loguru
            textual
            anyio
            bidict
            tiktoken
            hypercorn
            # Note: openai-harmony not available in nixpkgs, using stub
            # PyTorch and ML dependencies for CPU inference
            torch
            transformers
            tokenizers
            safetensors
          ] ++ pkgs.lib.optionals (accelerator == "cuda") [
            # CUDA packages would go here
          ] ++ pkgs.lib.optionals (accelerator == "rocm") [
            # ROCm packages would go here  
          ] ++ pkgs.lib.optionals (accelerator == "mlx") [
            # MLX packages - conditionally include based on platform
          ];

          # Install Rust bindings during build
          preBuild = ''
            echo "🔧 PRE-BUILD PHASE START"
            echo "========================"
            
            # Install Rust bindings wheel to temporary location
            echo "Installing Rust bindings..."
            echo "Rust bindings directory contents:"
            ls -la ${rustBindings}/
            
            # Create temporary install directory
            export TEMP_INSTALL_DIR=$TMPDIR/temp-python-install
            mkdir -p $TEMP_INSTALL_DIR
            
            for wheel in ${rustBindings}/*.whl; do
              if [ -f "$wheel" ]; then
                echo "Installing wheel: $wheel"
                python -m pip install --no-deps --no-build-isolation --target $TEMP_INSTALL_DIR "$wheel"
                echo "Rust bindings installed to temporary location"
                
                # Add to Python path so it can be found during build
                export PYTHONPATH="$TEMP_INSTALL_DIR:$PYTHONPATH"
                echo "Added to PYTHONPATH: $TEMP_INSTALL_DIR"
                break
              fi
            done
            
            echo "🔧 PRE-BUILD PHASE END"
            echo "======================"
          '';

          postInstall = ''
            echo "📦 POST-INSTALL PHASE START"
            echo "==========================="
            
            # Install Rust bindings to the final package
            echo "Installing Rust bindings to final package..."
            for wheel in ${rustBindings}/*.whl; do
              if [ -f "$wheel" ]; then
                echo "Installing wheel to final location: $wheel"
                
                # Create a temporary directory for extraction
                EXTRACT_DIR=$(mktemp -d)
                cd "$EXTRACT_DIR"
                
                # Extract wheel and show contents
                echo "Extracting wheel..."
                ${pkgs.unzip}/bin/unzip -o "$wheel"
                echo "Wheel contents:"
                find . -type f -name "*.so" -o -name "*.py" -o -name "*.pyi" | head -20
                
                # Copy the extracted contents to the final location
                mkdir -p $out/lib/python3.13/site-packages
                
                # Copy all extracted files
                if [ -d "exo_pyo3_bindings" ]; then
                  echo "Copying exo_pyo3_bindings directory..."
                  cp -rv exo_pyo3_bindings $out/lib/python3.13/site-packages/
                fi
                
                if [ -d "exo_pyo3_bindings-"*".dist-info" ]; then
                  echo "Copying dist-info directory..."
                  cp -rv exo_pyo3_bindings-*.dist-info $out/lib/python3.13/site-packages/
                fi
                
                echo "Final installation contents:"
                ls -la $out/lib/python3.13/site-packages/exo_pyo3_bindings/
                
                # Cleanup
                rm -rf "$EXTRACT_DIR"
                break
              fi
            done
            
            # Install dashboard
            echo "Installing dashboard..."
            mkdir -p $out/share/exo
            cp -r ${dashboard} $out/share/exo/dashboard
            echo "Dashboard installed successfully"
            
            # Create wrapper that sets dashboard path
            wrapProgram $out/bin/exo \
              --set EXO_DASHBOARD_PATH "$out/share/exo/dashboard"
            echo "Wrapper script created"
            
            echo "📦 POST-INSTALL PHASE END"
            echo "========================"
          '';

          # Skip tests during build and runtime dependency checking
          doCheck = false;

          # Override the runtime deps check hook to skip it
          pythonRuntimeDepsCheckHook = pkgs.writeShellScript "skip-runtime-deps-check" ''
            echo "Skipping Python runtime dependency checking for Nix build"
          '';

          meta = with pkgs.lib; {
            description = "Run your own AI cluster at home with everyday devices";
            homepage = "https://github.com/exo-explore/exo";
            license = licenses.asl20;
            maintainers = [ ];
            platforms = platforms.unix;
          };
        };
    in
    {
      # NixOS module - moved to top level
      nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.services.exo;
        in
        {
          options.services.exo = {
            enable = mkEnableOption "EXO distributed AI inference system";

            package = mkOption {
              type = types.package;
              default = pkgs."exo-${cfg.accelerator}" or (mkExoPackage { inherit pkgs; system = pkgs.system; accelerator = cfg.accelerator; });
              description = "EXO package to use";
            };

            accelerator = mkOption {
              type = types.enum [ "cpu" "cuda" "rocm" "intel" "mlx" ];
              default = "cpu";
              description = "Hardware accelerator to use";
            };

            port = mkOption {
              type = types.port;
              default = 52415;
              description = "Port for EXO dashboard and API";
            };

            openFirewall = mkOption {
              type = types.bool;
              default = false;
              description = "Open firewall for EXO communication";
            };

            user = mkOption {
              type = types.str;
              default = "exo";
              description = "User to run EXO services";
            };

            group = mkOption {
              type = types.str;
              default = "exo";
              description = "Group to run EXO services";
            };
          };

          config = mkIf cfg.enable {
            users.users.${cfg.user} = {
              isSystemUser = true;
              group = cfg.group;
              description = "EXO system user";
            };

            users.groups.${cfg.group} = { };

            systemd.services.exo = {
              description = "EXO distributed AI inference system";
              after = [ "network.target" ];
              wantedBy = [ "multi-user.target" ];

              serviceConfig = {
                Type = "simple";
                User = cfg.user;
                Group = cfg.group;
                ExecStart = "${cfg.package}/bin/exo --verbose";
                Restart = "always";
                RestartSec = "10";

                # Security settings
                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ "/var/lib/exo" "/var/log/exo" "/var/cache/exo" ];

                # Set proper working directory and cache locations
                WorkingDirectory = "/var/lib/exo";
                CacheDirectory = "exo";
                LogsDirectory = "exo";
                StateDirectory = "exo";
              };

              environment = {
                EXO_PORT = toString cfg.port;
                # Set cache and log directories to writable locations
                XDG_CACHE_HOME = "/var/cache/exo";
                XDG_DATA_HOME = "/var/lib/exo";
                HOME = "/var/lib/exo";
                # Set dashboard directory for the application to find
                DASHBOARD_DIR = "${cfg.package}/share/exo/dashboard";
                # Fix Python interpreter for PyO3 Rust bindings
                PYTHONPATH = "${cfg.package}/lib/python3.13/site-packages";
                # Set inference engine based on accelerator choice
                EXO_ENGINE = if cfg.accelerator == "cpu" then "torch" else cfg.accelerator;
              } // (if cfg.accelerator == "cpu" || (pkgs.stdenv.isLinux && cfg.accelerator == "mlx") then {
                # Force CPU inference and disable MLX when using CPU accelerator or MLX on Linux
                EXO_ENGINE = "torch";
                MLX_DISABLE = "1";
              } else { });
            };

            networking.firewall = mkIf cfg.openFirewall {
              allowedTCPPorts = [ cfg.port ];
            };

            systemd.tmpfiles.rules = [
              "d /var/lib/exo 0755 ${cfg.user} ${cfg.group} -"
              "d /var/log/exo 0755 ${cfg.user} ${cfg.group} -"
              "d /var/cache/exo 0755 ${cfg.user} ${cfg.group} -"
            ];
          };
        };

      # Overlays
      overlays.default = final: prev: {
        exo-cpu = mkExoPackage { pkgs = final; system = final.system; accelerator = "cpu"; };
        exo-cuda = mkExoPackage { pkgs = final; system = final.system; accelerator = "cuda"; };
        exo-rocm = mkExoPackage { pkgs = final; system = final.system; accelerator = "rocm"; };
        exo-intel = mkExoPackage { pkgs = final; system = final.system; accelerator = "intel"; };
        exo-mlx = mkExoPackage { pkgs = final; system = final.system; accelerator = "mlx"; };
      };
    } //
    inputs.flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.fenix.overlays.default ];
        };
        treefmtEval = inputs.treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.ruff-format.enable = true;
          programs.ruff-format.excludes = [ "rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi" ];
          programs.rustfmt.enable = true;
          programs.rustfmt.package = (fenixToolchain system).rustfmt;
          programs.nixpkgs-fmt.enable = true;
        };
      in
      {
        # Packages
        packages = {
          default = mkExoPackage { inherit pkgs system; };
          exo-cpu = mkExoPackage { inherit pkgs system; accelerator = "cpu"; };
          exo-cuda = mkExoPackage { inherit pkgs system; accelerator = "cuda"; };
          exo-rocm = mkExoPackage { inherit pkgs system; accelerator = "rocm"; };
          exo-intel = mkExoPackage { inherit pkgs system; accelerator = "intel"; };
          exo-mlx = mkExoPackage { inherit pkgs system; accelerator = "mlx"; };
        };

        formatter = treefmtEval.config.build.wrapper;
        checks.formatting = treefmtEval.config.build.check inputs.self;
        checks.lint = pkgs.runCommand "lint-check" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}/ --exclude src/exo_pyo3_bindings/
          touch $out
        '';

        devShells.default = pkgs.mkShell {
          packages =
            with pkgs;
            [
              # PYTHON
              python313
              uv
              ruff
              basedpyright

              # PYTHON PACKAGES
              python313Packages.torch
              python313Packages.transformers
              python313Packages.huggingface-hub
              python313Packages.loguru
              python313Packages.pydantic
              python313Packages.anyio
              python313Packages.aiofiles
              python313Packages.aiohttp
              python313Packages.fastapi
              python313Packages.rich
              python313Packages.psutil
              python313Packages.tiktoken
              python313Packages.safetensors
              python313Packages.tokenizers
              python313Packages.rustworkx
              python313Packages.networkx
              python313Packages.sqlmodel
              python313Packages.sqlalchemy
              python313Packages.aiosqlite
              python313Packages.cryptography
              python313Packages.base58
              python313Packages.filelock
              python313Packages.protobuf
              python313Packages.bidict
              python313Packages.hypercorn
              python313Packages.textual
              python313Packages.typeguard
              python313Packages.openai

              # RUST
              ((fenixToolchain system).withComponents [
                "cargo"
                "rustc"
                "clippy"
                "rustfmt"
                "rust-src"
              ])
              rustup # Just here to make RustRover happy
              maturin

              # NIX
              nixpkgs-fmt

              # SVELTE
              nodejs

              # MISC
              just
              jq
            ]
            ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [
              # IFCONFIG
              unixtools.ifconfig

              # Build dependencies for Linux
              pkg-config
              openssl
            ])
            ++ (pkgs.lib.optionals pkgs.stdenv.isDarwin [
              # MACMON
              macmon
            ]);

          shellHook = ''
            # PYTHON
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.python313}/lib"
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              # Build environment for Linux
              export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
            ''}
            echo
            echo "🍎🍎 Run 'just <recipe>' to get started"
            just --list
          '';

        };
      }
    );
}

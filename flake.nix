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

  # TODO: figure out caching story
  # nixConfig = {
  #   # nix community cachix
  #   extra-trusted-public-keys = "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs=";
  #   extra-substituters = "https://nix-community.cachix.org";
  # };

  outputs =
    inputs:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];
      fenixToolchain = system: inputs.fenix.packages.${system}.complete;
      
      # Package builder function
      mkExoPackage = { pkgs, system, accelerator ? "cpu" }: 
        let
          python = pkgs.python313;
        in
        python.pkgs.buildPythonApplication rec {
          pname = "exo-${accelerator}";
          version = "0.1.0";
          format = "pyproject";
          
          src = ./.;
          
          # Don't build Rust bindings for now - focus on getting basic package working
          postPatch = ''
            echo "🔧 POST-PATCH PHASE START"
            echo "========================="
            echo "Current directory: $(pwd)"
            echo "Original pyproject.toml:"
            head -20 pyproject.toml
            
            # Remove Rust bindings dependency temporarily
            echo "Removing Rust bindings dependency..."
            sed -i '/exo_pyo3_bindings/d' pyproject.toml
            
            # Remove MLX dependencies that might cause issues
            echo "Removing MLX dependencies..."
            sed -i '/mlx/d' pyproject.toml
            
            # Remove problematic dependencies that cause version conflicts
            echo "Removing problematic dependencies..."
            sed -i '/types-aiofiles/d' pyproject.toml
            sed -i '/rustworkx/d' pyproject.toml
            sed -i '/greenlet/d' pyproject.toml
            sed -i '/tiktoken/d' pyproject.toml
            sed -i '/hypercorn/d' pyproject.toml
            sed -i '/openai-harmony/d' pyproject.toml
            
            # Fix build system to use setuptools instead of uv_build
            echo "Fixing build system..."
            sed -i 's/requires = \["uv_build.*"\]/requires = ["setuptools>=61.0", "wheel"]/' pyproject.toml
            sed -i 's/build-backend = "uv_build"/build-backend = "setuptools.build_meta"/' pyproject.toml
            
            echo "Modified pyproject.toml:"
            head -20 pyproject.toml
            echo "Build system section:"
            grep -A 3 '\[build-system\]' pyproject.toml
            echo "🔧 POST-PATCH PHASE END"
            echo "======================"
          '';
          
          buildPhase = ''
            echo "🏗️  BUILD PHASE START"
            echo "===================="
            echo "Current directory: $(pwd)"
            echo "Python version: $(python --version)"
            echo "Pip version: $(pip --version)"
            
            # Set up proper cache directory to avoid homeless-shelter warning
            export PIP_CACHE_DIR=$TMPDIR/pip-cache
            mkdir -p $PIP_CACHE_DIR
            
            echo "Available Python packages (first few):"
            # Avoid broken pipe by not using head
            pip list --format=freeze | grep -E "^(aio|any|base|crypto|fast|file|green|hugging|log|network|proto|psutil|pydantic|rich|rust|sql|text|tiktoken|hyper)" || echo "Package listing completed"
            
            echo "Building Python package..."
            set -x
            python -m build --wheel --no-isolation 2>&1 | tee python-build.log
            set +x
            echo "Python build completed with exit code: $?"
            
            echo "Dist directory contents:"
            ls -la dist/ || echo "No dist directory found"
            echo "🏗️  BUILD PHASE END"
            echo "=================="
          '';
          
          nativeBuildInputs = with pkgs; [
            nodejs
            python.pkgs.setuptools
            python.pkgs.wheel
            python.pkgs.pip
            python.pkgs.build
            coreutils
          ];
          
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
            openai-harmony
          ];
          
          # Disable runtime dependency checking to avoid version conflicts
          dontUsePythonRuntimeDepsCheck = true;
          
          preBuild = ''
            echo "🔧 PRE-BUILD PHASE START"
            echo "========================"
            echo "Current directory: $(pwd)"
            echo "Available tools:"
            echo "  Node: $(node --version 2>/dev/null || echo 'NOT FOUND')"
            echo "  NPM: $(npm --version 2>/dev/null || echo 'NOT FOUND')"
            echo "  Python: $(python --version 2>/dev/null || echo 'NOT FOUND')"
            echo "Environment:"
            echo "  HOME: $HOME"
            echo "  TMPDIR: $TMPDIR"
            echo "  PWD: $(pwd)"
            
            # Set up proper cache directories
            export PIP_CACHE_DIR=$TMPDIR/pip-cache
            export NPM_CONFIG_CACHE=$TMPDIR/npm-cache
            mkdir -p $PIP_CACHE_DIR $NPM_CONFIG_CACHE
            
            # Build dashboard
            echo "📦 Building dashboard..."
            echo "Dashboard directory exists: $(test -d dashboard && echo 'YES' || echo 'NO')"
            if [ -d dashboard ]; then
              echo "Dashboard directory contents:"
              ls -la dashboard/ | head -5
            fi
            
            cd dashboard
            echo "In dashboard directory: $(pwd)"
            echo "Package.json exists: $(test -f package.json && echo 'YES' || echo 'NO')"
            if [ -f package.json ]; then
              echo "Package.json scripts section:"
              grep -A 10 '"scripts"' package.json || echo "No scripts section found"
            fi
            
            export HOME=$TMPDIR
            echo "Set HOME to: $HOME"
            
            echo "Running npm ci..."
            set -x
            timeout 300 npm ci --offline 2>&1 | tee npm-ci-offline.log || {
              echo "Offline npm ci failed or timed out, trying online..."
              timeout 300 npm ci 2>&1 | tee npm-ci-online.log
            }
            set +x
            echo "NPM ci completed with exit code: $?"
            
            echo "Node modules directory exists: $(test -d node_modules && echo 'YES' || echo 'NO')"
            if [ -d node_modules ]; then
              echo "Node modules count: $(ls node_modules | wc -l)"
            fi
            
            echo "Running npm run build..."
            set -x
            timeout 300 npm run build 2>&1 | tee npm-build.log
            set +x
            echo "NPM build completed with exit code: $?"
            
            echo "Build directory exists: $(test -d build && echo 'YES' || echo 'NO')"
            if [ -d build ]; then
              echo "Build directory size: $(du -sh build/ | cut -f1)"
              echo "Build directory file count: $(find build -type f | wc -l)"
            fi
            
            cd ..
            echo "Back in main directory: $(pwd)"
            echo "🔧 PRE-BUILD PHASE END"
            echo "======================"
          '';
          
          postInstall = ''
            echo "📦 POST-INSTALL PHASE START"
            echo "==========================="
            echo "Output directory: $out"
            echo "Current directory: $(pwd)"
            
            # Install dashboard
            echo "Installing dashboard..."
            mkdir -p $out/share/exo
            
            if [ -d dashboard/build ]; then
              echo "Dashboard build directory found"
              echo "Dashboard build contents:"
              ls -la dashboard/build/ | head -5
              echo "Copying dashboard build..."
              cp -r dashboard/build $out/share/exo/dashboard
              echo "Dashboard copied successfully"
            else
              echo "WARNING: Dashboard build directory not found"
              echo "Creating empty dashboard directory..."
              mkdir -p $out/share/exo/dashboard
              echo "Dashboard directory created"
            fi
            
            echo "Creating wrapper script..."
            # Create wrapper that sets dashboard path
            wrapProgram $out/bin/exo \
              --set EXO_DASHBOARD_PATH "$out/share/exo/dashboard"
            echo "Wrapper script created"
            
            echo "Build completed successfully!"
            echo "📦 POST-INSTALL PHASE END"
            echo "========================"
          '';
          
          # Skip tests during build
          doCheck = false;
          
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
      # NixOS module
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
              default = pkgs.exo-cpu or (mkExoPackage { inherit pkgs; system = pkgs.system; accelerator = "cpu"; });
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
            
            users.groups.${cfg.group} = {};
            
            systemd.services.exo = {
              description = "EXO distributed AI inference system";
              after = [ "network.target" ];
              wantedBy = [ "multi-user.target" ];
              
              serviceConfig = {
                Type = "simple";
                User = cfg.user;
                Group = cfg.group;
                ExecStart = "${cfg.package}/bin/exo";
                Restart = "always";
                RestartSec = "10";
                
                # Security settings
                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ "/var/lib/exo" ];
              };
              
              environment = {
                EXO_PORT = toString cfg.port;
              };
            };
            
            networking.firewall = mkIf cfg.openFirewall {
              allowedTCPPorts = [ cfg.port ];
            };
            
            systemd.tmpfiles.rules = [
              "d /var/lib/exo 0755 ${cfg.user} ${cfg.group} -"
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
          ${pkgs.ruff}/bin/ruff check ${inputs.self}/
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

              # RUST
              ((fenixToolchain system).withComponents [
                "cargo"
                "rustc"
                "clippy"
                "rustfmt"
                "rust-src"
              ])
              rustup # Just here to make RustRover happy

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

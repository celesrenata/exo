{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Pinned nixpkgs for swift-format (swift is broken on x86_64-linux in newer nixpkgs)
    nixpkgs-swift.url = "github:NixOS/nixpkgs/08dacfca559e1d7da38f3cf05f1f45ee9bfd213c";
  };

  nixConfig = {
    extra-trusted-public-keys = "exo.cachix.org-1:okq7hl624TBeAR3kV+g39dUFSiaZgLRkLsFBCuJ2NZI=";
    extra-substituters = "https://exo.cachix.org";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule
      ];

      flake.packages = inputs.nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-darwin" "aarch64-linux" ] (system:
        let
          pkgs = import inputs.nixpkgs { inherit system; };
          python = pkgs.python313;
        in
        {
          default = python.pkgs.buildPythonApplication {
            pname = "exo";
            version = "0.1.0";
            format = "pyproject";
            src = inputs.self;

            nativeBuildInputs = with pkgs; [
              python.pkgs.setuptools
              python.pkgs.wheel
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
            ];

            pythonImportsCheck = [ "exo" ];
          };
        }
      );

      flake.nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.services.exo;
        in
        {
          options.services.exo = {
            enable = mkEnableOption "EXO distributed AI inference system";

            package = mkOption {
              type = types.nullOr types.package;
              default = inputs.self.packages.${pkgs.system}.default or null;
              description = "EXO package to use (defaults to flake package if available)";
            };

            accelerator = mkOption {
              type = types.enum [ "cpu" "cuda" "rocm" "intel" "mlx" ];
              default = "cpu";
              description = "Hardware accelerator to use (for compatibility, not used by module)";
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
            assertions = [
              {
                assertion = cfg.package != null;
                message = "services.exo.package must be set to a valid EXO package";
              }
            ];

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

                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ "/var/lib/exo" "/var/log/exo" "/var/cache/exo" ];

                WorkingDirectory = "/var/lib/exo";
                CacheDirectory = "exo";
                LogsDirectory = "exo";
                StateDirectory = "exo";
              };

              environment = {
                EXO_PORT = toString cfg.port;
                XDG_CACHE_HOME = "/var/cache/exo";
                XDG_DATA_HOME = "/var/lib/exo";
                HOME = "/var/lib/exo";
              };
            };

            networking.firewall = mkIf cfg.openFirewall {
              allowedTCPPorts = [ cfg.port ];
            };
          };
        };

      perSystem =
        { config, inputs', pkgs, lib, system, ... }:
        let
          fenixToolchain = inputs'.fenix.packages.complete;
          # Use pinned nixpkgs for swift-format (swift is broken on x86_64-linux in newer nixpkgs)
          pkgsSwift = import inputs.nixpkgs-swift { inherit system; };
        in
        {
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixpkgs-fmt.enable = true;
              ruff-format = {
                enable = true;
                excludes = [ "rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi" ];
              };
              rustfmt = {
                enable = true;
                package = fenixToolchain.rustfmt;
              };
              prettier = {
                enable = true;
                includes = [ "*.ts" ];
              };
              swift-format = {
                enable = true;
                package = pkgsSwift.swiftPackages.swift-format;
              };
            };
          };

          checks.lint = pkgs.runCommand "lint-check" { } ''
            export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
            ${pkgs.ruff}/bin/ruff check ${inputs.self}/
            touch $out
          '';

          devShells.default = with pkgs; pkgs.mkShell {
            packages =
              [
                # FORMATTING
                config.treefmt.build.wrapper

                # PYTHON
                python313
                uv
                ruff
                basedpyright

                # RUST
                (fenixToolchain.withComponents [
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
              ${lib.optionalString pkgs.stdenv.isLinux ''
                # Build environment for Linux
                export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
                export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
              ''}
              echo
              echo "🍎🍎 Run 'just <recipe>' to get started"
              just --list
            '';
          };
        };
    };
}

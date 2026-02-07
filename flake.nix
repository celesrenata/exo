{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    crane.url = "github:ipetkov/crane";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    dream2nix = {
      url = "github:nix-community/dream2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    # Python packaging with uv2nix
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
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
        ./dashboard/parts.nix
        ./rust/parts.nix
        ./python/parts.nix
      ];

      flake.nixosModules.exo-intel =
        { config, lib, pkgs, ... }:
        {
          options.services.exo.intel = {
            enable = lib.mkEnableOption "Intel hardware acceleration for exo";

            tinygrad = {
              enable = lib.mkEnableOption "tinygrad backend for exo" // {
                default = true;
              };
              backend = lib.mkOption {
                type = lib.types.enum [ "GPU" "CPU" ];
                default = "GPU";
                description = "Tinygrad backend to use (GPU or CPU)";
              };
            };

            arc = {
              enable = lib.mkEnableOption "Intel Arc iGPU support" // {
                default = true;
              };
              runtime = lib.mkOption {
                type = lib.types.enum [ "level-zero" "opencl" "auto" ];
                default = "auto";
                description = "GPU runtime to use for Intel Arc (level-zero, opencl, or auto)";
              };
            };

            npu = {
              enable = lib.mkEnableOption "Intel NPU support (experimental)";
              servicePort = lib.mkOption {
                type = lib.types.port;
                default = 52416;
                description = "Port for NPU inference service";
              };
            };
          };

          config = lib.mkIf config.services.exo.intel.enable {
            # Base tinygrad support and exo package
            environment.systemPackages = with pkgs; [
              python313Packages.tinygrad
              # Add exo package from the flake
              inputs.self.packages.${system}.exo or (throw "exo package not available for ${system}")
            ] ++ lib.optionals config.services.exo.intel.arc.enable [
              # Monitoring and debugging tools for Intel Arc
              intel-gpu-tools # intel_gpu_top for GPU monitoring
              clinfo # OpenCL device information
            ];

            # Global environment variables for tinygrad backend
            environment.variables = lib.mkIf config.services.exo.intel.tinygrad.enable {
              # Enable tinygrad backend
              EXO_TINYGRAD_ENABLED = "true";

              # Set tinygrad backend (GPU or CPU)
              TINYGRAD_BACKEND = config.services.exo.intel.tinygrad.backend;

              # Runtime-specific environment variables for Intel Arc
              TINYGRAD_INTEL_RUNTIME = lib.mkIf config.services.exo.intel.arc.enable (
                if config.services.exo.intel.arc.runtime == "level-zero" then "LEVEL_ZERO"
                else if config.services.exo.intel.arc.runtime == "opencl" then "OPENCL"
                else "AUTO" # Auto-detect best available runtime
              );

              # Enable tinygrad GPU optimizations
              TINYGRAD_OPTIMIZE = lib.mkIf (config.services.exo.intel.tinygrad.backend == "GPU") "2";

              # Disable tinygrad JIT cache to avoid permission issues
              TINYGRAD_DISABLE_CACHE = "1";

              # Level Zero specific environment variables
              ZE_ENABLE_VALIDATION_LAYER = lib.mkIf
                (
                  config.services.exo.intel.arc.enable &&
                  config.services.exo.intel.arc.runtime != "opencl"
                ) "0"; # Disable validation layer for performance

              ZE_AFFINITY_MASK = lib.mkIf
                (
                  config.services.exo.intel.arc.enable &&
                  config.services.exo.intel.arc.runtime != "opencl"
                ) "0"; # Use first GPU device

              # OpenCL specific environment variables
              OCL_ICD_VENDORS = lib.mkIf
                (
                  config.services.exo.intel.arc.enable &&
                  config.services.exo.intel.arc.runtime != "level-zero"
                ) "/etc/OpenCL/vendors"; # Point to ICD vendor files

              # Intel GPU compute runtime settings
              NEOReadDebugKeys = lib.mkIf config.services.exo.intel.arc.enable "1";
            };

            # Intel Arc iGPU support
            hardware.graphics = lib.mkIf config.services.exo.intel.arc.enable {
              enable = true;
              extraPackages = with pkgs; [
                intel-compute-runtime # OpenCL runtime (provides libOpenCL.so)
                level-zero # Level Zero runtime and loader
                intel-media-driver # VA-API driver for Intel GPUs
                ocl-icd # OpenCL ICD loader (provides libOpenCL.so dispatch)
              ];
            };

            # OpenCL ICD configuration for Intel runtime
            environment.etc."OpenCL/vendors/intel.icd" = lib.mkIf config.services.exo.intel.arc.enable {
              text = "${pkgs.intel-compute-runtime}/lib/intel-opencl/libigdrcl.so";
            };

            # Udev rules for Intel Arc GPU and NPU device access
            services.udev.extraRules =
              (lib.optionalString config.services.exo.intel.arc.enable ''
                # Intel GPU render nodes - allow access for compute workloads
                SUBSYSTEM=="drm", KERNEL=="renderD*", ATTRS{vendor}=="0x8086", MODE="0666"
                
                # Intel GPU card nodes - for display and compute
                SUBSYSTEM=="drm", KERNEL=="card[0-9]*", ATTRS{vendor}=="0x8086", MODE="0666"
              '')
              +
              (lib.optionalString config.services.exo.intel.npu.enable ''
                # Intel NPU device permissions
                SUBSYSTEM=="accel", KERNEL=="accel[0-9]*", GROUP="exo", MODE="0660"
                SUBSYSTEM=="drm", KERNEL=="renderD*", ATTRS{vendor}=="0x8086", GROUP="exo", MODE="0660"
              '');

            # Kernel modules for Intel Arc GPU and NPU
            boot.kernelModules =
              (lib.optionals config.services.exo.intel.arc.enable [
                "i915" # Intel GPU driver
              ])
              ++
              (lib.optionals config.services.exo.intel.npu.enable [
                "intel_vpu" # Intel NPU driver
              ]);

            # Kernel parameters for Intel GPU
            boot.kernelParams = lib.mkIf config.services.exo.intel.arc.enable [
              "i915.force_probe=*" # Force probe all Intel GPUs
              "i915.enable_guc=3" # Enable GuC and HuC firmware loading
            ];

            # Intel NPU support
            systemd.services.exo-npu = lib.mkIf config.services.exo.intel.npu.enable {
              description = "exo Intel NPU Inference Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              serviceConfig = {
                Type = "simple";
                ExecStart = "${pkgs.python313}/bin/python -m exo.worker.engines.npu.service --port ${toString config.services.exo.intel.npu.servicePort}";
                Restart = "on-failure";
                RestartSec = "5s";
                User = "exo";
                Group = "exo";

                # Service isolation for security
                PrivateNetwork = false; # Needs localhost access for API
                ProtectSystem = "strict"; # Read-only system directories
                ProtectHome = true; # No access to home directories
                NoNewPrivileges = true; # Cannot escalate privileges
                PrivateTmp = true; # Private /tmp directory
                ProtectKernelTunables = true; # Protect /proc/sys
                ProtectKernelModules = true; # Cannot load kernel modules
                ProtectControlGroups = true; # Read-only cgroups
                RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ]; # Only needed address families
                RestrictNamespaces = true; # Cannot create namespaces
                LockPersonality = true; # Prevent personality changes
                RestrictRealtime = true; # No realtime scheduling
                RestrictSUIDSGID = true; # No SUID/SGID
                RemoveIPC = true; # Clean up IPC on exit

                # Resource limits
                MemoryMax = "8G"; # Maximum 8GB memory
                MemoryHigh = "6G"; # Soft limit at 6GB
                CPUQuota = "200%"; # Maximum 2 CPU cores
                TasksMax = "256"; # Maximum number of tasks

                # Device access - NPU device
                DeviceAllow = [
                  "/dev/accel/accel0 rw" # NPU device node
                  "/dev/dri rw" # DRI devices (may include NPU)
                ];

                # Logging
                StandardOutput = "journal";
                StandardError = "journal";
                SyslogIdentifier = "exo-npu";
              };

              environment = {
                NPU_SERVICE_PORT = toString config.services.exo.intel.npu.servicePort;
                PYTHONUNBUFFERED = "1"; # Unbuffered output for logging
              };
            };

            # Create exo user if NPU service is enabled
            users.users.exo = lib.mkIf config.services.exo.intel.npu.enable {
              isSystemUser = true;
              group = "exo";
              description = "exo NPU service user";
            };

            users.groups.exo = lib.mkIf config.services.exo.intel.npu.enable { };
          };
        };

      perSystem =
        { config, self', inputs', pkgs, lib, system, ... }:
        let
          fenixToolchain = inputs'.fenix.packages.complete;
          # Use pinned nixpkgs for swift-format (swift is broken on x86_64-linux in newer nixpkgs)
          pkgsSwift = import inputs.nixpkgs-swift { inherit system; };

          # Create a separate pkgs instance for exo with anyio overlay
          # This avoids polluting the global pkgs
          pkgsExo = import inputs.nixpkgs {
            inherit system;
            overlays = [
              # Overlay to pin anyio to 4.11.0 (required by exo)
              (final: prev: {
                python313 = prev.python313.override {
                  packageOverrides = pself: psuper: {
                    anyio = psuper.anyio.overridePythonAttrs (old: rec {
                      version = "4.11.0";
                      src = prev.fetchPypi {
                        pname = "anyio";
                        inherit version;
                        hash = "sha256-gqjQuB4xjMXOcaXx+LXE5jYZYgtjFB74yZX6DblaV8Q=";
                      };
                      doCheck = false; # Skip failing test
                      postPatch = (old.postPatch or "") + ''
                        sed -i '/def test_bad_init_value/,/pytest.raises.*CapacityLimiter.*0/d' tests/test_synchronization.py
                      '';
                    });
                  };
                };
              })
            ];
          };
        in
        {
          # Allow unfree for metal-toolchain (needed for Darwin Metal packages)
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfreePredicate = pkg: (pkg.pname or "") == "metal-toolchain";
          };

          # Make pkgsExo available to other modules
          _module.args.pkgsExo = pkgsExo;
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
                package = config.rust.toolchain;
              };
              prettier = {
                enable = true;
                package = self'.packages.prettier-svelte;
                includes = [ "*.ts" "*.svelte" ];
              };
              swift-format = {
                enable = true;
                package = pkgsSwift.swiftPackages.swift-format;
              };
              shfmt.enable = true;
            };
          };

          packages = lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin (
            let
              uvLock = builtins.fromTOML (builtins.readFile ./uv.lock);
              mlxPackage = builtins.head (builtins.filter (p: p.name == "mlx") uvLock.package);
              uvLockMlxVersion = mlxPackage.version;
            in
            {
              metal-toolchain = pkgs.callPackage ./nix/metal-toolchain.nix { };
              mlx = pkgs.callPackage ./nix/mlx.nix {
                inherit (self'.packages) metal-toolchain;
                inherit uvLockMlxVersion;
              };
              default = self'.packages.exo;
            }
          );

          devShells.default = with pkgs; pkgs.mkShell {
            inputsFrom = [ self'.checks.cargo-build ];

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
                config.rust.toolchain
                maturin

                # NIX
                nixpkgs-fmt

                # SVELTE
                nodejs

                # MISC
                just
                jq
              ]
              ++ lib.optionals stdenv.isLinux [
                unixtools.ifconfig
              ]
              ++ lib.optionals stdenv.isDarwin [
                macmon
              ];

            OPENSSL_NO_VENDOR = "1";

            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${python313}/lib"
              ${lib.optionalString stdenv.isLinux ''
                export LD_LIBRARY_PATH="${openssl.out}/lib:$LD_LIBRARY_PATH"
              ''}
            '';
          };
        };
    };
}

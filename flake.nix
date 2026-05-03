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

    # Intel PyTorch and XPU packages
    nixos-mordrag = {
      url = "github:MordragT/nixos";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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

            pytorch_xpu = {
              enable = lib.mkEnableOption "PyTorch XPU backend for exo" // {
                default = false;
              };
              preferredBackend = lib.mkOption {
                type = lib.types.bool;
                default = false;
                description = "Use PyTorch XPU as the preferred backend over tinygrad";
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
              enable = lib.mkEnableOption "Intel NPU support (experimental)" // {
                default = false;
              };
              servicePort = lib.mkOption {
                type = lib.types.port;
                default = 52416;
                description = "Port for NPU inference service";
              };
            };
          };

          config = lib.mkIf config.services.exo.intel.enable {
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

            # Additional system packages for PyTorch XPU support
            environment.systemPackages = lib.mkIf config.services.exo.intel.enable (
              with pkgs; [
                # Add exo package from the flake (includes patched tinygrad)
                inputs.self.packages.${system}.exo or (throw "exo package not available for ${system}")
              ] ++ lib.optionals config.services.exo.intel.arc.enable [
                # Monitoring and debugging tools for Intel Arc
                intel-gpu-tools # intel_gpu_top for GPU monitoring
                clinfo # OpenCL device information
                # Intel compute runtime and Level Zero for PyTorch XPU
                intel-compute-runtime
                level-zero
              ]
            );

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

              # PyTorch XPU environment variables
              EXO_PYTORCH_XPU_ENABLED = lib.mkIf config.services.exo.intel.pytorch_xpu.enable "true";
              PYTORCH_ENABLE_XPU = lib.mkIf config.services.exo.intel.pytorch_xpu.enable "1";
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

            # Main exo service
            systemd.services.exo = lib.mkIf config.services.exo.intel.enable {
              description = "exo Distributed AI Inference Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              serviceConfig = {
                Type = "simple";
                ExecStart = "${inputs.self.packages.${pkgs.system}.exo}/bin/exo -vv";
                Restart = "on-failure";
                RestartSec = "5s";
                User = "root"; # Needs root for GPU access
                Group = "root";

                # Environment variables
                Environment = [
                  "EXO_TINYGRAD_ENABLED=true"
                  "LD_LIBRARY_PATH=${pkgs.ocl-icd}/lib:${pkgs.intel-compute-runtime}/lib"
                  "OPENCL=1"
                  "GPU=1"
                  "OPENCL_DEVICE=0" # Force Intel Arc GPU (device 0)
                  "VISIBLE_DEVICES=0" # Tinygrad device visibility
                  "OCL_ICD_VENDORS=${pkgs.intel-compute-runtime}/etc/OpenCL/vendors" # Only Intel OpenCL
                ];

                # Logging
                StandardOutput = "journal";
                StandardError = "journal";
                SyslogIdentifier = "exo";
              };
            };

            # Intel NPU support
            systemd.services.exo-npu = lib.mkIf config.services.exo.intel.npu.enable {
              description = "exo Intel NPU Inference Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              serviceConfig = {
                Type = "simple";
                ExecStart = "${pkgs.python312}/bin/python -m exo.worker.engines.npu.service --port ${toString config.services.exo.intel.npu.servicePort}";
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

      # NixOS module for distributed inference with Gloo backend
      flake.nixosModules.exo-distributed = import ./nix/distributed-inference.nix;

      perSystem =
        { config, self', inputs', pkgs, lib, system, ... }:
        let
          fenixToolchain = inputs'.fenix.packages.complete;
          # Use pinned nixpkgs for swift-format (swift is broken on x86_64-linux in newer nixpkgs)
          pkgsSwift = import inputs.nixpkgs-swift { inherit system; };

          # Create a separate pkgs instance for exo with anyio and tinygrad overlays
          # This avoids polluting the global pkgs
          pkgsExo = import inputs.nixpkgs {
            inherit system;
            config = {
              # Disable checks globally to skip failing libffi tests
              doCheckByDefault = false;
              # Allow unfree for MKL (needed for PyTorch XPU)
              allowUnfreePredicate = pkg: (pkg.pname or "") == "mkl";
            };
            overlays = [
              # Import MordragT's overlay to get Intel runtime libraries (compute-runtime, level-zero, mkl, etc.)
              (import "${inputs.nixos-mordrag}/pkgs/overlay.nix")
              # Overlay to customize Python packages - disable failing tests
              (final: prev: {
                libffi = prev.libffi.overrideAttrs (old: {
                  outputs = old.outputs or [ "out" "dev" ];
                  doCheck = false;
                  doInstallCheck = false;
                });
                
                # Use standard python312 for PyTorch XPU compatibility
                # Python 3.13 is not yet supported by Intel's PyTorch XPU wheels
                python312 = prev.python312.override {
                  self = final.python312;
                  packageOverrides = pself: psuper: {
                    # Override mkDerivation to add LOCALE_ARCHIVE for all Python package builds
                    mkDerivation = args: psuper.mkDerivation (args // {
                      LOCALE_ARCHIVE = "${final.glibcLocales}/lib/locale/locale-archive";
                      LC_ALL = "en_US.UTF-8";
                      nativeBuildInputs = (args.nativeBuildInputs or []) ++ [ final.glibcLocales ];
                    });
                    
                    # Pin anyio to 4.11.0 (required by exo)
                    anyio = psuper.anyio.overridePythonAttrs (old: rec {
                      version = "4.11.0";
                      src = final.fetchPypi {
                        pname = "anyio";
                        inherit version;
                        hash = "sha256-gqjQuB4xjMXOcaXx+LXE5jYZYgtjFB74yZX6DblaV8Q=";
                      };
                      doCheck = false;
                      postPatch = (old.postPatch or "") + ''
                        sed -i '/def test_bad_init_value/,/pytest.raises.*CapacityLimiter.*0/d' tests/test_synchronization.py
                      '';
                    });
                    
                    # pycparser segfaults during unit tests
                    pycparser = psuper.pycparser.overridePythonAttrs (old: {
                      doCheck = false;
                      doInstallCheck = false;
                    });
                    
                    # sqlalchemy has a failing test
                    sqlalchemy = psuper.sqlalchemy.overridePythonAttrs (old: {
                      doCheck = false;
                      doInstallCheck = false;
                    });
                    
                    # uvloop - completely rebuild without tests
                    # Source: Based on nixpkgs uvloop, modified to skip flaky tests
                    uvloop = pself.buildPythonPackage rec {
                      pname = "uvloop";
                      version = "0.22.1";
                      pyproject = true;

                      src = final.fetchPypi {
                        inherit pname version;
                        hash = "sha256-bIS640W5FHCCsXNx491dQndb3c6R+IVJkBf0YH/a858=";
                      };

                      env.LIBUV_CONFIGURE_HOST = pself.python.stdenv.hostPlatform.config;

                      postPatch = ''
                        rm -rf vendor
                        substituteInPlace setup.py \
                          --replace-fail "use_system_libuv = False" "use_system_libuv = True"
                      '';

                      nativeBuildInputs = [
                        pself.cython
                        pself.setuptools
                      ];

                      buildInputs = [
                        final.libuv
                      ];

                      # DISABLE ALL TESTS
                      doCheck = false;
                      dontCheck = true;
                      doInstallCheck = false;

                      pythonImportsCheck = [
                        "uvloop"
                        "uvloop.loop"
                      ];

                      meta = {
                        description = "Ultra fast asyncio event loop (tests disabled for build stability)";
                        homepage = "https://github.com/MagicStack/uvloop";
                      };
                    };
                    
                    # fsspec - disable torch optional dependency to avoid pulling in standard PyTorch
                    # We build our own PyTorch XPU separately
                    fsspec = psuper.fsspec.overridePythonAttrs (old: {
                      # Remove torch from optional dependencies
                      passthru = (old.passthru or {}) // {
                        optional-dependencies = (old.passthru.optional-dependencies or {}) // {
                          # Remove torch from any optional dependency groups
                          full = builtins.filter (dep: dep.pname or "" != "torch") 
                            ((old.passthru.optional-dependencies or {}).full or []);
                        };
                      };
                    });
                    
                    # PyTorch with XPU support — override the nixpkgs torch
                    # so all transitive deps (transformers, etc.) use our XPU build
                    torch = pself.callPackage (inputs.self + /nix/pytorch-xpu.nix) {
                      inherit (final) intel-compute-runtime level-zero mkl oneDNN onetbb glibcLocales;
                    };

                    # safetensors tests import torch which needs libsycl.so.8 at runtime
                    safetensors = psuper.safetensors.overridePythonAttrs (old: {
                      doCheck = false;
                      doInstallCheck = false;
                    });

                    # transformers — nixpkgs has 4.x, we need 5.7+ for Qwen3.5
                    # Install as wheel to skip runtime deps check (deps satisfied at runtime)
                    transformers = psuper.buildPythonPackage {
                      pname = "transformers";
                      version = "5.7.0";
                      format = "wheel";
                      src = final.fetchurl {
                        url = "https://files.pythonhosted.org/packages/py3/t/transformers/transformers-5.7.0-py3-none-any.whl";
                        hash = "sha256-hpZgzY/JK63AQfVVG/dVpC9LlVjJM0G/P6Pu7XBlB5w=";
                      };
                      doCheck = false;
                      doInstallCheck = false;
                      dontUsePythonCatchConflicts = true;
                    };
                  };
                };
              })
            ];
          };
        in
        {
          # Allow unfree for metal-toolchain (needed for Darwin Metal packages) and mkl (needed for PyTorch XPU)
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfreePredicate = pkg: 
                let pname = pkg.pname or "";
                in (pname == "metal-toolchain") || (pname == "mkl");
              doCheckByDefault = false;
            };
            overlays = [
              # Override libffi globally to skip tests - FINAL overlay
              (final: prev: {
                libffi = prev.libffi.overrideAttrs (old: {
                  outputs = old.outputs or [ "out" "dev" ];
                  doCheck = false;
                  doInstallCheck = false;
                });
                
                # Fix locale for Python builds
                python312 = prev.python312.override {
                  packageOverrides = pself: psuper: {
                    mkDerivation = args: psuper.mkDerivation (args // {
                      LOCALE_ARCHIVE = "${final.glibcLocales}/lib/locale/locale-archive";
                      LC_ALL = "en_US.UTF-8";
                    });
                  };
                };
              })
            ];
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
          ) // lib.optionalAttrs pkgs.stdenv.isLinux {
            # PyTorch with Intel XPU support (Linux only)
            pytorch-xpu = pkgsExo.python312.pkgs.torch;
          };

          devShells.default =
            let
              # Create a Python environment with PyTorch XPU (no IPEX — discontinued)
              # On Linux, use PyTorch XPU package (Python 3.12)
              pythonWithPackages = if pkgs.stdenv.isLinux then
                pkgsExo.python312.withPackages (ps: [
                  self'.packages.pytorch-xpu
                ])
              else
                # On macOS, use standard Python (no XPU support needed)
                pkgsExo.python312;
            in
            pkgs.mkShell {
            inputsFrom = [ self'.checks.cargo-build ];

            packages =
              [
                # FORMATTING
                config.treefmt.build.wrapper

                # PYTHON - use Python with PyTorch XPU packages from pkgsExo
                pythonWithPackages
                pkgs.uv
                pkgs.ruff
                pkgs.basedpyright

                # RUST
                config.rust.toolchain
                pkgs.maturin

                # NIX
                pkgs.nixpkgs-fmt

                # SVELTE
                pkgs.nodejs

                # MISC
                pkgs.just
                pkgs.jq
              ]
              ++ lib.optionals pkgs.stdenv.isLinux [
                pkgs.unixtools.ifconfig
                # Intel GPU runtime libraries for PyTorch XPU - use pkgsExo to get libffi fix
                pkgsExo.intel-compute-runtime
                pkgsExo.level-zero
                pkgsExo.intel-gpu-tools # For monitoring with intel_gpu_top
                pkgsExo.clinfo # For OpenCL device information
              ]
              ++ lib.optionals pkgs.stdenv.isDarwin [
                pkgs.macmon
              ];

            OPENSSL_NO_VENDOR = "1";

            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pythonWithPackages}/lib"
              ${lib.optionalString pkgs.stdenv.isLinux ''
                export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
                # Add Intel GPU runtime libraries and oneAPI libraries for PyTorch XPU (from pkgsExo)
                export LD_LIBRARY_PATH="${pkgsExo.intel-compute-runtime}/lib:${pkgsExo.level-zero}/lib:${pkgsExo.mkl}/lib:${pkgsExo.oneDNN}/lib:${pkgsExo.onetbb}/lib:$LD_LIBRARY_PATH"
                # Enable PyTorch XPU (Intel GPU) support
                export PYTORCH_ENABLE_XPU=1
                echo "Intel Arc GPU support enabled for PyTorch XPU"
                echo "LD_LIBRARY_PATH includes Intel compute runtime, Level Zero, and oneAPI libraries (MKL, oneDNN, TBB)"
                echo "Python: ${pythonWithPackages}/bin/python"
              ''}
            '';
          };
        };
    };
}

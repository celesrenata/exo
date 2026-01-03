# EXO NixOS Flake - Complete NixOS integration for EXO distributed AI inference
{
  description = "EXO distributed AI inference system with NixOS integration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    
    # Rust toolchain for building Rust components
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, fenix }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      
      # Helper to create system-specific outputs
      forAllSystems = f: flake-utils.lib.genAttrs systems (system: f {
        inherit system;
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ fenix.overlays.default self.overlays.default ];
        };
      });
      
    in {
      # Overlay for EXO packages
      overlays.default = final: prev: {
        # Import all EXO packages
        exo-python = final.callPackage ./nix/packages/exo-python.nix { };
        exo-rust-bindings = final.callPackage ./nix/packages/exo-rust-bindings.nix { };
        exo-dashboard = final.callPackage ./nix/packages/exo-dashboard.nix { };
        exo-cpu = final.callPackage ./nix/packages/exo-cpu.nix { };
        exo-cuda = final.callPackage ./nix/packages/exo-cuda.nix { };
        exo-rocm = final.callPackage ./nix/packages/exo-rocm.nix { };
        exo-intel = final.callPackage ./nix/packages/exo-intel.nix { };
        exo-mlx = final.callPackage ./nix/packages/exo-mlx.nix { };
        exo-complete = final.callPackage ./nix/packages/exo-complete.nix { };
      };

      # System-specific packages
      packages = forAllSystems ({ system, pkgs }: {
        # Individual packages
        exo-python = pkgs.exo-python;
        exo-rust-bindings = pkgs.exo-rust-bindings;
        exo-dashboard = pkgs.exo-dashboard;
        exo-cpu = pkgs.exo-cpu;
        exo-cuda = pkgs.exo-cuda;
        exo-rocm = pkgs.exo-rocm;
        exo-intel = pkgs.exo-intel;
        exo-mlx = pkgs.exo-mlx;
        
        # Complete package with hardware detection
        exo-complete = pkgs.exo-complete;
        default = pkgs.exo-complete;
        
        # Test runner
        exo-test-runner = pkgs.writeShellScriptBin "exo-test-runner" ''
          exec ${./nix/tests/run-tests.sh} "$@"
        '';
      });

      # NixOS modules
      nixosModules = {
        exo-service = import ./nix/modules/exo-service.nix;
        exo-hardware = import ./nix/modules/exo-hardware.nix;
        exo-networking = import ./nix/modules/exo-networking.nix;
        exo-k3s = import ./nix/modules/exo-k3s.nix;
        
        # Default module that imports all others
        default = { ... }: {
          imports = [
            self.nixosModules.exo-service
            self.nixosModules.exo-hardware
            self.nixosModules.exo-networking
            self.nixosModules.exo-k3s
          ];
        };
      };

      # System checks and tests
      checks = forAllSystems ({ system, pkgs }: 
        let
          # Create test packages with proper dependencies
          exo-packages = {
            exo-cpu = pkgs.exo-cpu;
            exo-cuda = pkgs.exo-cuda;
            exo-rocm = pkgs.exo-rocm;
            exo-intel = pkgs.exo-intel;
            exo-mlx = pkgs.exo-mlx;
            exo-dashboard = pkgs.exo-dashboard;
            exo-rust-bindings = pkgs.exo-rust-bindings;
            exo-python = pkgs.exo-python;
            exo-complete = pkgs.exo-complete;
          };
          
          # Import test modules
          packageTests = import ./nix/tests/package-tests.nix {
            inherit (pkgs) lib;
            inherit pkgs system;
            inherit exo-packages;
          };
          
          integrationTests = import ./nix/tests/integration-tests.nix {
            inherit (pkgs) lib;
            inherit pkgs system;
            nixosModules = self.nixosModules;
          };
          
          hardwareTests = import ./nix/tests/hardware-tests.nix {
            inherit (pkgs) lib;
            inherit pkgs system;
            inherit exo-packages;
          };
          
          performanceTests = import ./nix/tests/performance-tests.nix {
            inherit (pkgs) lib;
            inherit pkgs system;
            inherit exo-packages;
          };
          
        in {
          # Package build tests
          test-build-all-packages = packageTests.build-all-packages;
          test-cross-compilation = packageTests.cross-compilation-tests;
          test-dependency-resolution = packageTests.dependency-resolution-tests;
          test-reproducibility = packageTests.reproducibility-tests;
          
          # Integration tests
          test-multi-node-cluster = integrationTests.multi-node-cluster-tests;
          test-k3s-integration = integrationTests.k3s-integration-tests;
          test-networking = integrationTests.networking-tests;
          test-service-discovery = integrationTests.service-discovery-tests;
          
          # Hardware tests
          test-gpu-detection = hardwareTests.gpu-detection-tests;
          test-hardware-acceleration = hardwareTests.hardware-acceleration-tests;
          test-fallback-scenarios = hardwareTests.fallback-scenario-tests;
          test-driver-configuration = hardwareTests.driver-configuration-tests;
          
          # Performance tests
          test-network-throughput = performanceTests.network-throughput-tests;
          test-rdma-performance = performanceTests.rdma-performance-tests;
          test-gpu-benchmarks = performanceTests.gpu-acceleration-benchmarks;
          test-cpu-benchmarks = performanceTests.cpu-fallback-benchmarks;
          
          # Comprehensive test runner
          test-all = import ./nix/tests/default.nix {
            inherit (pkgs) lib;
            inherit pkgs system;
            inherit exo-packages;
            nixosModules = self.nixosModules;
          };
        }
      );

      # Development shells
      devShells = forAllSystems ({ system, pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            # EXO packages for testing
            exo-complete
            
            # Development tools
            nix
            nixpkgs-fmt
            
            # Testing tools
            curl
            jq
            netcat
            iperf3
            
            # System tools
            pciutils
            util-linux
            procps
            
            # Build tools
            gcc
            pkg-config
            openssl
            
            # Test runner
            (writeShellScriptBin "test-exo" ''
              exec ${./nix/tests/run-tests.sh} "$@"
            '')
          ];
          
          shellHook = ''
            echo "🚀 EXO NixOS Development Environment"
            echo
            echo "Available commands:"
            echo "  test-exo                 - Run EXO test suite"
            echo "  test-exo --help          - Show test options"
            echo "  test-exo package         - Run package tests only"
            echo "  test-exo --performance   - Run all tests including performance"
            echo
            echo "Available packages:"
            echo "  exo-complete            - Complete EXO package with hardware detection"
            echo "  exo-detect-hardware     - Hardware detection script"
            echo "  exo-system-info         - System information script"
            echo
            echo "NixOS modules available in nixosModules.*"
            echo "Tests available in checks.*"
            echo
          '';
        };
        
        # Specialized shell for testing
        testing = pkgs.mkShell {
          packages = with pkgs; [
            # All EXO packages
            exo-complete
            exo-cpu
            exo-cuda
            exo-rocm
            exo-intel
            exo-mlx
            exo-dashboard
            
            # Comprehensive testing tools
            nixosTest
            qemu
            socat
            tcpdump
            wireshark-cli
            
            # Performance testing
            sysbench
            stress-ng
            htop
            iotop
            
            # Network testing
            iperf3
            netperf
            nmap
            
            # Hardware testing
            pciutils
            usbutils
            lshw
            dmidecode
            
            # Monitoring
            prometheus
            grafana
            
            # Test runner with all features
            (writeShellScriptBin "test-exo-full" ''
              export RUN_PERFORMANCE_TESTS=1
              export VERBOSE=1
              exec ${./nix/tests/run-tests.sh} "$@"
            '')
          ];
          
          shellHook = ''
            echo "🧪 EXO Testing Environment"
            echo
            echo "This shell includes comprehensive testing tools."
            echo "Use 'test-exo-full' to run all tests including performance benchmarks."
            echo
          '';
        };
      });

      # Apps for easy execution
      apps = forAllSystems ({ system, pkgs }: {
        default = {
          type = "app";
          program = "${self.packages.${system}.exo-complete}/bin/exo";
        };
        
        exo = {
          type = "app";
          program = "${self.packages.${system}.exo-complete}/bin/exo";
        };
        
        exo-master = {
          type = "app";
          program = "${self.packages.${system}.exo-complete}/bin/exo-master";
        };
        
        exo-worker = {
          type = "app";
          program = "${self.packages.${system}.exo-complete}/bin/exo-worker";
        };
        
        test-runner = {
          type = "app";
          program = "${self.packages.${system}.exo-test-runner}/bin/exo-test-runner";
        };
      });

      # Templates for easy project setup
      templates = {
        nixos-config = {
          path = ./nix/templates/nixos-config;
          description = "NixOS configuration template with EXO integration";
        };
        
        k3s-integration = {
          path = ./nix/templates/k3s-integration;
          description = "K3s cluster configuration with EXO integration";
        };
      };

      # Hydra jobsets for CI
      hydraJobs = self.checks;
    };
}
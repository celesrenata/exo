# Example NixOS configuration for PyTorch XPU backend with Intel Arc GPU
# This configuration enables the PyTorch XPU backend as the primary inference engine

{
  description = "NixOS configuration with PyTorch XPU backend for exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:exo-explore/exo";
  };

  outputs = { self, nixpkgs, exo }: {
    nixosConfigurations.your-hostname = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        # Import your hardware configuration
        ./hardware-configuration.nix

        # Import the exo Intel hardware module
        exo.nixosModules.exo-intel

        # Main configuration
        {
          # System identification
          networking.hostName = "your-hostname";

          # Enable Intel hardware acceleration for exo
          services.exo.intel = {
            # Enable the Intel hardware support module
            enable = true;

            # PyTorch XPU backend configuration (PRIMARY)
            pytorch_xpu = {
              # Enable PyTorch XPU backend
              enable = true;

              # Use PyTorch XPU as the preferred backend
              # This will try PyTorch XPU first, then fall back to tinygrad if unavailable
              preferredBackend = true;
            };

            # Tinygrad backend configuration (FALLBACK)
            tinygrad = {
              # Keep tinygrad enabled as fallback
              enable = true;

              # Backend selection:
              # - "GPU": Use GPU acceleration (default)
              # - "CPU": Use CPU-only execution
              backend = "GPU";
            };

            # Intel Arc iGPU configuration
            arc = {
              # Enable Intel Arc iGPU support
              enable = true;

              # Runtime selection:
              # - "level-zero": Use Level Zero API (recommended for best performance)
              # - "opencl": Use OpenCL API (fallback option)
              # - "auto": Automatically select best available runtime (default)
              runtime = "auto";
            };

            # Intel NPU configuration (optional, experimental)
            npu = {
              # Disable NPU support by default
              enable = false;

              # Port for NPU inference service
              servicePort = 52416;
            };
          };

          # Graphics configuration
          # This is automatically configured by the exo-intel module
          hardware.graphics = {
            enable = true;
            enable32Bit = true; # Enable 32-bit graphics support if needed
          };

          # Optional: Add monitoring and debugging tools
          environment.systemPackages = with pkgs; [
            intel-gpu-tools # For intel_gpu_top
            clinfo # For OpenCL device information
          ];

          # Optional: Enable verbose logging for debugging
          # environment.variables = {
          #   PYTORCH_ENABLE_XPU = "1";
          # };
        }
      ];
    };
  };
}

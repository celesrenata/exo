# Example NixOS configuration for Intel hardware acceleration with exo
# This configuration is designed for systems with Intel Core Ultra processors
# featuring Arc iGPU and NPU capabilities (e.g., Core Ultra 9 185H)

{
  description = "NixOS configuration with Intel hardware acceleration for exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:exo-explore/exo";
  };

  outputs = { self, nixpkgs, exo }: {
    nixosConfigurations.gremlin-1 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        # Import your hardware configuration
        ./hardware-configuration.nix

        # Import the exo Intel hardware module
        exo.nixosModules.exo-intel

        # Main configuration
        {
          # System identification
          networking.hostName = "gremlin-1";

          # Enable Intel hardware acceleration for exo
          services.exo.intel = {
            # Enable the Intel hardware support module
            enable = true;

            # Intel Arc iGPU configuration
            arc = {
              # Enable Intel Arc iGPU support
              enable = true;

              # Runtime selection:
              # - "level-zero": Use Level Zero API (recommended for best performance)
              # - "opencl": Use OpenCL API (fallback option)
              # - "auto": Automatically select best available runtime
              runtime = "level-zero";
            };

            # Intel NPU configuration (experimental)
            npu = {
              # Enable Intel NPU support
              # Note: This is experimental and requires Core Ultra processors
              enable = true;

              # Port for NPU inference service
              servicePort = 52416;
            };
          };

          # Graphics configuration
          # This is automatically configured by the exo-intel module,
          # but you can add additional graphics settings here
          hardware.graphics = {
            enable = true;
            enable32Bit = true; # Enable 32-bit graphics support if needed
          };

          # Optional: Add monitoring tools
          environment.systemPackages = with nixpkgs.legacyPackages.x86_64-linux; [
            intel-gpu-tools # For intel_gpu_top
            clinfo # For OpenCL device information
          ];

          # Optional: Enable OpenGL/Vulkan debugging
          # environment.variables = {
          #   LIBGL_DEBUG = "verbose";
          #   MESA_DEBUG = "1";
          # };
        }
      ];
    };
  };
}

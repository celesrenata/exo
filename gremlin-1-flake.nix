{
  description = "gremlin-1 NixOS configuration with Intel hardware support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    exo.url = "github:celesrenata/exo/ipex";
  };

  outputs = { self, nixpkgs, exo, ... }: {
    nixosConfigurations.gremlin-1 = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./hardware-configuration.nix
        exo.nixosModules.exo-intel
        {
          networking.hostName = "gremlin-1";

          # Enable Intel hardware support
          services.exo.intel = {
            enable = true;

            arc = {
              enable = true;
              runtime = "auto"; # Auto-detect Level Zero or OpenCL
            };

            npu = {
              enable = true; # Experimental NPU support
              servicePort = 52416;
            };
          };

          # Additional packages for testing and monitoring
          environment.systemPackages = with pkgs; [
            intel-gpu-tools # intel_gpu_top for GPU monitoring
            clinfo # OpenCL info
            pciutils # lspci
            usbutils # lsusb
            htop # System monitoring
            curl # API testing
            jq # JSON parsing
          ];

          # Ensure graphics support is enabled
          hardware.graphics.enable = true;

          # Allow unfree packages if needed
          nixpkgs.config.allowUnfree = true;
        }
      ];
    };
  };
}

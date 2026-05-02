# NixOS module for distributed inference with exo
#
# Configures the exo systemd service for pipeline-parallel distributed
# inference across heterogeneous GPU nodes using PyTorch's Gloo backend.
#
# Import this module into the gremlin flake and set:
#   services.exo.distributed.enable = true;
#   services.exo.distributed.masterAddr = "10.1.1.12";
#
# Requirements: 9.1, 9.2, 9.3

{ config, lib, pkgs, ... }:

let
  cfg = config.services.exo.distributed;
in
{
  options.services.exo.distributed = {
    enable = lib.mkEnableOption "distributed inference mode for exo";

    masterAddr = lib.mkOption {
      type = lib.types.str;
      description = ''
        IP address of the rank 0 node for torch.distributed rendezvous.
        This should be the ethernet IP of the node designated as rank 0
        (e.g., gremlin-1 at 10.1.1.12).
      '';
      example = "10.1.1.12";
    };

    masterPort = lib.mkOption {
      type = lib.types.port;
      default = 29500;
      description = ''
        Port for torch.distributed Gloo rendezvous on the rank 0 node.
        All nodes connect to MASTER_ADDR:MASTER_PORT during process group
        initialization.
      '';
    };

    ephemeralPortRange = lib.mkOption {
      type = lib.types.submodule {
        options = {
          from = lib.mkOption {
            type = lib.types.port;
            default = 49152;
            description = "Start of ephemeral port range for Gloo TCP communication.";
          };
          to = lib.mkOption {
            type = lib.types.port;
            default = 65535;
            description = "End of ephemeral port range for Gloo TCP communication.";
          };
        };
      };
      default = { from = 49152; to = 65535; };
      description = ''
        Ephemeral port range opened in the firewall for Gloo TCP
        inter-node communication. Gloo establishes direct TCP connections
        between all ranks through the aggregation switch.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # Set MASTER_ADDR and MASTER_PORT environment variables on the exo
    # systemd service so torch.distributed can find the rendezvous endpoint.
    systemd.services.exo.environment = {
      MASTER_ADDR = cfg.masterAddr;
      MASTER_PORT = toString cfg.masterPort;
    };

    # Run GPU verification before the main exo service starts.
    # This detects the GPU type, logs memory architecture (shared vs discrete)
    # and available memory, and verifies GPU drivers are functional.
    # Requirement: 9.4
    systemd.services.exo.serviceConfig.ExecStartPre = let
      gpuVerifyScript = pkgs.writeScript "exo-verify-gpu" ''
        #!${pkgs.python3}/bin/python3
        ${builtins.readFile ./verify-gpu-on-startup.py}
      '';
    in [
      "${gpuVerifyScript}"
    ];

    # Open the ephemeral port range and the master port in the firewall
    # for Gloo TCP communication between nodes.
    networking.firewall.allowedTCPPortRanges = [
      { from = cfg.ephemeralPortRange.from; to = cfg.ephemeralPortRange.to; }
    ];
    networking.firewall.allowedTCPPorts = [ cfg.masterPort ];
  };
}

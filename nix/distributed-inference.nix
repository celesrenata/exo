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

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        The exo package to use. Must provide bin/exo.
      '';
    };

    intelGpuPackages = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [];
      description = ''
        Intel GPU runtime packages (intel-compute-runtime, level-zero) to add
        to LD_LIBRARY_PATH for torch.xpu support. These provide the Level Zero
        driver backend that PyTorch XPU needs to detect Intel iGPUs.
      '';
      example = lib.literalExpression "[ pkgs.intel-compute-runtime pkgs.level-zero ]";
    };

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

    verbosity = lib.mkOption {
      type = lib.types.str;
      default = "-vv";
      description = "Verbosity flags passed to exo (e.g. -v, -vv).";
    };

    apiPort = lib.mkOption {
      type = lib.types.port;
      default = 52415;
      description = "Port for the exo API and dashboard.";
    };

    libp2pPort = lib.mkOption {
      type = lib.types.port;
      default = 4001;
      description = "Fixed TCP port for libp2p peer-to-peer communication.";
    };

    peers = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      description = ''
        List of static peer multiaddrs to dial on startup.
        Use when mDNS multicast doesn't work (e.g., switch blocks multicast).
        Format: /ip4/IP/tcp/PORT
      '';
      example = [ "/ip4/10.1.1.13/tcp/4001" "/ip4/10.1.1.14/tcp/4001" ];
    };
  };

  config = lib.mkIf cfg.enable {
    # Full exo systemd service definition for distributed inference
    systemd.services.exo = {
      description = "exo Distributed AI Inference Service";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        MASTER_ADDR = cfg.masterAddr;
        MASTER_PORT = toString cfg.masterPort;
        EXO_LIBP2P_PORT = toString cfg.libp2pPort;
      } // lib.optionalAttrs (cfg.peers != []) {
        EXO_PEERS = lib.concatStringsSep "," cfg.peers;
      } // lib.optionalAttrs (cfg.intelGpuPackages != []) {
        # Intel GPU runtime libraries for torch.xpu (Level Zero + compute runtime)
        LD_LIBRARY_PATH = lib.makeLibraryPath cfg.intelGpuPackages
          + ":" + lib.concatMapStringsSep ":" (pkg: "${pkg}/lib/intel-opencl") (
            builtins.filter (pkg: builtins.pathExists "${pkg}/lib/intel-opencl") cfg.intelGpuPackages
          );
      };

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/exo ${cfg.verbosity}";
        Restart = "on-failure";
        RestartSec = "5s";
        User = "root";
        Group = "root";

        # Logging
        StandardOutput = "journal";
        StandardError = "journal";
        SyslogIdentifier = "exo";
      };
    };

    # Open the ephemeral port range, master port, and API port in the firewall
    # Gloo uses OS-assigned ports for its mesh connections which may be below 49152
    networking.firewall.allowedTCPPortRanges = [
      { from = 1024; to = 65535; }
    ];
    networking.firewall.allowedTCPPorts = [ cfg.masterPort cfg.apiPort cfg.libp2pPort ];
    # mDNS for libp2p peer discovery
    networking.firewall.allowedUDPPorts = [ 5353 ];
  };
}

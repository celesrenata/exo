{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.exo;

  # Configuration file generation
  exoConfigFile = pkgs.writeText "exo-config.json" (builtins.toJSON {
    mode = cfg.mode;
    api_port = cfg.apiPort;
    discovery_port = cfg.networking.discoveryPort;
    bond_interface = cfg.networking.bondInterface;
    rdma_enabled = cfg.networking.rdmaEnabled;
    log_level = cfg.logging.level;
    data_dir = cfg.dataDir;
    model_cache_dir = cfg.modelCacheDir;
    k3s_integration = cfg.k3s.integration;
    k3s_namespace = cfg.k3s.namespace;
    memory_limit = cfg.hardware.memoryLimit;
  });

  # Environment file for services
  exoEnvironmentFile = pkgs.writeText "exo-environment" ''
    EXO_CONFIG_FILE=${exoConfigFile}
    EXO_DATA_DIR=${cfg.dataDir}
    EXO_LOG_LEVEL=${cfg.logging.level}
    EXO_API_PORT=${toString cfg.apiPort}
    EXO_DISCOVERY_PORT=${toString cfg.networking.discoveryPort}
    ${optionalString (cfg.networking.bondInterface != null) "EXO_BOND_INTERFACE=${cfg.networking.bondInterface}"}
    ${optionalString cfg.networking.rdmaEnabled "EXO_RDMA_ENABLED=1"}
    ${optionalString cfg.k3s.integration "EXO_K3S_INTEGRATION=1"}
    ${optionalString (cfg.k3s.namespace != null) "EXO_K3S_NAMESPACE=${cfg.k3s.namespace}"}
    ${optionalString (cfg.hardware.memoryLimit != null) "EXO_MEMORY_LIMIT=${cfg.hardware.memoryLimit}"}
  '';

in
{
  options.services.exo = {
    enable = mkEnableOption "EXO distributed AI inference system";

    package = mkOption {
      type = types.package;
      default = pkgs.exo-complete;
      description = ''
        EXO package to use. Defaults to the complete package with
        automatic hardware detection. Can be overridden to use specific
        hardware variants.
      '';
    };

    mode = mkOption {
      type = types.enum [ "master" "worker" "auto" ];
      default = "auto";
      description = ''
        Node operation mode:
        - master: Run as master node (coordinates cluster)
        - worker: Run as worker node (executes inference tasks)
        - auto: Automatically determine role based on cluster state
      '';
    };

    apiPort = mkOption {
      type = types.port;
      default = 52415;
      description = ''
        Port for the OpenAI-compatible API server.
        This port will be opened in the firewall if firewall is enabled.
      '';
    };

    user = mkOption {
      type = types.str;
      default = "exo";
      description = ''
        User account under which EXO services run.
        Will be created automatically if it doesn't exist.
      '';
    };

    group = mkOption {
      type = types.str;
      default = "exo";
      description = ''
        Group under which EXO services run.
        Will be created automatically if it doesn't exist.
      '';
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/exo";
      description = ''
        Directory for EXO runtime data, including cluster state,
        node information, and temporary files.
      '';
    };

    configDir = mkOption {
      type = types.path;
      default = "/etc/exo";
      description = ''
        Directory for EXO configuration files.
      '';
    };

    modelCacheDir = mkOption {
      type = types.path;
      default = "/var/cache/exo/models";
      description = ''
        Directory for cached AI models. This can be large and should
        be on a filesystem with sufficient space.
      '';
    };

    networking = {
      bondInterface = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "bond0";
        description = ''
          Bonded network interface to use for high-bandwidth model distribution.
          If null, EXO will use the default network interface.
        '';
      };

      discoveryPort = mkOption {
        type = types.port;
        default = 52416;
        description = ''
          Port for node discovery and cluster communication.
          This port will be opened in the firewall if firewall is enabled.
        '';
      };

      rdmaEnabled = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable RDMA over Thunderbolt if available.
          Provides low-latency communication for supported hardware.
        '';
      };

      openFirewall = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Open necessary ports in the firewall for EXO communication.
          Includes API port and discovery port.
        '';
      };
    };

    hardware = {
      autoDetect = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Automatically detect and configure hardware acceleration.
          When enabled, EXO will select the best available accelerator.
        '';
      };

      preferredAccelerator = mkOption {
        type = types.nullOr (types.enum [ "cuda" "rocm" "intel" "mlx" "cpu" ]);
        default = null;
        description = ''
          Preferred hardware accelerator to use.
          If null, automatic detection will be used.
        '';
      };

      memoryLimit = mkOption {
        type = types.nullOr types.str;
        default = "80%";
        example = "8G";
        description = ''
          Memory limit for EXO processes. Can be specified as a percentage
          (e.g., "80%") or absolute value (e.g., "8G").
        '';
      };
    };

    k3s = {
      integration = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Enable K3s integration for Kubernetes orchestration.
          Requires K3s to be configured on the system.
        '';
      };

      serviceDiscovery = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Register EXO services with K3s service discovery.
          Only effective when k3s.integration is enabled.
        '';
      };

      namespace = mkOption {
        type = types.nullOr types.str;
        default = "exo-system";
        description = ''
          Kubernetes namespace for EXO services.
          If null, uses the default namespace.
        '';
      };

      serviceAccount = mkOption {
        type = types.str;
        default = "exo";
        description = ''
          Kubernetes service account for EXO services.
        '';
      };
    };

    logging = {
      level = mkOption {
        type = types.enum [ "debug" "info" "warning" "error" ];
        default = "info";
        description = ''
          Log level for EXO services.
          Higher levels include lower level messages.
        '';
      };

      journalRateLimit = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable systemd journal rate limiting to prevent log flooding.
        '';
      };

      maxLogSize = mkOption {
        type = types.str;
        default = "100M";
        description = ''
          Maximum size for individual log files before rotation.
        '';
      };

      maxLogFiles = mkOption {
        type = types.int;
        default = 10;
        description = ''
          Maximum number of rotated log files to keep.
        '';
      };
    };

    security = {
      enableSandbox = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable systemd sandboxing for EXO services.
          Provides additional security isolation.
        '';
      };

      allowedDevices = mkOption {
        type = types.listOf types.str;
        default = [ "/dev/dri" "/dev/nvidia*" "/dev/kfd" ];
        description = ''
          Device paths that EXO services are allowed to access.
          Automatically includes detected GPU devices.
        '';
      };

      networkNamespace = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Run EXO services in a separate network namespace.
          Provides network isolation but may complicate networking setup.
        '';
      };
    };

    dashboard = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable the web dashboard for cluster management.
          Dashboard will be accessible on the configured port.
        '';
      };

      port = mkOption {
        type = types.port;
        default = 8080;
        description = ''
          Port for the web dashboard server.
          This port will be opened in the firewall if firewall is enabled.
        '';
      };

      ssl = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = ''
            Enable SSL/TLS for dashboard and API access.
            Requires certificate configuration.
          '';
        };

        certificatePath = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = ''
            Path to SSL certificate file.
            Required when SSL is enabled.
          '';
        };

        keyPath = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = ''
            Path to SSL private key file.
            Required when SSL is enabled.
          '';
        };

        autoGenerate = mkOption {
          type = types.bool;
          default = false;
          description = ''
            Automatically generate self-signed SSL certificates.
            Only recommended for development environments.
          '';
        };

        acme = {
          enable = mkOption {
            type = types.bool;
            default = false;
            description = ''
              Use ACME (Let's Encrypt) for automatic SSL certificate management.
              Requires a valid domain name and internet connectivity.
            '';
          };

          domain = mkOption {
            type = types.nullOr types.str;
            default = null;
            example = "exo.example.com";
            description = ''
              Domain name for ACME certificate.
              Required when ACME is enabled.
            '';
          };

          email = mkOption {
            type = types.nullOr types.str;
            default = null;
            example = "admin@example.com";
            description = ''
              Email address for ACME registration.
              Required when ACME is enabled.
            '';
          };
        };
      };

      authentication = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = ''
            Enable authentication for dashboard access.
            Requires API key configuration.
          '';
        };

        apiKeys = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = ''
            List of valid API keys for dashboard authentication.
            Keys should be securely generated random strings.
          '';
        };

        sessionTimeout = mkOption {
          type = types.int;
          default = 3600;
          description = ''
            Session timeout in seconds for authenticated users.
          '';
        };
      };
    };
  };

  config = mkIf cfg.enable {
    # Import hardware detection, networking, and K3s modules
    imports = [ ./exo-hardware.nix ./exo-networking.nix ./exo-k3s.nix ];

    # Enable hardware detection by default
    services.exo.hardware.autoDetect = mkDefault cfg.hardware.autoDetect;
    services.exo.hardware.preferredAccelerator = mkDefault cfg.hardware.preferredAccelerator;
    services.exo.hardware.memoryLimit = mkDefault cfg.hardware.memoryLimit;

    # User and group creation
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      description = "EXO distributed AI inference system user";
      home = cfg.dataDir;
      createHome = false;
      shell = pkgs.bash;
      extraGroups = [ "video" ] ++ optionals (config.services.exo.hardware.enableAmdSupport || config.services.exo.hardware.enableIntelSupport) [ "render" ];
      uid = mkDefault 993; # Use a consistent UID for reproducibility
    };

    users.groups.${cfg.group} = {
      gid = mkDefault 993; # Use a consistent GID for reproducibility
    };

    # Additional security groups
    users.groups.exo-admin = mkIf cfg.security.enableSandbox {
      members = [ cfg.user ];
    };

    # Directory creation and permissions with enhanced security
    systemd.tmpfiles.rules = [
      # Main directories with proper ownership and permissions
      "d ${cfg.dataDir} 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.configDir} 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.modelCacheDir} 0750 ${cfg.user} ${cfg.group} -"
      "d /var/log/exo 0750 ${cfg.user} ${cfg.group} -"

      # Runtime directories
      "d /run/exo 0755 ${cfg.user} ${cfg.group} -"
      "d /run/exo/sockets 0750 ${cfg.user} ${cfg.group} -"

      # Configuration files with restricted permissions
      "f ${cfg.configDir}/config.json 0640 ${cfg.user} ${cfg.group} - ${exoConfigFile}"
      "f ${cfg.configDir}/environment 0640 ${cfg.user} ${cfg.group} - ${exoEnvironmentFile}"

      # Lock files and PID files
      "d /var/lock/exo 0755 ${cfg.user} ${cfg.group} -"
      "d /run/lock/exo 0755 ${cfg.user} ${cfg.group} -"

      # Cache directories with appropriate permissions
      "d ${cfg.modelCacheDir}/downloads 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.modelCacheDir}/models 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.modelCacheDir}/temp 0750 ${cfg.user} ${cfg.group} -"

      # Log rotation setup
      "d /var/log/exo/archive 0750 ${cfg.user} ${cfg.group} -"
    ];

    # Enhanced security with capability management
    security.sudo.extraRules = mkIf cfg.security.enableSandbox [
      {
        users = [ cfg.user ];
        commands = [
          {
            command = "${pkgs.systemd}/bin/systemctl restart exo-worker.service";
            options = [ "NOPASSWD" ];
          }
          {
            command = "${pkgs.systemd}/bin/systemctl reload exo-master.service";
            options = [ "NOPASSWD" ];
          }
        ];
      }
    ];

    # PAM configuration for additional security
    security.pam.services.exo = mkIf cfg.security.enableSandbox {
      text = ''
        auth    required pam_deny.so
        account required pam_deny.so
        password required pam_deny.so
        session required pam_deny.so
      '';
    };

    # AppArmor profiles for additional sandboxing (if AppArmor is enabled)
    security.apparmor.profiles = mkIf (cfg.security.enableSandbox && config.security.apparmor.enable) {
      exo-master = {
        profile = ''
          #include <tunables/global>
          
          ${cfg.package}/bin/exo {
            #include <abstractions/base>
            #include <abstractions/nameservice>
            
            # Network access
            network inet stream,
            network inet dgram,
            network inet6 stream,
            network inet6 dgram,
            
            # File access
            ${cfg.dataDir}/** rw,
            ${cfg.configDir}/** r,
            ${cfg.modelCacheDir}/** rw,
            /var/log/exo/** w,
            /run/exo/** rw,
            
            # System access
            /proc/sys/kernel/random/uuid r,
            /sys/devices/system/cpu/online r,
            /sys/devices/system/node/node*/meminfo r,
            
            # GPU device access
            /dev/dri/** rw,
            /dev/nvidia* rw,
            /dev/kfd rw,
            
            # Deny dangerous capabilities
            deny capability sys_admin,
            deny capability sys_module,
            deny capability sys_rawio,
          }
        '';
      };
    };

    # SELinux policies (if SELinux is enabled)
    # Note: This would require more extensive SELinux policy development
    # For now, we'll ensure compatibility with SELinux permissive mode

    # File system ACLs for fine-grained permissions
    systemd.services.exo-setup-acls = mkIf cfg.security.enableSandbox {
      description = "Setup EXO ACLs";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = pkgs.writeShellScript "setup-exo-acls" ''
          # Set up ACLs for EXO directories
          ${pkgs.acl}/bin/setfacl -R -m u:${cfg.user}:rwx ${cfg.dataDir}
          ${pkgs.acl}/bin/setfacl -R -m u:${cfg.user}:rx ${cfg.configDir}
          ${pkgs.acl}/bin/setfacl -R -m u:${cfg.user}:rwx ${cfg.modelCacheDir}
          ${pkgs.acl}/bin/setfacl -R -m g:${cfg.group}:rx ${cfg.configDir}
          ${pkgs.acl}/bin/setfacl -R -m g:${cfg.group}:rwx ${cfg.dataDir}
          ${pkgs.acl}/bin/setfacl -R -m g:${cfg.group}:rwx ${cfg.modelCacheDir}
          
          # Deny access to other users
          ${pkgs.acl}/bin/setfacl -R -m o::--- ${cfg.configDir}
          ${pkgs.acl}/bin/setfacl -R -m o::--- ${cfg.dataDir}
        '';
      };
    };

    # Device permissions and udev rules for GPU access
    services.udev.extraRules = mkMerge [
      # Base GPU device permissions
      ''
        # EXO GPU device access rules
        SUBSYSTEM=="drm", KERNEL=="card*", GROUP="video", MODE="0664", TAG+="uaccess"
        SUBSYSTEM=="drm", KERNEL=="render*", GROUP="render", MODE="0664", TAG+="uaccess"
      ''

      # NVIDIA specific rules
      (mkIf config.services.exo.hardware.enableNvidiaSupport ''
        # NVIDIA GPU devices for EXO
        SUBSYSTEM=="nvidia", KERNEL=="nvidia*", GROUP="video", MODE="0664"
        SUBSYSTEM=="nvidia", KERNEL=="nvidiactl", GROUP="video", MODE="0664"
        SUBSYSTEM=="nvidia", KERNEL=="nvidia-uvm", GROUP="video", MODE="0664"
        SUBSYSTEM=="nvidia", KERNEL=="nvidia-uvm-tools", GROUP="video", MODE="0664"
        SUBSYSTEM=="nvidia", KERNEL=="nvidia-modeset", GROUP="video", MODE="0664"
      '')

      # AMD specific rules
      (mkIf config.services.exo.hardware.enableAmdSupport ''
        # AMD GPU devices for EXO
        SUBSYSTEM=="misc", KERNEL=="kfd", GROUP="render", MODE="0664"
        SUBSYSTEM=="drm", KERNEL=="card*", ATTRS{vendor}=="0x1002", GROUP="video", MODE="0664"
        SUBSYSTEM=="drm", KERNEL=="render*", ATTRS{vendor}=="0x1002", GROUP="render", MODE="0664"
      '')

      # Intel specific rules
      (mkIf config.services.exo.hardware.enableIntelSupport ''
        # Intel GPU devices for EXO
        SUBSYSTEM=="drm", KERNEL=="card*", ATTRS{vendor}=="0x8086", GROUP="video", MODE="0664"
        SUBSYSTEM=="drm", KERNEL=="render*", ATTRS{vendor}=="0x8086", GROUP="render", MODE="0664"
      '')
    ];

    # Namespace isolation setup
    systemd.services.exo-namespace-setup = mkIf cfg.security.networkNamespace {
      description = "Setup EXO Network Namespace";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = pkgs.writeShellScript "setup-exo-namespace" ''
          # Create network namespace for EXO
          ${pkgs.iproute2}/bin/ip netns add exo-ns || true
          
          # Setup loopback interface in namespace
          ${pkgs.iproute2}/bin/ip netns exec exo-ns ${pkgs.iproute2}/bin/ip link set lo up
          
          # Create veth pair for namespace communication
          ${pkgs.iproute2}/bin/ip link add veth-exo type veth peer name veth-exo-ns || true
          ${pkgs.iproute2}/bin/ip link set veth-exo-ns netns exo-ns
          
          # Configure namespace networking
          ${pkgs.iproute2}/bin/ip addr add 192.168.100.1/24 dev veth-exo
          ${pkgs.iproute2}/bin/ip link set veth-exo up
          
          ${pkgs.iproute2}/bin/ip netns exec exo-ns ${pkgs.iproute2}/bin/ip addr add 192.168.100.2/24 dev veth-exo-ns
          ${pkgs.iproute2}/bin/ip netns exec exo-ns ${pkgs.iproute2}/bin/ip link set veth-exo-ns up
          ${pkgs.iproute2}/bin/ip netns exec exo-ns ${pkgs.iproute2}/bin/ip route add default via 192.168.100.1
        '';

        ExecStop = pkgs.writeShellScript "cleanup-exo-namespace" ''
          # Cleanup network namespace
          ${pkgs.iproute2}/bin/ip link delete veth-exo || true
          ${pkgs.iproute2}/bin/ip netns delete exo-ns || true
        '';
      };
    };

    # Firewall configuration
    networking.firewall = mkIf cfg.networking.openFirewall {
      allowedTCPPorts = [ cfg.apiPort cfg.networking.discoveryPort ] ++ optional cfg.dashboard.enable cfg.dashboard.port;
      allowedUDPPorts = [ cfg.networking.discoveryPort ];
    };

    # Package installation
    environment.systemPackages = [ cfg.package ];

    # Systemd services
    systemd.targets.exo = {
      description = "EXO Distributed AI Inference System";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
    };

    systemd.services.exo-master = mkIf (cfg.mode == "master" || cfg.mode == "auto") {
      description = "EXO Master Node";
      wantedBy = [ "exo.target" ];
      after = [ "network-online.target" ] ++ optional cfg.hardware.autoDetect "exo-hardware-detect.service";
      wants = [ "network-online.target" ];
      requires = optional cfg.hardware.autoDetect "exo-hardware-detect.service";
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/exo --mode master --api-port ${toString cfg.apiPort}";
        ExecReload = "${pkgs.coreutils}/bin/kill -HUP $MAINPID";
        Restart = "always";
        RestartSec = "5s";
        RestartSteps = 5;
        RestartMaxDelaySec = "30s";

        # Logging
        StandardOutput = mkIf cfg.logging.journalRateLimit "journal";
        StandardError = mkIf cfg.logging.journalRateLimit "journal";

        # Environment
        EnvironmentFile = "${cfg.configDir}/environment";
        Environment = [
          "EXO_MODE=master"
          "EXO_CONFIG_DIR=${cfg.configDir}"
          "EXO_DATA_DIR=${cfg.dataDir}"
          "EXO_MODEL_CACHE_DIR=${cfg.modelCacheDir}"
        ];

        # Working directory
        WorkingDirectory = cfg.dataDir;

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir cfg.modelCacheDir "/var/log/exo" ];
        ReadOnlyPaths = [ cfg.configDir ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        RestrictNamespaces = !cfg.security.networkNamespace;
        LockPersonality = true;
        MemoryDenyWriteExecute = false; # Needed for AI inference
        RemoveIPC = true;

        # Resource limits
        LimitNOFILE = 65536;
        LimitNPROC = 4096;
      } // optionalAttrs (cfg.hardware.memoryLimit != null) {
        MemoryMax = cfg.hardware.memoryLimit;
        MemoryHigh = cfg.hardware.memoryLimit;
      } // optionalAttrs cfg.security.enableSandbox {
        # Additional sandboxing
        ProtectProc = "invisible";
        ProcSubset = "pid";
        ProtectHostname = true;
        ProtectClock = true;
        SystemCallArchitectures = "native";
        SystemCallFilter = [ "@system-service" "~@privileged" "~@resources" ];
      } // optionalAttrs cfg.security.networkNamespace {
        PrivateNetwork = true;
      };

      # Logging configuration
      environment.SYSTEMD_LOG_LEVEL = cfg.logging.level;
    };

    systemd.services.exo-worker = mkIf (cfg.mode == "worker" || cfg.mode == "auto") {
      description = "EXO Worker Node";
      wantedBy = [ "exo.target" ];
      after = [ "network-online.target" "exo-master.service" ] ++ optional cfg.hardware.autoDetect "exo-hardware-detect.service";
      wants = [ "network-online.target" ];
      requires = optional cfg.hardware.autoDetect "exo-hardware-detect.service";
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/exo --mode worker";
        ExecReload = "${pkgs.coreutils}/bin/kill -HUP $MAINPID";
        Restart = "always";
        RestartSec = "5s";
        RestartSteps = 5;
        RestartMaxDelaySec = "30s";

        # Logging
        StandardOutput = mkIf cfg.logging.journalRateLimit "journal";
        StandardError = mkIf cfg.logging.journalRateLimit "journal";

        # Environment
        EnvironmentFile = "${cfg.configDir}/environment";
        Environment = [
          "EXO_MODE=worker"
          "EXO_CONFIG_DIR=${cfg.configDir}"
          "EXO_DATA_DIR=${cfg.dataDir}"
          "EXO_MODEL_CACHE_DIR=${cfg.modelCacheDir}"
        ];

        # Working directory
        WorkingDirectory = cfg.dataDir;

        # Security settings (similar to master but with device access)
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir cfg.modelCacheDir "/var/log/exo" ];
        ReadOnlyPaths = [ cfg.configDir ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = false; # May need realtime for GPU operations
        RestrictNamespaces = !cfg.security.networkNamespace;
        LockPersonality = true;
        MemoryDenyWriteExecute = false; # Needed for AI inference
        RemoveIPC = true;

        # Device access for GPU
        DeviceAllow = map (dev: "${dev} rw") cfg.security.allowedDevices;

        # Resource limits
        LimitNOFILE = 65536;
        LimitNPROC = 4096;
      } // optionalAttrs (cfg.hardware.memoryLimit != null) {
        MemoryMax = cfg.hardware.memoryLimit;
        MemoryHigh = cfg.hardware.memoryLimit;
      } // optionalAttrs cfg.security.enableSandbox {
        # Additional sandboxing (less restrictive for GPU access)
        ProtectProc = "invisible";
        ProcSubset = "pid";
        ProtectHostname = true;
        ProtectClock = true;
        SystemCallArchitectures = "native";
        SystemCallFilter = [ "@system-service" "@io-event" "~@privileged" ];
      } // optionalAttrs cfg.security.networkNamespace {
        PrivateNetwork = true;
      };

      # Logging configuration
      environment.SYSTEMD_LOG_LEVEL = cfg.logging.level;
    };

    systemd.services.exo-api = mkIf cfg.dashboard.enable {
      description = "EXO API Server";
      wantedBy = [ "exo.target" ];
      after = [ "exo-master.service" ];
      requires = [ "exo-master.service" ];
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/exo --spawn-api --api-port ${toString cfg.apiPort}";
        ExecReload = "${pkgs.coreutils}/bin/kill -HUP $MAINPID";
        Restart = "always";
        RestartSec = "5s";
        RestartSteps = 5;
        RestartMaxDelaySec = "30s";

        # Logging
        StandardOutput = mkIf cfg.logging.journalRateLimit "journal";
        StandardError = mkIf cfg.logging.journalRateLimit "journal";

        # Environment
        EnvironmentFile = "${cfg.configDir}/environment";
        Environment = [
          "EXO_MODE=api"
          "EXO_CONFIG_DIR=${cfg.configDir}"
          "EXO_DATA_DIR=${cfg.dataDir}"
          "EXO_API_PORT=${toString cfg.apiPort}"
        ];

        # Working directory
        WorkingDirectory = cfg.dataDir;

        # Security settings (most restrictive as API is network-facing)
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir "/var/log/exo" ];
        ReadOnlyPaths = [ cfg.configDir cfg.modelCacheDir ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        RestrictNamespaces = !cfg.security.networkNamespace;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;

        # Network binding
        IPAddressDeny = mkIf cfg.security.enableSandbox "any";
        IPAddressAllow = mkIf cfg.security.enableSandbox [ "localhost" "10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16" ];

        # Resource limits
        LimitNOFILE = 65536;
        LimitNPROC = 1024;
      } // optionalAttrs cfg.security.enableSandbox {
        # Additional sandboxing
        ProtectProc = "invisible";
        ProcSubset = "pid";
        ProtectHostname = true;
        ProtectClock = true;
        SystemCallArchitectures = "native";
        SystemCallFilter = [ "@system-service" "~@privileged" "~@resources" ];
      } // optionalAttrs cfg.security.networkNamespace {
        PrivateNetwork = true;
      };

      # Logging configuration
      environment.SYSTEMD_LOG_LEVEL = cfg.logging.level;
    };

    # Dashboard web server service
    systemd.services.exo-dashboard = mkIf cfg.dashboard.enable {
      description = "EXO Dashboard Web Server";
      wantedBy = [ "exo.target" ];
      after = [ "exo-api.service" ];
      wants = [ "exo-api.service" ];
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/exo-dashboard";
        ExecReload = "${pkgs.coreutils}/bin/kill -HUP $MAINPID";
        Restart = "always";
        RestartSec = "5s";
        RestartSteps = 5;
        RestartMaxDelaySec = "30s";

        # Logging
        StandardOutput = mkIf cfg.logging.journalRateLimit "journal";
        StandardError = mkIf cfg.logging.journalRateLimit "journal";

        # Environment
        EnvironmentFile = "${cfg.configDir}/environment";
        Environment = [
          "EXO_DASHBOARD_PORT=${toString cfg.dashboard.port}"
          "EXO_API_PORT=${toString cfg.apiPort}"
          "NODE_ENV=production"
        ] ++ optionals cfg.dashboard.ssl.enable [
          "EXO_SSL_CERT=${cfg.dashboard.ssl.certificatePath}"
          "EXO_SSL_KEY=${cfg.dashboard.ssl.keyPath}"
        ] ++ optionals cfg.dashboard.authentication.enable [
          "EXO_DASHBOARD_AUTH=true"
          "EXO_DASHBOARD_API_KEYS=${concatStringsSep "," cfg.dashboard.authentication.apiKeys}"
          "EXO_DASHBOARD_SESSION_TIMEOUT=${toString cfg.dashboard.authentication.sessionTimeout}"
        ];

        # Working directory
        WorkingDirectory = cfg.dataDir;

        # Security settings for web server
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/log/exo" ];
        ReadOnlyPaths = [ cfg.configDir "${cfg.package}/share/exo/dashboard" ] ++ optionals cfg.dashboard.ssl.enable [
          cfg.dashboard.ssl.certificatePath
          cfg.dashboard.ssl.keyPath
        ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        RestrictNamespaces = !cfg.security.networkNamespace;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;

        # Network binding for dashboard
        IPAddressDeny = mkIf cfg.security.enableSandbox "any";
        IPAddressAllow = mkIf cfg.security.enableSandbox [ "localhost" "10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16" ];

        # Resource limits
        LimitNOFILE = 65536;
        LimitNPROC = 512;
      } // optionalAttrs cfg.security.enableSandbox {
        # Additional sandboxing
        ProtectProc = "invisible";
        ProcSubset = "pid";
        ProtectHostname = true;
        ProtectClock = true;
        SystemCallArchitectures = "native";
        SystemCallFilter = [ "@system-service" "~@privileged" "~@resources" ];
      } // optionalAttrs cfg.security.networkNamespace {
        PrivateNetwork = true;
      };

      # Logging configuration
      environment.SYSTEMD_LOG_LEVEL = cfg.logging.level;
    };

    # ACME certificate configuration
    security.acme = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.acme.enable) {
      acceptTerms = true;
      defaults.email = cfg.dashboard.ssl.acme.email;

      certs.${cfg.dashboard.ssl.acme.domain} = {
        domain = cfg.dashboard.ssl.acme.domain;
        group = cfg.group;

        # Use HTTP-01 challenge by default
        webroot = "/var/lib/acme/acme-challenge";

        # Post-renewal hook to restart services
        postRun = ''
          systemctl reload-or-restart exo-dashboard.service exo-api.service || true
        '';
      };
    };

    # Update SSL paths when using ACME
    services.exo.dashboard.ssl.certificatePath = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.acme.enable)
      (mkDefault "/var/lib/acme/${cfg.dashboard.ssl.acme.domain}/cert.pem");
    services.exo.dashboard.ssl.keyPath = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.acme.enable)
      (mkDefault "/var/lib/acme/${cfg.dashboard.ssl.acme.domain}/key.pem");

    # Nginx configuration for ACME challenge (if needed)
    services.nginx = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.acme.enable) {
      enable = mkDefault true;

      virtualHosts.${cfg.dashboard.ssl.acme.domain} = {
        locations."/.well-known/acme-challenge" = {
          root = "/var/lib/acme/acme-challenge";
        };

        # Redirect HTTP to HTTPS after certificate is obtained
        locations."/" = {
          return = "301 https://$server_name$request_uri";
        };
      };
    };

    # SSL certificate generation and renewal service
    systemd.services.exo-ssl-setup = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.autoGenerate) {
      description = "Generate EXO SSL Certificates";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo-dashboard.service" "exo-api.service" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root";
        ExecStart = pkgs.writeShellScript "generate-exo-ssl" ''
          CERT_DIR="${cfg.configDir}/ssl"
          CERT_FILE="$CERT_DIR/cert.pem"
          KEY_FILE="$CERT_DIR/key.pem"
          CSR_FILE="$CERT_DIR/cert.csr"
          
          # Create SSL directory
          mkdir -p "$CERT_DIR"
          
          # Function to generate certificate
          generate_cert() {
            echo "Generating self-signed SSL certificate for EXO dashboard..."
            
            # Generate private key
            ${pkgs.openssl}/bin/openssl genrsa -out "$KEY_FILE" 4096
            
            # Generate certificate signing request
            ${pkgs.openssl}/bin/openssl req -new -key "$KEY_FILE" -out "$CSR_FILE" \
              -subj "/C=US/ST=Local/L=Local/O=EXO/CN=localhost"
            
            # Generate self-signed certificate with SAN
            ${pkgs.openssl}/bin/openssl x509 -req -in "$CSR_FILE" -signkey "$KEY_FILE" \
              -out "$CERT_FILE" -days 365 \
              -extensions v3_req -extfile <(cat <<EOF
          [v3_req]
          basicConstraints = CA:FALSE
          keyUsage = nonRepudiation, digitalSignature, keyEncipherment
          subjectAltName = @alt_names
          
          [alt_names]
          DNS.1 = localhost
          DNS.2 = *.localhost
          IP.1 = 127.0.0.1
          IP.2 = ::1
          EOF
          )
            
            # Clean up CSR
            rm -f "$CSR_FILE"
            
            # Set proper permissions
            chown ${cfg.user}:${cfg.group} "$CERT_FILE" "$KEY_FILE"
            chmod 640 "$CERT_FILE" "$KEY_FILE"
            
            echo "SSL certificate generated at $CERT_FILE"
          }
          
          # Check if certificate exists and is valid
          if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
            # Check certificate expiration (renew if expires within 30 days)
            if ${pkgs.openssl}/bin/openssl x509 -checkend 2592000 -noout -in "$CERT_FILE" >/dev/null 2>&1; then
              echo "SSL certificate is valid and not expiring soon"
            else
              echo "SSL certificate is expiring soon, regenerating..."
              generate_cert
            fi
          else
            generate_cert
          fi
          
          # Verify certificate
          if ${pkgs.openssl}/bin/openssl x509 -in "$CERT_FILE" -text -noout >/dev/null 2>&1; then
            echo "SSL certificate verification successful"
          else
            echo "SSL certificate verification failed"
            exit 1
          fi
        '';
      };
    };

    # SSL certificate renewal timer
    systemd.timers.exo-ssl-renewal = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.autoGenerate) {
      description = "EXO SSL Certificate Renewal Timer";
      wantedBy = [ "timers.target" ];

      timerConfig = {
        OnBootSec = "1h";
        OnUnitActiveSec = "1d";
        Unit = "exo-ssl-setup.service";
        Persistent = true;
      };
    };

    # Update SSL paths when auto-generating
    services.exo.dashboard.ssl.certificatePath = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.autoGenerate)
      (mkDefault "${cfg.configDir}/ssl/cert.pem");
    services.exo.dashboard.ssl.keyPath = mkIf (cfg.dashboard.ssl.enable && cfg.dashboard.ssl.autoGenerate)
      (mkDefault "${cfg.configDir}/ssl/key.pem");

    # Health check service
    systemd.services.exo-healthcheck = {
      description = "EXO Health Check";
      after = [ "exo-master.service" "exo-worker.service" "exo-api.service" ] ++ optional cfg.dashboard.enable "exo-dashboard.service";

      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = pkgs.writeShellScript "exo-healthcheck" ''
          # Basic health check for EXO services
          set -e
          
          # Check if master is responding
          if systemctl is-active --quiet exo-master.service; then
            ${pkgs.curl}/bin/curl -f -s http://localhost:${toString cfg.apiPort}/health || exit 1
            echo "Master service health check passed"
          fi
          
          # Check worker status
          if systemctl is-active --quiet exo-worker.service; then
            echo "Worker service health check passed"
          fi
          
          # Check API service
          if systemctl is-active --quiet exo-api.service; then
            ${pkgs.curl}/bin/curl -f -s http://localhost:${toString cfg.apiPort}/api/health || exit 1
            echo "API service health check passed"
          fi
          
          # Check dashboard service
          ${optionalString cfg.dashboard.enable ''
            if systemctl is-active --quiet exo-dashboard.service; then
              DASHBOARD_PROTOCOL="${if cfg.dashboard.ssl.enable then "https" else "http"}"
              ${pkgs.curl}/bin/curl -f -s -k "$DASHBOARD_PROTOCOL://localhost:${toString cfg.dashboard.port}/health" || exit 1
              echo "Dashboard service health check passed"
            fi
          ''}
          
          echo "EXO health check passed"
        '';

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
      };
    };

    systemd.timers.exo-healthcheck = {
      description = "EXO Health Check Timer";
      wantedBy = [ "timers.target" ];

      timerConfig = {
        OnBootSec = "5min";
        OnUnitActiveSec = "5min";
        Unit = "exo-healthcheck.service";
      };
    };

    # Advanced logging and monitoring configuration
    systemd.services.exo-log-manager = {
      description = "EXO Log Management Service";
      wantedBy = [ "multi-user.target" ];
      after = [ "systemd-journald.service" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root";
        ExecStart = pkgs.writeShellScript "setup-exo-logging" ''
          # Configure journal settings for EXO services
          mkdir -p /etc/systemd/journald.conf.d
          
          cat > /etc/systemd/journald.conf.d/exo.conf << EOF
          [Journal]
          # EXO-specific journal configuration
          SystemMaxUse=1G
          SystemKeepFree=2G
          SystemMaxFileSize=${cfg.logging.maxLogSize}
          SystemMaxFiles=${toString cfg.logging.maxLogFiles}
          MaxRetentionSec=7day
          
          # Rate limiting for EXO services
          RateLimitInterval=30s
          RateLimitBurst=10000
          EOF
          
          # Setup log rotation for EXO-specific logs
          mkdir -p /etc/logrotate.d
          cat > /etc/logrotate.d/exo << EOF
          /var/log/exo/*.log {
              daily
              missingok
              rotate ${toString cfg.logging.maxLogFiles}
              compress
              delaycompress
              notifempty
              create 0640 ${cfg.user} ${cfg.group}
              postrotate
                  systemctl reload-or-restart exo-master.service exo-worker.service exo-api.service || true
              endscript
          }
          EOF
          
          # Reload journald to apply new configuration
          systemctl reload-or-restart systemd-journald || true
        '';
      };
    };

    # Monitoring and metrics collection service
    systemd.services.exo-metrics = {
      description = "EXO Metrics Collection";
      wantedBy = [ "exo.target" ];
      after = [ "exo-master.service" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        Restart = "always";
        RestartSec = "10s";

        ExecStart = pkgs.writeShellScript "exo-metrics" ''
          #!/bin/bash
          # EXO metrics collection script
          
          METRICS_DIR="/var/log/exo/metrics"
          mkdir -p "$METRICS_DIR"
          
          while true; do
            TIMESTAMP=$(date -Iseconds)
            
            # Collect system metrics
            {
              echo "timestamp=$TIMESTAMP"
              echo "cpu_usage=$(${pkgs.procps}/bin/top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
              echo "memory_usage=$(${pkgs.procps}/bin/free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')"
              echo "disk_usage=$(${pkgs.coreutils}/bin/df ${cfg.dataDir} | tail -1 | awk '{print $5}' | sed 's/%//')"
              
              # GPU metrics (if available)
              if command -v nvidia-smi >/dev/null 2>&1; then
                echo "gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)"
                echo "gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
              fi
              
              # Network metrics
              if [ -n "${optionalString (cfg.networking.bondInterface != null) cfg.networking.bondInterface}" ]; then
                BOND_IFACE="${optionalString (cfg.networking.bondInterface != null) cfg.networking.bondInterface}"
                echo "network_rx_bytes=$(${pkgs.coreutils}/bin/cat /sys/class/net/$BOND_IFACE/statistics/rx_bytes)"
                echo "network_tx_bytes=$(${pkgs.coreutils}/bin/cat /sys/class/net/$BOND_IFACE/statistics/tx_bytes)"
              fi
              
              # EXO-specific metrics
              if systemctl is-active --quiet exo-master.service; then
                echo "master_status=active"
              else
                echo "master_status=inactive"
              fi
              
              if systemctl is-active --quiet exo-worker.service; then
                echo "worker_status=active"
              else
                echo "worker_status=inactive"
              fi
              
              if systemctl is-active --quiet exo-api.service; then
                echo "api_status=active"
              else
                echo "api_status=inactive"
              fi
              
            } >> "$METRICS_DIR/metrics-$(date +%Y%m%d).log"
            
            sleep 60
          done
        '';

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/log/exo" ];
        ReadOnlyPaths = [ "/sys" "/proc" cfg.dataDir ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
      };
    };

    # Log analysis and alerting service
    systemd.services.exo-log-analyzer = {
      description = "EXO Log Analysis and Alerting";
      after = [ "exo-master.service" "exo-worker.service" ];

      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = pkgs.writeShellScript "exo-log-analyzer" ''
          #!/bin/bash
          # Analyze EXO logs for errors and performance issues
          
          LOG_DIR="/var/log/exo"
          ALERT_FILE="$LOG_DIR/alerts.log"
          
          # Check for critical errors in the last hour
          ERRORS=$(journalctl -u exo-master.service -u exo-worker.service -u exo-api.service \
                   --since "1 hour ago" --priority=err --no-pager -q | wc -l)
          
          if [ "$ERRORS" -gt 10 ]; then
            echo "$(date -Iseconds): HIGH ERROR RATE: $ERRORS errors in the last hour" >> "$ALERT_FILE"
          fi
          
          # Check for memory usage warnings
          MEMORY_WARNINGS=$(journalctl -u exo-master.service -u exo-worker.service \
                           --since "1 hour ago" --grep="memory" --no-pager -q | wc -l)
          
          if [ "$MEMORY_WARNINGS" -gt 5 ]; then
            echo "$(date -Iseconds): MEMORY WARNINGS: $MEMORY_WARNINGS memory-related warnings" >> "$ALERT_FILE"
          fi
          
          # Check for GPU errors
          GPU_ERRORS=$(journalctl -u exo-worker.service \
                      --since "1 hour ago" --grep="GPU\|CUDA\|ROCm" --priority=err --no-pager -q | wc -l)
          
          if [ "$GPU_ERRORS" -gt 0 ]; then
            echo "$(date -Iseconds): GPU ERRORS: $GPU_ERRORS GPU-related errors" >> "$ALERT_FILE"
          fi
          
          # Check service restart frequency
          RESTARTS=$(journalctl -u exo-master.service -u exo-worker.service -u exo-api.service \
                    --since "1 hour ago" --grep="Started\|Stopped" --no-pager -q | wc -l)
          
          if [ "$RESTARTS" -gt 20 ]; then
            echo "$(date -Iseconds): HIGH RESTART RATE: $RESTARTS service restarts in the last hour" >> "$ALERT_FILE"
          fi
          
          # Cleanup old alerts (keep last 7 days)
          find "$LOG_DIR" -name "alerts.log" -mtime +7 -delete 2>/dev/null || true
          
          echo "Log analysis completed at $(date -Iseconds)"
        '';

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/log/exo" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
      };
    };

    systemd.timers.exo-log-analyzer = {
      description = "EXO Log Analysis Timer";
      wantedBy = [ "timers.target" ];

      timerConfig = {
        OnBootSec = "10min";
        OnUnitActiveSec = "1h";
        Unit = "exo-log-analyzer.service";
      };
    };

    # Performance monitoring service
    systemd.services.exo-performance-monitor = {
      description = "EXO Performance Monitor";
      wantedBy = [ "exo.target" ];
      after = [ "exo-master.service" "exo-worker.service" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        Restart = "always";
        RestartSec = "30s";

        ExecStart = pkgs.writeShellScript "exo-performance-monitor" ''
          #!/bin/bash
          # Monitor EXO performance metrics
          
          PERF_LOG="/var/log/exo/performance.log"
          
          while true; do
            TIMESTAMP=$(date -Iseconds)
            
            # Monitor API response times
            if systemctl is-active --quiet exo-api.service; then
              API_RESPONSE=$(${pkgs.curl}/bin/curl -w "%{time_total}" -s -o /dev/null \
                           http://localhost:${toString cfg.apiPort}/health 2>/dev/null || echo "timeout")
              echo "$TIMESTAMP api_response_time=$API_RESPONSE" >> "$PERF_LOG"
            fi
            
            # Monitor model loading times
            MODEL_LOAD_TIME=$(journalctl -u exo-worker.service --since "5 minutes ago" \
                             --grep="Model loaded" --no-pager -q | tail -1 | \
                             grep -o "loaded in [0-9.]*s" | grep -o "[0-9.]*" || echo "0")
            
            if [ "$MODEL_LOAD_TIME" != "0" ]; then
              echo "$TIMESTAMP model_load_time=$MODEL_LOAD_TIME" >> "$PERF_LOG"
            fi
            
            # Monitor inference throughput
            INFERENCE_COUNT=$(journalctl -u exo-worker.service --since "1 minute ago" \
                             --grep="inference completed" --no-pager -q | wc -l)
            echo "$TIMESTAMP inference_per_minute=$INFERENCE_COUNT" >> "$PERF_LOG"
            
            # Rotate performance log if it gets too large
            if [ -f "$PERF_LOG" ] && [ $(stat -c%s "$PERF_LOG") -gt 10485760 ]; then # 10MB
              mv "$PERF_LOG" "$PERF_LOG.old"
              touch "$PERF_LOG"
              chown ${cfg.user}:${cfg.group} "$PERF_LOG"
            fi
            
            sleep 300 # 5 minutes
          done
        '';

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/log/exo" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
      };
    };

    # Prometheus metrics exporter (optional)
    systemd.services.exo-prometheus-exporter = mkIf (config.services.prometheus.enable or false) {
      description = "EXO Prometheus Metrics Exporter";
      wantedBy = [ "exo.target" ];
      after = [ "exo-master.service" ];

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        Restart = "always";
        RestartSec = "10s";

        ExecStart = pkgs.writeShellScript "exo-prometheus-exporter" ''
          #!/bin/bash
          # Export EXO metrics in Prometheus format
          
          METRICS_PORT=9090
          METRICS_FILE="/tmp/exo-metrics.prom"
          
          while true; do
            {
              echo "# HELP exo_service_status EXO service status (1=active, 0=inactive)"
              echo "# TYPE exo_service_status gauge"
              
              if systemctl is-active --quiet exo-master.service; then
                echo "exo_service_status{service=\"master\"} 1"
              else
                echo "exo_service_status{service=\"master\"} 0"
              fi
              
              if systemctl is-active --quiet exo-worker.service; then
                echo "exo_service_status{service=\"worker\"} 1"
              else
                echo "exo_service_status{service=\"worker\"} 0"
              fi
              
              if systemctl is-active --quiet exo-api.service; then
                echo "exo_service_status{service=\"api\"} 1"
              else
                echo "exo_service_status{service=\"api\"} 0"
              fi
              
              # Add more metrics as needed
              echo "# HELP exo_uptime_seconds EXO service uptime in seconds"
              echo "# TYPE exo_uptime_seconds counter"
              
              MASTER_UPTIME=$(systemctl show exo-master.service --property=ActiveEnterTimestamp --value | \
                             xargs -I {} date -d "{}" +%s 2>/dev/null || echo "0")
              if [ "$MASTER_UPTIME" != "0" ]; then
                CURRENT_TIME=$(date +%s)
                UPTIME=$((CURRENT_TIME - MASTER_UPTIME))
                echo "exo_uptime_seconds{service=\"master\"} $UPTIME"
              fi
              
            } > "$METRICS_FILE"
            
            # Serve metrics via simple HTTP server
            ${pkgs.python3}/bin/python3 -m http.server $METRICS_PORT --directory /tmp &
            HTTP_PID=$!
            
            sleep 30
            kill $HTTP_PID 2>/dev/null || true
          done
        '';

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/tmp" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
      };
    };

    # Centralized logging configuration
    services.journald.extraConfig = mkIf cfg.logging.journalRateLimit ''
      # EXO-specific journal configuration
      SystemMaxUse=2G
      SystemKeepFree=4G
      SystemMaxFileSize=${cfg.logging.maxLogSize}
      SystemMaxFiles=${toString cfg.logging.maxLogFiles}
      MaxRetentionSec=7day
      
      # Rate limiting
      RateLimitInterval=30s
      RateLimitBurst=10000
    '';

    # Logrotate configuration for EXO logs
    services.logrotate.settings.exo = {
      files = "/var/log/exo/*.log";
      frequency = "daily";
      rotate = cfg.logging.maxLogFiles;
      compress = true;
      delaycompress = true;
      missingok = true;
      notifempty = true;
      create = "0640 ${cfg.user} ${cfg.group}";
      postrotate = "systemctl reload-or-restart exo-master.service exo-worker.service exo-api.service || true";
    };

    # Configuration validation assertions
    assertions = [
      {
        assertion = cfg.dashboard.ssl.enable -> (cfg.dashboard.ssl.certificatePath != null && cfg.dashboard.ssl.keyPath != null);
        message = "SSL certificate and key paths must be specified when SSL is enabled for the dashboard.";
      }
      {
        assertion = cfg.dashboard.ssl.acme.enable -> (cfg.dashboard.ssl.acme.domain != null && cfg.dashboard.ssl.acme.email != null);
        message = "ACME domain and email must be specified when ACME is enabled.";
      }
      {
        assertion = !(cfg.dashboard.ssl.autoGenerate && cfg.dashboard.ssl.acme.enable);
        message = "Cannot enable both autoGenerate and ACME for SSL certificates. Choose one method.";
      }
      {
        assertion = cfg.dashboard.authentication.enable -> (length cfg.dashboard.authentication.apiKeys > 0);
        message = "At least one API key must be specified when dashboard authentication is enabled.";
      }
      {
        assertion = cfg.k3s.integration -> config.services.k3s.enable;
        message = "K3s must be enabled when EXO K3s integration is requested.";
      }
      {
        assertion = cfg.networking.bondInterface != null -> hasAttr cfg.networking.bondInterface config.systemd.network.bonds;
        message = "Specified bond interface must be configured in systemd.network.bonds when bondInterface is set.";
      }
      {
        assertion = cfg.mode != "master" || cfg.apiPort != cfg.networking.discoveryPort;
        message = "API port and discovery port must be different.";
      }
      {
        assertion = cfg.dashboard.enable -> cfg.dashboard.port != cfg.apiPort;
        message = "Dashboard port and API port must be different.";
      }
    ];

    # Configuration warnings
    warnings = mkMerge [
      (mkIf (cfg.hardware.memoryLimit == null) [
        "No memory limit set for EXO services. Consider setting hardware.memoryLimit to prevent excessive memory usage."
      ])
      (mkIf (cfg.networking.bondInterface == null && cfg.k3s.integration) [
        "No bonded interface specified but K3s integration is enabled. Consider configuring bondInterface for optimal performance."
      ])
      (mkIf (!cfg.dashboard.ssl.enable && cfg.networking.openFirewall) [
        "Dashboard SSL is disabled but firewall ports are open. Consider enabling SSL for secure access."
      ])
    ];
  };
}

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.exo.hardware;

  # Hardware detection script for NixOS module
  hardwareDetectionScript = pkgs.writeShellScript "exo-hardware-detect" ''
    # Hardware detection for EXO NixOS module
    
    detect_nvidia() {
        # Check for NVIDIA GPU and driver
        if ${pkgs.pciutils}/bin/lspci | grep -i nvidia > /dev/null 2>&1; then
            if ${pkgs.kmod}/bin/lsmod | grep -q nvidia; then
                if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
                    return 0
                fi
            fi
        fi
        return 1
    }
    
    detect_amd() {
        # Check for AMD GPU and driver
        if ${pkgs.pciutils}/bin/lspci | grep -i amd | grep -i vga > /dev/null 2>&1; then
            if ${pkgs.kmod}/bin/lsmod | grep -q amdgpu; then
                if [ -d "/dev/dri" ] && [ -e "/dev/kfd" ]; then
                    return 0
                fi
            fi
        fi
        return 1
    }
    
    detect_intel() {
        # Check for Intel Arc/Xe GPU and driver
        if ${pkgs.pciutils}/bin/lspci | grep -i intel | grep -E "(arc|xe|dg|iris)" > /dev/null 2>&1; then
            if ${pkgs.kmod}/bin/lsmod | grep -E "(i915|xe)" > /dev/null 2>&1; then
                if [ -d "/sys/class/drm" ] && ls /dev/dri/render* 2>/dev/null | head -1 > /dev/null; then
                    return 0
                fi
            fi
        fi
        return 1
    }
    
    # Determine best accelerator
    if detect_nvidia; then
        echo "cuda"
    elif detect_amd; then
        echo "rocm"  
    elif detect_intel; then
        echo "intel"
    else
        echo "cpu"
    fi
  '';

in
{
  options.services.exo.hardware = {
    autoDetect = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Automatically detect and configure hardware acceleration.
        When enabled, EXO will automatically select the best available
        hardware accelerator (CUDA, ROCm, Intel, or CPU fallback).
      '';
    };

    preferredAccelerator = mkOption {
      type = types.nullOr (types.enum [ "cuda" "rocm" "intel" "mlx" "cpu" ]);
      default = null;
      description = ''
        Preferred hardware accelerator to use.
        If null, automatic detection will be used.
        If specified, this accelerator will be used regardless of detection results.
      '';
    };

    package = mkOption {
      type = types.package;
      default = pkgs.exo-complete;
      description = ''
        EXO package to use. By default, uses the complete package with
        automatic hardware detection. Can be overridden to use specific
        hardware variants like pkgs.exo-cuda, pkgs.exo-rocm, etc.
      '';
    };

    enableNvidiaSupport = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Enable NVIDIA CUDA support when NVIDIA GPUs are detected.
        Requires NVIDIA drivers to be installed separately.
      '';
    };

    enableAmdSupport = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Enable AMD ROCm support when AMD GPUs are detected.
        Requires AMD GPU drivers (amdgpu) to be loaded.
      '';
    };

    enableIntelSupport = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Enable Intel GPU support when Intel Arc/Xe GPUs are detected.
        Requires Intel GPU drivers (i915/xe) and compute runtime.
      '';
    };

    cpuOptimizations = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Enable CPU optimizations when using CPU-only inference.
        Includes automatic thread count optimization and CPU feature detection.
      '';
    };

    memoryLimit = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "80%";
      description = ''
        Memory limit for EXO processes. Can be specified as a percentage
        (e.g., "80%") or absolute value (e.g., "8G"). If null, no limit is set.
      '';
    };
  };

  config = mkIf config.services.exo.enable {
    # Hardware detection service
    systemd.services.exo-hardware-detect = mkIf cfg.autoDetect {
      description = "EXO Hardware Detection";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo-master.service" "exo-worker.service" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${hardwareDetectionScript}";
        StandardOutput = "file:/var/lib/exo/hardware-type";
        User = "root";
      };

      preStart = ''
        mkdir -p /var/lib/exo
        chown exo:exo /var/lib/exo
      '';
    };

    # Environment variables for hardware configuration
    environment.variables = mkMerge [
      # Base configuration
      (mkIf (cfg.preferredAccelerator != null) {
        EXO_FORCE_ACCELERATOR = cfg.preferredAccelerator;
      })

      # Memory limit
      (mkIf (cfg.memoryLimit != null) {
        EXO_MEMORY_LIMIT = cfg.memoryLimit;
      })

      # CPU optimizations
      (mkIf cfg.cpuOptimizations {
        EXO_CPU_OPTIMIZATIONS = "1";
      })
    ];

    # Hardware-specific system configuration
    hardware = mkMerge [
      # NVIDIA support
      (mkIf (cfg.enableNvidiaSupport && config.hardware.nvidia.modesetting.enable) {
        nvidia = {
          # Ensure NVIDIA settings are compatible with EXO
          powerManagement.enable = mkDefault true;
          powerManagement.finegrained = mkDefault false;
        };
      })

      # AMD GPU support
      (mkIf cfg.enableAmdSupport {
        amdgpu = {
          # Enable AMD GPU support
          enable = mkDefault true;
        };
      })

      # Intel GPU support  
      (mkIf cfg.enableIntelSupport {
        intel-gpu-tools.enable = mkDefault true;
      })
    ];

    # Required packages for hardware detection
    environment.systemPackages = with pkgs; [
      pciutils
      kmod
      # Hardware-specific tools
    ] ++ optionals cfg.enableNvidiaSupport [
      # NVIDIA tools are typically provided by nvidia drivers
    ] ++ optionals cfg.enableAmdSupport [
      rocmPackages.rocm-smi
    ] ++ optionals cfg.enableIntelSupport [
      intel-compute-runtime
      level-zero
    ];

    # Kernel modules for GPU support
    boot.kernelModules = mkMerge [
      (mkIf cfg.enableAmdSupport [ "amdgpu" ])
      (mkIf cfg.enableIntelSupport [ "i915" ])
    ];

    # Device permissions for GPU access
    services.udev.extraRules = mkMerge [
      # AMD GPU device permissions
      (mkIf cfg.enableAmdSupport ''
        # AMD GPU devices
        SUBSYSTEM=="drm", KERNEL=="card*", GROUP="video", MODE="0664"
        SUBSYSTEM=="drm", KERNEL=="render*", GROUP="render", MODE="0664"
        SUBSYSTEM=="misc", KERNEL=="kfd", GROUP="render", MODE="0664"
      '')

      # Intel GPU device permissions
      (mkIf cfg.enableIntelSupport ''
        # Intel GPU devices
        SUBSYSTEM=="drm", KERNEL=="card*", ATTRS{vendor}=="0x8086", GROUP="video", MODE="0664"
        SUBSYSTEM=="drm", KERNEL=="render*", ATTRS{vendor}=="0x8086", GROUP="render", MODE="0664"
      '')
    ];

    # User groups for GPU access
    users.groups = mkMerge [
      (mkIf (cfg.enableAmdSupport || cfg.enableIntelSupport) {
        render = { };
      })
    ];

    # Add exo user to necessary groups
    users.users.exo = mkIf config.services.exo.enable {
      extraGroups = mkMerge [
        [ "video" ]
        (mkIf (cfg.enableAmdSupport || cfg.enableIntelSupport) [ "render" ])
      ];
    };

    # Assertions for configuration validation
    assertions = [
      {
        assertion = cfg.preferredAccelerator == null || cfg.autoDetect == false || cfg.preferredAccelerator != null;
        message = "When preferredAccelerator is set, autoDetect should typically be disabled to avoid conflicts.";
      }

      {
        assertion = !(cfg.enableNvidiaSupport && cfg.preferredAccelerator == "cuda") || config.hardware.nvidia.modesetting.enable;
        message = "NVIDIA modesetting must be enabled when using CUDA acceleration.";
      }
    ];

    # Warnings for common configuration issues
    warnings = mkMerge [
      (mkIf (cfg.enableNvidiaSupport && !config.hardware.nvidia.modesetting.enable) [
        "EXO NVIDIA support is enabled but NVIDIA modesetting is disabled. CUDA acceleration may not work properly."
      ])

      (mkIf (cfg.preferredAccelerator == "rocm" && !cfg.enableAmdSupport) [
        "ROCm accelerator is preferred but AMD support is disabled."
      ])

      (mkIf (cfg.preferredAccelerator == "intel" && !cfg.enableIntelSupport) [
        "Intel accelerator is preferred but Intel GPU support is disabled."
      ])
    ];
  };
}

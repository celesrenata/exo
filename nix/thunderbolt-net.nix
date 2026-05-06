# NixOS module for Thunderbolt 4 networking
#
# Configures TB4 ports as network interfaces for high-bandwidth inter-node
# communication (40 Gbps per port). Used by the tensor-parallel inference
# path to run Gloo all-reduce operations over TB4 instead of ethernet.
#
# Import this module into the gremlin flake and set:
#   services.exo.thunderbolt.enable = true;
#   services.exo.thunderbolt.nodeAssignments."gremlin-1" = [ "10.4.0.1" "10.4.0.2" "10.4.0.3" ];
#
# Requirements: 1.1, 1.3, 12.1, 12.2, 12.5

{ config, lib, pkgs, ... }:

let
  cfg = config.services.exo.thunderbolt;
in
{
  options.services.exo.thunderbolt = {
    enable = lib.mkEnableOption "Thunderbolt 4 networking for tensor-parallel inference";

    subnet = lib.mkOption {
      type = lib.types.str;
      default = "10.4.0.0/24";
      description = ''
        The dedicated subnet for Thunderbolt 4 inter-node traffic.
        Must be separate from the existing ethernet subnet (10.1.1.x)
        to isolate TB4 traffic from management traffic.
      '';
      example = "10.4.0.0/24";
    };

    nodeAssignments = lib.mkOption {
      type = lib.types.attrsOf (lib.types.listOf lib.types.str);
      default = { };
      description = ''
        Mapping of hostname to list of TB4 IP addresses. Each node gets
        one IP per active TB4 port. These are assigned to TB4 network
        interfaces as they appear via systemd-networkd.
      '';
      example = lib.literalExpression ''
        {
          "gremlin-1" = [ "10.4.0.1" "10.4.0.2" "10.4.0.3" ];
          "gremlin-2" = [ "10.4.0.4" "10.4.0.5" "10.4.0.6" ];
          "gremlin-3" = [ "10.4.0.7" "10.4.0.8" "10.4.0.9" ];
          "gremlin-4" = [ "10.4.0.10" "10.4.0.11" "10.4.0.12" ];
        }
      '';
    };

    firewallPorts = lib.mkOption {
      type = lib.types.submodule {
        options = {
          from = lib.mkOption {
            type = lib.types.port;
            default = 49152;
            description = "Start of ephemeral port range for Gloo TCP communication over TB4.";
          };
          to = lib.mkOption {
            type = lib.types.port;
            default = 65535;
            description = "End of ephemeral port range for Gloo TCP communication over TB4.";
          };
        };
      };
      default = { from = 49152; to = 65535; };
      description = ''
        TCP port range opened in the firewall for Gloo all-reduce
        communication over TB4 interfaces. Gloo establishes direct TCP
        connections between all tensor-parallel ranks.
      '';
    };
  };

  config = lib.mkIf cfg.enable (
    let
      # Extract the prefix length from the subnet (e.g., "10.4.0.0/24" → "24")
      prefixLength = lib.last (lib.splitString "/" cfg.subnet);

      # Get the list of TB4 IPs assigned to this node by hostname lookup.
      # Falls back to empty list if this hostname has no assignment.
      thisNodeIPs = cfg.nodeAssignments.${config.networking.hostName} or [];

      # Generate systemd-networkd .network units — one per TB4 interface.
      # Each TB4 port appears as thunderbolt0, thunderbolt1, etc. We create
      # a .network unit for each expected interface based on the number of
      # IPs assigned to this node.
      #
      # Example: if gremlin-1 has 3 IPs assigned, we generate units for
      # thunderbolt0 (10.4.0.1/24), thunderbolt1 (10.4.0.2/24), thunderbolt2 (10.4.0.3/24)
      tb4NetworkUnits = lib.listToAttrs (
        lib.imap0 (index: ip:
          lib.nameValuePair "50-thunderbolt${toString index}" {
            matchConfig = {
              Name = "thunderbolt${toString index}";
              Driver = "thunderbolt-net";
            };
            networkConfig = {
              Address = "${ip}/${prefixLength}";
              # No gateway — TB4 is a local subnet only, not a route to the internet.
              # DHCP is disabled; we use static IPs exclusively.
              DHCP = "no";
              LinkLocalAddressing = "no";
              # Disable IPv6 link-local to keep the interface clean for Gloo.
              IPv6AcceptRA = false;
            };
            linkConfig = {
              # Bring the interface up as soon as it appears.
              RequiredForOnline = "no";
            };
          }
        ) thisNodeIPs
      );
    in
    {
      # Load thunderbolt kernel modules at boot for TB4 networking.
      # - thunderbolt: Core Thunderbolt/USB4 driver (device enumeration, security)
      # - thunderbolt-net: Presents TB4 connections as network interfaces
      # Requirement 12.1
      boot.kernelModules = [ "thunderbolt" "thunderbolt-net" ];

      # Enable systemd-networkd to manage TB4 interfaces.
      # This does not conflict with NetworkManager or other network config
      # because we only match thunderbolt* interfaces by name and driver.
      # Requirement 12.3
      systemd.network.enable = true;

      # Generate .network units for each TB4 interface on this node.
      # Each unit matches a specific thunderboltN interface by name and driver,
      # assigning the corresponding static IP from nodeAssignments.
      # Requirement 1.2, 1.3, 12.3
      systemd.network.networks = tb4NetworkUnits;

      # Authorize Thunderbolt devices automatically for networking.
      # TB4 has security levels that can block new connections until approved.
      # This udev rule sets the security level to allow networking without
      # manual approval, enabling TB4 interfaces to come up automatically.
      # Requirement 12.5
      #
      # The second rule tags Thunderbolt network interfaces so systemd-networkd
      # can match them reliably by driver name.
      # Requirement 1.2
      services.udev.extraRules = ''
        # Authorize Thunderbolt devices for networking (set authorized flag)
        # When a new Thunderbolt device appears, authorize it immediately so
        # the thunderbolt-net driver can create network interfaces.
        ACTION=="add", SUBSYSTEM=="thunderbolt", ATTR{authorized}=="0", ATTR{authorized}="1"

        # Tag Thunderbolt network interfaces for systemd-networkd matching.
        # The thunderbolt-net driver creates interfaces named thunderbolt*.
        ACTION=="add", SUBSYSTEM=="net", KERNEL=="thunderbolt*", TAG+="systemd", ENV{SYSTEMD_ALIAS}="/sys/subsystem/net/devices/%k"
      '';

      # Open Gloo TCP port range on TB4 interfaces for tensor-parallel
      # all-reduce communication. Also allow ICMP for reachability probes
      # used by the topology discoverer.
      # Requirement 1.4, 12.4
      networking.firewall.interfaces = lib.listToAttrs (
        lib.imap0 (index: _ip:
          lib.nameValuePair "thunderbolt${toString index}" {
            allowedTCPPortRanges = [
              { from = cfg.firewallPorts.from; to = cfg.firewallPorts.to; }
            ];
          }
        ) thisNodeIPs
      );

      # Allow ICMP (ping) on TB4 interfaces for topology discovery probes.
      # NixOS firewall allows ICMP by default when the firewall is enabled,
      # but we explicitly ensure it via extraCommands for TB4 interfaces.
      # Requirement 12.4
      networking.firewall.extraCommands = lib.concatStringsSep "\n" (
        lib.imap0 (index: _ip:
          "iptables -A nixos-fw -i thunderbolt${toString index} -p icmp -j nixos-fw-accept"
        ) thisNodeIPs
      );

      # systemd oneshot service that waits for at least one TB4 interface
      # to come up with carrier, then logs the status of all TB4 interfaces.
      # Provides a dependency target for the exo service.
      # Requirement 1.5, 12.6
      systemd.services.tb4-ready = {
        description = "Wait for Thunderbolt 4 network interfaces";
        after = [ "systemd-networkd.service" "systemd-udevd.service" ];
        wants = [ "systemd-networkd.service" ];
        wantedBy = [ "multi-user.target" ];

        serviceConfig = {
          Type = "oneshot";
          RemainAfterExit = true;
          TimeoutStartSec = 120;
        };

        # The script polls for TB4 interfaces with carrier. It exits
        # successfully when at least one is found, or after timeout.
        # Link-down events are logged gracefully without failing.
        script = ''
          #!${pkgs.bash}/bin/bash
          set -u

          TIMEOUT=110  # slightly less than systemd TimeoutStartSec
          POLL_INTERVAL=2
          ELAPSED=0

          echo "tb4-ready: Waiting for Thunderbolt 4 interfaces..."

          while [ $ELAPSED -lt $TIMEOUT ]; do
            FOUND=0

            for iface in /sys/class/net/thunderbolt*; do
              [ -e "$iface" ] || continue
              IFNAME=$(basename "$iface")

              # Check carrier (link up)
              CARRIER=$(cat "$iface/carrier" 2>/dev/null || echo "0")
              OPERSTATE=$(cat "$iface/operstate" 2>/dev/null || echo "unknown")

              if [ "$CARRIER" = "1" ]; then
                # Get IP addresses assigned to this interface
                IPS=$(${pkgs.iproute2}/bin/ip -4 addr show "$IFNAME" 2>/dev/null | \
                      ${pkgs.gnugrep}/bin/grep -oP 'inet \K[0-9./]+' || echo "no-ip")
                echo "tb4-ready: $IFNAME is UP (carrier=1, operstate=$OPERSTATE, ips=$IPS)"
                FOUND=$((FOUND + 1))
              else
                # Link down — log gracefully, don't fail
                echo "tb4-ready: $IFNAME is DOWN (carrier=$CARRIER, operstate=$OPERSTATE)"
              fi
            done

            if [ $FOUND -gt 0 ]; then
              echo "tb4-ready: $FOUND TB4 interface(s) ready"
              exit 0
            fi

            sleep $POLL_INTERVAL
            ELAPSED=$((ELAPSED + POLL_INTERVAL))
          done

          # Timeout reached — log all interface states and exit successfully
          # to avoid blocking boot. The exo service will handle missing TB4
          # by falling back to pipeline parallelism.
          echo "tb4-ready: WARNING - No TB4 interfaces with carrier detected after ''${TIMEOUT}s"
          echo "tb4-ready: Listing all thunderbolt interfaces:"
          for iface in /sys/class/net/thunderbolt*; do
            [ -e "$iface" ] || continue
            IFNAME=$(basename "$iface")
            OPERSTATE=$(cat "$iface/operstate" 2>/dev/null || echo "unknown")
            echo "tb4-ready:   $IFNAME operstate=$OPERSTATE"
          done
          # Exit 0 even on timeout — don't block boot, exo handles degraded mode
          exit 0
        '';
      };

      # The exo service should start after TB4 interfaces are ready.
      # Using 'wants' (not 'requires') so exo still starts if TB4 is
      # unavailable — it will fall back to pipeline parallelism.
      # Requirement 1.6, 12.6
      systemd.services.exo = {
        after = [ "tb4-ready.service" ];
        wants = [ "tb4-ready.service" ];
      };
    }
  );
}

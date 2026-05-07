# NixOS module evaluation test for thunderbolt-net.nix
#
# Run with:
#   nix-instantiate --eval nix/tests/thunderbolt-net-test.nix --strict
#
# This evaluates the thunderbolt-net module with a test configuration and
# asserts the expected outputs match requirements 12.1, 12.2, 12.3, 12.4, 12.6.

let
  pkgs = import <nixpkgs> { };

  # Evaluate a minimal NixOS configuration with our thunderbolt-net module.
  # Uses eval-config.nix which is the standard NixOS module evaluation entry point.
  evalConfig = import (pkgs.path + "/nixos/lib/eval-config.nix");

  testSystem = evalConfig {
    system = "x86_64-linux";
    modules = [
      ../thunderbolt-net.nix
      ({ lib, ... }: {
        # Minimal config to make NixOS evaluation happy
        system.stateVersion = "24.11";
        boot.loader.grub.device = "nodev";
        fileSystems."/" = { device = "/dev/sda1"; fsType = "ext4"; };

        # Set hostname so nodeAssignments can resolve
        networking.hostName = "gremlin-1";

        # Enable the thunderbolt module with test configuration
        services.exo.thunderbolt = {
          enable = true;
          subnet = "10.4.0.0/24";
          nodeAssignments = {
            "gremlin-1" = [ "10.4.0.1" "10.4.0.2" "10.4.0.3" ];
            "gremlin-2" = [ "10.4.0.4" "10.4.0.5" "10.4.0.6" ];
            "gremlin-3" = [ "10.4.0.7" "10.4.0.8" "10.4.0.9" ];
            "gremlin-4" = [ "10.4.0.10" "10.4.0.11" "10.4.0.12" ];
          };
          firewallPorts = { from = 49152; to = 65535; };
        };
      })
    ];
  };

  cfg = testSystem.config;
  lib = pkgs.lib;

  # ─── Assertion helpers ───────────────────────────────────────────────────────

  assertMsg = cond: msg:
    if cond then true
    else builtins.throw "ASSERTION FAILED: ${msg}";

  # ─── Test 1: Kernel modules (Requirement 12.1) ──────────────────────────────
  #
  # boot.kernelModules must contain "thunderbolt" and "thunderbolt-net"

  hasThunderboltModule = builtins.elem "thunderbolt" cfg.boot.kernelModules;
  hasThunderboltNetModule = builtins.elem "thunderbolt-net" cfg.boot.kernelModules;

  test_kernel_modules =
    assertMsg hasThunderboltModule
      "boot.kernelModules does not contain 'thunderbolt'" &&
    assertMsg hasThunderboltNetModule
      "boot.kernelModules does not contain 'thunderbolt-net'";

  # ─── Test 2: Firewall rules include Gloo port range (Requirement 12.4) ──────
  #
  # networking.firewall.interfaces.thunderboltN must have TCP port range 49152-65535

  firewallInterfaces = cfg.networking.firewall.interfaces;

  # gremlin-1 has 3 IPs, so we expect thunderbolt0, thunderbolt1, thunderbolt2
  hasThunderbolt0Firewall = builtins.hasAttr "thunderbolt0" firewallInterfaces;
  hasThunderbolt1Firewall = builtins.hasAttr "thunderbolt1" firewallInterfaces;
  hasThunderbolt2Firewall = builtins.hasAttr "thunderbolt2" firewallInterfaces;

  # Check that the port range is correct on thunderbolt0
  tb0PortRanges = firewallInterfaces.thunderbolt0.allowedTCPPortRanges;
  hasGlooPortRange = builtins.any (range:
    range.from == 49152 && range.to == 65535
  ) tb0PortRanges;

  test_firewall_rules =
    assertMsg hasThunderbolt0Firewall
      "networking.firewall.interfaces does not contain 'thunderbolt0'" &&
    assertMsg hasThunderbolt1Firewall
      "networking.firewall.interfaces does not contain 'thunderbolt1'" &&
    assertMsg hasThunderbolt2Firewall
      "networking.firewall.interfaces does not contain 'thunderbolt2'" &&
    assertMsg hasGlooPortRange
      "thunderbolt0 firewall does not include Gloo port range 49152-65535";

  # ─── Test 3: systemd-networkd configuration (Requirement 12.3) ──────────────
  #
  # systemd.network.networks must have entries for TB4 interfaces

  networkdEnabled = cfg.systemd.network.enable;
  networks = cfg.systemd.network.networks;

  hasNetwork0 = builtins.hasAttr "50-thunderbolt0" networks;
  hasNetwork1 = builtins.hasAttr "50-thunderbolt1" networks;
  hasNetwork2 = builtins.hasAttr "50-thunderbolt2" networks;

  # Verify the first network unit has correct match and address
  net0 = networks."50-thunderbolt0";
  net0MatchesName = net0.matchConfig.Name == "thunderbolt0";
  net0MatchesDriver = net0.matchConfig.Driver == "thunderbolt-net";
  net0HasCorrectAddress = net0.networkConfig.Address == "10.4.0.1/24";

  test_networkd_config =
    assertMsg networkdEnabled
      "systemd.network.enable is not true" &&
    assertMsg hasNetwork0
      "systemd.network.networks does not contain '50-thunderbolt0'" &&
    assertMsg hasNetwork1
      "systemd.network.networks does not contain '50-thunderbolt1'" &&
    assertMsg hasNetwork2
      "systemd.network.networks does not contain '50-thunderbolt2'" &&
    assertMsg net0MatchesName
      "50-thunderbolt0 matchConfig.Name is not 'thunderbolt0'" &&
    assertMsg net0MatchesDriver
      "50-thunderbolt0 matchConfig.Driver is not 'thunderbolt-net'" &&
    assertMsg net0HasCorrectAddress
      "50-thunderbolt0 networkConfig.Address is not '10.4.0.1/24'";

  # ─── Test 4: Service dependency ordering (Requirement 12.6) ─────────────────
  #
  # systemd.services.exo.after must include "tb4-ready.service"
  # systemd.services.tb4-ready must exist

  exoService = cfg.systemd.services.exo;
  tb4ReadyService = cfg.systemd.services.tb4-ready;

  exoAfterIncludesTb4Ready = builtins.elem "tb4-ready.service" exoService.after;
  tb4ReadyExists = builtins.hasAttr "tb4-ready" cfg.systemd.services;
  tb4ReadyAfterNetworkd = builtins.elem "systemd-networkd.service" tb4ReadyService.after;

  test_service_ordering =
    assertMsg tb4ReadyExists
      "systemd.services.tb4-ready does not exist" &&
    assertMsg exoAfterIncludesTb4Ready
      "systemd.services.exo.after does not include 'tb4-ready.service'" &&
    assertMsg tb4ReadyAfterNetworkd
      "systemd.services.tb4-ready.after does not include 'systemd-networkd.service'";

  # ─── All tests ──────────────────────────────────────────────────────────────

in
  assert test_kernel_modules;
  assert test_firewall_rules;
  assert test_networkd_config;
  assert test_service_ordering;
  {
    result = "All thunderbolt-net module evaluation tests passed";
    tests_run = [
      "kernel_modules: thunderbolt and thunderbolt-net present in boot.kernelModules"
      "firewall_rules: Gloo port range 49152-65535 on thunderbolt interfaces"
      "networkd_config: systemd-networkd networks generated for TB4 interfaces"
      "service_ordering: exo.after includes tb4-ready.service, tb4-ready exists"
    ];
  }

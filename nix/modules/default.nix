# EXO NixOS Modules
# This file provides easy access to all EXO NixOS modules

{
  # Main service module - provides the core EXO service configuration
  exo-service = ./exo-service.nix;
  
  # Hardware detection and configuration module
  exo-hardware = ./exo-hardware.nix;
  
  # Network interface detection and optimization module
  exo-networking = ./exo-networking.nix;
  
  # K3s integration module - provides Kubernetes orchestration support
  exo-k3s = ./exo-k3s.nix;
}
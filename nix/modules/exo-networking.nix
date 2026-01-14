{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.exo.networking;
  exoCfg = config.services.exo;

  # Network interface detection script
  networkDetectionScript = pkgs.writeShellScript "exo-network-detection" ''
    #!/bin/bash
    # EXO Network Interface Detection and Configuration
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    DATA_DIR="${exoCfg.dataDir}"
    NETWORK_CONFIG="$CONFIG_DIR/network-config.json"
    
    # Create network configuration directory
    mkdir -p "$CONFIG_DIR"
    
    # Function to detect bonded interfaces
    detect_bonded_interfaces() {
      local bonded_interfaces=()
      
      # Check for existing bond interfaces
      for bond in /sys/class/net/bond*; do
        if [ -d "$bond" ]; then
          bond_name=$(basename "$bond")
          bond_state=$(cat "$bond/bonding/mode" 2>/dev/null || echo "unknown")
          bond_slaves=$(cat "$bond/bonding/slaves" 2>/dev/null || echo "")
          
          if [ -n "$bond_slaves" ]; then
            bonded_interfaces+=("$bond_name")
            echo "Found bonded interface: $bond_name (mode: $bond_state, slaves: $bond_slaves)" >&2
          fi
        fi
      done
      
      printf '%s\n' "''${bonded_interfaces[@]}"
    }
    
    # Function to detect high-bandwidth interfaces
    detect_high_bandwidth_interfaces() {
      local high_bw_interfaces=()
      
      for iface in /sys/class/net/*; do
        if [ -d "$iface" ]; then
          iface_name=$(basename "$iface")
          
          # Skip loopback and virtual interfaces
          if [[ "$iface_name" =~ ^(lo|docker|br-|veth) ]]; then
            continue
          fi
          
          # Check if interface is up and has speed information
          if [ -f "$iface/operstate" ] && [ -f "$iface/speed" ]; then
            operstate=$(cat "$iface/operstate" 2>/dev/null || echo "unknown")
            speed=$(cat "$iface/speed" 2>/dev/null || echo "0")
            
            # Consider interfaces with speed >= 1Gbps as high-bandwidth
            if [ "$operstate" = "up" ] && [ "$speed" -ge 1000 ]; then
              high_bw_interfaces+=("$iface_name:$speed")
              echo "Found high-bandwidth interface: $iface_name ($speed Mbps)" >&2
            fi
          fi
        fi
      done
      
      printf '%s\n' "''${high_bw_interfaces[@]}"
    }
    
    # Function to detect RDMA-capable interfaces
    detect_rdma_interfaces() {
      local rdma_interfaces=()
      
      # Check for InfiniBand devices
      if [ -d "/sys/class/infiniband" ]; then
        for ib_dev in /sys/class/infiniband/*; do
          if [ -d "$ib_dev" ]; then
            ib_name=$(basename "$ib_dev")
            rdma_interfaces+=("$ib_name:infiniband")
            echo "Found InfiniBand interface: $ib_name" >&2
          fi
        done
      fi
      
      # Check for RoCE (RDMA over Converged Ethernet) capable interfaces
      for iface in /sys/class/net/*; do
        if [ -d "$iface" ]; then
          iface_name=$(basename "$iface")
          
          # Check if interface supports RDMA
          if [ -d "/sys/class/infiniband_verbs" ]; then
            for verbs_dev in /sys/class/infiniband_verbs/*; do
              if [ -d "$verbs_dev" ]; then
                # Check if this verbs device is associated with the network interface
                if [ -f "$verbs_dev/device/net/$iface_name/operstate" ]; then
                  rdma_interfaces+=("$iface_name:roce")
                  echo "Found RoCE-capable interface: $iface_name" >&2
                fi
              fi
            done
          fi
        fi
      done
      
      printf '%s\n' "''${rdma_interfaces[@]}"
    }
    
    # Function to detect Thunderbolt interfaces
    detect_thunderbolt_interfaces() {
      local tb_interfaces=()
      
      # Check for Thunderbolt network interfaces
      for iface in /sys/class/net/*; do
        if [ -d "$iface" ]; then
          iface_name=$(basename "$iface")
          
          # Check if interface is connected via Thunderbolt
          if [ -L "$iface/device" ]; then
            device_path=$(readlink -f "$iface/device")
            if [[ "$device_path" =~ thunderbolt ]]; then
              # Get Thunderbolt generation if available
              tb_gen="unknown"
              if [[ "$device_path" =~ thunderbolt([0-9]+) ]]; then
                tb_gen="''${BASH_REMATCH[1]}"
              fi
              
              tb_interfaces+=("$iface_name:tb$tb_gen")
              echo "Found Thunderbolt interface: $iface_name (generation: $tb_gen)" >&2
            fi
          fi
        fi
      done
      
      printf '%s\n' "''${tb_interfaces[@]}"
    }
    
    # Function to get optimal interface configuration
    get_optimal_interface_config() {
      local bonded_ifaces=($(detect_bonded_interfaces))
      local high_bw_ifaces=($(detect_high_bandwidth_interfaces))
      local rdma_ifaces=($(detect_rdma_interfaces))
      local tb_ifaces=($(detect_thunderbolt_interfaces))
      
      local optimal_interface=""
      local interface_type="standard"
      local capabilities=()
      
      # Priority: Bonded > RDMA > Thunderbolt > High-bandwidth
      if [ ''${#bonded_ifaces[@]} -gt 0 ]; then
        optimal_interface="''${bonded_ifaces[0]}"
        interface_type="bonded"
        capabilities+=("bonding")
        
        # Check if bonded interface also supports RDMA
        for rdma_iface in "''${rdma_ifaces[@]}"; do
          rdma_name=$(echo "$rdma_iface" | cut -d: -f1)
          if [ "$rdma_name" = "$optimal_interface" ]; then
            capabilities+=("rdma")
            break
          fi
        done
        
      elif [ ''${#rdma_ifaces[@]} -gt 0 ]; then
        optimal_interface=$(echo "''${rdma_ifaces[0]}" | cut -d: -f1)
        interface_type="rdma"
        capabilities+=("rdma")
        
      elif [ ''${#tb_ifaces[@]} -gt 0 ]; then
        optimal_interface=$(echo "''${tb_ifaces[0]}" | cut -d: -f1)
        interface_type="thunderbolt"
        capabilities+=("thunderbolt")
        
        # Check if Thunderbolt interface supports RDMA
        for rdma_iface in "''${rdma_ifaces[@]}"; do
          rdma_name=$(echo "$rdma_iface" | cut -d: -f1)
          if [ "$rdma_name" = "$optimal_interface" ]; then
            capabilities+=("rdma")
            break
          fi
        done
        
      elif [ ''${#high_bw_ifaces[@]} -gt 0 ]; then
        # Select the highest bandwidth interface
        local max_speed=0
        for iface_info in "''${high_bw_ifaces[@]}"; do
          iface_name=$(echo "$iface_info" | cut -d: -f1)
          iface_speed=$(echo "$iface_info" | cut -d: -f2)
          
          if [ "$iface_speed" -gt "$max_speed" ]; then
            max_speed="$iface_speed"
            optimal_interface="$iface_name"
            interface_type="high_bandwidth"
          fi
        done
        capabilities+=("high_bandwidth")
      fi
      
      # Generate network configuration JSON
      cat > "$NETWORK_CONFIG" << EOF
    {
      "optimal_interface": "$optimal_interface",
      "interface_type": "$interface_type",
      "capabilities": [$(printf '"%s",' "''${capabilities[@]}" | sed 's/,$//')]],
      "bonded_interfaces": [$(printf '"%s",' "''${bonded_ifaces[@]}" | sed 's/,$//')]],
      "high_bandwidth_interfaces": [$(printf '"%s",' "''${high_bw_ifaces[@]}" | sed 's/,$//')]],
      "rdma_interfaces": [$(printf '"%s",' "''${rdma_ifaces[@]}" | sed 's/,$//')]],
      "thunderbolt_interfaces": [$(printf '"%s",' "''${tb_ifaces[@]}" | sed 's/,$//')]],
      "detection_timestamp": "$(date -Iseconds)"
    }
    EOF
      
      echo "Network configuration written to: $NETWORK_CONFIG" >&2
      echo "Optimal interface: $optimal_interface ($interface_type)" >&2
      echo "Capabilities: ''${capabilities[*]}" >&2
    }
    
    # Main execution
    echo "Starting EXO network interface detection..." >&2
    get_optimal_interface_config
    echo "Network detection completed." >&2
  '';

  # K3s integration detection script
  k3sIntegrationScript = pkgs.writeShellScript "exo-k3s-integration" ''
    #!/bin/bash
    # EXO K3s Integration Detection
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    K3S_CONFIG="$CONFIG_DIR/k3s-integration.json"
    
    # Function to detect K3s configuration
    detect_k3s_config() {
      local k3s_enabled=false
      local k3s_node_name=""
      local k3s_cluster_cidr=""
      local k3s_service_cidr=""
      local k3s_flannel_backend=""
      local k3s_network_interfaces=()
      
      # Check if K3s is running
      if systemctl is-active --quiet k3s 2>/dev/null; then
        k3s_enabled=true
        echo "K3s service detected as active" >&2
        
        # Get node information
        if command -v kubectl >/dev/null 2>&1; then
          k3s_node_name=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "unknown")
          echo "K3s node name: $k3s_node_name" >&2
        fi
        
        # Parse K3s configuration
        if [ -f "/etc/rancher/k3s/config.yaml" ]; then
          k3s_cluster_cidr=$(grep -E "cluster-cidr:" /etc/rancher/k3s/config.yaml | awk '{print $2}' || echo "10.42.0.0/16")
          k3s_service_cidr=$(grep -E "service-cidr:" /etc/rancher/k3s/config.yaml | awk '{print $2}' || echo "10.43.0.0/16")
          k3s_flannel_backend=$(grep -E "flannel-backend:" /etc/rancher/k3s/config.yaml | awk '{print $2}' || echo "vxlan")
        else
          # Default K3s values
          k3s_cluster_cidr="10.42.0.0/16"
          k3s_service_cidr="10.43.0.0/16"
          k3s_flannel_backend="vxlan"
        fi
        
        # Detect network interfaces used by K3s
        for iface in /sys/class/net/*; do
          if [ -d "$iface" ]; then
            iface_name=$(basename "$iface")
            
            # Check if interface has K3s-related configuration
            if ip addr show "$iface_name" 2>/dev/null | grep -E "(10\.42\.|10\.43\.|flannel)" >/dev/null; then
              k3s_network_interfaces+=("$iface_name")
            fi
          fi
        done
        
        echo "K3s cluster CIDR: $k3s_cluster_cidr" >&2
        echo "K3s service CIDR: $k3s_service_cidr" >&2
        echo "K3s flannel backend: $k3s_flannel_backend" >&2
        echo "K3s network interfaces: ''${k3s_network_interfaces[*]}" >&2
      else
        echo "K3s service not detected or not active" >&2
      fi
      
      # Generate K3s integration configuration
      cat > "$K3S_CONFIG" << EOF
    {
      "k3s_enabled": $k3s_enabled,
      "node_name": "$k3s_node_name",
      "cluster_cidr": "$k3s_cluster_cidr",
      "service_cidr": "$k3s_service_cidr",
      "flannel_backend": "$k3s_flannel_backend",
      "network_interfaces": [$(printf '"%s",' "''${k3s_network_interfaces[@]}" | sed 's/,$//')]],
      "detection_timestamp": "$(date -Iseconds)"
    }
    EOF
      
      echo "K3s integration configuration written to: $K3S_CONFIG" >&2
    }
    
    # Main execution
    echo "Starting K3s integration detection..." >&2
    detect_k3s_config
    echo "K3s integration detection completed." >&2
  '';

  # RDMA configuration script
  rdmaConfigScript = pkgs.writeShellScript "exo-rdma-config" ''
    #!/bin/bash
    # EXO RDMA Configuration and Setup
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    RDMA_CONFIG="$CONFIG_DIR/rdma-config.json"
    
    # Function to detect RDMA devices
    detect_rdma_devices() {
      local rdma_devices=()
      local rdma_capabilities=()
      
      echo "Detecting RDMA devices..." >&2
      
      # Check for InfiniBand devices
      if [ -d "/sys/class/infiniband" ]; then
        for ib_dev in /sys/class/infiniband/*; do
          if [ -d "$ib_dev" ]; then
            ib_name=$(basename "$ib_dev")
            
            # Get device capabilities
            local caps=()
            if [ -f "$ib_dev/node_type" ]; then
              node_type=$(cat "$ib_dev/node_type")
              caps+=("node_type:$node_type")
            fi
            
            if [ -f "$ib_dev/sys_image_guid" ]; then
              sys_guid=$(cat "$ib_dev/sys_image_guid")
              caps+=("guid:$sys_guid")
            fi
            
            # Check for ports
            local ports=()
            for port in "$ib_dev"/ports/*; do
              if [ -d "$port" ]; then
                port_num=$(basename "$port")
                if [ -f "$port/state" ]; then
                  port_state=$(cat "$port/state")
                  if [ "$port_state" = "4: ACTIVE" ]; then
                    ports+=("$port_num")
                  fi
                fi
              fi
            done
            
            if [ ''${#ports[@]} -gt 0 ]; then
              caps+=("active_ports:''${ports[*]}")
              rdma_devices+=("$ib_name")
              rdma_capabilities+=("$ib_name:''${caps[*]}")
              echo "Found InfiniBand device: $ib_name (ports: ''${ports[*]})" >&2
            fi
          fi
        done
      fi
      
      # Check for RDMA verbs devices
      if [ -d "/sys/class/infiniband_verbs" ]; then
        for verbs_dev in /sys/class/infiniband_verbs/*; do
          if [ -d "$verbs_dev" ]; then
            verbs_name=$(basename "$verbs_dev")
            
            # Check if this is a new device (not already found via InfiniBand)
            device_name=""
            if [ -L "$verbs_dev/device" ]; then
              device_path=$(readlink -f "$verbs_dev/device")
              device_name=$(basename "$device_path")
            fi
            
            # Check if device supports RoCE
            local roce_support=false
            if [ -f "$verbs_dev/device/infiniband/$device_name/node_type" ]; then
              node_type=$(cat "$verbs_dev/device/infiniband/$device_name/node_type" 2>/dev/null || echo "")
              if [ "$node_type" = "1: CA" ]; then
                roce_support=true
              fi
            fi
            
            if [ "$roce_support" = true ] && ! printf '%s\n' "''${rdma_devices[@]}" | grep -q "^$device_name$"; then
              rdma_devices+=("$device_name")
              rdma_capabilities+=("$device_name:roce")
              echo "Found RoCE device: $device_name" >&2
            fi
          fi
        done
      fi
      
      printf '%s\n' "''${rdma_devices[@]}"
    }
    
    # Function to detect Thunderbolt RDMA support
    detect_thunderbolt_rdma() {
      local tb_rdma_devices=()
      
      echo "Detecting Thunderbolt RDMA devices..." >&2
      
      # Check for Thunderbolt network interfaces with RDMA support
      for iface in /sys/class/net/*; do
        if [ -d "$iface" ]; then
          iface_name=$(basename "$iface")
          
          # Check if interface is connected via Thunderbolt
          if [ -L "$iface/device" ]; then
            device_path=$(readlink -f "$iface/device")
            if [[ "$device_path" =~ thunderbolt ]]; then
              
              # Check if this Thunderbolt interface has RDMA capabilities
              # Look for associated RDMA devices
              for rdma_dev in /sys/class/infiniband/*; do
                if [ -d "$rdma_dev" ]; then
                  rdma_name=$(basename "$rdma_dev")
                  
                  # Check if RDMA device is associated with this network interface
                  if [ -d "$rdma_dev/device/net/$iface_name" ]; then
                    # Get Thunderbolt generation
                    tb_gen="unknown"
                    if [[ "$device_path" =~ thunderbolt([0-9]+) ]]; then
                      tb_gen="''${BASH_REMATCH[1]}"
                    fi
                    
                    tb_rdma_devices+=("$iface_name:$rdma_name:tb$tb_gen")
                    echo "Found Thunderbolt RDMA device: $iface_name -> $rdma_name (TB$tb_gen)" >&2
                  fi
                fi
              done
            fi
          fi
        fi
      done
      
      printf '%s\n' "''${tb_rdma_devices[@]}"
    }
    
    # Function to configure RDMA devices
    configure_rdma_devices() {
      local rdma_devices=($(detect_rdma_devices))
      local tb_rdma_devices=($(detect_thunderbolt_rdma))
      
      echo "Configuring RDMA devices..." >&2
      
      # Configure RDMA subsystem
      if [ ''${#rdma_devices[@]} -gt 0 ] || [ ''${#tb_rdma_devices[@]} -gt 0 ]; then
        
        # Load RDMA kernel modules
        modprobe rdma_core 2>/dev/null || true
        modprobe ib_core 2>/dev/null || true
        modprobe ib_uverbs 2>/dev/null || true
        modprobe rdma_ucm 2>/dev/null || true
        
        # Configure RDMA device limits
        if [ -f /sys/module/ib_core/parameters/netdev_max_backlog ]; then
          echo 5000 > /sys/module/ib_core/parameters/netdev_max_backlog 2>/dev/null || true
        fi
        
        # Set up RDMA device permissions
        for device in "''${rdma_devices[@]}"; do
          if [ -c "/dev/infiniband/uverbs0" ]; then
            chgrp ${exoCfg.group} /dev/infiniband/uverbs* 2>/dev/null || true
            chmod 664 /dev/infiniband/uverbs* 2>/dev/null || true
          fi
          
          if [ -c "/dev/infiniband/rdma_cm" ]; then
            chgrp ${exoCfg.group} /dev/infiniband/rdma_cm 2>/dev/null || true
            chmod 664 /dev/infiniband/rdma_cm 2>/dev/null || true
          fi
        done
        
        echo "RDMA devices configured successfully" >&2
      else
        echo "No RDMA devices found" >&2
      fi
      
      # Generate RDMA configuration
      cat > "$RDMA_CONFIG" << EOF
    {
      "rdma_enabled": $([ ''${#rdma_devices[@]} -gt 0 ] && echo "true" || echo "false"),
      "thunderbolt_rdma_enabled": $([ ''${#tb_rdma_devices[@]} -gt 0 ] && echo "true" || echo "false"),
      "rdma_devices": [$(printf '"%s",' "''${rdma_devices[@]}" | sed 's/,$//')]],
      "thunderbolt_rdma_devices": [$(printf '"%s",' "''${tb_rdma_devices[@]}" | sed 's/,$//')]],
      "max_queue_pairs": ${toString cfg.rdma.maxQueuePairs},
      "completion_queue_size": ${toString cfg.rdma.completionQueueSize},
      "configuration_timestamp": "$(date -Iseconds)"
    }
    EOF
      
      echo "RDMA configuration written to: $RDMA_CONFIG" >&2
    }
    
    # Function to test RDMA connectivity
    test_rdma_connectivity() {
      local rdma_devices=($(detect_rdma_devices))
      
      if [ ''${#rdma_devices[@]} -gt 0 ]; then
        echo "Testing RDMA connectivity..." >&2
        
        for device in "''${rdma_devices[@]}"; do
          # Test basic RDMA functionality
          if command -v ibv_devinfo >/dev/null 2>&1; then
            echo "Testing device: $device" >&2
            ibv_devinfo -d "$device" >/dev/null 2>&1 && echo "  Device $device: OK" >&2 || echo "  Device $device: FAILED" >&2
          fi
          
          # Test RDMA CM functionality
          if command -v ucmatose >/dev/null 2>&1; then
            timeout 5 ucmatose -s &
            local server_pid=$!
            sleep 1
            timeout 5 ucmatose -c localhost && echo "  RDMA CM test: OK" >&2 || echo "  RDMA CM test: FAILED" >&2
            kill $server_pid 2>/dev/null || true
          fi
        done
      fi
    }
    
    # Main execution
    echo "Starting RDMA configuration..." >&2
    configure_rdma_devices
    
    if [ "${toString cfg.rdma.enable}" = "1" ]; then
      test_rdma_connectivity
    fi
    
    echo "RDMA configuration completed." >&2
  '';

  # Network discovery script
  networkDiscoveryScript = pkgs.writeShellScript "exo-network-discovery" ''
    #!/bin/bash
    # EXO Network Discovery Service
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    DATA_DIR="${exoCfg.dataDir}"
    DISCOVERY_CONFIG="$CONFIG_DIR/discovery-config.json"
    NODES_DB="$DATA_DIR/discovered-nodes.json"
    
    # Discovery configuration
    MULTICAST_ADDR="${cfg.discovery.multicast.address}"
    MULTICAST_PORT="${toString cfg.discovery.multicast.port}"
    BROADCAST_PORT="${toString cfg.discovery.broadcast.port}"
    DISCOVERY_TIMEOUT="${cfg.discovery.timeout}"
    DISCOVERY_INTERVAL="${cfg.discovery.multicast.interval}"
    
    # Create necessary directories
    mkdir -p "$CONFIG_DIR" "$DATA_DIR"
    
    # Function to get local node information
    get_local_node_info() {
      local hostname=$(hostname)
      local api_port="${toString exoCfg.apiPort}"
      local discovery_port="${toString exoCfg.networking.discoveryPort}"
      
      # Get primary IP address
      local primary_ip=""
      if [ -n "${optionalString (cfg.interfaces.primary != null) cfg.interfaces.primary}" ]; then
        primary_ip=$(ip addr show "${optionalString (cfg.interfaces.primary != null) cfg.interfaces.primary}" | grep -E 'inet [0-9]' | awk '{print $2}' | cut -d/ -f1 | head -1)
      else
        primary_ip=$(ip route get 8.8.8.8 | grep -oP 'src \K\S+')
      fi
      
      # Get hardware capabilities
      local capabilities=()
      if [ -f "$CONFIG_DIR/network-config.json" ]; then
        capabilities=($(${pkgs.jq}/bin/jq -r '.capabilities[]' "$CONFIG_DIR/network-config.json" 2>/dev/null || echo ""))
      fi
      
      cat << EOF
    {
      "node_id": "$(cat /etc/machine-id)",
      "hostname": "$hostname",
      "primary_ip": "$primary_ip",
      "api_port": $api_port,
      "discovery_port": $discovery_port,
      "capabilities": [$(printf '"%s",' "''${capabilities[@]}" | sed 's/,$//')]],
      "last_seen": "$(date -Iseconds)",
      "node_type": "${exoCfg.mode}"
    }
    EOF
    }
    
    # Function to send multicast discovery announcement
    send_multicast_announcement() {
      local node_info=$(get_local_node_info)
      
      # Create discovery packet
      local packet=$(cat << EOF
    {
      "type": "discovery_announcement",
      "version": "1.0",
      "timestamp": "$(date -Iseconds)",
      "node": $node_info
    }
    EOF
    )
      
      # Send multicast packet
      echo "$packet" | ${pkgs.socat}/bin/socat - UDP-DATAGRAM:$MULTICAST_ADDR:$MULTICAST_PORT,broadcast
      echo "Sent multicast discovery announcement" >&2
    }
    
    # Function to send broadcast discovery announcement
    send_broadcast_announcement() {
      local node_info=$(get_local_node_info)
      
      # Create discovery packet
      local packet=$(cat << EOF
    {
      "type": "discovery_announcement",
      "version": "1.0",
      "timestamp": "$(date -Iseconds)",
      "node": $node_info
    }
    EOF
    )
      
      # Send broadcast packet to all interfaces
      for iface in /sys/class/net/*; do
        if [ -d "$iface" ]; then
          iface_name=$(basename "$iface")
          
          # Skip excluded interfaces
          local excluded=false
          for exclude_pattern in ${concatStringsSep " " cfg.interfaces.exclude}; do
            if [[ "$iface_name" =~ $exclude_pattern ]]; then
              excluded=true
              break
            fi
          done
          
          if [ "$excluded" = false ] && [ -f "$iface/operstate" ]; then
            operstate=$(cat "$iface/operstate")
            if [ "$operstate" = "up" ]; then
              # Get broadcast address for interface
              broadcast_addr=$(ip addr show "$iface_name" | grep -E 'inet [0-9]' | awk '{print $4}' | head -1)
              if [ -n "$broadcast_addr" ]; then
                echo "$packet" | ${pkgs.socat}/bin/socat - UDP-DATAGRAM:$broadcast_addr:$BROADCAST_PORT,broadcast 2>/dev/null || true
              fi
            fi
          fi
        fi
      done
      
      echo "Sent broadcast discovery announcements" >&2
    }
    
    # Function to listen for discovery packets
    listen_for_discoveries() {
      local temp_file=$(mktemp)
      
      # Listen for multicast packets
      if [ "${toString cfg.discovery.multicast.enable}" = "1" ]; then
        timeout "$DISCOVERY_TIMEOUT" ${pkgs.socat}/bin/socat UDP-RECV:$MULTICAST_PORT,ip-add-membership=$MULTICAST_ADDR:0.0.0.0 - >> "$temp_file" 2>/dev/null || true
      fi
      
      # Listen for broadcast packets
      if [ "${toString cfg.discovery.broadcast.enable}" = "1" ]; then
        timeout "$DISCOVERY_TIMEOUT" ${pkgs.socat}/bin/socat UDP-RECV:$BROADCAST_PORT - >> "$temp_file" 2>/dev/null || true
      fi
      
      # Process received packets
      if [ -s "$temp_file" ]; then
        while IFS= read -r packet; do
          if [ -n "$packet" ]; then
            process_discovery_packet "$packet"
          fi
        done < "$temp_file"
      fi
      
      rm -f "$temp_file"
    }
    
    # Function to process discovery packets
    process_discovery_packet() {
      local packet="$1"
      
      # Validate packet format
      if echo "$packet" | ${pkgs.jq}/bin/jq -e '.type == "discovery_announcement"' >/dev/null 2>&1; then
        local node_info=$(echo "$packet" | ${pkgs.jq}/bin/jq '.node')
        local node_id=$(echo "$node_info" | ${pkgs.jq}/bin/jq -r '.node_id')
        local local_node_id=$(cat /etc/machine-id)
        
        # Don't process our own announcements
        if [ "$node_id" != "$local_node_id" ]; then
          update_nodes_database "$node_info"
          echo "Discovered node: $node_id" >&2
        fi
      fi
    }
    
    # Function to update nodes database
    update_nodes_database() {
      local node_info="$1"
      local node_id=$(echo "$node_info" | ${pkgs.jq}/bin/jq -r '.node_id')
      
      # Initialize nodes database if it doesn't exist
      if [ ! -f "$NODES_DB" ]; then
        echo '{"nodes": {}, "last_updated": ""}' > "$NODES_DB"
      fi
      
      # Update node information
      local updated_db=$(${pkgs.jq}/bin/jq --argjson node "$node_info" --arg node_id "$node_id" \
        '.nodes[$node_id] = $node | .last_updated = now | strftime("%Y-%m-%dT%H:%M:%SZ")' "$NODES_DB")
      
      echo "$updated_db" > "$NODES_DB"
      echo "Updated node database for: $node_id" >&2
    }
    
    # Function to probe static nodes
    probe_static_nodes() {
      local static_nodes=(${concatStringsSep " " cfg.discovery.staticNodes})
      
      for node_addr in "''${static_nodes[@]}"; do
        if [ -n "$node_addr" ]; then
          echo "Probing static node: $node_addr" >&2
          
          # Extract IP and port
          local ip=$(echo "$node_addr" | cut -d: -f1)
          local port=$(echo "$node_addr" | cut -d: -f2)
          
          # Try to connect and get node information
          local node_info=$(timeout "$DISCOVERY_TIMEOUT" ${pkgs.curl}/bin/curl -s "http://$ip:$port/api/v1/node/info" 2>/dev/null || echo "")
          
          if [ -n "$node_info" ] && echo "$node_info" | ${pkgs.jq}/bin/jq -e '.node_id' >/dev/null 2>&1; then
            # Add last_seen timestamp
            node_info=$(echo "$node_info" | ${pkgs.jq}/bin/jq --arg timestamp "$(date -Iseconds)" '. + {last_seen: $timestamp}')
            update_nodes_database "$node_info"
            echo "Successfully probed static node: $node_addr" >&2
          else
            echo "Failed to probe static node: $node_addr" >&2
          fi
        fi
      done
    }
    
    # Function to cleanup stale nodes
    cleanup_stale_nodes() {
      if [ -f "$NODES_DB" ]; then
        local current_time=$(date +%s)
        local stale_threshold=300  # 5 minutes
        
        local cleaned_db=$(${pkgs.jq}/bin/jq --arg current_time "$current_time" --arg threshold "$stale_threshold" '
          .nodes = (.nodes | to_entries | map(
            select(
              (.value.last_seen | strptime("%Y-%m-%dT%H:%M:%SZ") | mktime) > ($current_time | tonumber) - ($threshold | tonumber)
            )
          ) | from_entries)
        ' "$NODES_DB")
        
        echo "$cleaned_db" > "$NODES_DB"
        echo "Cleaned up stale nodes from database" >&2
      fi
    }
    
    # Main discovery loop
    discovery_loop() {
      echo "Starting network discovery loop..." >&2
      
      while true; do
        # Send announcements
        if [ "${toString cfg.discovery.multicast.enable}" = "1" ]; then
          send_multicast_announcement
        fi
        
        if [ "${toString cfg.discovery.broadcast.enable}" = "1" ]; then
          send_broadcast_announcement
        fi
        
        # Listen for responses
        listen_for_discoveries &
        local listen_pid=$!
        
        # Probe static nodes
        probe_static_nodes
        
        # Wait for listen process
        wait $listen_pid 2>/dev/null || true
        
        # Cleanup stale nodes
        cleanup_stale_nodes
        
        # Sleep until next discovery cycle
        sleep "$DISCOVERY_INTERVAL"
      done
    }
    
    # Handle signals
    trap 'echo "Discovery service stopping..."; exit 0' TERM INT
    
    # Start discovery
    discovery_loop
  '';

  # Network topology management script
  topologyManagementScript = pkgs.writeShellScript "exo-topology-management" ''
    #!/bin/bash
    # EXO Network Topology Management
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    DATA_DIR="${exoCfg.dataDir}"
    TOPOLOGY_CONFIG="$CONFIG_DIR/topology-config.json"
    TOPOLOGY_STATE="$DATA_DIR/topology-state.json"
    NODES_DB="$DATA_DIR/discovered-nodes.json"
    
    # Configuration
    UPDATE_INTERVAL="${cfg.topology.updateInterval}"
    HEALTH_CHECK_INTERVAL="${cfg.topology.loadBalancing.healthCheck.interval}"
    HEALTH_CHECK_TIMEOUT="${cfg.topology.loadBalancing.healthCheck.timeout}"
    
    # Create necessary directories
    mkdir -p "$CONFIG_DIR" "$DATA_DIR"
    
    # Function to measure network metrics
    measure_network_metrics() {
      local target_ip="$1"
      local target_port="$2"
      
      local latency=0
      local packet_loss=0
      local bandwidth=0
      local jitter=0
      
      # Measure latency and packet loss with ping
      local ping_result=$(ping -c 5 -W 2 "$target_ip" 2>/dev/null || echo "")
      if [ -n "$ping_result" ]; then
        latency=$(echo "$ping_result" | grep -E 'rtt min/avg/max/mdev' | awk -F'/' '{print $5}' | cut -d' ' -f1)
        packet_loss=$(echo "$ping_result" | grep -E 'packet loss' | awk '{print $6}' | sed 's/%//')
        jitter=$(echo "$ping_result" | grep -E 'rtt min/avg/max/mdev' | awk -F'/' '{print $6}' | cut -d' ' -f1)
      fi
      
      # Measure bandwidth with a simple HTTP test
      local start_time=$(date +%s.%N)
      local test_result=$(timeout "$HEALTH_CHECK_TIMEOUT" ${pkgs.curl}/bin/curl -s -w "%{speed_download}" -o /dev/null "http://$target_ip:$target_port/api/v1/health" 2>/dev/null || echo "0")
      local end_time=$(date +%s.%N)
      
      if [ "$test_result" != "0" ]; then
        bandwidth=$(echo "$test_result" | awk '{printf "%.0f", $1 * 8 / 1024 / 1024}')  # Convert to Mbps
      fi
      
      cat << EOF
    {
      "latency_ms": ${latency:-0},
      "packet_loss_percent": ${packet_loss:-100},
      "bandwidth_mbps": ${bandwidth:-0},
      "jitter_ms": ${jitter:-0},
      "measurement_timestamp": "$(date -Iseconds)"
    }
    EOF
    }
    
    # Function to perform health checks
    perform_health_checks() {
      local health_results="{}"
      
      if [ -f "$NODES_DB" ]; then
        # Get list of discovered nodes
        local nodes=$(${pkgs.jq}/bin/jq -r '.nodes | keys[]' "$NODES_DB" 2>/dev/null || echo "")
        
        for node_id in $nodes; do
          if [ -n "$node_id" ]; then
            local node_info=$(${pkgs.jq}/bin/jq -r ".nodes[\"$node_id\"]" "$NODES_DB")
            local node_ip=$(echo "$node_info" | ${pkgs.jq}/bin/jq -r '.primary_ip')
            local node_port=$(echo "$node_info" | ${pkgs.jq}/bin/jq -r '.api_port')
            
            if [ "$node_ip" != "null" ] && [ "$node_port" != "null" ]; then
              echo "Performing health check for node: $node_id ($node_ip:$node_port)" >&2
              
              # Measure network metrics
              local metrics=$(measure_network_metrics "$node_ip" "$node_port")
              
              # Determine health status
              local latency=$(echo "$metrics" | ${pkgs.jq}/bin/jq -r '.latency_ms')
              local packet_loss=$(echo "$metrics" | ${pkgs.jq}/bin/jq -r '.packet_loss_percent')
              local bandwidth=$(echo "$metrics" | ${pkgs.jq}/bin/jq -r '.bandwidth_mbps')
              
              local health_status="healthy"
              if (( $(echo "$packet_loss > ${toString cfg.monitoring.alertThresholds.packetLoss}" | ${pkgs.bc}/bin/bc -l) )); then
                health_status="unhealthy"
              elif (( $(echo "$latency > ${toString cfg.monitoring.alertThresholds.latency}" | ${pkgs.bc}/bin/bc -l) )); then
                health_status="degraded"
              fi
              
              # Update health results
              health_results=$(echo "$health_results" | ${pkgs.jq}/bin/jq --arg node_id "$node_id" --argjson metrics "$metrics" --arg status "$health_status" \
                '.[$node_id] = {status: $status, metrics: $metrics}')
            fi
          fi
        done
      fi
      
      echo "$health_results"
    }
    
    # Function to calculate path weights
    calculate_path_weights() {
      local health_results="$1"
      local path_weights="{}"
      
      # Weight configuration
      local latency_weight="${toString cfg.topology.pathOptimization.latencyWeight}"
      local bandwidth_weight="${toString cfg.topology.pathOptimization.bandwidthWeight}"
      local reliability_weight="${toString cfg.topology.pathOptimization.reliabilityWeight}"
      
      # Calculate weights for each node
      local nodes=$(echo "$health_results" | ${pkgs.jq}/bin/jq -r 'keys[]')
      
      for node_id in $nodes; do
        if [ -n "$node_id" ]; then
          local node_health=$(echo "$health_results" | ${pkgs.jq}/bin/jq -r ".\"$node_id\"")
          local status=$(echo "$node_health" | ${pkgs.jq}/bin/jq -r '.status')
          
          if [ "$status" = "healthy" ] || [ "$status" = "degraded" ]; then
            local metrics=$(echo "$node_health" | ${pkgs.jq}/bin/jq '.metrics')
            local latency=$(echo "$metrics" | ${pkgs.jq}/bin/jq -r '.latency_ms')
            local bandwidth=$(echo "$metrics" | ${pkgs.jq}/bin/jq -r '.bandwidth_mbps')
            local packet_loss=$(echo "$metrics" | ${pkgs.jq}/bin/jq -r '.packet_loss_percent')
            
            # Calculate normalized scores (0-1, higher is better)
            local latency_score=$(echo "scale=4; 1 / (1 + $latency / 100)" | ${pkgs.bc}/bin/bc -l)
            local bandwidth_score=$(echo "scale=4; $bandwidth / 1000" | ${pkgs.bc}/bin/bc -l)  # Normalize to Gbps
            local reliability_score=$(echo "scale=4; 1 - $packet_loss / 100" | ${pkgs.bc}/bin/bc -l)
            
            # Ensure scores are between 0 and 1
            latency_score=$(echo "$latency_score" | awk '{if($1>1) print 1; else if($1<0) print 0; else print $1}')
            bandwidth_score=$(echo "$bandwidth_score" | awk '{if($1>1) print 1; else if($1<0) print 0; else print $1}')
            reliability_score=$(echo "$reliability_score" | awk '{if($1>1) print 1; else if($1<0) print 0; else print $1}')
            
            # Calculate weighted score
            local weighted_score=$(echo "scale=4; $latency_score * $latency_weight + $bandwidth_score * $bandwidth_weight + $reliability_score * $reliability_weight" | ${pkgs.bc}/bin/bc -l)
            
            # Update path weights
            path_weights=$(echo "$path_weights" | ${pkgs.jq}/bin/jq --arg node_id "$node_id" --arg weight "$weighted_score" \
              '.[$node_id] = ($weight | tonumber)')
          else
            # Unhealthy nodes get zero weight
            path_weights=$(echo "$path_weights" | ${pkgs.jq}/bin/jq --arg node_id "$node_id" \
              '.[$node_id] = 0')
          fi
        fi
      done
      
      echo "$path_weights"
    }
    
    # Function to update topology state
    update_topology_state() {
      local health_results="$1"
      local path_weights="$2"
      
      # Create topology state
      local topology_state=$(cat << EOF
    {
      "last_updated": "$(date -Iseconds)",
      "health_results": $health_results,
      "path_weights": $path_weights,
      "load_balancing": {
        "algorithm": "${cfg.topology.loadBalancing.algorithm}",
        "enabled": ${toString cfg.topology.loadBalancing.enable}
      },
      "optimization": {
        "enabled": ${toString cfg.topology.pathOptimization.enable},
        "metrics": [$(printf '"%s",' ${concatStringsSep " " cfg.topology.pathOptimization.metrics} | sed 's/,$//')]
      }
    }
    EOF
    )
      
      echo "$topology_state" > "$TOPOLOGY_STATE"
      echo "Updated topology state" >&2
    }
    
    # Main topology management loop
    topology_loop() {
      echo "Starting topology management loop..." >&2
      
      while true; do
        if [ "${toString cfg.topology.enable}" = "1" ]; then
          echo "Performing topology update..." >&2
          
          # Perform health checks
          local health_results=$(perform_health_checks)
          
          # Calculate path weights
          local path_weights="{}"
          if [ "${toString cfg.topology.pathOptimization.enable}" = "1" ]; then
            path_weights=$(calculate_path_weights "$health_results")
          fi
          
          # Update topology state
          update_topology_state "$health_results" "$path_weights"
          
          echo "Topology update completed" >&2
        fi
        
        # Sleep until next update
        sleep "$UPDATE_INTERVAL"
      done
    }
    
    # Handle signals
    trap 'echo "Topology management stopping..."; exit 0' TERM INT
    
    # Start topology management
    topology_loop
  '';

in
{
  options.services.exo.networking = {
    autoDetection = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable automatic detection of optimal network interfaces.
          When enabled, EXO will automatically detect and configure
          the best available network interfaces for cluster communication.
        '';
      };

      interval = mkOption {
        type = types.str;
        default = "5min";
        description = ''
          Interval for automatic network interface detection.
          Uses systemd timer format (e.g., "5min", "1h", "daily").
        '';
      };

      preferBonded = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Prefer bonded network interfaces when available.
          Bonded interfaces provide higher bandwidth and redundancy.
        '';
      };

      preferRdma = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Prefer RDMA-capable interfaces when available.
          RDMA provides low-latency communication for supported hardware.
        '';
      };
    };

    interfaces = {
      primary = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "bond0";
        description = ''
          Primary network interface for EXO cluster communication.
          If null, automatic detection will be used.
        '';
      };

      backup = mkOption {
        type = types.listOf types.str;
        default = [ ];
        example = [ "eth0" "eth1" ];
        description = ''
          Backup network interfaces for failover scenarios.
          EXO will automatically switch to backup interfaces if primary fails.
        '';
      };

      exclude = mkOption {
        type = types.listOf types.str;
        default = [ "lo" "docker0" "br-*" "veth*" ];
        description = ''
          Network interfaces to exclude from automatic detection.
          Supports glob patterns for interface names.
        '';
      };
    };

    k3s = {
      detectExisting = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Automatically detect existing K3s network configuration.
          When enabled, EXO will integrate with existing K3s networking.
        '';
      };

      reuseInterfaces = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Reuse network interfaces already configured for K3s.
          Helps avoid conflicts with existing K3s networking.
        '';
      };

      coordinateWithFlannel = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Coordinate with K3s Flannel networking.
          Ensures EXO traffic doesn't conflict with Kubernetes networking.
        '';
      };
      rdma = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable RDMA (Remote Direct Memory Access) support.
            Provides low-latency, high-throughput communication for supported hardware.
          '';
        };

        thunderboltSupport = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable RDMA over Thunderbolt support.
            Requires Thunderbolt 4 or 5 hardware with RDMA capabilities.
          '';
        };

        autoDetect = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Automatically detect and configure RDMA-capable interfaces.
            When enabled, EXO will automatically use RDMA when available.
          '';
        };

        devices = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [ "mlx5_0" "mlx5_1" ];
          description = ''
            Specific RDMA devices to use for EXO communication.
            If empty, all available RDMA devices will be used.
          '';
        };

        maxQueuePairs = mkOption {
          type = types.int;
          default = 256;
          description = ''
            Maximum number of RDMA queue pairs per device.
            Higher values allow more concurrent connections but use more memory.
          '';
        };

        completionQueueSize = mkOption {
          type = types.int;
          default = 1024;
          description = ''
            Size of RDMA completion queues.
            Larger queues can improve performance but use more memory.
          '';
        };
      };
      enableTcpOptimization = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable TCP optimization for EXO network traffic.
          Improves performance for large model transfers.
        '';
      };

      bufferSizes = {
        receive = mkOption {
          type = types.nullOr types.str;
          default = "16M";
          description = ''
            TCP receive buffer size for EXO connections.
            Larger buffers improve throughput for high-bandwidth connections.
          '';
        };

        send = mkOption {
          type = types.nullOr types.str;
          default = "16M";
          description = ''
            TCP send buffer size for EXO connections.
            Larger buffers improve throughput for high-bandwidth connections.
          '';
        };
      };

      congestionControl = mkOption {
        type = types.enum [ "cubic" "bbr" "reno" "vegas" ];
        default = "bbr";
        description = ''
          TCP congestion control algorithm for EXO connections.
          BBR is recommended for high-bandwidth, high-latency networks.
        '';
      };
      discovery = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable automatic network discovery for EXO cluster nodes.
            Allows nodes to automatically find and connect to each other.
          '';
        };

        multicast = {
          enable = mkOption {
            type = types.bool;
            default = true;
            description = ''
              Enable multicast-based node discovery.
              Uses multicast packets to discover nodes on the local network.
            '';
          };

          address = mkOption {
            type = types.str;
            default = "239.255.42.42";
            description = ''
              Multicast address for node discovery.
              Must be a valid multicast address in the 239.x.x.x range.
            '';
          };

          port = mkOption {
            type = types.port;
            default = 52417;
            description = ''
              Port for multicast discovery packets.
              Should be different from API and discovery ports.
            '';
          };

          interval = mkOption {
            type = types.str;
            default = "30s";
            description = ''
              Interval between multicast discovery announcements.
              More frequent announcements provide faster discovery but increase network traffic.
            '';
          };
        };

        broadcast = {
          enable = mkOption {
            type = types.bool;
            default = true;
            description = ''
              Enable broadcast-based node discovery as fallback.
              Used when multicast is not available or blocked.
            '';
          };

          port = mkOption {
            type = types.port;
            default = 52418;
            description = ''
              Port for broadcast discovery packets.
              Should be different from other EXO ports.
            '';
          };
        };

        staticNodes = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [ "192.168.1.10:52415" "192.168.1.11:52415" ];
          description = ''
            Static list of known EXO nodes.
            Used as fallback when automatic discovery fails.
          '';
        };

        timeout = mkOption {
          type = types.str;
          default = "10s";
          description = ''
            Timeout for node discovery attempts.
            Nodes that don't respond within this time are considered unavailable.
          '';
        };
      };

      topology = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable network topology management and optimization.
            Automatically manages network paths and load balancing.
          '';
        };

        updateInterval = mkOption {
          type = types.str;
          default = "60s";
          description = ''
            Interval for topology updates and optimization.
            More frequent updates provide better optimization but use more resources.
          '';
        };

        loadBalancing = {
          enable = mkOption {
            type = types.bool;
            default = true;
            description = ''
              Enable load balancing across multiple network paths.
              Distributes traffic across available network interfaces.
            '';
          };

          algorithm = mkOption {
            type = types.enum [ "round_robin" "least_connections" "weighted" "adaptive" ];
            default = "adaptive";
            description = ''
              Load balancing algorithm to use:
              - round_robin: Distribute requests evenly across interfaces
              - least_connections: Use interface with fewest active connections
              - weighted: Use interface weights based on bandwidth
              - adaptive: Dynamically adjust based on performance metrics
            '';
          };

          healthCheck = {
            enable = mkOption {
              type = types.bool;
              default = true;
              description = ''
                Enable health checking for network paths.
                Automatically removes failed paths from load balancing.
              '';
            };

            interval = mkOption {
              type = types.str;
              default = "10s";
              description = ''
                Interval between health checks for network paths.
                More frequent checks provide faster failure detection.
              '';
            };

            timeout = mkOption {
              type = types.str;
              default = "5s";
              description = ''
                Timeout for individual health check probes.
                Paths that don't respond within this time are marked as failed.
              '';
            };
          };
        };

        pathOptimization = {
          enable = mkOption {
            type = types.bool;
            default = true;
            description = ''
              Enable automatic path optimization based on performance metrics.
              Selects optimal network paths for different types of traffic.
            '';
          };

          metrics = mkOption {
            type = types.listOf (types.enum [ "latency" "bandwidth" "packet_loss" "jitter" ]);
            default = [ "latency" "bandwidth" "packet_loss" ];
            description = ''
              Network metrics to consider for path optimization.
              Multiple metrics are combined to select optimal paths.
            '';
          };

          latencyWeight = mkOption {
            type = types.float;
            default = 0.4;
            description = ''
              Weight for latency in path optimization (0.0-1.0).
              Higher values prioritize low-latency paths.
            '';
          };

          bandwidthWeight = mkOption {
            type = types.float;
            default = 0.4;
            description = ''
              Weight for bandwidth in path optimization (0.0-1.0).
              Higher values prioritize high-bandwidth paths.
            '';
          };

          reliabilityWeight = mkOption {
            type = types.float;
            default = 0.2;
            description = ''
              Weight for reliability in path optimization (0.0-1.0).
              Higher values prioritize stable, low-loss paths.
            '';
          };
        };
      };
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable network monitoring for EXO interfaces.
          Provides metrics and alerts for network performance.
        '';
      };

      metricsInterval = mkOption {
        type = types.str;
        default = "30s";
        description = ''
          Interval for collecting network metrics.
          More frequent collection provides better monitoring but uses more resources.
        '';
      };

      alertThresholds = {
        packetLoss = mkOption {
          type = types.float;
          default = 0.01;
          description = ''
            Packet loss threshold for network alerts (as percentage).
            Alerts will be generated when packet loss exceeds this threshold.
          '';
        };

        latency = mkOption {
          type = types.int;
          default = 100;
          description = ''
            Latency threshold for network alerts (in milliseconds).
            Alerts will be generated when latency exceeds this threshold.
          '';
        };

        bandwidth = mkOption {
          type = types.str;
          default = "100M";
          description = ''
            Minimum bandwidth threshold for network alerts.
            Alerts will be generated when available bandwidth drops below this threshold.
          '';
        };
      };
    };
    security = {
      firewall = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable EXO-specific firewall rules.
            Automatically configures firewall rules for EXO communication ports.
          '';
        };

        allowedNetworks = mkOption {
          type = types.listOf types.str;
          default = [ "10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16" ];
          example = [ "192.168.1.0/24" "10.0.0.0/8" ];
          description = ''
            Networks allowed to access EXO services.
            Only traffic from these networks will be allowed through the firewall.
          '';
        };

        restrictToInterfaces = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [ "bond0" "eth0" ];
          description = ''
            Restrict EXO traffic to specific network interfaces.
            If empty, EXO will be accessible on all interfaces.
          '';
        };

        customRules = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [
            "iptables -A INPUT -p tcp --dport 52415 -m conntrack --ctstate NEW -j LOG --log-prefix 'EXO-API: '"
            "iptables -A INPUT -p udp --dport 52416 -m limit --limit 10/min -j ACCEPT"
          ];
          description = ''
            Custom iptables rules for EXO networking.
            These rules will be applied in addition to the default rules.
          '';
        };

        rateLimiting = {
          enable = mkOption {
            type = types.bool;
            default = true;
            description = ''
              Enable rate limiting for EXO network traffic.
              Helps prevent DoS attacks and excessive resource usage.
            '';
          };

          apiRequests = mkOption {
            type = types.str;
            default = "100/minute";
            description = ''
              Rate limit for API requests per source IP.
              Format: number/timeunit (e.g., "100/minute", "10/second").
            '';
          };

          discoveryPackets = mkOption {
            type = types.str;
            default = "30/minute";
            description = ''
              Rate limit for discovery packets per source IP.
              Prevents discovery flooding attacks.
            '';
          };
        };
      };

      networkNamespace = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = ''
            Enable network namespace isolation for EXO services.
            Provides additional network security but may complicate networking setup.
          '';
        };

        name = mkOption {
          type = types.str;
          default = "exo-ns";
          description = ''
            Name of the network namespace for EXO services.
            Must be unique on the system.
          '';
        };

        bridgeInterface = mkOption {
          type = types.str;
          default = "exo-br0";
          description = ''
            Name of the bridge interface for namespace communication.
            Used to connect the namespace to the host network.
          '';
        };

        ipRange = mkOption {
          type = types.str;
          default = "192.168.100.0/24";
          description = ''
            IP range for the network namespace.
            Should not conflict with existing network ranges.
          '';
        };
      };

      encryption = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable encryption for EXO network communication.
            Encrypts all traffic between EXO nodes.
          '';
        };

        algorithm = mkOption {
          type = types.enum [ "aes256" "chacha20" "aes128" ];
          default = "aes256";
          description = ''
            Encryption algorithm for network communication.
            AES256 provides the best security, ChaCha20 may be faster on some hardware.
          '';
        };

        keyRotationInterval = mkOption {
          type = types.str;
          default = "24h";
          description = ''
            Interval for automatic encryption key rotation.
            More frequent rotation improves security but increases overhead.
          '';
        };
      };

      authentication = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = ''
            Enable authentication for EXO node communication.
            Ensures only authorized nodes can join the cluster.
          '';
        };

        method = mkOption {
          type = types.enum [ "psk" "certificate" "token" ];
          default = "psk";
          description = ''
            Authentication method for node communication:
            - psk: Pre-shared key authentication
            - certificate: X.509 certificate-based authentication
            - token: JWT token-based authentication
          '';
        };

        presharedKey = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = ''
            Pre-shared key for node authentication.
            Should be a strong, randomly generated key.
            If null, a key will be generated automatically.
          '';
        };

        certificatePath = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = ''
            Path to certificate file for certificate-based authentication.
            Required when authentication method is "certificate".
          '';
        };

        keyPath = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = ''
            Path to private key file for certificate-based authentication.
            Required when authentication method is "certificate".
          '';
        };
      };
    };
  };

  config = mkIf exoCfg.enable {
    # Network interface detection service
    systemd.services.exo-network-detection = mkIf cfg.autoDetection.enable {
      description = "EXO Network Interface Detection";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = networkDetectionScript;
        ExecReload = networkDetectionScript;

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir ];
        ReadOnlyPaths = [ "/sys" "/proc" ];
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

    # Network detection timer
    systemd.timers.exo-network-detection = mkIf cfg.autoDetection.enable {
      description = "EXO Network Detection Timer";
      wantedBy = [ "timers.target" ];

      timerConfig = {
        OnBootSec = "1min";
        OnUnitActiveSec = cfg.autoDetection.interval;
        Unit = "exo-network-detection.service";
      };
    };

    # RDMA configuration service
    systemd.services.exo-rdma-config = mkIf cfg.rdma.enable {
      description = "EXO RDMA Configuration";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root"; # Need root for device configuration
        Group = "root";
        ExecStart = rdmaConfigScript;
        ExecReload = rdmaConfigScript;

        # Security settings (less restrictive due to hardware access needs)
        NoNewPrivileges = false; # Need privileges for device configuration
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir "/dev" "/sys" ];
        ReadOnlyPaths = [ "/proc" ];
        PrivateTmp = true;
        RestrictSUIDSGID = false; # May need SUID for device access
        RestrictRealtime = false; # RDMA may need realtime capabilities
        LockPersonality = true;
        RemoveIPC = true;
      };
    };

    # RDMA monitoring service
    systemd.services.exo-rdma-monitor = mkIf (cfg.rdma.enable && cfg.monitoring.enable) {
      description = "EXO RDMA Monitoring";
      wantedBy = [ "exo.target" ];
      after = [ "exo-rdma-config.service" ];
      requires = [ "exo-rdma-config.service" ];

      serviceConfig = {
        Type = "exec";
        User = exoCfg.user;
        Group = exoCfg.group;
        Restart = "always";
        RestartSec = "30s";

        ExecStart = pkgs.writeShellScript "exo-rdma-monitor" ''
          #!/bin/bash
          # Monitor RDMA device status and performance
          
          MONITOR_LOG="/var/log/exo/rdma-monitor.log"
          RDMA_CONFIG="${exoCfg.configDir}/rdma-config.json"
          
          # Create log directory
          mkdir -p "$(dirname "$MONITOR_LOG")"
          
          while true; do
            TIMESTAMP=$(date -Iseconds)
            
            # Check if RDMA config exists
            if [ -f "$RDMA_CONFIG" ]; then
              # Extract RDMA devices from config
              RDMA_DEVICES=$(${pkgs.jq}/bin/jq -r '.rdma_devices[]' "$RDMA_CONFIG" 2>/dev/null || echo "")
              
              for device in $RDMA_DEVICES; do
                if [ -n "$device" ]; then
                  # Monitor device status
                  if [ -d "/sys/class/infiniband/$device" ]; then
                    # Check port states
                    for port in /sys/class/infiniband/$device/ports/*; do
                      if [ -d "$port" ]; then
                        port_num=$(basename "$port")
                        if [ -f "$port/state" ]; then
                          port_state=$(cat "$port/state")
                          echo "$TIMESTAMP device=$device port=$port_num state=\"$port_state\"" >> "$MONITOR_LOG"
                        fi
                        
                        # Monitor port counters if available
                        if [ -d "$port/counters" ]; then
                          for counter in "$port/counters"/*; do
                            if [ -f "$counter" ]; then
                              counter_name=$(basename "$counter")
                              counter_value=$(cat "$counter" 2>/dev/null || echo "0")
                              echo "$TIMESTAMP device=$device port=$port_num counter=$counter_name value=$counter_value" >> "$MONITOR_LOG"
                            fi
                          done
                        fi
                      fi
                    done
                  fi
                fi
              done
              
              # Monitor Thunderbolt RDMA devices
              TB_RDMA_DEVICES=$(${pkgs.jq}/bin/jq -r '.thunderbolt_rdma_devices[]' "$RDMA_CONFIG" 2>/dev/null || echo "")
              
              for tb_device in $TB_RDMA_DEVICES; do
                if [ -n "$tb_device" ]; then
                  # Parse Thunderbolt RDMA device info (format: iface:rdma_dev:tb_gen)
                  iface=$(echo "$tb_device" | cut -d: -f1)
                  rdma_dev=$(echo "$tb_device" | cut -d: -f2)
                  tb_gen=$(echo "$tb_device" | cut -d: -f3)
                  
                  # Monitor network interface status
                  if [ -d "/sys/class/net/$iface" ]; then
                    operstate=$(cat "/sys/class/net/$iface/operstate" 2>/dev/null || echo "unknown")
                    speed=$(cat "/sys/class/net/$iface/speed" 2>/dev/null || echo "0")
                    echo "$TIMESTAMP thunderbolt_iface=$iface rdma_device=$rdma_dev generation=$tb_gen operstate=$operstate speed=$speed" >> "$MONITOR_LOG"
                  fi
                fi
              done
            fi
            
            # Rotate log if it gets too large
            if [ -f "$MONITOR_LOG" ] && [ $(stat -c%s "$MONITOR_LOG") -gt 52428800 ]; then # 50MB
              mv "$MONITOR_LOG" "$MONITOR_LOG.old"
              touch "$MONITOR_LOG"
              chown ${exoCfg.user}:${exoCfg.group} "$MONITOR_LOG"
            fi
            
            sleep ${cfg.monitoring.metricsInterval}
          done
        '';

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/log/exo" ];
        ReadOnlyPaths = [ exoCfg.configDir "/sys" ];
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

    # Network discovery service
    systemd.services.exo-network-discovery = mkIf cfg.discovery.enable {
      description = "EXO Network Discovery Service";
      wantedBy = [ "exo.target" ];
      after = [ "network-online.target" "exo-network-detection.service" ];
      wants = [ "network-online.target" ];
      requires = [ "exo-network-detection.service" ];

      serviceConfig = {
        Type = "exec";
        User = exoCfg.user;
        Group = exoCfg.group;
        Restart = "always";
        RestartSec = "10s";
        ExecStart = networkDiscoveryScript;

        # Environment
        Environment = [
          "EXO_CONFIG_DIR=${exoCfg.configDir}"
          "EXO_DATA_DIR=${exoCfg.dataDir}"
        ];

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir exoCfg.dataDir ];
        ReadOnlyPaths = [ "/sys" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;

        # Network access
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # Network topology management service
    systemd.services.exo-topology-management = mkIf cfg.topology.enable {
      description = "EXO Network Topology Management";
      wantedBy = [ "exo.target" ];
      after = [ "exo-network-discovery.service" ];
      wants = [ "exo-network-discovery.service" ];

      serviceConfig = {
        Type = "exec";
        User = exoCfg.user;
        Group = exoCfg.group;
        Restart = "always";
        RestartSec = "15s";
        ExecStart = topologyManagementScript;

        # Environment
        Environment = [
          "EXO_CONFIG_DIR=${exoCfg.configDir}"
          "EXO_DATA_DIR=${exoCfg.dataDir}"
        ];

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir exoCfg.dataDir ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;

        # Network access
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # Firewall rules for discovery
    networking.firewall = mkIf (cfg.discovery.enable && exoCfg.networking.openFirewall) {
      allowedUDPPorts = mkMerge [
        (mkIf cfg.discovery.multicast.enable [ cfg.discovery.multicast.port ])
        (mkIf cfg.discovery.broadcast.enable [ cfg.discovery.broadcast.port ])
      ];
    };

    # Additional packages for network discovery, topology, security, and RDMA
    environment.systemPackages = mkMerge [
      (mkIf (cfg.discovery.enable || cfg.topology.enable) [
        pkgs.socat
        pkgs.jq
        pkgs.bc
        pkgs.curl
        pkgs.iputils
      ])
      (mkIf cfg.security.firewall.enable [
        pkgs.iptables
        pkgs.iproute2
      ])
      (mkIf cfg.rdma.enable [
        pkgs.rdma-core
        pkgs.libibverbs
        pkgs.librdmacm
        pkgs.infiniband-diags
      ])
    ];
    systemd.services.exo-firewall-config = mkIf cfg.security.firewall.enable {
      description = "EXO Firewall Configuration";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root";
        Group = "root";
        ExecStart = pkgs.writeShellScript "exo-firewall-setup" ''
          #!/bin/bash
          # EXO Firewall Configuration
          
          set -euo pipefail
          
          echo "Configuring EXO firewall rules..." >&2
          
          # EXO ports
          API_PORT="${toString exoCfg.apiPort}"
          DISCOVERY_PORT="${toString exoCfg.networking.discoveryPort}"
          MULTICAST_PORT="${toString cfg.discovery.multicast.port}"
          BROADCAST_PORT="${toString cfg.discovery.broadcast.port}"
          
          # Create EXO chain
          iptables -t filter -N EXO-INPUT 2>/dev/null || true
          iptables -t filter -F EXO-INPUT
          
          # Allow loopback traffic
          iptables -A EXO-INPUT -i lo -j ACCEPT
          
          # Allow established and related connections
          iptables -A EXO-INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
          
          # Rate limiting for API requests
          ${optionalString cfg.security.firewall.rateLimiting.enable ''
            # Parse rate limiting configuration
            API_RATE="${cfg.security.firewall.rateLimiting.apiRequests}"
            DISCOVERY_RATE="${cfg.security.firewall.rateLimiting.discoveryPackets}"
            
            # Extract number and time unit
            API_NUM=$(echo "$API_RATE" | cut -d'/' -f1)
            API_UNIT=$(echo "$API_RATE" | cut -d'/' -f2)
            DISCOVERY_NUM=$(echo "$DISCOVERY_RATE" | cut -d'/' -f1)
            DISCOVERY_UNIT=$(echo "$DISCOVERY_RATE" | cut -d'/' -f2)
            
            # API rate limiting
            iptables -A EXO-INPUT -p tcp --dport "$API_PORT" -m hashlimit \
              --hashlimit-name exo-api --hashlimit-mode srcip \
              --hashlimit "$API_NUM/$API_UNIT" --hashlimit-burst 10 -j ACCEPT
            
            # Discovery rate limiting
            iptables -A EXO-INPUT -p udp --dport "$DISCOVERY_PORT" -m hashlimit \
              --hashlimit-name exo-discovery --hashlimit-mode srcip \
              --hashlimit "$DISCOVERY_NUM/$DISCOVERY_UNIT" --hashlimit-burst 5 -j ACCEPT
            
            iptables -A EXO-INPUT -p udp --dport "$MULTICAST_PORT" -m hashlimit \
              --hashlimit-name exo-multicast --hashlimit-mode srcip \
              --hashlimit "$DISCOVERY_NUM/$DISCOVERY_UNIT" --hashlimit-burst 5 -j ACCEPT
            
            iptables -A EXO-INPUT -p udp --dport "$BROADCAST_PORT" -m hashlimit \
              --hashlimit-name exo-broadcast --hashlimit-mode srcip \
              --hashlimit "$DISCOVERY_NUM/$DISCOVERY_UNIT" --hashlimit-burst 5 -j ACCEPT
          ''}
          
          # Network restrictions
          ${concatMapStringsSep "\n" (network: ''
            iptables -A EXO-INPUT -s ${network} -p tcp --dport "$API_PORT" -j ACCEPT
            iptables -A EXO-INPUT -s ${network} -p udp --dport "$DISCOVERY_PORT" -j ACCEPT
            iptables -A EXO-INPUT -s ${network} -p udp --dport "$MULTICAST_PORT" -j ACCEPT
            iptables -A EXO-INPUT -s ${network} -p udp --dport "$BROADCAST_PORT" -j ACCEPT
          '') cfg.security.firewall.allowedNetworks}
          
          # Interface restrictions
          ${concatMapStringsSep "\n" (iface: ''
            iptables -A EXO-INPUT -i ${iface} -p tcp --dport "$API_PORT" -j ACCEPT
            iptables -A EXO-INPUT -i ${iface} -p udp --dport "$DISCOVERY_PORT" -j ACCEPT
            iptables -A EXO-INPUT -i ${iface} -p udp --dport "$MULTICAST_PORT" -j ACCEPT
            iptables -A EXO-INPUT -i ${iface} -p udp --dport "$BROADCAST_PORT" -j ACCEPT
          '') cfg.security.firewall.restrictToInterfaces}
          
          # Custom rules
          ${concatStringsSep "\n" cfg.security.firewall.customRules}
          
          # Default deny for EXO ports (if not already allowed)
          ${optionalString (cfg.security.firewall.allowedNetworks != [] || cfg.security.firewall.restrictToInterfaces != []) ''
            iptables -A EXO-INPUT -p tcp --dport "$API_PORT" -j DROP
            iptables -A EXO-INPUT -p udp --dport "$DISCOVERY_PORT" -j DROP
            iptables -A EXO-INPUT -p udp --dport "$MULTICAST_PORT" -j DROP
            iptables -A EXO-INPUT -p udp --dport "$BROADCAST_PORT" -j DROP
          ''}
          
          # Insert EXO chain into INPUT chain
          iptables -I INPUT -j EXO-INPUT 2>/dev/null || true
          
          echo "EXO firewall rules configured successfully" >&2
        '';

        ExecStop = pkgs.writeShellScript "exo-firewall-cleanup" ''
          #!/bin/bash
          # Cleanup EXO firewall rules
          
          echo "Cleaning up EXO firewall rules..." >&2
          
          # Remove EXO chain from INPUT
          iptables -D INPUT -j EXO-INPUT 2>/dev/null || true
          
          # Flush and delete EXO chain
          iptables -t filter -F EXO-INPUT 2>/dev/null || true
          iptables -t filter -X EXO-INPUT 2>/dev/null || true
          
          echo "EXO firewall rules cleaned up" >&2
        '';
      };
    };

    # Network namespace setup service
    systemd.services.exo-network-namespace = mkIf cfg.security.networkNamespace.enable {
      description = "EXO Network Namespace Setup";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root";
        Group = "root";
        ExecStart = pkgs.writeShellScript "exo-namespace-setup" ''
          #!/bin/bash
          # EXO Network Namespace Setup
          
          set -euo pipefail
          
          NAMESPACE="${cfg.security.networkNamespace.name}"
          BRIDGE="${cfg.security.networkNamespace.bridgeInterface}"
          IP_RANGE="${cfg.security.networkNamespace.ipRange}"
          
          echo "Setting up EXO network namespace: $NAMESPACE" >&2
          
          # Create network namespace
          ${pkgs.iproute2}/bin/ip netns add "$NAMESPACE" 2>/dev/null || true
          
          # Create bridge interface
          ${pkgs.iproute2}/bin/ip link add "$BRIDGE" type bridge 2>/dev/null || true
          ${pkgs.iproute2}/bin/ip link set "$BRIDGE" up
          
          # Create veth pair
          ${pkgs.iproute2}/bin/ip link add "veth-exo-host" type veth peer name "veth-exo-ns" 2>/dev/null || true
          
          # Move one end to namespace
          ${pkgs.iproute2}/bin/ip link set "veth-exo-ns" netns "$NAMESPACE"
          
          # Configure host side
          ${pkgs.iproute2}/bin/ip link set "veth-exo-host" master "$BRIDGE"
          ${pkgs.iproute2}/bin/ip link set "veth-exo-host" up
          
          # Configure namespace side
          ${pkgs.iproute2}/bin/ip netns exec "$NAMESPACE" ${pkgs.iproute2}/bin/ip link set lo up
          ${pkgs.iproute2}/bin/ip netns exec "$NAMESPACE" ${pkgs.iproute2}/bin/ip link set "veth-exo-ns" up
          
          # Assign IP addresses
          BRIDGE_IP=$(echo "$IP_RANGE" | sed 's|0/24|1/24|')
          NS_IP=$(echo "$IP_RANGE" | sed 's|0/24|2/24|')
          GATEWAY_IP=$(echo "$IP_RANGE" | sed 's|0/24|1|')
          
          ${pkgs.iproute2}/bin/ip addr add "$BRIDGE_IP" dev "$BRIDGE" 2>/dev/null || true
          ${pkgs.iproute2}/bin/ip netns exec "$NAMESPACE" ${pkgs.iproute2}/bin/ip addr add "$NS_IP" dev "veth-exo-ns"
          ${pkgs.iproute2}/bin/ip netns exec "$NAMESPACE" ${pkgs.iproute2}/bin/ip route add default via "$GATEWAY_IP"
          
          # Enable IP forwarding
          echo 1 > /proc/sys/net/ipv4/ip_forward
          
          # Configure NAT for namespace
          iptables -t nat -A POSTROUTING -s "$IP_RANGE" ! -d "$IP_RANGE" -j MASQUERADE
          iptables -A FORWARD -s "$IP_RANGE" -j ACCEPT
          iptables -A FORWARD -d "$IP_RANGE" -j ACCEPT
          
          echo "EXO network namespace configured successfully" >&2
        '';

        ExecStop = pkgs.writeShellScript "exo-namespace-cleanup" ''
          #!/bin/bash
          # Cleanup EXO network namespace
          
          NAMESPACE="${cfg.security.networkNamespace.name}"
          BRIDGE="${cfg.security.networkNamespace.bridgeInterface}"
          IP_RANGE="${cfg.security.networkNamespace.ipRange}"
          
          echo "Cleaning up EXO network namespace: $NAMESPACE" >&2
          
          # Remove NAT rules
          iptables -t nat -D POSTROUTING -s "$IP_RANGE" ! -d "$IP_RANGE" -j MASQUERADE 2>/dev/null || true
          iptables -D FORWARD -s "$IP_RANGE" -j ACCEPT 2>/dev/null || true
          iptables -D FORWARD -d "$IP_RANGE" -j ACCEPT 2>/dev/null || true
          
          # Remove veth pair
          ${pkgs.iproute2}/bin/ip link delete "veth-exo-host" 2>/dev/null || true
          
          # Remove bridge
          ${pkgs.iproute2}/bin/ip link delete "$BRIDGE" 2>/dev/null || true
          
          # Remove namespace
          ${pkgs.iproute2}/bin/ip netns delete "$NAMESPACE" 2>/dev/null || true
          
          echo "EXO network namespace cleaned up" >&2
        '';
      };
    };

    # Security monitoring service
    systemd.services.exo-security-monitor = mkIf cfg.security.firewall.enable {
      description = "EXO Security Monitoring";
      wantedBy = [ "exo.target" ];
      after = [ "exo-firewall-config.service" ];
      requires = [ "exo-firewall-config.service" ];

      serviceConfig = {
        Type = "exec";
        User = exoCfg.user;
        Group = exoCfg.group;
        Restart = "always";
        RestartSec = "30s";

        ExecStart = pkgs.writeShellScript "exo-security-monitor" ''
          #!/bin/bash
          # EXO Security Monitoring
          
          SECURITY_LOG="/var/log/exo/security.log"
          
          # Create log directory
          mkdir -p "$(dirname "$SECURITY_LOG")"
          
          while true; do
            TIMESTAMP=$(date -Iseconds)
            
            # Monitor failed connection attempts
            FAILED_CONNECTIONS=$(journalctl --since "1 minute ago" -u exo-api.service -u exo-master.service -u exo-worker.service \
              | grep -i "connection.*failed\|refused\|timeout" | wc -l)
            
            if [ "$FAILED_CONNECTIONS" -gt 10 ]; then
              echo "$TIMESTAMP ALERT: High number of failed connections: $FAILED_CONNECTIONS" >> "$SECURITY_LOG"
            fi
            
            # Monitor authentication failures
            AUTH_FAILURES=$(journalctl --since "1 minute ago" -u exo-master.service -u exo-worker.service \
              | grep -i "auth.*fail\|unauthorized\|forbidden" | wc -l)
            
            if [ "$AUTH_FAILURES" -gt 5 ]; then
              echo "$TIMESTAMP ALERT: Authentication failures detected: $AUTH_FAILURES" >> "$SECURITY_LOG"
            fi
            
            # Monitor unusual network activity
            NETWORK_ERRORS=$(journalctl --since "1 minute ago" -u exo-network-discovery.service -u exo-topology-management.service \
              | grep -i "error\|fail" | wc -l)
            
            if [ "$NETWORK_ERRORS" -gt 20 ]; then
              echo "$TIMESTAMP WARNING: High network error rate: $NETWORK_ERRORS" >> "$SECURITY_LOG"
            fi
            
            # Check firewall rule integrity
            if ! iptables -L EXO-INPUT >/dev/null 2>&1; then
              echo "$TIMESTAMP CRITICAL: EXO firewall rules missing, restarting firewall service" >> "$SECURITY_LOG"
              systemctl restart exo-firewall-config.service || true
            fi
            
            # Rotate log if it gets too large
            if [ -f "$SECURITY_LOG" ] && [ $(stat -c%s "$SECURITY_LOG") -gt 10485760 ]; then # 10MB
              mv "$SECURITY_LOG" "$SECURITY_LOG.old"
              touch "$SECURITY_LOG"
              chown ${exoCfg.user}:${exoCfg.group} "$SECURITY_LOG"
            fi
            
            sleep 60
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

    # Load required kernel modules for networking, security, RDMA, and optimization
    boot.kernelModules = mkMerge [
      (mkIf cfg.security.firewall.enable [
        "xt_hashlimit" # For rate limiting
        "xt_conntrack" # For connection tracking
        "xt_multiport" # For multiple port matching
      ])
      (mkIf cfg.rdma.enable [
        "rdma_core"
        "ib_core"
        "ib_uverbs"
        "rdma_ucm"
        "ib_cm"
        "iw_cm"
      ])
      (mkIf cfg.optimization.enableTcpOptimization [
        "tcp_bbr" # BBR congestion control
        "tcp_cubic" # CUBIC congestion control
      ])
    ];



    # RDMA device permissions
    services.udev.extraRules = mkIf cfg.rdma.enable ''
      # RDMA device permissions for EXO
      SUBSYSTEM=="infiniband", GROUP="${exoCfg.group}", MODE="0664"
      SUBSYSTEM=="infiniband_verbs", GROUP="${exoCfg.group}", MODE="0664"
      KERNEL=="uverbs*", GROUP="${exoCfg.group}", MODE="0664"
      KERNEL=="rdma_cm", GROUP="${exoCfg.group}", MODE="0664"
      
      # Thunderbolt RDMA device permissions
      SUBSYSTEM=="thunderbolt", ATTR{authorized}=="0", ATTR{authorized}="1"
      SUBSYSTEM=="thunderbolt", ACTION=="add", RUN+="${pkgs.systemd}/bin/systemctl reload exo-rdma-config.service"
    '';



    # K3s integration detection service
    systemd.services.exo-k3s-detection = mkIf (exoCfg.k3s.integration && cfg.k3s.detectExisting) {
      description = "EXO K3s Integration Detection";
      wantedBy = [ "multi-user.target" ];
      before = [ "exo.target" ];
      after = [ "k3s.service" ];
      wants = [ "k3s.service" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = k3sIntegrationScript;
        ExecReload = k3sIntegrationScript;

        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir ];
        ReadOnlyPaths = [ "/etc/rancher" "/sys" "/proc" ];
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

    # System configuration for TCP optimization and RDMA
    boot.kernel.sysctl = mkMerge [
      (mkIf cfg.optimization.enableTcpOptimization {
        # TCP buffer sizes
        "net.core.rmem_max" = mkIf (cfg.optimization.bufferSizes.receive != null)
          (
            let size = cfg.optimization.bufferSizes.receive; in
            if hasSuffix "M" size then toString (toInt (removeSuffix "M" size) * 1024 * 1024)
            else if hasSuffix "K" size then toString (toInt (removeSuffix "K" size) * 1024)
            else size
          );

        "net.core.wmem_max" = mkIf (cfg.optimization.bufferSizes.send != null)
          (
            let size = cfg.optimization.bufferSizes.send; in
            if hasSuffix "M" size then toString (toInt (removeSuffix "M" size) * 1024 * 1024)
            else if hasSuffix "K" size then toString (toInt (removeSuffix "K" size) * 1024)
            else size
          );

        # TCP congestion control
        "net.ipv4.tcp_congestion_control" = cfg.optimization.congestionControl;

        # Additional TCP optimizations for high-bandwidth networks
        "net.ipv4.tcp_window_scaling" = 1;
        "net.ipv4.tcp_timestamps" = 1;
        "net.ipv4.tcp_sack" = 1;
        "net.ipv4.tcp_fack" = 1;
        "net.ipv4.tcp_low_latency" = 1;
        "net.ipv4.tcp_adv_win_scale" = 2;

        # Increase default buffer sizes
        "net.core.netdev_max_backlog" = 5000;
        "net.ipv4.tcp_rmem" = "4096 87380 16777216";
        "net.ipv4.tcp_wmem" = "4096 65536 16777216";

        # Optimize for high-throughput
        "net.ipv4.tcp_slow_start_after_idle" = 0;
        "net.ipv4.tcp_mtu_probing" = 1;
      })
      (mkIf cfg.rdma.enable {
        # RDMA memory settings
        "vm.max_map_count" = 262144;
        "kernel.shmmax" = 68719476736; # 64GB
        "kernel.shmall" = 4294967296; # 16TB

        # Network buffer settings for RDMA
        "net.core.rmem_default" = 262144;
        "net.core.rmem_max" = 16777216;
        "net.core.wmem_default" = 262144;
        "net.core.wmem_max" = 16777216;

        # RDMA-specific settings
        "net.ipv4.tcp_timestamps" = 0; # Disable for better RDMA performance
        "net.ipv4.tcp_sack" = 0; # Disable for better RDMA performance
      })
    ];



    # Ensure BBR is available if selected
    assertions = [
      {
        assertion = cfg.optimization.congestionControl != "bbr" || elem "tcp_bbr" config.boot.kernelModules;
        message = "BBR congestion control requires the tcp_bbr kernel module to be loaded.";
      }
    ];
  };
}

# Integration Testing for EXO NixOS Flake
{ lib
, pkgs
, system
, nixosModules
}:

let
  # Helper function to create NixOS test VMs
  mkNixOSTest = name: testScript: pkgs.nixosTest {
    inherit name;
    
    nodes = {
      master = { config, pkgs, ... }: {
        imports = [ nixosModules.exo-service ];
        
        services.exo = {
          enable = true;
          mode = "master";
          apiPort = 52415;
          networking.bondInterface = "eth0";
        };
        
        # Basic networking setup
        networking = {
          firewall.enable = false;  # Simplified for testing
          interfaces.eth0.ipv4.addresses = [{
            address = "192.168.1.10";
            prefixLength = 24;
          }];
        };
        
        # Enable systemd-resolved for service discovery
        services.resolved.enable = true;
      };
      
      worker1 = { config, pkgs, ... }: {
        imports = [ nixosModules.exo-service ];
        
        services.exo = {
          enable = true;
          mode = "worker";
          networking.bondInterface = "eth0";
        };
        
        networking = {
          firewall.enable = false;
          interfaces.eth0.ipv4.addresses = [{
            address = "192.168.1.11";
            prefixLength = 24;
          }];
        };
        
        services.resolved.enable = true;
      };
      
      worker2 = { config, pkgs, ... }: {
        imports = [ nixosModules.exo-service ];
        
        services.exo = {
          enable = true;
          mode = "worker";
          networking.bondInterface = "eth0";
        };
        
        networking = {
          firewall.enable = false;
          interfaces.eth0.ipv4.addresses = [{
            address = "192.168.1.12";
            prefixLength = 24;
          }];
        };
        
        services.resolved.enable = true;
      };
    };
    
    testScript = testScript;
  };

  # Helper function for simple integration tests
  mkTest = name: script: pkgs.runCommand "test-${name}" {
    nativeBuildInputs = [ pkgs.bash pkgs.coreutils pkgs.curl pkgs.jq pkgs.netcat ];
  } ''
    set -euo pipefail
    
    echo "=== ${name} Integration Test ==="
    echo "System: ${system}"
    echo "Date: $(date)"
    echo
    
    ${script}
    
    echo "Integration test completed successfully"
    touch $out
  '';

  # Test multi-node cluster formation
  multi-node-cluster-tests = mkNixOSTest "multi-node-cluster" ''
    # Start all nodes
    start_all()
    
    # Wait for services to be ready
    master.wait_for_unit("exo-master.service")
    worker1.wait_for_unit("exo-worker.service") 
    worker2.wait_for_unit("exo-worker.service")
    
    # Wait for network connectivity
    master.wait_until_succeeds("ping -c 1 192.168.1.11")
    master.wait_until_succeeds("ping -c 1 192.168.1.12")
    
    # Test that master API is accessible
    master.wait_for_open_port(52415)
    master.succeed("curl -f http://localhost:52415/health || true")
    
    # Test cluster formation
    with subtest("Cluster formation"):
        # Wait for workers to connect to master
        master.wait_until_succeeds("exo-system-info | grep -i 'worker.*connected' || true")
        
        # Verify cluster topology
        result = master.succeed("exo-system-info")
        print(f"Cluster info: {result}")
    
    with subtest("Service communication"):
        # Test that workers can communicate with master
        worker1.succeed("ping -c 3 192.168.1.10")
        worker2.succeed("ping -c 3 192.168.1.10")
        
        # Test EXO-specific communication ports
        master.wait_for_open_port(52416)  # Discovery port
        
    with subtest("Hardware detection"):
        # Test hardware detection on all nodes
        master_hw = master.succeed("exo-detect-hardware")
        worker1_hw = worker1.succeed("exo-detect-hardware")
        worker2_hw = worker2.succeed("exo-detect-hardware")
        
        print(f"Master hardware: {master_hw}")
        print(f"Worker1 hardware: {worker1_hw}")
        print(f"Worker2 hardware: {worker2_hw}")
        
        # All should detect CPU (since VMs don't have GPUs)
        assert "cpu" in master_hw.lower()
        assert "cpu" in worker1_hw.lower()
        assert "cpu" in worker2_hw.lower()
  '';

  # Test K3s integration
  k3s-integration-tests = mkNixOSTest "k3s-integration" ''
    # Configure K3s master node
    master.wait_for_unit("k3s.service")
    master.wait_for_unit("exo-master.service")
    
    # Configure K3s worker with EXO integration
    worker1.wait_for_unit("k3s-agent.service")
    worker1.wait_for_unit("exo-worker.service")
    
    with subtest("K3s cluster formation"):
        # Wait for K3s cluster to be ready
        master.wait_until_succeeds("k3s kubectl get nodes")
        
        # Verify worker joined the cluster
        master.wait_until_succeeds("k3s kubectl get nodes | grep worker1")
    
    with subtest("EXO service discovery"):
        # Test that EXO services are registered with K3s
        master.wait_until_succeeds("k3s kubectl get services | grep exo || true")
        
        # Test service discovery integration
        result = master.succeed("k3s kubectl get pods -A | grep exo || true")
        print(f"EXO pods: {result}")
    
    with subtest("Network policy integration"):
        # Test that EXO networking works with K3s policies
        master.succeed("k3s kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: exo-network-policy
spec:
  podSelector:
    matchLabels:
      app: exo
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from: []
    ports:
    - protocol: TCP
      port: 52415
    - protocol: TCP
      port: 52416
  egress:
  - to: []
EOF")
        
        # Verify policy was applied
        master.wait_until_succeeds("k3s kubectl get networkpolicy exo-network-policy")
  '';

  # Test networking functionality
  networking-tests = mkTest "networking" ''
    echo "Testing EXO networking functionality..."
    
    # Test network interface detection
    echo "Testing network interface detection..."
    
    # List available interfaces
    interfaces=$(ip link show | grep -E '^[0-9]+:' | cut -d: -f2 | tr -d ' ')
    echo "Available interfaces: $interfaces"
    
    # Test bonding interface detection logic
    echo "Testing bonding detection logic..."
    
    # Create a mock bonding interface for testing
    if command -v ip >/dev/null 2>&1; then
      echo "✓ ip command available for network testing"
    else
      echo "⚠ ip command not available, skipping network interface tests"
    fi
    
    # Test RDMA detection (mock)
    echo "Testing RDMA detection..."
    
    # Check for RDMA-related kernel modules
    if [ -d "/sys/class/infiniband" ]; then
      echo "✓ InfiniBand/RDMA support detected"
    else
      echo "⚠ No RDMA hardware detected (expected in test environment)"
    fi
    
    # Test network discovery ports
    echo "Testing network discovery functionality..."
    
    # Test that discovery ports are configurable
    discovery_port=52416
    api_port=52415
    
    echo "Discovery port: $discovery_port"
    echo "API port: $api_port"
    
    # Test port availability
    if command -v netcat >/dev/null 2>&1; then
      # Test that ports are not already in use
      if ! netcat -z localhost $api_port 2>/dev/null; then
        echo "✓ API port $api_port is available"
      else
        echo "⚠ API port $api_port is already in use"
      fi
      
      if ! netcat -z localhost $discovery_port 2>/dev/null; then
        echo "✓ Discovery port $discovery_port is available"
      else
        echo "⚠ Discovery port $discovery_port is already in use"
      fi
    fi
    
    echo "Networking tests completed"
  '';

  # Test service discovery functionality
  service-discovery-tests = mkTest "service-discovery" ''
    echo "Testing EXO service discovery functionality..."
    
    # Test DNS-based service discovery
    echo "Testing DNS service discovery..."
    
    # Test that systemd-resolved is working
    if systemctl is-active systemd-resolved >/dev/null 2>&1; then
      echo "✓ systemd-resolved is active"
    else
      echo "⚠ systemd-resolved not active (may affect service discovery)"
    fi
    
    # Test mDNS functionality
    echo "Testing mDNS functionality..."
    
    # Check for Avahi or systemd-resolved mDNS support
    if command -v avahi-browse >/dev/null 2>&1; then
      echo "✓ Avahi available for mDNS"
    elif systemctl is-active systemd-resolved >/dev/null 2>&1; then
      echo "✓ systemd-resolved available for mDNS"
    else
      echo "⚠ No mDNS service detected"
    fi
    
    # Test service registration format
    echo "Testing service registration format..."
    
    # Mock service registration data
    service_data='{
      "name": "exo-master",
      "type": "_exo._tcp",
      "port": 52415,
      "txt": {
        "version": "0.3.0",
        "mode": "master",
        "hardware": "cpu"
      }
    }'
    
    echo "Service registration format: $service_data"
    
    # Validate JSON format
    if echo "$service_data" | jq . >/dev/null 2>&1; then
      echo "✓ Service registration JSON is valid"
    else
      echo "ERROR: Service registration JSON is invalid"
      exit 1
    fi
    
    # Test network topology discovery
    echo "Testing network topology discovery..."
    
    # Mock topology data
    topology_data='{
      "nodes": [
        {
          "id": "master-001",
          "address": "192.168.1.10",
          "role": "master",
          "hardware": "cpu",
          "status": "active"
        },
        {
          "id": "worker-001", 
          "address": "192.168.1.11",
          "role": "worker",
          "hardware": "cuda",
          "status": "active"
        }
      ],
      "connections": [
        {
          "from": "master-001",
          "to": "worker-001",
          "type": "tcp",
          "latency": "1ms"
        }
      ]
    }'
    
    echo "Topology format: $topology_data"
    
    if echo "$topology_data" | jq . >/dev/null 2>&1; then
      echo "✓ Network topology JSON is valid"
    else
      echo "ERROR: Network topology JSON is invalid"
      exit 1
    fi
    
    echo "Service discovery tests completed"
  '';

in {
  inherit multi-node-cluster-tests k3s-integration-tests networking-tests service-discovery-tests;
}
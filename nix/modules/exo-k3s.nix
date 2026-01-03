{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.exo.k3s;
  exoCfg = config.services.exo;
  
  # K3s service discovery script
  k3sServiceDiscoveryScript = pkgs.writeShellScript "exo-k3s-service-discovery" ''
    #!/bin/bash
    # EXO K3s Service Discovery Integration
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    DATA_DIR="${exoCfg.dataDir}"
    K3S_CONFIG="$CONFIG_DIR/k3s-integration.json"
    SERVICE_REGISTRY="$DATA_DIR/k3s-service-registry.json"
    
    # K3s configuration
    NAMESPACE="${cfg.namespace}"
    SERVICE_ACCOUNT="${cfg.serviceAccount}"
    CLUSTER_ROLE="${cfg.clusterRole}"
    
    # Create necessary directories
    mkdir -p "$CONFIG_DIR" "$DATA_DIR"
    
    # Function to check K3s availability
    check_k3s_availability() {
      if ! systemctl is-active --quiet k3s 2>/dev/null; then
        echo "K3s service is not active" >&2
        return 1
      fi
      
      if ! command -v kubectl >/dev/null 2>&1; then
        echo "kubectl command not available" >&2
        return 1
      fi
      
      # Test kubectl connectivity
      if ! timeout 10 kubectl get nodes >/dev/null 2>&1; then
        echo "Cannot connect to K3s API server" >&2
        return 1
      fi
      
      return 0
    }
    
    # Function to create EXO namespace
    create_exo_namespace() {
      echo "Creating EXO namespace: $NAMESPACE" >&2
      
      # Check if namespace exists
      if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo "Namespace $NAMESPACE already exists" >&2
        return 0
      fi
      
      # Create namespace
      kubectl create namespace "$NAMESPACE" || {
        echo "Failed to create namespace $NAMESPACE" >&2
        return 1
      }
      
      # Label namespace for EXO
      kubectl label namespace "$NAMESPACE" app.kubernetes.io/name=exo app.kubernetes.io/component=distributed-ai || true
      
      echo "Namespace $NAMESPACE created successfully" >&2
    }
    
    # Function to create service account
    create_service_account() {
      echo "Creating EXO service account: $SERVICE_ACCOUNT" >&2
      
      # Check if service account exists
      if kubectl get serviceaccount "$SERVICE_ACCOUNT" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "Service account $SERVICE_ACCOUNT already exists" >&2
        return 0
      fi
      
      # Create service account
      cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: $SERVICE_ACCOUNT
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: distributed-ai
    automountServiceAccountToken: true
    EOF
      
      echo "Service account $SERVICE_ACCOUNT created successfully" >&2
    }
    
    # Function to create cluster role and binding
    create_cluster_role() {
      echo "Creating EXO cluster role: $CLUSTER_ROLE" >&2
      
      # Check if cluster role exists
      if kubectl get clusterrole "$CLUSTER_ROLE" >/dev/null 2>&1; then
        echo "Cluster role $CLUSTER_ROLE already exists" >&2
      else
        # Create cluster role
        cat <<EOF | kubectl apply -f -
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRole
    metadata:
      name: $CLUSTER_ROLE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: distributed-ai
    rules:
    - apiGroups: [""]
      resources: ["nodes", "pods", "services", "endpoints", "configmaps"]
      verbs: ["get", "list", "watch"]
    - apiGroups: [""]
      resources: ["services", "endpoints"]
      verbs: ["create", "update", "patch", "delete"]
    - apiGroups: ["apps"]
      resources: ["deployments", "replicasets"]
      verbs: ["get", "list", "watch"]
    - apiGroups: ["networking.k8s.io"]
      resources: ["networkpolicies"]
      verbs: ["get", "list", "watch"]
    - apiGroups: ["metrics.k8s.io"]
      resources: ["nodes", "pods"]
      verbs: ["get", "list"]
    EOF
        
        echo "Cluster role $CLUSTER_ROLE created successfully" >&2
      fi
      
      # Check if cluster role binding exists
      if kubectl get clusterrolebinding "$CLUSTER_ROLE-binding" >/dev/null 2>&1; then
        echo "Cluster role binding $CLUSTER_ROLE-binding already exists" >&2
      else
        # Create cluster role binding
        cat <<EOF | kubectl apply -f -
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRoleBinding
    metadata:
      name: $CLUSTER_ROLE-binding
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: distributed-ai
    roleRef:
      apiGroup: rbac.authorization.k8s.io
      kind: ClusterRole
      name: $CLUSTER_ROLE
    subjects:
    - kind: ServiceAccount
      name: $SERVICE_ACCOUNT
      namespace: $NAMESPACE
    EOF
        
        echo "Cluster role binding $CLUSTER_ROLE-binding created successfully" >&2
      fi
    }
    
    # Function to register EXO services
    register_exo_services() {
      local node_name=$(hostname)
      local node_ip=$(ip route get 8.8.8.8 | grep -oP 'src \K\S+')
      local api_port="${toString exoCfg.apiPort}"
      local discovery_port="${toString exoCfg.networking.discoveryPort}"
      
      echo "Registering EXO services for node: $node_name (IP: $node_ip)" >&2
      
      # Get node hardware information for service labels
      local gpu_info=$(lspci | grep -i vga | head -1 | cut -d: -f3 | xargs || echo "cpu-only")
      local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
      local cpu_cores=$(nproc)
      
      # Determine EXO capabilities based on hardware
      local capabilities="inference"
      if echo "$gpu_info" | grep -qi nvidia; then
        capabilities="$capabilities,cuda"
      elif echo "$gpu_info" | grep -qi amd; then
        capabilities="$capabilities,rocm"
      elif echo "$gpu_info" | grep -qi intel; then
        capabilities="$capabilities,intel-gpu"
      fi
      
      # Register EXO API service with enhanced metadata
      cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: Service
    metadata:
      name: exo-api-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: api
        app.kubernetes.io/instance: $node_name
        exo.ai/node-name: $node_name
        exo.ai/service-type: api
        exo.ai/capabilities: $capabilities
        exo.ai/memory-gb: $memory_gb
        exo.ai/cpu-cores: $cpu_cores
        exo.ai/gpu-info: "$(echo "$gpu_info" | tr ' ' '-')"
      annotations:
        exo.ai/registered-at: "$(date -Iseconds)"
        exo.ai/node-ip: $node_ip
        exo.ai/discovery-port: $discovery_port
        service.kubernetes.io/topology-aware-hints: auto
    spec:
      type: ClusterIP
      ports:
      - name: api
        port: $api_port
        targetPort: $api_port
        protocol: TCP
      - name: discovery
        port: $discovery_port
        targetPort: $discovery_port
        protocol: UDP
      selector:
        exo.ai/node-name: $node_name
      sessionAffinity: ClientIP
    ---
    apiVersion: v1
    kind: Endpoints
    metadata:
      name: exo-api-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: api
        exo.ai/node-name: $node_name
      annotations:
        endpoints.kubernetes.io/last-change-trigger-time: "$(date -Iseconds)"
    subsets:
    - addresses:
      - ip: $node_ip
        nodeName: $node_name
        targetRef:
          kind: Node
          name: $node_name
      ports:
      - name: api
        port: $api_port
        protocol: TCP
      - name: discovery
        port: $discovery_port
        protocol: UDP
    EOF
      
      # Register EXO master service (if this node is master)
      if [ "${exoCfg.mode}" = "master" ] || [ "${exoCfg.mode}" = "auto" ]; then
        cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: Service
    metadata:
      name: exo-master-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: master
        exo.ai/node-name: $node_name
        exo.ai/service-type: master
    spec:
      type: ClusterIP
      ports:
      - name: coordination
        port: $discovery_port
        targetPort: $discovery_port
        protocol: UDP
      selector:
        exo.ai/node-name: $node_name
        exo.ai/role: master
    ---
    apiVersion: v1
    kind: Endpoints
    metadata:
      name: exo-master-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: master
        exo.ai/node-name: $node_name
    subsets:
    - addresses:
      - ip: $node_ip
        nodeName: $node_name
      ports:
      - name: coordination
        port: $discovery_port
        protocol: UDP
    EOF
      fi
      
      # Register EXO worker service (if this node is worker)
      if [ "${exoCfg.mode}" = "worker" ] || [ "${exoCfg.mode}" = "auto" ]; then
        cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: Service
    metadata:
      name: exo-worker-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: worker
        exo.ai/node-name: $node_name
        exo.ai/service-type: worker
    spec:
      type: ClusterIP
      ports:
      - name: inference
        port: $api_port
        targetPort: $api_port
        protocol: TCP
      selector:
        exo.ai/node-name: $node_name
        exo.ai/role: worker
    ---
    apiVersion: v1
    kind: Endpoints
    metadata:
      name: exo-worker-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: worker
        exo.ai/node-name: $node_name
    subsets:
    - addresses:
      - ip: $node_ip
        nodeName: $node_name
      ports:
      - name: inference
        port: $api_port
        protocol: TCP
    EOF
      fi
      
      echo "EXO services registered successfully" >&2
    }
    
    # Function to discover EXO services in K3s
    discover_exo_services() {
      echo "Discovering EXO services in K3s cluster..." >&2
      
      # Get all EXO services
      local services=$(kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/name=exo -o json 2>/dev/null || echo '{"items":[]}')
      local endpoints=$(kubectl get endpoints -n "$NAMESPACE" -l app.kubernetes.io/name=exo -o json 2>/dev/null || echo '{"items":[]}')
      local nodes=$(kubectl get nodes -o json 2>/dev/null || echo '{"items":[]}')
      
      # Detect topology changes
      local current_topology_hash=$(echo "$services$endpoints$nodes" | ${pkgs.openssl}/bin/openssl dgst -sha256 | cut -d' ' -f2)
      local previous_topology_hash=""
      
      if [ -f "$SERVICE_REGISTRY" ]; then
        previous_topology_hash=$(${pkgs.jq}/bin/jq -r '.topology_hash // ""' "$SERVICE_REGISTRY" 2>/dev/null || echo "")
      fi
      
      # Check for topology changes
      local topology_changed=false
      if [ "$current_topology_hash" != "$previous_topology_hash" ]; then
        topology_changed=true
        echo "Network topology change detected (hash: $current_topology_hash)" >&2
      fi
      
      # Create enhanced service registry with topology information
      local registry=$(cat <<EOF
    {
      "last_updated": "$(date -Iseconds)",
      "namespace": "$NAMESPACE",
      "topology_hash": "$current_topology_hash",
      "topology_changed": $topology_changed,
      "services": $services,
      "endpoints": $endpoints,
      "nodes": $nodes,
      "discovery_method": "k3s_api",
      "cluster_info": {
        "total_nodes": $(echo "$nodes" | ${pkgs.jq}/bin/jq '.items | length'),
        "exo_services": $(echo "$services" | ${pkgs.jq}/bin/jq '.items | length'),
        "active_endpoints": $(echo "$endpoints" | ${pkgs.jq}/bin/jq '[.items[].subsets[]?.addresses[]?] | length')
      }
    }
    EOF
    )
      
      echo "$registry" > "$SERVICE_REGISTRY"
      echo "Service registry updated with $(echo "$services" | ${pkgs.jq}/bin/jq '.items | length') services" >&2
      
      # Trigger topology adaptation if changes detected
      if [ "$topology_changed" = "true" ]; then
        adapt_to_topology_changes
      fi
    }
    
    # Function to adapt to network topology changes
    adapt_to_topology_changes() {
      echo "Adapting to network topology changes..." >&2
      
      # Get current cluster state
      local cluster_nodes=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
      local exo_services=$(kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/name=exo -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
      
      # Update EXO configuration with new topology
      local topology_config=$(cat <<EOF
    {
      "cluster_nodes": [$(echo "$cluster_nodes" | tr ' ' '\n' | sed 's/^/"/;s/$/"/' | paste -sd,)],
      "exo_services": [$(echo "$exo_services" | tr ' ' '\n' | sed 's/^/"/;s/$/"/' | paste -sd,)],
      "adaptation_timestamp": "$(date -Iseconds)"
    }
    EOF
    )
      
      echo "$topology_config" > "$CONFIG_DIR/k3s-topology.json"
      
      # Notify EXO services about topology changes
      local node_name=$(hostname)
      if systemctl is-active --quiet exo-master 2>/dev/null; then
        echo "Notifying EXO master about topology changes..." >&2
        systemctl reload exo-master || true
      fi
      
      if systemctl is-active --quiet exo-worker 2>/dev/null; then
        echo "Notifying EXO worker about topology changes..." >&2
        systemctl reload exo-worker || true
      fi
      
      # Update service registrations to reflect new topology
      register_exo_services
      
      echo "Topology adaptation completed" >&2
    }
    
    # Function to monitor service health
    monitor_service_health() {
      echo "Monitoring EXO service health in K3s..." >&2
      
      local node_name=$(hostname)
      local unhealthy_services=()
      
      # Check API service health
      if kubectl get service "exo-api-$node_name" -n "$NAMESPACE" >/dev/null 2>&1; then
        local api_endpoints=$(kubectl get endpoints "exo-api-$node_name" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
        if [ -z "$api_endpoints" ]; then
          unhealthy_services+=("exo-api-$node_name")
        fi
      fi
      
      # Check master service health (if applicable)
      if [ "${exoCfg.mode}" = "master" ] || [ "${exoCfg.mode}" = "auto" ]; then
        if kubectl get service "exo-master-$node_name" -n "$NAMESPACE" >/dev/null 2>&1; then
          local master_endpoints=$(kubectl get endpoints "exo-master-$node_name" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
          if [ -z "$master_endpoints" ]; then
            unhealthy_services+=("exo-master-$node_name")
          fi
        fi
      fi
      
      # Check worker service health (if applicable)
      if [ "${exoCfg.mode}" = "worker" ] || [ "${exoCfg.mode}" = "auto" ]; then
        if kubectl get service "exo-worker-$node_name" -n "$NAMESPACE" >/dev/null 2>&1; then
          local worker_endpoints=$(kubectl get endpoints "exo-worker-$node_name" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
          if [ -z "$worker_endpoints" ]; then
            unhealthy_services+=("exo-worker-$node_name")
          fi
        fi
      fi
      
      # Report unhealthy services
      if [ ''${#unhealthy_services[@]} -gt 0 ]; then
        echo "WARNING: Unhealthy services detected: ''${unhealthy_services[*]}" >&2
        
        # Attempt to re-register unhealthy services
        echo "Attempting to re-register unhealthy services..." >&2
        register_exo_services
      else
        echo "All EXO services are healthy" >&2
      fi
    }
    
    # Function to cleanup services on shutdown
    cleanup_services() {
      echo "Cleaning up EXO services from K3s..." >&2
      
      local node_name=$(hostname)
      
      # Remove services for this node
      kubectl delete service "exo-api-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      kubectl delete service "exo-master-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      kubectl delete service "exo-worker-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      
      # Remove endpoints for this node
      kubectl delete endpoints "exo-api-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      kubectl delete endpoints "exo-master-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      kubectl delete endpoints "exo-worker-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      
      echo "EXO services cleaned up successfully" >&2
    }
    
    # Main service discovery function
    service_discovery_main() {
      echo "Starting EXO K3s service discovery..." >&2
      
      # Check K3s availability
      if ! check_k3s_availability; then
        echo "K3s is not available, skipping service discovery" >&2
        return 1
      fi
      
      # Setup K3s resources
      create_exo_namespace
      create_service_account
      create_cluster_role
      
      # Register services
      register_exo_services
      
      # Discover existing services
      discover_exo_services
      
      echo "EXO K3s service discovery completed successfully" >&2
    }
    
    # Handle different modes
    case "''${1:-main}" in
      "main")
        service_discovery_main
        ;;
      "monitor")
        if check_k3s_availability; then
          monitor_service_health
          discover_exo_services
        fi
        ;;
      "cleanup")
        if check_k3s_availability; then
          cleanup_services
        fi
        ;;
      *)
        echo "Usage: $0 [main|monitor|cleanup]" >&2
        exit 1
        ;;
    esac
  '';

  # K3s network policy integration script
  k3sNetworkPolicyScript = pkgs.writeShellScript "exo-k3s-network-policy" ''
    #!/bin/bash
    # EXO K3s Network Policy Integration
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    NAMESPACE="${cfg.namespace}"
    
    # Function to detect existing K3s network policies
    detect_existing_policies() {
      echo "Detecting existing K3s network policies..." >&2
      
      local existing_policies=$(kubectl get networkpolicies --all-namespaces -o json 2>/dev/null || echo '{"items":[]}')
      local k3s_cni=$(kubectl get nodes -o jsonpath='{.items[0].status.nodeInfo.containerRuntimeVersion}' 2>/dev/null | grep -o 'containerd\|cri-o' || echo "unknown")
      
      # Save existing policies for coordination
      echo "$existing_policies" > "$CONFIG_DIR/k3s-existing-policies.json"
      
      # Detect CNI plugin
      local cni_plugin="flannel"  # K3s default
      if kubectl get ds -n kube-system | grep -q calico; then
        cni_plugin="calico"
      elif kubectl get ds -n kube-system | grep -q cilium; then
        cni_plugin="cilium"
      elif kubectl get ds -n kube-system | grep -q weave; then
        cni_plugin="weave"
      fi
      
      echo "Detected CNI plugin: $cni_plugin" >&2
      echo "$cni_plugin" > "$CONFIG_DIR/k3s-cni-plugin"
      
      return 0
    }
    
    # Function to create EXO network policies that coordinate with K3s
    create_coordinated_network_policies() {
      echo "Creating coordinated EXO network policies..." >&2
      
      # Detect existing policies first
      detect_existing_policies
      
      local cni_plugin=$(cat "$CONFIG_DIR/k3s-cni-plugin" 2>/dev/null || echo "flannel")
      
      # Base EXO internal communication policy
      cat <<EOF | kubectl apply -f -
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: exo-internal-communication
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: network-policy
        exo.ai/policy-type: internal
      annotations:
        exo.ai/cni-plugin: $cni_plugin
        exo.ai/created-at: "$(date -Iseconds)"
    spec:
      podSelector:
        matchLabels:
          app.kubernetes.io/name: exo
      policyTypes:
      - Ingress
      - Egress
      ingress:
      # Allow communication from other EXO pods
      - from:
        - namespaceSelector:
            matchLabels:
              name: $NAMESPACE
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: exo
        ports:
        - protocol: TCP
          port: ${toString exoCfg.apiPort}
        - protocol: UDP
          port: ${toString exoCfg.networking.discoveryPort}
      # Allow communication from K3s system pods
      - from:
        - namespaceSelector:
            matchLabels:
              name: kube-system
        - podSelector:
            matchLabels:
              k8s-app: kube-proxy
        - podSelector:
            matchLabels:
              app: local-path-provisioner
      # Allow health checks from kubelet
      - from: []
        ports:
        - protocol: TCP
          port: 10250  # kubelet health check port
      egress:
      # Allow communication to other EXO pods
      - to:
        - namespaceSelector:
            matchLabels:
              name: $NAMESPACE
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: exo
        ports:
        - protocol: TCP
          port: ${toString exoCfg.apiPort}
        - protocol: UDP
          port: ${toString exoCfg.networking.discoveryPort}
      # Allow DNS resolution
      - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
        - podSelector:
            matchLabels:
              k8s-app: kube-dns
        ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
      # Allow external model downloads and API access
      - to: []
        ports:
        - protocol: TCP
          port: 80
        - protocol: TCP
          port: 443
        - protocol: TCP
          port: 22  # For git+ssh model downloads
    EOF
      
      # K3s system access policy with enhanced coordination
      cat <<EOF | kubectl apply -f -
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: exo-k3s-system-access
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: network-policy
        exo.ai/policy-type: k3s-system
      annotations:
        exo.ai/coordinates-with: "kube-system,kube-node-lease,kube-public"
    spec:
      podSelector:
        matchLabels:
          app.kubernetes.io/name: exo
      policyTypes:
      - Ingress
      ingress:
      # Allow access from K3s system components
      - from:
        - namespaceSelector:
            matchLabels:
              name: kube-system
        ports:
        - protocol: TCP
          port: ${toString exoCfg.apiPort}
      # Allow access from kube-node-lease namespace
      - from:
        - namespaceSelector:
            matchLabels:
              name: kube-node-lease
      # Allow access from kube-public namespace
      - from:
        - namespaceSelector:
            matchLabels:
              name: kube-public
      # Allow access from K3s ingress controllers
      - from:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              app.kubernetes.io/name: traefik
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              app.kubernetes.io/name: nginx-ingress
        ports:
        - protocol: TCP
          port: ${toString exoCfg.apiPort}
    EOF
      
      # Network isolation policy for enhanced security
      cat <<EOF | kubectl apply -f -
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: exo-network-isolation
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: network-policy
        exo.ai/policy-type: isolation
    spec:
      podSelector:
        matchLabels:
          app.kubernetes.io/name: exo
      policyTypes:
      - Ingress
      - Egress
      # Default deny all, specific allows are defined in other policies
      ingress: []
      egress: []
    EOF
      
      # Allow monitoring and metrics collection
      if [ "${toString cfg.monitoring.enable}" = "1" ]; then
        cat <<EOF | kubectl apply -f -
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: exo-monitoring-access
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: network-policy
    spec:
      podSelector:
        matchLabels:
          app.kubernetes.io/name: exo
      policyTypes:
      - Ingress
      ingress:
      - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
        - namespaceSelector:
            matchLabels:
              name: prometheus
        - podSelector:
            matchLabels:
              app: prometheus
        - podSelector:
            matchLabels:
              app: grafana
        ports:
        - protocol: TCP
          port: ${toString cfg.monitoring.metricsPort}
    EOF
      fi
      
      echo "EXO network policies created successfully" >&2
    }
    
    # Function to coordinate with existing K3s network policies
    coordinate_with_existing_policies() {
      echo "Coordinating with existing K3s network policies..." >&2
      
      # Check for conflicting policies
      local conflicting_policies=$(kubectl get networkpolicies --all-namespaces -o json | \
        ${pkgs.jq}/bin/jq -r '.items[] | select(.spec.podSelector.matchLabels."app.kubernetes.io/name" == "exo" and .metadata.namespace != "'$NAMESPACE'") | .metadata.name' 2>/dev/null || echo "")
      
      if [ -n "$conflicting_policies" ]; then
        echo "WARNING: Found potentially conflicting network policies: $conflicting_policies" >&2
        
        # Create coordination annotations
        for policy in $conflicting_policies; do
          kubectl annotate networkpolicy "$policy" exo.ai/coordination-required=true --overwrite 2>/dev/null || true
        done
      fi
      
      # Check for existing ingress controllers and update policies accordingly
      local ingress_controllers=$(kubectl get pods --all-namespaces -l app.kubernetes.io/component=controller -o jsonpath='{.items[*].metadata.labels.app\.kubernetes\.io/name}' 2>/dev/null || echo "")
      
      if echo "$ingress_controllers" | grep -q traefik; then
        echo "Detected Traefik ingress controller, updating policies..." >&2
        kubectl patch networkpolicy exo-k3s-system-access -n "$NAMESPACE" --type='merge' -p='{
          "spec": {
            "ingress": [
              {
                "from": [
                  {
                    "namespaceSelector": {},
                    "podSelector": {
                      "matchLabels": {
                        "app.kubernetes.io/name": "traefik"
                      }
                    }
                  }
                ],
                "ports": [
                  {
                    "protocol": "TCP",
                    "port": '${toString exoCfg.apiPort}'
                  }
                ]
              }
            ]
          }
        }' 2>/dev/null || true
      fi
      
      # Update policies to work with service mesh if detected
      if kubectl get pods --all-namespaces -l app=istio-proxy >/dev/null 2>&1; then
        echo "Detected Istio service mesh, updating policies for sidecar communication..." >&2
        kubectl patch networkpolicy exo-internal-communication -n "$NAMESPACE" --type='merge' -p='{
          "spec": {
            "ingress": [
              {
                "from": [
                  {
                    "podSelector": {
                      "matchLabels": {
                        "app": "istio-proxy"
                      }
                    }
                  }
                ],
                "ports": [
                  {
                    "protocol": "TCP",
                    "port": 15090
                  }
                ]
              }
            ]
          }
        }' 2>/dev/null || true
      fi
      
      echo "Policy coordination completed" >&2
    }
    
    # Function to create EXO network policies
    create_network_policies() {
      echo "Creating EXO network policies..." >&2
      
      # Create coordinated policies
      create_coordinated_network_policies
      
      # Coordinate with existing policies
      coordinate_with_existing_policies
    }
    
    # Function to validate network policies and coordination
    validate_network_policies() {
      echo "Validating EXO network policies and coordination..." >&2
      
      local policies=("exo-internal-communication" "exo-k3s-system-access" "exo-network-isolation")
      if [ "${toString cfg.monitoring.enable}" = "1" ]; then
        policies+=("exo-monitoring-access")
      fi
      
      local validation_failed=false
      
      for policy in "''${policies[@]}"; do
        if kubectl get networkpolicy "$policy" -n "$NAMESPACE" >/dev/null 2>&1; then
          echo "Network policy $policy: OK" >&2
          
          # Validate policy effectiveness
          local policy_rules=$(kubectl get networkpolicy "$policy" -n "$NAMESPACE" -o json | ${pkgs.jq}/bin/jq '.spec')
          if [ "$policy_rules" = "null" ] || [ -z "$policy_rules" ]; then
            echo "WARNING: Network policy $policy has no rules" >&2
            validation_failed=true
          fi
        else
          echo "Network policy $policy: MISSING" >&2
          validation_failed=true
        fi
      done
      
      # Validate coordination with K3s
      local k3s_namespaces=$(kubectl get namespaces -l name=kube-system -o name 2>/dev/null | wc -l)
      if [ "$k3s_namespaces" -eq 0 ]; then
        echo "WARNING: K3s system namespace not properly labeled for network policy coordination" >&2
        validation_failed=true
      fi
      
      # Check for policy conflicts
      local conflicting_policies=$(kubectl get networkpolicies --all-namespaces -o json | \
        ${pkgs.jq}/bin/jq -r '.items[] | select(.spec.podSelector.matchLabels."app.kubernetes.io/name" == "exo" and .metadata.namespace != "'$NAMESPACE'") | .metadata.name' 2>/dev/null || echo "")
      
      if [ -n "$conflicting_policies" ]; then
        echo "WARNING: Found conflicting network policies that may interfere with EXO: $conflicting_policies" >&2
        validation_failed=true
      fi
      
      if [ "$validation_failed" = "true" ]; then
        echo "Network policy validation failed" >&2
        return 1
      fi
      
      echo "All EXO network policies are valid and properly coordinated" >&2
      return 0
    }
    
    # Function to cleanup network policies
    cleanup_network_policies() {
      echo "Cleaning up EXO network policies..." >&2
      
      kubectl delete networkpolicy -n "$NAMESPACE" -l app.kubernetes.io/name=exo 2>/dev/null || true
      
      echo "EXO network policies cleaned up" >&2
    }
    
    # Main execution
    case "''${1:-create}" in
      "create")
        create_network_policies
        ;;
      "validate")
        validate_network_policies
        ;;
      "cleanup")
        cleanup_network_policies
        ;;
      *)
        echo "Usage: $0 [create|validate|cleanup]" >&2
        exit 1
        ;;
    esac
  '';

  # K3s orchestration support script
  k3sOrchestrationScript = pkgs.writeShellScript "exo-k3s-orchestration" ''
    #!/bin/bash
    # EXO K3s Orchestration Support
    
    set -euo pipefail
    
    CONFIG_DIR="${exoCfg.configDir}"
    DATA_DIR="${exoCfg.dataDir}"
    NAMESPACE="${cfg.namespace}"
    
    # Function to create EXO deployment manifests
    create_deployment_manifests() {
      echo "Creating EXO deployment manifests..." >&2
      
      local node_name=$(hostname)
      local node_selector="kubernetes.io/hostname=$node_name"
      
      # Create ConfigMap for EXO configuration
      cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: exo-config
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: config
    data:
      exo-config.json: |
        {
          "mode": "${exoCfg.mode}",
          "api_port": ${toString exoCfg.apiPort},
          "discovery_port": ${toString exoCfg.networking.discoveryPort},
          "log_level": "${exoCfg.logging.level}",
          "k3s_integration": true,
          "k3s_namespace": "$NAMESPACE"
        }
    EOF
      
      # Create PersistentVolumeClaim for model cache
      cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: exo-model-cache-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: storage
        exo.ai/node-name: $node_name
    spec:
      accessModes:
      - ReadWriteOnce
      resources:
        requests:
          storage: ${cfg.orchestration.modelCacheSize}
      storageClassName: ${cfg.orchestration.storageClass}
    EOF
      
      # Create Deployment for EXO services
      cat <<EOF | kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: exo-$node_name
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: distributed-ai
        exo.ai/node-name: $node_name
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: exo
          exo.ai/node-name: $node_name
      template:
        metadata:
          labels:
            app.kubernetes.io/name: exo
            app.kubernetes.io/component: distributed-ai
            exo.ai/node-name: $node_name
            exo.ai/role: ${exoCfg.mode}
        spec:
          serviceAccountName: ${cfg.serviceAccount}
          nodeSelector:
            $node_selector
          tolerations:
          - key: node-role.kubernetes.io/master
            operator: Exists
            effect: NoSchedule
          - key: node-role.kubernetes.io/control-plane
            operator: Exists
            effect: NoSchedule
          containers:
          - name: exo
            image: ${cfg.orchestration.image}
            imagePullPolicy: ${cfg.orchestration.imagePullPolicy}
            ports:
            - name: api
              containerPort: ${toString exoCfg.apiPort}
              protocol: TCP
            - name: discovery
              containerPort: ${toString exoCfg.networking.discoveryPort}
              protocol: UDP
            env:
            - name: EXO_MODE
              value: "${exoCfg.mode}"
            - name: EXO_API_PORT
              value: "${toString exoCfg.apiPort}"
            - name: EXO_DISCOVERY_PORT
              value: "${toString exoCfg.networking.discoveryPort}"
            - name: EXO_LOG_LEVEL
              value: "${exoCfg.logging.level}"
            - name: EXO_K3S_INTEGRATION
              value: "true"
            - name: EXO_K3S_NAMESPACE
              value: "$NAMESPACE"
            - name: EXO_NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: EXO_POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            volumeMounts:
            - name: config
              mountPath: /etc/exo
              readOnly: true
            - name: data
              mountPath: /var/lib/exo
            - name: model-cache
              mountPath: /var/cache/exo/models
            resources:
              requests:
                memory: ${cfg.orchestration.resources.requests.memory}
                cpu: ${cfg.orchestration.resources.requests.cpu}
              limits:
                memory: ${cfg.orchestration.resources.limits.memory}
                cpu: ${cfg.orchestration.resources.limits.cpu}
            livenessProbe:
              httpGet:
                path: /health
                port: api
              initialDelaySeconds: 30
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            readinessProbe:
              httpGet:
                path: /ready
                port: api
              initialDelaySeconds: 10
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 3
          volumes:
          - name: config
            configMap:
              name: exo-config
          - name: data
            emptyDir: {}
          - name: model-cache
            persistentVolumeClaim:
              claimName: exo-model-cache-$node_name
    EOF
      
      echo "EXO deployment manifests created successfully" >&2
    }
    
    # Function to manage resource allocation with K3s coordination
    manage_resource_allocation() {
      echo "Managing EXO resource allocation with K3s coordination..." >&2
      
      local node_name=$(hostname)
      
      # Get current resource usage from K3s metrics
      local cpu_usage=$(kubectl top node "$node_name" --no-headers | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "0")
      local memory_usage=$(kubectl top node "$node_name" --no-headers | awk '{print $5}' | sed 's/%//' 2>/dev/null || echo "0")
      
      echo "Node $node_name resource usage: CPU $cpu_usage%, Memory $memory_usage%" >&2
      
      # Get K3s resource allocations
      local k3s_cpu_requests=$(kubectl describe node "$node_name" | grep -A 10 "Allocated resources:" | grep "cpu" | awk '{print $2}' | sed 's/[^0-9]//g' 2>/dev/null || echo "0")
      local k3s_memory_requests=$(kubectl describe node "$node_name" | grep -A 10 "Allocated resources:" | grep "memory" | awk '{print $2}' | sed 's/[^0-9]//g' 2>/dev/null || echo "0")
      
      # Calculate available resources after K3s allocations
      local total_cpu=$(kubectl describe node "$node_name" | grep -A 5 "Capacity:" | grep "cpu:" | awk '{print $2}' | sed 's/[^0-9]//g' 2>/dev/null || echo "1000")
      local total_memory=$(kubectl describe node "$node_name" | grep -A 5 "Capacity:" | grep "memory:" | awk '{print $2}' | sed 's/Ki//' 2>/dev/null || echo "1000000")
      
      local available_cpu=$((total_cpu - k3s_cpu_requests))
      local available_memory=$((total_memory - k3s_memory_requests))
      
      echo "Available resources after K3s allocation: CPU ${available_cpu}m, Memory ${available_memory}Ki" >&2
      
      # Coordinate with K3s scheduler for optimal resource allocation
      local optimal_cpu_request=$((available_cpu * 60 / 100))  # Use 60% of available CPU
      local optimal_memory_request=$((available_memory * 60 / 100))  # Use 60% of available memory
      local optimal_cpu_limit=$((optimal_cpu_request * 150 / 100))  # 150% of request for bursting
      local optimal_memory_limit=$((optimal_memory_request * 120 / 100))  # 120% of request for bursting
      
      # Ensure minimum viable resources
      if [ "$optimal_cpu_request" -lt 100 ]; then
        optimal_cpu_request=100  # Minimum 100m CPU
      fi
      if [ "$optimal_memory_request" -lt 512000 ]; then
        optimal_memory_request=512000  # Minimum 512Mi memory
      fi
      
      echo "Optimal EXO resource allocation: CPU ${optimal_cpu_request}m (limit: ${optimal_cpu_limit}m), Memory ${optimal_memory_request}Ki (limit: ${optimal_memory_limit}Ki)" >&2
      
      # Check if resources are under pressure
      if [ "$cpu_usage" -gt 80 ] || [ "$memory_usage" -gt 80 ]; then
        echo "High resource usage detected, implementing resource pressure handling..." >&2
        
        # Scale down EXO deployment temporarily
        kubectl scale deployment "exo-$node_name" -n "$NAMESPACE" --replicas=0 2>/dev/null || true
        sleep 10
        
        # Reduce resource requests under pressure
        optimal_cpu_request=$((optimal_cpu_request * 70 / 100))
        optimal_memory_request=$((optimal_memory_request * 70 / 100))
        
        echo "Reduced resource allocation due to pressure: CPU ${optimal_cpu_request}m, Memory ${optimal_memory_request}Ki" >&2
      fi
      
      # Update deployment with coordinated resource allocation
      kubectl patch deployment "exo-$node_name" -n "$NAMESPACE" --type='merge' -p="{
        \"spec\": {
          \"template\": {
            \"spec\": {
              \"containers\": [{
                \"name\": \"exo\",
                \"resources\": {
                  \"requests\": {
                    \"memory\": \"''${optimal_memory_request}Ki\",
                    \"cpu\": \"''${optimal_cpu_request}m\"
                  },
                  \"limits\": {
                    \"memory\": \"''${optimal_memory_limit}Ki\",
                    \"cpu\": \"''${optimal_cpu_limit}m\"
                  }
                }
              }]
            }
          }
        }
      }" 2>/dev/null || true
      
      # Scale back up if we scaled down
      if [ "$cpu_usage" -gt 80 ] || [ "$memory_usage" -gt 80 ]; then
        kubectl scale deployment "exo-$node_name" -n "$NAMESPACE" --replicas=1 2>/dev/null || true
      fi
      
      # Create or update resource quota for the namespace
      cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: ResourceQuota
    metadata:
      name: exo-resource-quota
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: resource-management
    spec:
      hard:
        requests.cpu: "${optimal_cpu_request}m"
        requests.memory: "${optimal_memory_request}Ki"
        limits.cpu: "${optimal_cpu_limit}m"
        limits.memory: "${optimal_memory_limit}Ki"
        persistentvolumeclaims: "10"
        services: "20"
    EOF
      
      echo "Resource allocation coordination with K3s completed" >&2
    }
    
    # Function to monitor cluster-wide resources with K3s coordination
    monitor_cluster_resources() {
      echo "Monitoring cluster-wide EXO resources with K3s coordination..." >&2
      
      # Get all EXO deployments
      local deployments=$(kubectl get deployments -n "$NAMESPACE" -l app.kubernetes.io/name=exo -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
      
      # Track cluster-wide resource usage
      local total_cpu_requests=0
      local total_memory_requests=0
      local total_replicas=0
      local total_ready_replicas=0
      
      for deployment in $deployments; do
        if [ -n "$deployment" ]; then
          local replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo "0")
          local ready_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
          
          echo "Deployment $deployment: $ready_replicas/$replicas replicas ready" >&2
          
          total_replicas=$((total_replicas + replicas))
          total_ready_replicas=$((total_ready_replicas + ready_replicas))
          
          # Get resource requests for this deployment
          local cpu_request=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}' 2>/dev/null | sed 's/m//' || echo "0")
          local memory_request=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}' 2>/dev/null | sed 's/Ki//' || echo "0")
          
          total_cpu_requests=$((total_cpu_requests + cpu_request))
          total_memory_requests=$((total_memory_requests + memory_request))
          
          # Restart unhealthy deployments
          if [ "$ready_replicas" -lt "$replicas" ]; then
            echo "Restarting unhealthy deployment: $deployment" >&2
            kubectl rollout restart deployment "$deployment" -n "$NAMESPACE" || true
          fi
        fi
      done
      
      echo "Cluster-wide EXO resource usage: CPU ${total_cpu_requests}m, Memory ${total_memory_requests}Ki" >&2
      echo "Cluster health: $total_ready_replicas/$total_replicas replicas ready" >&2
      
      # Implement cluster-wide load balancing
      if [ "$total_ready_replicas" -gt 1 ]; then
        echo "Implementing cluster-wide load balancing..." >&2
        
        # Create or update cluster-wide service for load balancing
        cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: Service
    metadata:
      name: exo-cluster-api
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: cluster-api
        exo.ai/service-type: cluster-wide
      annotations:
        exo.ai/load-balancing: "round-robin"
        exo.ai/total-replicas: "$total_replicas"
        exo.ai/ready-replicas: "$total_ready_replicas"
    spec:
      type: ClusterIP
      ports:
      - name: api
        port: ${toString exoCfg.apiPort}
        targetPort: ${toString exoCfg.apiPort}
        protocol: TCP
      selector:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: distributed-ai
      sessionAffinity: None  # Enable round-robin load balancing
    EOF
        
        # Create horizontal pod autoscaler for cluster-wide scaling
        cat <<EOF | kubectl apply -f -
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: exo-cluster-hpa
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: autoscaling
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: exo-$(hostname)
      minReplicas: 1
      maxReplicas: 3
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
      - type: Resource
        resource:
          name: memory
          target:
            type: Utilization
            averageUtilization: 80
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
          - type: Percent
            value: 50
            periodSeconds: 60
        scaleUp:
          stabilizationWindowSeconds: 60
          policies:
          - type: Percent
            value: 100
            periodSeconds: 60
    EOF
      fi
      
      # Monitor PVC usage across the cluster
      local pvcs=$(kubectl get pvc -n "$NAMESPACE" -l app.kubernetes.io/name=exo -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
      local total_storage_used=0
      
      for pvc in $pvcs; do
        if [ -n "$pvc" ]; then
          local pvc_status=$(kubectl get pvc "$pvc" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
          local pvc_size=$(kubectl get pvc "$pvc" -n "$NAMESPACE" -o jsonpath='{.spec.resources.requests.storage}' 2>/dev/null | sed 's/Gi//' || echo "0")
          
          echo "PVC $pvc status: $pvc_status (size: ${pvc_size}Gi)" >&2
          
          total_storage_used=$((total_storage_used + pvc_size))
          
          if [ "$pvc_status" != "Bound" ]; then
            echo "WARNING: PVC $pvc is not bound" >&2
          fi
        fi
      done
      
      echo "Total cluster storage used: ${total_storage_used}Gi" >&2
      
      # Implement storage cleanup if usage is high
      if [ "$total_storage_used" -gt 200 ]; then  # More than 200Gi used
        echo "High storage usage detected, implementing cleanup..." >&2
        
        # Create cleanup job
        cat <<EOF | kubectl apply -f -
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: exo-storage-cleanup-$(date +%s)
      namespace: $NAMESPACE
      labels:
        app.kubernetes.io/name: exo
        app.kubernetes.io/component: cleanup
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: cleanup
            image: alpine:latest
            command: ["/bin/sh"]
            args:
            - -c
            - |
              echo "Cleaning up old model cache files..."
              find /var/cache/exo/models -type f -mtime +7 -delete || true
              echo "Cleanup completed"
            volumeMounts:
            - name: model-cache
              mountPath: /var/cache/exo/models
          volumes:
          - name: model-cache
            persistentVolumeClaim:
              claimName: exo-model-cache-$(hostname)
    EOF
      fi
    }
    
    # Function to cleanup orchestration resources
    cleanup_orchestration() {
      echo "Cleaning up EXO orchestration resources..." >&2
      
      local node_name=$(hostname)
      
      # Delete deployment
      kubectl delete deployment "exo-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      
      # Delete PVC (optional, preserves data)
      if [ "${toString cfg.orchestration.cleanupStorage}" = "1" ]; then
        kubectl delete pvc "exo-model-cache-$node_name" -n "$NAMESPACE" 2>/dev/null || true
      fi
      
      # Delete ConfigMap
      kubectl delete configmap exo-config -n "$NAMESPACE" 2>/dev/null || true
      
      echo "EXO orchestration resources cleaned up" >&2
    }
    
    # Main execution
    case "''${1:-create}" in
      "create")
        create_deployment_manifests
        ;;
      "manage")
        manage_resource_allocation
        ;;
      "monitor")
        monitor_cluster_resources
        ;;
      "cleanup")
        cleanup_orchestration
        ;;
      *)
        echo "Usage: $0 [create|manage|monitor|cleanup]" >&2
        exit 1
        ;;
    esac
  '';

in {
  options.services.exo.k3s = {
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
      type = types.str;
      default = "exo-system";
      description = ''
        Kubernetes namespace for EXO services.
        Will be created automatically if it doesn't exist.
      '';
    };

    serviceAccount = mkOption {
      type = types.str;
      default = "exo";
      description = ''
        Kubernetes service account for EXO services.
        Will be created automatically with appropriate permissions.
      '';
    };

    clusterRole = mkOption {
      type = types.str;
      default = "exo-cluster-role";
      description = ''
        Kubernetes cluster role for EXO services.
        Defines permissions for cluster-wide operations.
      '';
    };

    networkPolicy = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable Kubernetes network policies for EXO services.
          Provides network isolation and security.
        '';
      };

      allowK3sSystem = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Allow access from K3s system components.
          Required for proper integration with Kubernetes.
        '';
      };

      allowMonitoring = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Allow access from monitoring systems.
          Enables metrics collection and monitoring.
        '';
      };
    };

    orchestration = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Enable Kubernetes orchestration for EXO workloads.
          Manages EXO services as Kubernetes deployments.
        '';
      };

      image = mkOption {
        type = types.str;
        default = "exo:latest";
        description = ''
          Container image for EXO services.
          Should contain the EXO application and dependencies.
        '';
      };

      imagePullPolicy = mkOption {
        type = types.enum [ "Always" "IfNotPresent" "Never" ];
        default = "IfNotPresent";
        description = ''
          Image pull policy for EXO containers.
          Controls when Kubernetes pulls the container image.
        '';
      };

      storageClass = mkOption {
        type = types.str;
        default = "local-path";
        description = ''
          Storage class for EXO persistent volumes.
          Used for model cache and data storage.
        '';
      };

      modelCacheSize = mkOption {
        type = types.str;
        default = "50Gi";
        description = ''
          Size of the model cache persistent volume.
          Should be large enough to store AI models.
        '';
      };

      cleanupStorage = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Cleanup persistent storage on service removal.
          WARNING: This will delete all cached models.
        '';
      };

      resources = {
        requests = {
          memory = mkOption {
            type = types.str;
            default = "2Gi";
            description = ''
              Memory request for EXO containers.
              Minimum memory guaranteed for the container.
            '';
          };

          cpu = mkOption {
            type = types.str;
            default = "500m";
            description = ''
              CPU request for EXO containers.
              Minimum CPU guaranteed for the container.
            '';
          };
        };

        limits = {
          memory = mkOption {
            type = types.str;
            default = "8Gi";
            description = ''
              Memory limit for EXO containers.
              Maximum memory the container can use.
            '';
          };

          cpu = mkOption {
            type = types.str;
            default = "2000m";
            description = ''
              CPU limit for EXO containers.
              Maximum CPU the container can use.
            '';
          };
        };
      };
    };

    monitoring = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable monitoring integration with K3s.
          Exposes metrics for Prometheus and other monitoring systems.
        '';
      };

      metricsPort = mkOption {
        type = types.port;
        default = 9090;
        description = ''
          Port for exposing Prometheus metrics.
          Should not conflict with other services.
        '';
      };

      serviceMonitor = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Create ServiceMonitor resources for Prometheus Operator.
          Enables automatic metrics discovery.
        '';
      };
    };

    autoRegistration = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Enable automatic service registration and deregistration.
          Services are automatically managed based on node status.
        '';
      };

      interval = mkOption {
        type = types.str;
        default = "30s";
        description = ''
          Interval for service registration updates.
          More frequent updates provide better accuracy but increase API load.
        '';
      };

      healthCheckInterval = mkOption {
        type = types.str;
        default = "60s";
        description = ''
          Interval for service health checks.
          Unhealthy services are automatically re-registered.
        '';
      };
    };
  };

  config = mkIf (exoCfg.enable && cfg.integration) {
    # K3s service discovery service
    systemd.services.exo-k3s-service-discovery = mkIf cfg.serviceDiscovery {
      description = "EXO K3s Service Discovery";
      wantedBy = [ "exo.target" ];
      after = [ "k3s.service" "network-online.target" ];
      wants = [ "k3s.service" "network-online.target" ];
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = "${k3sServiceDiscoveryScript} main";
        ExecReload = "${k3sServiceDiscoveryScript} main";
        ExecStop = "${k3sServiceDiscoveryScript} cleanup";
        
        # Environment
        Environment = [
          "KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
          "EXO_CONFIG_DIR=${exoCfg.configDir}"
          "EXO_DATA_DIR=${exoCfg.dataDir}"
        ];
        
        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir exoCfg.dataDir ];
        ReadOnlyPaths = [ "/etc/rancher" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
        
        # Network access for K3s API
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # K3s service monitoring timer
    systemd.timers.exo-k3s-service-monitor = mkIf (cfg.serviceDiscovery && cfg.autoRegistration.enable) {
      description = "EXO K3s Service Monitor Timer";
      wantedBy = [ "timers.target" ];
      
      timerConfig = {
        OnBootSec = "2min";
        OnUnitActiveSec = cfg.autoRegistration.healthCheckInterval;
        Unit = "exo-k3s-service-monitor.service";
      };
    };

    systemd.services.exo-k3s-service-monitor = mkIf (cfg.serviceDiscovery && cfg.autoRegistration.enable) {
      description = "EXO K3s Service Monitor";
      
      serviceConfig = {
        Type = "oneshot";
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = "${k3sServiceDiscoveryScript} monitor";
        
        # Environment
        Environment = [
          "KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
          "EXO_CONFIG_DIR=${exoCfg.configDir}"
          "EXO_DATA_DIR=${exoCfg.dataDir}"
        ];
        
        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir exoCfg.dataDir ];
        ReadOnlyPaths = [ "/etc/rancher" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
        
        # Network access for K3s API
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # K3s network policy service
    systemd.services.exo-k3s-network-policy = mkIf cfg.networkPolicy.enable {
      description = "EXO K3s Network Policy";
      wantedBy = [ "exo.target" ];
      after = [ "exo-k3s-service-discovery.service" ];
      requires = [ "exo-k3s-service-discovery.service" ];
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = "${k3sNetworkPolicyScript} create";
        ExecReload = "${k3sNetworkPolicyScript} create";
        ExecStop = "${k3sNetworkPolicyScript} cleanup";
        
        # Environment
        Environment = [
          "KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
        ];
        
        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadOnlyPaths = [ "/etc/rancher" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
        
        # Network access for K3s API
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # K3s orchestration service
    systemd.services.exo-k3s-orchestration = mkIf cfg.orchestration.enable {
      description = "EXO K3s Orchestration";
      wantedBy = [ "exo.target" ];
      after = [ "exo-k3s-service-discovery.service" ];
      requires = [ "exo-k3s-service-discovery.service" ];
      partOf = [ "exo.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = "${k3sOrchestrationScript} create";
        ExecReload = "${k3sOrchestrationScript} manage";
        ExecStop = "${k3sOrchestrationScript} cleanup";
        
        # Environment
        Environment = [
          "KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
          "EXO_CONFIG_DIR=${exoCfg.configDir}"
          "EXO_DATA_DIR=${exoCfg.dataDir}"
        ];
        
        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir exoCfg.dataDir ];
        ReadOnlyPaths = [ "/etc/rancher" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
        
        # Network access for K3s API
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # K3s orchestration monitoring timer
    systemd.timers.exo-k3s-orchestration-monitor = mkIf cfg.orchestration.enable {
      description = "EXO K3s Orchestration Monitor Timer";
      wantedBy = [ "timers.target" ];
      
      timerConfig = {
        OnBootSec = "5min";
        OnUnitActiveSec = "5min";
        Unit = "exo-k3s-orchestration-monitor.service";
      };
    };

    systemd.services.exo-k3s-orchestration-monitor = mkIf cfg.orchestration.enable {
      description = "EXO K3s Orchestration Monitor";
      
      serviceConfig = {
        Type = "oneshot";
        User = exoCfg.user;
        Group = exoCfg.group;
        ExecStart = "${k3sOrchestrationScript} monitor";
        
        # Environment
        Environment = [
          "KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
          "EXO_CONFIG_DIR=${exoCfg.configDir}"
          "EXO_DATA_DIR=${exoCfg.dataDir}"
        ];
        
        # Security settings
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ exoCfg.configDir exoCfg.dataDir ];
        ReadOnlyPaths = [ "/etc/rancher" ];
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictRealtime = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RemoveIPC = true;
        
        # Network access for K3s API
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };
    };

    # Required packages for K3s integration
    environment.systemPackages = [
      pkgs.kubectl
      pkgs.jq
      pkgs.curl
    ];

    # User permissions for K3s integration
    users.users.${exoCfg.user}.extraGroups = [ "k3s" ];

    # Additional systemd tmpfiles for K3s integration
    systemd.tmpfiles.rules = [
      "d ${exoCfg.configDir}/k3s 0750 ${exoCfg.user} ${exoCfg.group} -"
      "d ${exoCfg.dataDir}/k3s 0750 ${exoCfg.user} ${exoCfg.group} -"
    ];
  };
}
# EXO K3s Integration Guide

This guide explains how to integrate EXO distributed AI inference system with existing K3s (Kubernetes) clusters on NixOS. EXO can work alongside K3s to provide AI inference capabilities while leveraging existing network infrastructure and orchestration.

## Overview

EXO K3s integration provides:

- **Service Discovery**: Automatic registration with Kubernetes API
- **Network Integration**: Utilization of existing bonded interfaces and network policies
- **Resource Coordination**: Intelligent resource sharing between K3s and EXO workloads
- **Load Balancing**: Distribution of AI inference requests across cluster nodes
- **Monitoring Integration**: Unified monitoring with existing K3s infrastructure

## Prerequisites

### Existing K3s Setup

This guide assumes you have:

- A working K3s cluster on NixOS
- Bonded network interfaces configured
- Basic Kubernetes knowledge
- Administrative access to cluster nodes

### Network Requirements

- **Bonded Interfaces**: EXO will utilize existing bonded network configuration
- **Port Availability**: Ensure EXO ports don't conflict with K3s services
- **Network Policies**: Compatible with existing K3s network policies

## Integration Methods

### Method 1: Standalone Integration (Recommended)

This method runs EXO as a separate service that registers with K3s for service discovery.

#### Step 1: Configure EXO with K3s Integration

Add to your NixOS configuration:

```nix
{ config, pkgs, ... }:

{
  # Existing K3s configuration
  services.k3s = {
    enable = true;
    role = "server";  # or "agent"
    # ... your existing K3s config
  };

  # EXO with K3s integration
  services.exo = {
    enable = true;
    mode = "auto";
    
    # Network integration with existing K3s setup
    networking = {
      # Use the same bonded interface as K3s
      bondInterface = config.services.k3s.networking.bondInterface or "bond0";
      rdmaEnabled = true;
      
      # Avoid port conflicts with K3s
      apiPort = 52415;
      discoveryPort = 52416;
      
      # Integrate with K3s networking
      openFirewall = true;
    };
    
    # Enable K3s integration
    k3s = {
      integration = true;
      serviceDiscovery = true;
      namespace = "exo-system";
      
      # Coordinate with K3s resource management
      resourceCoordination = true;
    };
    
    # Hardware configuration
    hardware = {
      autoDetect = true;
      # Reserve some resources for K3s workloads
      memoryLimit = "70%";  # Leave 30% for K3s
    };
    
    # Dashboard configuration
    dashboard = {
      enable = true;
      port = 8080;  # Ensure no conflict with K3s dashboard
    };
  };
}
```

#### Step 2: Create EXO Namespace

Create a dedicated namespace for EXO services:

```bash
kubectl create namespace exo-system
```

#### Step 3: Apply Service Definitions

EXO will automatically register services, but you can also create explicit service definitions:

```yaml
# exo-services.yaml
apiVersion: v1
kind: Service
metadata:
  name: exo-api
  namespace: exo-system
  labels:
    app: exo
    component: api
spec:
  selector:
    app: exo
    component: api
  ports:
  - name: api
    port: 52415
    targetPort: 52415
    protocol: TCP
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: exo-dashboard
  namespace: exo-system
  labels:
    app: exo
    component: dashboard
spec:
  selector:
    app: exo
    component: dashboard
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  type: LoadBalancer  # or NodePort for external access

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: exo-config
  namespace: exo-system
data:
  exo.yaml: |
    api:
      port: 52415
      bind_address: "0.0.0.0"
    networking:
      discovery_port: 52416
      bond_interface: "bond0"
      rdma_enabled: true
    hardware:
      auto_detect: true
      memory_limit: "70%"
```

Apply the services:

```bash
kubectl apply -f exo-services.yaml
```

### Method 2: Kubernetes-Native Deployment

Deploy EXO as Kubernetes workloads using the NixOS-built containers.

#### Step 1: Build Container Images

```bash
# Build EXO container image using Nix
nix build github:exo-explore/exo#exo-container

# Load into containerd (K3s default)
sudo ctr -n k8s.io images import result
```

#### Step 2: Create Deployment Manifests

```yaml
# exo-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: exo-master
  namespace: exo-system
  labels:
    app: exo
    component: master
spec:
  replicas: 1  # Only one master per cluster
  selector:
    matchLabels:
      app: exo
      component: master
  template:
    metadata:
      labels:
        app: exo
        component: master
    spec:
      serviceAccountName: exo
      containers:
      - name: exo-master
        image: exo:latest
        command: ["exo", "--mode", "master"]
        ports:
        - containerPort: 52415
          name: api
        - containerPort: 52416
          name: discovery
        env:
        - name: EXO_LOG_LEVEL
          value: "info"
        - name: EXO_API_PORT
          value: "52415"
        - name: EXO_DISCOVERY_PORT
          value: "52416"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: exo-data
          mountPath: /var/lib/exo
        - name: exo-config
          mountPath: /etc/exo
      volumes:
      - name: exo-data
        persistentVolumeClaim:
          claimName: exo-data
      - name: exo-config
        configMap:
          name: exo-config

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: exo-worker
  namespace: exo-system
  labels:
    app: exo
    component: worker
spec:
  selector:
    matchLabels:
      app: exo
      component: worker
  template:
    metadata:
      labels:
        app: exo
        component: worker
    spec:
      serviceAccountName: exo
      hostNetwork: true  # For RDMA and bonded interface access
      containers:
      - name: exo-worker
        image: exo:latest
        command: ["exo", "--mode", "worker"]
        env:
        - name: EXO_LOG_LEVEL
          value: "info"
        - name: EXO_MASTER_ENDPOINT
          value: "exo-master.exo-system.svc.cluster.local:52415"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "32Gi"  # Adjust based on available memory
            cpu: "8"
            nvidia.com/gpu: 1  # If using GPU nodes
        volumeMounts:
        - name: exo-data
          mountPath: /var/lib/exo
        - name: dev-nvidia
          mountPath: /dev/nvidia0  # GPU access
        securityContext:
          privileged: true  # Required for GPU access
      volumes:
      - name: exo-data
        hostPath:
          path: /var/lib/exo
          type: DirectoryOrCreate
      - name: dev-nvidia
        hostPath:
          path: /dev/nvidia0
          type: CharDevice
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

#### Step 3: Create RBAC and Storage

```yaml
# exo-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: exo
  namespace: exo-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: exo
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "daemonsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["nodes", "pods"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: exo
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: exo
subjects:
- kind: ServiceAccount
  name: exo
  namespace: exo-system

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exo-data
  namespace: exo-system
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi  # Adjust based on model storage needs
  storageClassName: local-path  # K3s default storage class
```

Apply all manifests:

```bash
kubectl apply -f exo-rbac.yaml
kubectl apply -f exo-deployment.yaml
```

## Network Configuration

### Bonded Interface Integration

EXO automatically detects and uses existing bonded interfaces configured for K3s.

#### Verify Bonded Interface Configuration

```bash
# Check existing bonded interfaces
ip link show type bond

# Verify bond configuration
cat /proc/net/bonding/bond0

# Check K3s network configuration
kubectl get nodes -o wide
```

#### Configure EXO to Use Specific Interface

```nix
services.exo.networking = {
  bondInterface = "bond0";  # Match your K3s bond interface
  
  # Advanced bonding options
  bondOptions = {
    mode = "802.3ad";  # Match K3s bonding mode
    miimon = 100;
    lacp_rate = "fast";
  };
};
```

### RDMA over Thunderbolt Configuration

For ultra-low latency communication between nodes:

```nix
services.exo.networking = {
  rdmaEnabled = true;
  
  # Thunderbolt-specific configuration
  thunderbolt = {
    enable = true;
    interfaces = [ "thunderbolt0" "thunderbolt1" ];
    rdmaMode = "RoCE";  # RDMA over Converged Ethernet
  };
};
```

### Network Policy Integration

Create network policies that allow EXO communication:

```yaml
# exo-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: exo-communication
  namespace: exo-system
spec:
  podSelector:
    matchLabels:
      app: exo
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: exo-system
    - podSelector:
        matchLabels:
          app: exo
    ports:
    - protocol: TCP
      port: 52415  # API port
    - protocol: TCP
      port: 52416  # Discovery port
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: exo-system
    - podSelector:
        matchLabels:
          app: exo
    ports:
    - protocol: TCP
      port: 52415
    - protocol: TCP
      port: 52416
  - to: []  # Allow external communication for model downloads
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

## Resource Coordination

### Memory and CPU Coordination

Configure EXO to coordinate with K3s resource allocation:

```nix
services.exo = {
  # Resource coordination with K3s
  k3s.resourceCoordination = true;
  
  hardware = {
    # Reserve resources for K3s workloads
    memoryLimit = "70%";  # Leave 30% for K3s pods
    cpuLimit = "75%";     # Leave 25% for K3s system processes
    
    # GPU sharing configuration
    gpu = {
      sharing = {
        enable = true;
        strategy = "time-slicing";  # or "mps" for NVIDIA MPS
        maxSharedProcesses = 4;
      };
    };
  };
};
```

### Storage Coordination

Configure shared storage for models and data:

```yaml
# exo-storage.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: exo-models
spec:
  capacity:
    storage: 500Gi
  accessModes:
  - ReadWriteMany  # Shared across nodes
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs-client  # or your preferred storage class
  nfs:
    server: your-nfs-server.local
    path: /exports/exo-models

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exo-models
  namespace: exo-system
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: nfs-client
```

## Service Discovery and Load Balancing

### Automatic Service Registration

EXO automatically registers with K3s service discovery:

```bash
# Verify EXO service registration
kubectl get services -n exo-system

# Check endpoints
kubectl get endpoints -n exo-system

# View service discovery logs
kubectl logs -n exo-system -l app=exo,component=master
```

### Load Balancing Configuration

Configure load balancing for AI inference requests:

```yaml
# exo-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: exo-api
  namespace: exo-system
  annotations:
    kubernetes.io/ingress.class: traefik  # K3s default ingress
    traefik.ingress.kubernetes.io/router.middlewares: exo-system-auth@kubernetescrd
spec:
  rules:
  - host: exo-api.your-domain.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: exo-api
            port:
              number: 52415
  - host: exo-dashboard.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: exo-dashboard
            port:
              number: 8080

---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: auth
  namespace: exo-system
spec:
  basicAuth:
    secret: exo-auth
```

## Monitoring and Observability

### Prometheus Integration

Integrate EXO metrics with existing Prometheus setup:

```yaml
# exo-monitoring.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: exo
  namespace: exo-system
  labels:
    app: exo
spec:
  selector:
    matchLabels:
      app: exo
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics

---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: exo-alerts
  namespace: exo-system
spec:
  groups:
  - name: exo
    rules:
    - alert: EXONodeDown
      expr: up{job="exo"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "EXO node is down"
        description: "EXO node {{ $labels.instance }} has been down for more than 5 minutes"
    
    - alert: EXOHighMemoryUsage
      expr: exo_memory_usage_percent > 90
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "EXO high memory usage"
        description: "EXO node {{ $labels.instance }} memory usage is above 90%"
```

### Grafana Dashboard

Import EXO dashboard into existing Grafana:

```json
{
  "dashboard": {
    "title": "EXO Cluster Overview",
    "panels": [
      {
        "title": "Cluster Topology",
        "type": "graph",
        "targets": [
          {
            "expr": "exo_cluster_nodes",
            "legendFormat": "Active Nodes"
          }
        ]
      },
      {
        "title": "Inference Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(exo_inference_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "exo_gpu_utilization_percent",
            "legendFormat": "GPU {{ $labels.device }}"
          }
        ]
      }
    ]
  }
}
```

## Migration from Standalone to K3s

### Step 1: Backup Existing Configuration

```bash
# Backup EXO data
sudo cp -r /var/lib/exo /var/lib/exo.backup

# Export current configuration
sudo nixos-rebuild dry-build --show-trace > current-config.log
```

### Step 2: Update Configuration

Modify your NixOS configuration to enable K3s integration:

```nix
services.exo = {
  # ... existing configuration ...
  
  # Add K3s integration
  k3s = {
    integration = true;
    serviceDiscovery = true;
    
    # Migration settings
    migration = {
      preserveData = true;
      dataPath = "/var/lib/exo";
    };
  };
};
```

### Step 3: Gradual Migration

1. **Enable K3s integration** without changing service mode
2. **Verify service discovery** works correctly
3. **Migrate workloads** gradually to K3s orchestration
4. **Update client applications** to use K3s service endpoints

### Step 4: Validation

```bash
# Verify K3s integration
kubectl get services -n exo-system

# Test API through K3s service
kubectl port-forward -n exo-system svc/exo-api 52415:52415 &
curl http://localhost:52415/v1/models

# Check cluster topology
kubectl exec -n exo-system deployment/exo-master -- exo --list-nodes
```

## Troubleshooting

### Common Integration Issues

#### Service Discovery Problems

```bash
# Check K3s API server connectivity
kubectl cluster-info

# Verify EXO service registration
kubectl get services -n exo-system -o yaml

# Check DNS resolution
kubectl exec -n exo-system deployment/exo-master -- nslookup kubernetes.default.svc.cluster.local
```

#### Network Connectivity Issues

```bash
# Test inter-node communication
kubectl exec -n exo-system deployment/exo-master -- ping exo-worker-node-ip

# Check bonded interface status
ip link show bond0
cat /proc/net/bonding/bond0

# Verify RDMA functionality
ibstat  # If using InfiniBand
```

#### Resource Conflicts

```bash
# Check resource allocation
kubectl top nodes
kubectl top pods -n exo-system

# Verify GPU sharing
nvidia-smi  # On GPU nodes
kubectl describe node gpu-node-name
```

### Performance Optimization

#### Network Optimization

```nix
services.exo.networking = {
  # Optimize for K3s integration
  bufferSizes = {
    send = "64MB";
    receive = "64MB";
  };
  
  # TCP optimization
  tcpOptimization = {
    enable = true;
    congestionControl = "bbr";
    windowScaling = true;
  };
};
```

#### Resource Optimization

```nix
services.exo.hardware = {
  # Optimize for K3s coexistence
  scheduling = {
    cpuAffinity = [ 4 5 6 7 ];  # Use specific CPU cores
    numaAware = true;
  };
  
  # Memory optimization
  memoryOptimization = {
    hugepages = true;
    swappiness = 10;
  };
};
```

## Best Practices

### Security

1. **Network Segmentation**: Use network policies to isolate EXO traffic
2. **RBAC**: Implement least-privilege access for EXO services
3. **Secrets Management**: Use K3s secrets for sensitive configuration
4. **Pod Security**: Enable pod security standards

### Reliability

1. **Health Checks**: Implement comprehensive health checks
2. **Resource Limits**: Set appropriate resource limits and requests
3. **Backup Strategy**: Regular backup of EXO data and configuration
4. **Monitoring**: Comprehensive monitoring and alerting

### Performance

1. **Resource Allocation**: Proper resource allocation between K3s and EXO
2. **Network Optimization**: Optimize network configuration for AI workloads
3. **Storage Performance**: Use high-performance storage for model cache
4. **GPU Sharing**: Implement efficient GPU sharing strategies

## Next Steps

- [Hardware Compatibility Guide](nixos-hardware-compatibility.md) - Optimize for your specific hardware
- [Performance Tuning](nixos-performance-tuning.md) - Advanced performance optimization
- [Multi-Node Setup](nixos-multi-node.md) - Scale across multiple nodes
- [Security Hardening](nixos-security.md) - Production security configuration
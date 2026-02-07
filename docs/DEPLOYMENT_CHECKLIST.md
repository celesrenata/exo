# Gremlin Cluster Deployment Checklist

Use this checklist to track deployment progress for the Intel hardware support.

## Pre-Deployment

- [ ] All code changes committed to git
- [ ] All tests pass locally
- [ ] Documentation is complete
- [ ] flake.nix is finalized
- [ ] Release tagged in git (e.g., v0.1.0-intel)
- [ ] Changes pushed to remote repository

## Task 9: Single-Node Validation (gremlin-1)

### 9.1 Build
- [ ] Flake builds successfully on gremlin-1
- [ ] All dependencies resolve correctly
- [ ] Tinygrad backend compiles
- [ ] No build errors or warnings

### 9.2 Service Start
- [ ] exo service starts without errors
- [ ] Service initializes correctly
- [ ] Logs show proper initialization
- [ ] No crashes on startup

### 9.3 Web Service
- [ ] API accessible at http://10.1.1.12:52415
- [ ] Health endpoint responds
- [ ] OpenAI-compatible API available
- [ ] Dashboard loads correctly

### 9.4 GPU Detection
- [ ] Intel Arc iGPU detected
- [ ] Level Zero runtime available
- [ ] OpenCL fallback works (if Level Zero fails)
- [ ] GPU appears in metrics
- [ ] Device info correct in dashboard

### 9.5 NPU Detection
- [ ] Intel NPU hardware detected
- [ ] Kernel modules loaded (intel_vpu)
- [ ] OpenVINO can access NPU
- [ ] NPU appears in capability report
- [ ] NPU service starts (if enabled)

### 9.6 Model Download
- [ ] TinyLlama-1.1B downloads successfully
- [ ] Model loads into tinygrad backend
- [ ] Model ready for inference
- [ ] No download errors

### 9.7 Inference
- [ ] Test inference completes successfully
- [ ] Tokens generated correctly
- [ ] Performance acceptable (>5 tokens/sec)
- [ ] GPU being used (not CPU fallback)
- [ ] Metrics show GPU activity

## Task 10: Git Environment and Multi-Node Deployment

### 10.1 Git Repository
- [ ] All changes committed
- [ ] flake.nix complete and tested
- [ ] Release version tagged
- [ ] Pushed to remote repository
- [ ] Git URL documented for deployment

### 10.2 gremlin-1 Git Flake
- [ ] Local drive mapping removed
- [ ] Configuration updated to use git flake
- [ ] Rebuild successful
- [ ] Functionality unchanged from local test
- [ ] Git flake URL documented

### 10.3 Deploy to gremlin-2, 3, 4

#### gremlin-2 (10.1.1.13)
- [ ] Configuration updated
- [ ] Flake imported from git
- [ ] Rebuild successful
- [ ] Service starts correctly
- [ ] Node accessible via API

#### gremlin-3 (10.1.1.14)
- [ ] Configuration updated
- [ ] Flake imported from git
- [ ] Rebuild successful
- [ ] Service starts correctly
- [ ] Node accessible via API

#### gremlin-4 (10.1.1.15)
- [ ] Configuration updated
- [ ] Flake imported from git
- [ ] Rebuild successful
- [ ] Service starts correctly
- [ ] Node accessible via API

### 10.4 Cluster Formation
- [ ] All nodes accessible
- [ ] Nodes discover each other
- [ ] Cluster forms correctly
- [ ] Node count correct in cluster status
- [ ] Dashboard shows all nodes
- [ ] Peer connections established

### 10.5 Cluster Stability

#### Initial Test (1 hour)
- [ ] All nodes remain connected
- [ ] No crashes or restarts
- [ ] Model sharding works
- [ ] Inference requests succeed
- [ ] Performance stable

#### Extended Test (4+ hours)
- [ ] All nodes remain connected
- [ ] No memory leaks
- [ ] No performance degradation
- [ ] Error rate < 1%
- [ ] Cluster remains stable

## Validation Tests

### Single Node Tests
- [ ] `./tests/test_gremlin_single_node.sh gremlin-1` passes
- [ ] `./tests/test_gremlin_single_node.sh gremlin-2` passes
- [ ] `./tests/test_gremlin_single_node.sh gremlin-3` passes
- [ ] `./tests/test_gremlin_single_node.sh gremlin-4` passes

### Cluster Tests
- [ ] `./tests/test_gremlin_cluster.sh 5` passes (5 min quick test)
- [ ] `./tests/test_gremlin_cluster.sh 60` passes (1 hour test)
- [ ] `./tests/test_gremlin_cluster.sh 240` passes (4 hour test)

## Performance Validation

### Single Node Performance
- [ ] TinyLlama-1.1B: >10 tokens/sec on GPU
- [ ] GPU utilization >50% during inference
- [ ] Memory usage stable
- [ ] No memory leaks over 1 hour

### Cluster Performance
- [ ] Model sharding works across nodes
- [ ] Inference latency acceptable
- [ ] Load balancing works
- [ ] No bottlenecks identified

## Issues and Resolutions

### gremlin-1
- Issue: 
- Resolution: 
- Status: 

### gremlin-2
- Issue: 
- Resolution: 
- Status: 

### gremlin-3
- Issue: 
- Resolution: 
- Status: 

### gremlin-4
- Issue: 
- Resolution: 
- Status: 

### Cluster
- Issue: 
- Resolution: 
- Status: 

## Sign-Off

### Single Node (gremlin-1)
- [ ] All tests passed
- [ ] Performance acceptable
- [ ] No critical issues
- [ ] Ready for cluster deployment
- Signed: _________________ Date: _________

### Cluster Deployment
- [ ] All nodes deployed
- [ ] Cluster formed successfully
- [ ] Stability tests passed
- [ ] Performance acceptable
- [ ] Ready for production use
- Signed: _________________ Date: _________

## Notes

### Deployment Date
- Started: _________________
- Completed: _________________

### Configuration
- Git Repository: _________________
- Branch/Tag: _________________
- Commit Hash: _________________

### Performance Metrics
- Single Node Tokens/Sec: _________________
- Cluster Tokens/Sec: _________________
- GPU Utilization: _________________
- Memory Usage: _________________

### Known Issues
1. 
2. 
3. 

### Future Improvements
1. 
2. 
3. 

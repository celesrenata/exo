# Large Model Testing for 4-Node Cluster

## Cluster Specifications
- **4 nodes** with **96GB RAM each** = **384GB total**
- **CPU inference** (torch engine)
- **Distributed sharding** across nodes

## Added Large Models for Testing

### Small Multi-Node Models (16-64GB)
Perfect for testing 2-3 node sharding:

1. **Llama 3.1 8B** (16GB)
   - CPU: `llama-3.1-8b-cpu` 
   - CUDA: `llama-3.1-8b-cuda`
   - Expected: 1-2 nodes

2. **Qwen2.5 32B** (64GB) 
   - CPU: `qwen2.5-32b-cpu`
   - CUDA: `qwen2.5-32b-cuda`
   - Expected: 2-3 nodes

### Large Multi-Node Models (90-140GB)
Perfect for testing 3-4 node sharding:

3. **Mixtral 8x7B** (90GB)
   - CPU: `mixtral-8x7b-cpu`
   - CUDA: `mixtral-8x7b-cuda` 
   - Expected: 3-4 nodes
   - **MoE architecture** - great for testing

4. **Llama 3.1 70B** (140GB)
   - CPU: `llama-3.1-70b-cpu`
   - CUDA: `llama-3.1-70b-cuda`
   - Expected: 4 nodes (full cluster)

## Testing Strategy

### Phase 1: Single Node Validation
- Start with `llama-3.1-8b-cpu` on 1 node
- Verify download, load, and chat functionality

### Phase 2: Multi-Node Sharding
- Test `qwen2.5-32b-cpu` across 2-3 nodes
- Verify sharding distribution in dashboard
- Test chat performance across nodes

### Phase 3: Full Cluster Utilization  
- Test `llama-3.1-70b-cpu` across all 4 nodes
- Verify optimal resource utilization
- Benchmark inference performance

### Phase 4: MoE Testing
- Test `mixtral-8x7b-cpu` for Mixture of Experts
- Validate expert routing across nodes

## Validation Points

### Sharding Verification
- Check `/state` endpoint for instance distribution
- Verify `nodeToRunner` mapping spans multiple nodes
- Confirm memory usage per node

### Performance Testing
- Measure tokens per second across cluster
- Test concurrent chat sessions
- Monitor memory and CPU usage

### Failure Recovery
- Test node disconnection/reconnection
- Verify graceful degradation
- Test cluster reformation

## Expected Behavior

### Memory Distribution
- **16GB model**: Single node (plenty of headroom)
- **64GB model**: 2-3 nodes (comfortable fit)
- **90GB model**: 3-4 nodes (good utilization)
- **140GB model**: All 4 nodes (near capacity)

### Sharding Strategy
- **Pipeline sharding**: Layers distributed across nodes
- **Tensor sharding**: Model weights split across nodes
- **Automatic selection**: Based on model size and topology

## Commands for Testing

```bash
# Check available models
curl -s http://localhost:52415/v1/models | jq '.data[] | select(.tags[] | contains("cpu")) | {id, name, storage_size_megabytes}'

# Create large model instance
curl -X POST http://localhost:52415/place_instance \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llama-3.1-70b-cpu",
    "sharding": "Pipeline", 
    "instance_meta": "CpuRing",
    "min_nodes": 4
  }'

# Check sharding distribution
curl -s http://localhost:52415/state | jq '.instances | .[] | .CpuRingInstance.shardAssignments.nodeToRunner'

# Test chat with large model
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "Explain distributed inference"}],
    "stream": false
  }'
```

## Success Criteria

✅ **Multi-node sharding works correctly**
✅ **Large models load and run across cluster** 
✅ **Chat performance is acceptable**
✅ **Resource utilization is optimal**
✅ **System handles node failures gracefully**

This setup will thoroughly validate EXO's distributed inference capabilities!
# Quick Model Reference for Intel Arc

## Recommended Models for Tinygrad on Intel Hardware

### 🚀 Quick Start (Testing)
```
Qwen/Qwen2.5-0.5B-Instruct
```
- Size: 494 MB
- VRAM: ~1.5 GB
- Best for: Quick testing, development

### ⚡ Best for 8GB VRAM (Arc A750)
```
meta-llama/Llama-3.2-3B-Instruct
```
- Size: 3.2 GB
- VRAM: ~8.5 GB
- Best for: Production on 8GB cards

### 🎯 Best for 16GB VRAM (Arc A770)
```
Qwen/Qwen2.5-7B-Instruct
meta-llama/Meta-Llama-3.1-8B-Instruct
```
- Size: 7-8 GB
- VRAM: ~15-16 GB
- Best for: High-quality production inference

### 🌟 Multi-Node Distributed
```
meta-llama/Meta-Llama-3.1-70B-Instruct
```
- Size: 70 GB
- VRAM: ~140 GB (distributed)
- Best for: Cluster deployments

## All Available Models

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| Qwen/Qwen2.5-0.5B-Instruct | 494 MB | 1.5 GB | Testing, dev |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5 GB | 4 GB | Light production |
| meta-llama/Llama-3.2-3B-Instruct | 3.2 GB | 8.5 GB | 8GB cards |
| Qwen/Qwen2.5-7B-Instruct | 7.6 GB | 15 GB | 16GB cards |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 8 GB | 16 GB | 16GB cards |
| mistralai/Mistral-7B-Instruct-v0.3 | 7 GB | 14 GB | Alternative 7B |
| meta-llama/Meta-Llama-3.1-70B-Instruct | 70 GB | 140 GB | Distributed |

## Quick Commands

### List available models
```bash
curl http://localhost:52415/v1/models | jq '.data[] | {id, storage_size_megabytes}'
```

### Add a model
```bash
curl -X POST http://localhost:52415/models/add \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

### Create instance
```bash
curl -X POST http://localhost:52415/place_instance \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1
  }'
```

### Test inference
```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Important Notes

⚠️ **MLX models won't work with tinygrad!**
- Avoid models with `mlx-community/` prefix
- Use standard HuggingFace model IDs

✅ **Tinygrad backend must be enabled:**
```bash
export EXO_TINYGRAD_ENABLED=true
export TINYGRAD_BACKEND=GPU  # For Intel Arc
```

📊 **VRAM estimates are for FP16 inference**
- Actual usage may vary with context length
- Quantization can reduce requirements

🔧 **For best performance on Intel Arc:**
- Use `TINYGRAD_BACKEND=GPU`
- Enable OpenCL or Level Zero runtime
- Monitor with `intel_gpu_top`

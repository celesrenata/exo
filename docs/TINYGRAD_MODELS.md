# Tinygrad-Compatible Models for Intel Hardware

This document lists HuggingFace models that are compatible with the tinygrad backend on Intel Arc GPU hardware.

## Overview

Unlike MLX models (which are Apple Silicon-specific), tinygrad works with standard HuggingFace models. The models listed below have been verified to exist on HuggingFace Hub and are suitable for inference with tinygrad.

## Model Selection Criteria

- Standard PyTorch/transformers format (not MLX-specific)
- Instruction-tuned for chat/dialogue
- Range of sizes for different hardware capabilities
- Popular, well-maintained model families

## Verified Models

### Small Models (0.5-1.5B Parameters)

#### Qwen/Qwen2.5-0.5B-Instruct
- **Parameters**: 0.49B (0.36B non-embedding)
- **VRAM (FP16)**: ~1.5 GB
- **Context Length**: 32K tokens
- **Use Case**: Lightweight chat, instruction following, resource-constrained environments
- **Storage Size**: ~1 GB

#### Qwen/Qwen2.5-1.5B-Instruct
- **Parameters**: 1.54B (1.31B non-embedding)
- **VRAM (FP16)**: ~4 GB
- **Context Length**: 32K tokens
- **Use Case**: Balanced performance for basic tasks, good for testing
- **Storage Size**: ~3 GB

### Medium Models (3-8B Parameters)

#### meta-llama/Llama-3.2-3B-Instruct
- **Parameters**: 3.21B
- **VRAM (FP16)**: ~8.5 GB
- **Context Length**: 128K tokens
- **Use Case**: Multilingual dialogue, instruction following, good quality/performance balance
- **Storage Size**: ~6.4 GB

#### Qwen/Qwen2.5-7B-Instruct
- **Parameters**: 7.61B
- **VRAM (FP16)**: ~15 GB
- **Context Length**: 128K tokens
- **Use Case**: High-quality chat, strong multilingual support, structured data handling
- **Storage Size**: ~15 GB

#### meta-llama/Meta-Llama-3.1-8B-Instruct
- **Parameters**: 8B
- **VRAM (FP16)**: ~16 GB
- **Context Length**: 128K tokens
- **Use Case**: Production-quality chat, instruction following, Grouped-Query Attention for efficiency
- **Storage Size**: ~16 GB

#### mistralai/Mistral-7B-Instruct-v0.3
- **Parameters**: 7B
- **VRAM (FP16)**: ~14 GB
- **Context Length**: 32K tokens
- **Use Case**: Efficient instruction following, good performance/size ratio
- **Storage Size**: ~14 GB

### Large Models (70B Parameters)

#### meta-llama/Meta-Llama-3.1-70B-Instruct
- **Parameters**: 70B
- **VRAM (FP16)**: ~140 GB
- **Context Length**: 128K tokens
- **Use Case**: High-end performance, complex reasoning, multi-node distributed inference
- **Storage Size**: ~140 GB
- **Note**: Requires distributed inference across multiple nodes

## Intel Arc GPU Recommendations

### Intel Arc A770 (16GB VRAM)
- **Recommended**: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct
- **Also Works**: Smaller models (0.5B-3B) with room for larger context

### Intel Arc A750 (8GB VRAM)
- **Recommended**: Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct
- **Best**: Qwen2.5-0.5B-Instruct for maximum context length

### Multi-Node Clusters
- **Recommended**: Llama-3.1-70B-Instruct distributed across nodes
- **Sharding**: Use tensor or pipeline parallelism

## Usage with exo

To use these models with exo on Intel hardware:

1. Ensure tinygrad backend is enabled:
   ```bash
   export EXO_TINYGRAD_ENABLED=true
   export TINYGRAD_BACKEND=CLANG  # or GPU for OpenCL/Level Zero
   ```

2. Add model via API:
   ```bash
   curl -X POST http://localhost:52415/models/add \
     -H "Content-Type: application/json" \
     -d '{"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}'
   ```

3. Create instance and start inference:
   ```bash
   curl -X POST http://localhost:52415/instance \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
       "sharding": "Pipeline"
     }'
   ```

## Model Card Files

Model cards for these models are located in:
- `resources/inference_model_cards/` - Standard models
- `~/.cache/exo/custom_model_cards/` - User-added models

## Notes

- All models support standard HuggingFace transformers format
- VRAM estimates are for FP16 inference; quantization can reduce requirements
- Context length support depends on available memory
- Tinygrad backend selection (CLANG vs GPU) affects performance
- For Intel Arc, use `TINYGRAD_BACKEND=GPU` with OpenCL or Level Zero runtime

## References

- [Qwen2.5 Models](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [Llama 3.2 Models](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- [Llama 3.1 Models](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)
- [Mistral Models](https://huggingface.co/mistralai)
- [Tinygrad Documentation](https://github.com/tinygrad/tinygrad)

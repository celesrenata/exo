# Requirements Document

## Introduction

This specification defines the requirements for implementing a complete Llama transformer architecture in tinygrad to enable proper text generation on Intel Arc GPUs. Currently, the tinygrad backend loads model weights successfully but uses a placeholder that generates random output. This feature will replace the placeholder with a functional transformer implementation.

## Glossary

- **Tinygrad Backend**: The inference engine using tinygrad for GPU acceleration
- **Llama Architecture**: The transformer model architecture used by Meta's Llama models
- **Transformer Layer**: A neural network layer containing self-attention and feed-forward components
- **KV Cache**: Key-Value cache for efficient autoregressive generation
- **Safetensors**: Binary format for storing model weights
- **Intel Arc GPU**: Intel's discrete GPU hardware
- **bfloat16**: 16-bit brain floating point format used by modern LLMs

## Requirements

### Requirement 1: Llama Model Architecture Implementation

**User Story:** As a developer, I want the tinygrad backend to implement the Llama transformer architecture, so that loaded models can perform actual inference instead of generating random output.

#### Acceptance Criteria

1. WHEN the System loads a Llama model, THE Tinygrad Backend SHALL create transformer layers with correct dimensions based on model configuration
2. WHEN the System processes input tokens, THE Transformer SHALL apply multi-head self-attention with correct head dimensions
3. WHEN the System computes attention, THE Transformer SHALL use rotary position embeddings (RoPE) for position encoding
4. WHEN the System processes attention output, THE Transformer SHALL apply feed-forward network with SwiGLU activation
5. WHERE the model uses grouped-query attention, THE Transformer SHALL correctly handle different numbers of query and key-value heads

### Requirement 2: Weight Loading and Initialization

**User Story:** As a developer, I want the transformer to load actual model weights from safetensors files, so that the model uses trained parameters instead of random initialization.

#### Acceptance Criteria

1. WHEN the System loads model weights, THE Weight Loader SHALL map HuggingFace weight names to tinygrad model parameters
2. WHEN the System encounters bfloat16 weights, THE Weight Loader SHALL convert them to float32 format
3. WHEN the System loads sharded models, THE Weight Loader SHALL correctly combine weights from multiple safetensors files
4. WHEN the System applies weights, THE Transformer SHALL verify weight shapes match expected layer dimensions
5. WHERE weights are missing for optional components, THE Transformer SHALL initialize them with appropriate defaults

### Requirement 3: Efficient KV Cache Implementation

**User Story:** As a user, I want text generation to be fast and memory-efficient, so that I can generate long sequences without performance degradation.

#### Acceptance Criteria

1. WHEN the System generates the first token, THE KV Cache SHALL store key and value tensors for all layers
2. WHEN the System generates subsequent tokens, THE Transformer SHALL reuse cached keys and values from previous tokens
3. WHEN the System updates the cache, THE KV Cache SHALL append new key-value pairs without recomputing previous tokens
4. WHEN the System reaches maximum sequence length, THE KV Cache SHALL handle cache eviction or extension appropriately
5. WHERE multiple requests are active, THE KV Cache SHALL maintain separate cache state per request ID

### Requirement 4: Token Generation Pipeline

**User Story:** As a user, I want to send prompts and receive coherent text responses, so that I can use the model for text generation tasks.

#### Acceptance Criteria

1. WHEN the System receives a text prompt, THE Generator SHALL encode it to token IDs using the tokenizer
2. WHEN the System processes tokens, THE Transformer SHALL produce logits for next token prediction
3. WHEN the System samples from logits, THE Generator SHALL apply temperature and top-p sampling parameters
4. WHEN the System generates tokens, THE Generator SHALL decode token IDs back to text incrementally
5. WHERE the model generates an EOS token, THE Generator SHALL stop generation and return the complete response

### Requirement 5: Model Configuration Support

**User Story:** As a developer, I want the transformer to support different Llama model sizes, so that I can run 1B, 3B, 8B, and larger models.

#### Acceptance Criteria

1. WHEN the System loads a model, THE Configuration Parser SHALL extract model dimensions from config.json
2. WHEN the System creates layers, THE Transformer SHALL use configuration values for hidden size, number of layers, and attention heads
3. WHEN the System handles different model sizes, THE Transformer SHALL correctly scale memory allocation and computation
4. WHERE the model uses non-standard configurations, THE Configuration Parser SHALL support custom parameter values
5. WHEN the System encounters unsupported configurations, THE Transformer SHALL provide clear error messages

### Requirement 6: Integration with Existing Backend

**User Story:** As a developer, I want the transformer implementation to integrate seamlessly with the existing tinygrad backend, so that no changes are needed to the runner or API layers.

#### Acceptance Criteria

1. WHEN the System loads a checkpoint, THE Model Loader SHALL return a transformer instance compatible with existing interfaces
2. WHEN the System calls infer_tensor, THE Transformer SHALL accept numpy arrays and return numpy arrays
3. WHEN the System manages inference state, THE Transformer SHALL use the existing state dictionary format
4. WHEN the System collects metrics, THE Transformer SHALL work with existing metrics collection infrastructure
5. WHERE the backend switches devices, THE Transformer SHALL support CPU, GPU, and METAL execution

### Requirement 7: Correctness Validation

**User Story:** As a developer, I want to verify that the transformer produces correct output, so that I can trust the implementation matches reference implementations.

#### Acceptance Criteria

1. WHEN the System generates text from a known prompt, THE Transformer SHALL produce output similar to reference implementations
2. WHEN the System processes the same input twice, THE Transformer SHALL produce identical output (with fixed random seed)
3. WHEN the System loads weights, THE Weight Loader SHALL verify checksums or validate weight integrity
4. WHERE numerical precision differs, THE Transformer SHALL maintain output quality within acceptable tolerance
5. WHEN the System encounters errors, THE Transformer SHALL provide detailed error messages for debugging

### Requirement 8: Performance Optimization

**User Story:** As a user, I want text generation to be fast, so that I can have responsive interactions with the model.

#### Acceptance Criteria

1. WHEN the System generates tokens, THE Transformer SHALL achieve at least 10 tokens per second on Intel Arc GPU
2. WHEN the System uses KV cache, THE Generator SHALL avoid recomputing attention for previous tokens
3. WHEN the System performs matrix operations, THE Transformer SHALL utilize GPU acceleration effectively
4. WHERE memory is limited, THE Transformer SHALL support gradient checkpointing or memory-efficient attention
5. WHEN the System profiles performance, THE Metrics Collector SHALL track tokens per second and memory usage

## Out of Scope

The following items are explicitly out of scope for this specification:

- Support for non-Llama architectures (GPT, BERT, etc.)
- Quantization (4-bit, 8-bit) support
- Fine-tuning or training capabilities
- Multi-GPU distributed inference
- Speculative decoding or other advanced generation techniques
- Vision or multimodal model support

## Success Criteria

The implementation will be considered successful when:

1. A Llama-3.2-3B model can generate coherent English text responses
2. Generation speed exceeds 10 tokens/second on Intel Arc GPU
3. Output quality is comparable to reference implementations (MLX, transformers)
4. The implementation passes all unit tests and integration tests
5. Memory usage is within expected bounds for the model size

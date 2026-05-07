# Design Document

## Overview

This document describes the design for implementing a complete Llama transformer architecture in tinygrad. The implementation will replace the current placeholder model with a functional transformer that can perform actual text generation using loaded model weights.

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Tinygrad Backend                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Model Loader (existing)                    │ │
│  │  - Load safetensors weights                            │ │
│  │  - Convert bfloat16 to float32                         │ │
│  │  - Map HuggingFace weight names                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Llama Transformer (NEW)                       │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Embedding Layer                                  │  │ │
│  │  │  - Token embeddings                               │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Transformer Layers (N layers)                    │  │ │
│  │  │  ┌────────────────────────────────────────────┐  │  │ │
│  │  │  │  RMSNorm                                    │  │  │ │
│  │  │  │  Multi-Head Attention with RoPE             │  │  │ │
│  │  │  │  Residual Connection                        │  │  │ │
│  │  │  │  RMSNorm                                    │  │  │ │
│  │  │  │  Feed-Forward (SwiGLU)                      │  │  │ │
│  │  │  │  Residual Connection                        │  │  │ │
│  │  │  └────────────────────────────────────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Output Layer                                     │  │ │
│  │  │  - RMSNorm                                        │  │ │
│  │  │  - LM Head (logits projection)                   │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              KV Cache Manager (NEW)                     │ │
│  │  - Store key/value tensors per layer                   │ │
│  │  - Manage cache per request ID                         │ │
│  │  - Handle cache updates and eviction                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Generator (existing)                       │ │
│  │  - Token sampling                                       │ │
│  │  - Temperature/top-p                                    │ │
│  │  - Streaming responses                                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Llama Transformer Module

**File**: `src/exo/worker/engines/tinygrad/llama_transformer.py`

```python
class LlamaTransformer:
    """Llama transformer implementation in tinygrad."""
    
    def __init__(self, config: LlamaConfig):
        """Initialize transformer with configuration.
        
        Args:
            config: Model configuration (dimensions, layers, etc.)
        """
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerLayer(config) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self, 
        input_ids: Tensor, 
        cache: Optional[KVCache] = None,
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, KVCache]:
        """Forward pass through transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            cache: Optional KV cache from previous forward passes
            position_ids: Optional position IDs for RoPE
            
        Returns:
            Tuple of (logits, updated_cache)
            - logits: [batch_size, seq_len, vocab_size]
            - updated_cache: Updated KV cache for next forward pass
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize or update cache
        if cache is None:
            cache = KVCache(self.config.num_layers)
            
        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, layer_cache = layer(
                hidden_states,
                cache=cache.get_layer_cache(layer_idx),
                position_ids=position_ids
            )
            cache.update_layer_cache(layer_idx, layer_cache)
            
        # Final norm and projection to vocabulary
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, cache
```

### 2. Transformer Layer

**File**: `src/exo/worker/engines/tinygrad/llama_transformer.py`

```python
class TransformerLayer:
    """Single Llama transformer layer."""
    
    def __init__(self, config: LlamaConfig):
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        
    def forward(
        self,
        hidden_states: Tensor,
        cache: Optional[LayerCache] = None,
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, LayerCache]:
        """Forward pass through layer with residual connections."""
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states, 
            cache=cache,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_cache
```

### 3. Multi-Head Attention with RoPE

**File**: `src/exo/worker/engines/tinygrad/llama_transformer.py`

```python
class Attention:
    """Multi-head attention with rotary position embeddings."""
    
    def __init__(self, config: LlamaConfig):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        
        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)
        
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)
        
    def forward(
        self,
        hidden_states: Tensor,
        cache: Optional[LayerCache] = None,
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, LayerCache]:
        """Compute multi-head attention with RoPE and KV cache."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        query, key = self.rope(query, key, position_ids)
        
        # Update KV cache
        if cache is not None:
            key = cache.update_key(key)
            value = cache.update_value(value)
            
        # Grouped-query attention (repeat KV heads if needed)
        if self.num_heads != self.num_kv_heads:
            key = repeat_kv(key, self.num_heads // self.num_kv_heads)
            value = repeat_kv(value, self.num_heads // self.num_kv_heads)
            
        # Compute attention scores
        attn_output = scaled_dot_product_attention(query, key, value)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output, cache
```

### 4. Feed-Forward Network (SwiGLU)

**File**: `src/exo/worker/engines/tinygrad/llama_transformer.py`

```python
class MLP:
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: LlamaConfig):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply SwiGLU: down(silu(gate(x)) * up(x))"""
        gate = self.gate_proj(hidden_states)
        gate = silu(gate)  # SiLU activation
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)
```

### 5. RoPE (Rotary Position Embeddings)

**File**: `src/exo/worker/engines/tinygrad/llama_transformer.py`

```python
class RotaryEmbedding:
    """Rotary position embeddings."""
    
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta
        self._cos_cached = None
        self._sin_cached = None
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key."""
        seq_len = query.shape[1]
        
        # Compute or retrieve cached cos/sin
        cos, sin = self._get_cos_sin(seq_len, position_ids)
        
        # Apply rotation
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key
```

### 6. KV Cache Manager

**File**: `src/exo/worker/engines/tinygrad/kv_cache.py`

```python
class KVCache:
    """Manages key-value cache for efficient generation."""
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.layer_caches = [LayerCache() for _ in range(num_layers)]
        self.position = 0
        
    def get_layer_cache(self, layer_idx: int) -> LayerCache:
        """Get cache for specific layer."""
        return self.layer_caches[layer_idx]
        
    def update_layer_cache(self, layer_idx: int, cache: LayerCache):
        """Update cache for specific layer."""
        self.layer_caches[layer_idx] = cache
        
    def increment_position(self):
        """Increment position counter."""
        self.position += 1

class LayerCache:
    """Cache for a single transformer layer."""
    
    def __init__(self):
        self.key_cache: Optional[Tensor] = None
        self.value_cache: Optional[Tensor] = None
        
    def update_key(self, new_key: Tensor) -> Tensor:
        """Append new keys to cache."""
        if self.key_cache is None:
            self.key_cache = new_key
        else:
            self.key_cache = Tensor.cat([self.key_cache, new_key], dim=1)
        return self.key_cache
        
    def update_value(self, new_value: Tensor) -> Tensor:
        """Append new values to cache."""
        if self.value_cache is None:
            self.value_cache = new_value
        else:
            self.value_cache = Tensor.cat([self.value_cache, new_value], dim=1)
        return self.value_cache
```

### 7. Weight Loading Integration

**File**: `src/exo/worker/engines/tinygrad/model_loader.py` (modifications)

```python
def _create_model_with_weights(
    shard_metadata: ShardMetadata,
    weights: dict[str, np.ndarray],
    device: str,
    model_size: str,
) -> Any:
    """Create Llama transformer and load weights."""
    # Parse model configuration
    config = _parse_model_config(shard_metadata, model_size)
    
    # Create transformer
    from exo.worker.engines.tinygrad.llama_transformer import LlamaTransformer
    model = LlamaTransformer(config)
    
    # Load weights into model
    _load_weights_into_model(model, weights, device)
    
    return model

def _load_weights_into_model(
    model: LlamaTransformer,
    weights: dict[str, np.ndarray],
    device: str
):
    """Map HuggingFace weights to tinygrad model parameters."""
    from tinygrad import Tensor
    
    weight_mapping = {
        'model.embed_tokens.weight': model.embed_tokens.weight,
        'model.norm.weight': model.norm.weight,
        'lm_head.weight': model.lm_head.weight,
        # Layer weights (repeated for each layer)
        # 'model.layers.{i}.input_layernorm.weight': ...
        # 'model.layers.{i}.self_attn.q_proj.weight': ...
        # etc.
    }
    
    for hf_name, np_weight in weights.items():
        # Convert to tinygrad tensor
        tensor = Tensor(np_weight, device=device)
        
        # Find corresponding model parameter and assign
        param = _find_model_parameter(model, hf_name)
        if param is not None:
            param.assign(tensor)
```

## Data Models

### LlamaConfig

```python
@dataclass
class LlamaConfig:
    """Configuration for Llama model."""
    vocab_size: int = 128256
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'LlamaConfig':
        """Create config from HuggingFace config.json."""
        return cls(
            vocab_size=config_dict.get('vocab_size', 128256),
            hidden_size=config_dict.get('hidden_size', 3072),
            # ... map all fields
        )
```

## Error Handling

1. **Weight Shape Mismatch**: Validate weight shapes match expected dimensions before loading
2. **Missing Weights**: Provide clear error messages listing missing required weights
3. **OOM Errors**: Catch out-of-memory errors and suggest smaller batch size or model
4. **Numerical Issues**: Add checks for NaN/Inf in forward pass outputs

## Testing Strategy

1. **Unit Tests**:
   - Test each component (RMSNorm, Attention, MLP) independently
   - Verify RoPE computation matches reference implementation
   - Test KV cache updates and retrieval

2. **Integration Tests**:
   - Load actual Llama-3.2-3B model weights
   - Generate text from known prompts
   - Compare output to reference implementation (transformers library)

3. **Performance Tests**:
   - Measure tokens/second on Intel Arc GPU
   - Profile memory usage during generation
   - Verify KV cache reduces computation time

## Performance Considerations

1. **Memory Optimization**:
   - Use KV cache to avoid recomputing attention
   - Consider flash attention for long sequences
   - Batch multiple requests when possible

2. **Computation Optimization**:
   - Fuse operations where possible (RMSNorm + Linear)
   - Use tinygrad's JIT compilation
   - Optimize matrix multiplication layouts

3. **GPU Utilization**:
   - Ensure operations run on GPU (not CPU fallback)
   - Monitor GPU memory usage
   - Profile kernel execution times

## Migration Path

1. **Phase 1**: Implement core transformer without KV cache (slower but simpler)
2. **Phase 2**: Add KV cache for efficient generation
3. **Phase 3**: Optimize performance (fused operations, better memory layout)
4. **Phase 4**: Add support for additional model sizes and variants

## References

- Llama 2 Paper: https://arxiv.org/abs/2307.09288
- RoPE Paper: https://arxiv.org/abs/2104.09864
- Tinygrad Documentation: https://github.com/tinygrad/tinygrad
- HuggingFace Transformers Llama Implementation
- exo-cuda Reference Implementation

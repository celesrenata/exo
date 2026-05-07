# Implementation Plan

- [x] 1. Set up transformer module structure
  - Create `llama_transformer.py` with base classes
  - Define `LlamaConfig` dataclass for model configuration
  - Implement configuration parser to read from HuggingFace config.json
  - _Requirements: 1.1, 5.1, 5.2_

- [x] 2. Implement core transformer components
  - [x] 2.1 Implement RMSNorm layer
    - Write RMSNorm class with forward method
    - Add epsilon parameter for numerical stability
    - _Requirements: 1.2, 1.4_
  
  - [x] 2.2 Implement embedding layer
    - Create token embedding layer
    - Handle vocabulary size and hidden dimensions
    - _Requirements: 1.1_
  
  - [x] 2.3 Implement linear projection layers
    - Create Linear layer wrapper for tinygrad
    - Support bias and no-bias variants
    - _Requirements: 1.1, 1.2_

- [x] 3. Implement rotary position embeddings (RoPE)
  - [x] 3.1 Create RotaryEmbedding class
    - Compute frequency bands based on theta parameter
    - Cache cos/sin values for efficiency
    - _Requirements: 1.3_
  
  - [x] 3.2 Implement rotation application
    - Write apply_rotary_emb function
    - Handle complex number rotation in real space
    - Support position_ids for custom positions
    - _Requirements: 1.3_

- [x] 4. Implement multi-head attention
  - [x] 4.1 Create Attention class structure
    - Initialize Q, K, V, O projection layers
    - Set up head dimensions and counts
    - _Requirements: 1.2, 1.5_
  
  - [x] 4.2 Implement attention computation
    - Reshape tensors for multi-head processing
    - Compute scaled dot-product attention
    - Apply attention mask if needed
    - _Requirements: 1.2_
  
  - [x] 4.3 Add grouped-query attention support
    - Implement repeat_kv function for KV head expansion
    - Handle different numbers of Q and KV heads
    - _Requirements: 1.5_
  
  - [x] 4.4 Integrate RoPE into attention
    - Apply rotary embeddings to Q and K
    - Pass position IDs through attention forward
    - _Requirements: 1.3_

- [x] 5. Implement feed-forward network (MLP)
  - [x] 5.1 Create MLP class with SwiGLU
    - Initialize gate, up, and down projection layers
    - _Requirements: 1.4_
  
  - [x] 5.2 Implement SwiGLU activation
    - Apply SiLU (Swish) activation to gate projection
    - Element-wise multiply gate and up projections
    - Project down to hidden size
    - _Requirements: 1.4_

- [x] 6. Implement transformer layer
  - [x] 6.1 Create TransformerLayer class
    - Initialize layer norm, attention, and MLP components
    - _Requirements: 1.1, 1.2, 1.4_
  
  - [x] 6.2 Implement layer forward pass
    - Apply pre-attention layer norm
    - Compute attention with residual connection
    - Apply pre-MLP layer norm
    - Compute MLP with residual connection
    - _Requirements: 1.2, 1.4_

- [x] 7. Implement KV cache system
  - [x] 7.1 Create KVCache and LayerCache classes
    - Define cache data structures
    - Implement cache initialization
    - _Requirements: 3.1, 3.5_
  
  - [x] 7.2 Implement cache update logic
    - Write methods to append new key-value pairs
    - Handle first token vs. subsequent tokens
    - _Requirements: 3.2, 3.3_
  
  - [x] 7.3 Integrate cache with attention
    - Modify attention forward to accept and return cache
    - Concatenate cached keys/values with new ones
    - _Requirements: 3.2, 3.3_
  
  - [x] 7.4 Add cache management per request
    - Track cache state by request ID
    - Implement cache eviction for completed requests
    - _Requirements: 3.4, 3.5_

- [x] 8. Implement complete LlamaTransformer
  - [x] 8.1 Create LlamaTransformer class
    - Initialize embedding, layers, norm, and lm_head
    - Set up configuration
    - _Requirements: 1.1, 5.2_
  
  - [x] 8.2 Implement transformer forward pass
    - Embed input tokens
    - Process through all transformer layers
    - Apply final layer norm
    - Project to vocabulary logits
    - _Requirements: 1.1, 1.2, 1.4, 4.2_
  
  - [x] 8.3 Add position ID generation
    - Compute position IDs based on cache state
    - Handle both prefill and generation phases
    - _Requirements: 1.3, 3.2_

- [x] 9. Implement weight loading
  - [x] 9.1 Create weight name mapping
    - Map HuggingFace weight names to model parameters
    - Handle layer indexing in weight names
    - _Requirements: 2.1, 2.4_
  
  - [x] 9.2 Implement weight assignment
    - Convert numpy arrays to tinygrad Tensors
    - Assign weights to model parameters
    - Verify weight shapes match expectations
    - _Requirements: 2.1, 2.2, 2.4_
  
  - [x] 9.3 Handle sharded model loading
    - Load weights from multiple safetensors files
    - Combine sharded weights correctly
    - _Requirements: 2.3_
  
  - [x] 9.4 Add weight validation
    - Check for missing required weights
    - Validate weight shapes and dtypes
    - Provide clear error messages
    - _Requirements: 2.4, 2.5, 7.3_

- [x] 10. Integrate with existing backend
  - [x] 10.1 Update model_loader.py
    - Replace PlaceholderModel with LlamaTransformer
    - Update _create_model_with_weights function
    - Parse model configuration from config.json
    - _Requirements: 5.1, 5.2, 6.1_
  
  - [x] 10.2 Ensure interface compatibility
    - Verify model accepts numpy arrays
    - Verify model returns numpy arrays
    - Test with existing generator.py
    - _Requirements: 6.2, 6.3_
  
  - [x] 10.3 Update inference state handling
    - Adapt KVCache to work with existing state dict format
    - Ensure state can be serialized/deserialized
    - _Requirements: 6.4_

- [x] 11. Add configuration support for different model sizes
  - [x] 11.1 Create configuration presets
    - Define configs for 1B, 3B, 8B, 70B models
    - _Requirements: 5.3_
  
  - [x] 11.2 Implement config auto-detection
    - Parse config.json from model directory
    - Extract all necessary parameters
    - Handle missing or custom parameters
    - _Requirements: 5.1, 5.2, 5.4_
  
  - [x] 11.3 Add config validation
    - Verify required parameters are present
    - Check parameter value ranges
    - Provide helpful error messages
    - _Requirements: 5.5_

- [-] 12. Testing and validation
  - [ ]* 12.1 Write unit tests for components
    - Test RMSNorm computation
    - Test RoPE application
    - Test attention mechanism
    - Test MLP forward pass
    - _Requirements: 7.1, 7.2_
  
  - [ ]* 12.2 Write integration tests
    - Test full forward pass with dummy weights
    - Test weight loading from actual model
    - Test generation with known prompts
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 12.3 Validate output correctness
    - Compare output to HuggingFace transformers
    - Test with multiple prompts
    - Verify deterministic output with fixed seed
    - _Requirements: 7.1, 7.2, 7.4_
  
  - [x] 12.4 Test different model sizes
    - Verify 1B model loads and generates
    - Verify 3B model loads and generates
    - Test configuration parsing for each size
    - _Requirements: 5.3_

- [x] 13. Performance optimization
  - **Context**: See `#[[file:IMPLEMENTATION_CONTEXT.md]]` for testing workflow, deployment process, and integration points
  - [x] 13.1 Profile generation speed
    - Measure tokens per second
    - Identify bottlenecks
    - _Requirements: 8.1, 8.3_
  
  - [x] 13.2 Optimize memory usage
    - Monitor GPU memory during generation
    - Verify KV cache reduces computation
    - _Requirements: 8.2, 8.5_
  
  - [x] 13.3 Verify GPU utilization
    - Ensure operations run on GPU
    - Check for CPU fallbacks
    - Profile kernel execution
    - _Requirements: 8.3, 6.5_
  
  - [ ]* 13.4 Add performance benchmarks
    - Create benchmark script
    - Test various sequence lengths
    - Compare to baseline performance
    - _Requirements: 8.1, 8.5_

- [ ] 14. Documentation and deployment
  - **Context**: See `#[[file:IMPLEMENTATION_CONTEXT.md]]` for architecture overview, common patterns, and deployment workflow
  - [ ] 14.1 Document transformer architecture
    - Add docstrings to all classes and methods
    - Create architecture diagram
    - _Requirements: 6.1_
  
  - [ ] 14.2 Create usage examples
    - Write example for loading and generating
    - Document configuration options
    - _Requirements: 4.1, 4.4_
  
  - [ ] 14.3 Update deployment guide
    - Add transformer to gremlin-1 deployment
    - Test end-to-end on Intel Arc GPU
    - _Requirements: 6.5_
  
  - [ ] 14.4 Verify production readiness
    - Test with real user prompts
    - Verify error handling
    - Check memory leaks
    - _Requirements: 7.3, 7.5_

# Implementation Plan

- [ ] 1. Set up transformer module structure
  - Create `llama_transformer.py` with base classes
  - Define `LlamaConfig` dataclass for model configuration
  - Implement configuration parser to read from HuggingFace config.json
  - _Requirements: 1.1, 5.1, 5.2_

- [ ] 2. Implement core transformer components
  - [ ] 2.1 Implement RMSNorm layer
    - Write RMSNorm class with forward method
    - Add epsilon parameter for numerical stability
    - _Requirements: 1.2, 1.4_
  
  - [ ] 2.2 Implement embedding layer
    - Create token embedding layer
    - Handle vocabulary size and hidden dimensions
    - _Requirements: 1.1_
  
  - [ ] 2.3 Implement linear projection layers
    - Create Linear layer wrapper for tinygrad
    - Support bias and no-bias variants
    - _Requirements: 1.1, 1.2_

- [ ] 3. Implement rotary position embeddings (RoPE)
  - [ ] 3.1 Create RotaryEmbedding class
    - Compute frequency bands based on theta parameter
    - Cache cos/sin values for efficiency
    - _Requirements: 1.3_
  
  - [ ] 3.2 Implement rotation application
    - Write apply_rotary_emb function
    - Handle complex number rotation in real space
    - Support position_ids for custom positions
    - _Requirements: 1.3_

- [ ] 4. Implement multi-head attention
  - [ ] 4.1 Create Attention class structure
    - Initialize Q, K, V, O projection layers
    - Set up head dimensions and counts
    - _Requirements: 1.2, 1.5_
  
  - [ ] 4.2 Implement attention computation
    - Reshape tensors for multi-head processing
    - Compute scaled dot-product attention
    - Apply attention mask if needed
    - _Requirements: 1.2_
  
  - [ ] 4.3 Add grouped-query attention support
    - Implement repeat_kv function for KV head expansion
    - Handle different numbers of Q and KV heads
    - _Requirements: 1.5_
  
  - [ ] 4.4 Integrate RoPE into attention
    - Apply rotary embeddings to Q and K
    - Pass position IDs through attention forward
    - _Requirements: 1.3_

- [ ] 5. Implement feed-forward network (MLP)
  - [ ] 5.1 Create MLP class with SwiGLU
    - Initialize gate, up, and down projection layers
    - _Requirements: 1.4_
  
  - [ ] 5.2 Implement SwiGLU activation
    - Apply SiLU (Swish) activation to gate projection
    - Element-wise multiply gate and up projections
    - Project down to hidden size
    - _Requirements: 1.4_

- [ ] 6. Implement transformer layer
  - [ ] 6.1 Create TransformerLayer class
    - Initialize layer norm, attention, and MLP components
    - _Requirements: 1.1, 1.2, 1.4_
  
  - [ ] 6.2 Implement layer forward pass
    - Apply pre-attention layer norm
    - Compute attention with residual connection
    - Apply pre-MLP layer norm
    - Compute MLP with residual connection
    - _Requirements: 1.2, 1.4_

- [ ] 7. Implement KV cache system
  - [ ] 7.1 Create KVCache and LayerCache classes
    - Define cache data structures
    - Implement cache initialization
    - _Requirements: 3.1, 3.5_
  
  - [ ] 7.2 Implement cache update logic
    - Write methods to append new key-value pairs
    - Handle first token vs. subsequent tokens
    - _Requirements: 3.2, 3.3_
  
  - [ ] 7.3 Integrate cache with attention
    - Modify attention forward to accept and return cache
    - Concatenate cached keys/values with new ones
    - _Requirements: 3.2, 3.3_
  
  - [ ] 7.4 Add cache management per request
    - Track cache state by request ID
    - Implement cache eviction for completed requests
    - _Requirements: 3.4, 3.5_

- [ ] 8. Implement complete LlamaTransformer
  - [ ] 8.1 Create LlamaTransformer class
    - Initialize embedding, layers, norm, and lm_head
    - Set up configuration
    - _Requirements: 1.1, 5.2_
  
  - [ ] 8.2 Implement transformer forward pass
    - Embed input tokens
    - Process through all transformer layers
    - Apply final layer norm
    - Project to vocabulary logits
    - _Requirements: 1.1, 1.2, 1.4, 4.2_
  
  - [ ] 8.3 Add position ID generation
    - Compute position IDs based on cache state
    - Handle both prefill and generation phases
    - _Requirements: 1.3, 3.2_

- [ ] 9. Implement weight loading
  - [ ] 9.1 Create weight name mapping
    - Map HuggingFace weight names to model parameters
    - Handle layer indexing in weight names
    - _Requirements: 2.1, 2.4_
  
  - [ ] 9.2 Implement weight assignment
    - Convert numpy arrays to tinygrad Tensors
    - Assign weights to model parameters
    - Verify weight shapes match expectations
    - _Requirements: 2.1, 2.2, 2.4_
  
  - [ ] 9.3 Handle sharded model loading
    - Load weights from multiple safetensors files
    - Combine sharded weights correctly
    - _Requirements: 2.3_
  
  - [ ] 9.4 Add weight validation
    - Check for missing required weights
    - Validate weight shapes and dtypes
    - Provide clear error messages
    - _Requirements: 2.4, 2.5, 7.3_

- [ ] 10. Integrate with existing backend
  - [ ] 10.1 Update model_loader.py
    - Replace PlaceholderModel with LlamaTransformer
    - Update _create_model_with_weights function
    - Parse model configuration from config.json
    - _Requirements: 5.1, 5.2, 6.1_
  
  - [ ] 10.2 Ensure interface compatibility
    - Verify model accepts numpy arrays
    - Verify model returns numpy arrays
    - Test with existing generator.py
    - _Requirements: 6.2, 6.3_
  
  - [ ] 10.3 Update inference state handling
    - Adapt KVCache to work with existing state dict format
    - Ensure state can be serialized/deserialized
    - _Requirements: 6.4_

- [ ] 11. Add configuration support for different model sizes
  - [ ] 11.1 Create configuration presets
    - Define configs for 1B, 3B, 8B, 70B models
    - _Requirements: 5.3_
  
  - [ ] 11.2 Implement config auto-detection
    - Parse config.json from model directory
    - Extract all necessary parameters
    - Handle missing or custom parameters
    - _Requirements: 5.1, 5.2, 5.4_
  
  - [ ] 11.3 Add config validation
    - Verify required parameters are present
    - Check parameter value ranges
    - Provide helpful error messages
    - _Requirements: 5.5_

- [ ] 12. Testing and validation
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
  
  - [ ] 12.3 Validate output correctness
    - Compare output to HuggingFace transformers
    - Test with multiple prompts
    - Verify deterministic output with fixed seed
    - _Requirements: 7.1, 7.2, 7.4_
  
  - [ ] 12.4 Test different model sizes
    - Verify 1B model loads and generates
    - Verify 3B model loads and generates
    - Test configuration parsing for each size
    - _Requirements: 5.3_

- [ ] 13. Performance optimization
  - [ ] 13.1 Profile generation speed
    - Measure tokens per second
    - Identify bottlenecks
    - _Requirements: 8.1, 8.3_
  
  - [ ] 13.2 Optimize memory usage
    - Monitor GPU memory during generation
    - Verify KV cache reduces computation
    - _Requirements: 8.2, 8.5_
  
  - [ ] 13.3 Verify GPU utilization
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

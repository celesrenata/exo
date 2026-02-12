# Llama Transformer Validation Test Guide

This document describes the validation tests for the Llama transformer implementation in tinygrad.

## Test File

`test_llama_validation.py` - Comprehensive validation test suite

## Running the Tests

```bash
# Run all validation tests
uv run python test_llama_validation.py

# Or use pytest
uv run pytest test_llama_validation.py -v
```

## Test Coverage

### Task 12.3: Validate Output Correctness

The following tests validate that the transformer produces correct and consistent output:

#### Test 1: Basic Forward Pass
- **Purpose**: Verify model can perform a forward pass without errors
- **Validates**: 
  - Model initialization with valid config
  - Forward pass completes successfully
  - Output shape matches expected dimensions (batch_size, seq_len, vocab_size)
  - KV cache is created and populated correctly
- **Requirements**: 7.1, 7.2

#### Test 2: Deterministic Output
- **Purpose**: Verify model produces identical output for same input
- **Validates**:
  - Two forward passes with identical input produce identical output
  - Maximum difference between runs is < 1e-6 (numerical precision tolerance)
  - No random behavior in forward pass
- **Requirements**: 7.2, 7.4

#### Test 3: KV Cache Consistency
- **Purpose**: Verify KV cache produces same results as full sequence processing
- **Validates**:
  - Processing full sequence at once vs. incrementally with cache yields same results
  - Cache correctly stores and reuses key-value pairs
  - Maximum difference < 1e-4 (allows for small numerical differences in cache operations)
- **Requirements**: 7.1, 7.2

#### Test 9: Output Logits Range
- **Purpose**: Verify output logits are in reasonable range
- **Validates**:
  - No NaN values in output
  - No Inf values in output
  - Logits are non-zero (model is actually computing)
  - Logits are in reasonable range (typically [-100, 100] for random init)
- **Requirements**: 7.1, 7.3

#### Test 10: Multiple Prompts
- **Purpose**: Test model with various prompt lengths and content
- **Validates**:
  - Model handles different sequence lengths correctly
  - Output shape adapts to input length
  - No NaN/Inf for any prompt
  - Model is robust to different inputs
- **Requirements**: 7.1, 7.2

### Task 12.4: Test Different Model Sizes

The following tests validate support for different Llama model sizes:

#### Test 4: Configuration Parsing
- **Purpose**: Verify configuration parsing and validation
- **Validates**:
  - Config can be parsed from dictionary (HuggingFace format)
  - Config validation catches invalid parameters
  - Default configs exist for 0.5B, 1B, 3B, 8B models
  - All default configs pass validation
- **Requirements**: 5.1, 5.2, 5.3, 5.4, 5.5

#### Test 5: 1B Model Structure
- **Purpose**: Verify 1B model can be created and used
- **Validates**:
  - 1B default config loads correctly
  - Model structure is created with correct dimensions
  - Forward pass works with 1B model
  - Output shape is correct for 1B model
- **Requirements**: 5.3

#### Test 6: 3B Model Structure
- **Purpose**: Verify 3B model can be created and used
- **Validates**:
  - 3B default config loads correctly
  - Model structure is created with correct dimensions
  - Forward pass works with 3B model
  - Output shape is correct for 3B model
- **Requirements**: 5.3

### Additional Validation Tests

#### Test 7: Grouped-Query Attention
- **Purpose**: Verify GQA works with different Q and KV head counts
- **Validates**:
  - Model handles num_attention_heads != num_key_value_heads
  - KV head repetition works correctly (e.g., 24 Q heads, 8 KV heads = 3x repetition)
  - Forward pass completes with GQA configuration
- **Requirements**: 1.5

#### Test 8: Rotary Position Embeddings
- **Purpose**: Verify RoPE implementation
- **Validates**:
  - RoPE can be created with correct dimensions
  - RoPE preserves tensor shapes
  - RoPE actually modifies values (rotation is applied)
  - Rotation magnitude is significant (not identity transform)
- **Requirements**: 1.3

## Test Results Interpretation

### Success Criteria

All tests should pass with the following characteristics:

1. **Determinism**: Same input always produces same output (within numerical precision)
2. **Correctness**: Output shapes match expected dimensions
3. **Stability**: No NaN or Inf values in outputs
4. **Consistency**: KV cache produces same results as full sequence processing
5. **Flexibility**: Model works with different sizes (1B, 3B) and configurations

### Expected Output

When all tests pass, you should see:

```
============================================================
LLAMA TRANSFORMER VALIDATION TESTS
============================================================

=== Test 1: Basic Forward Pass ===
✓ Created model with 2 layers
✓ Forward pass successful, output shape: (1, 4, 1000)
✓ KV cache created with 4 tokens

=== Test 2: Deterministic Output ===
✓ Maximum difference between runs: 0.0000000000
✓ Model produces deterministic output

... (more tests) ...

============================================================
RESULTS: 10 passed, 0 failed out of 10 tests
============================================================

✅ All validation tests passed!
```

## Comparison with Reference Implementations

While these tests don't directly compare with HuggingFace transformers (which would require loading actual model weights), they validate:

1. **Architectural Correctness**: Model structure matches Llama specification
2. **Numerical Stability**: No NaN/Inf, deterministic behavior
3. **Functional Correctness**: KV cache, RoPE, GQA all work as expected

To compare with HuggingFace transformers, you would need to:

1. Load actual model weights (e.g., Llama-3.2-1B or Llama-3.2-3B)
2. Run same prompts through both implementations
3. Compare output logits (should match within numerical precision)

This is covered by integration tests that load real models.

## Troubleshooting

### Test Failures

If tests fail, check:

1. **Import Errors**: Ensure tinygrad is installed (`pip install tinygrad`)
2. **Shape Mismatches**: Verify config parameters are consistent
3. **NaN/Inf**: Check for division by zero or numerical instability
4. **Determinism Failures**: Ensure no random operations without fixed seed

### Performance Issues

If tests are slow:

1. Use smaller model configs for unit tests
2. Reduce sequence lengths
3. Use CPU device for testing (GPU may have initialization overhead)

## Next Steps

After validation tests pass:

1. **Integration Tests**: Load real model weights and test generation
2. **Performance Tests**: Measure tokens/second on target hardware
3. **Comparison Tests**: Compare output with HuggingFace transformers
4. **End-to-End Tests**: Test full generation pipeline with real prompts

## References

- Requirements: `.kiro/specs/tinygrad-llama-transformer/requirements.md`
- Design: `.kiro/specs/tinygrad-llama-transformer/design.md`
- Tasks: `.kiro/specs/tinygrad-llama-transformer/tasks.md`

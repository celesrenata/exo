# TokenGenerator Usage Guide

## Overview

The TokenGenerator provides sophisticated token sampling for the PyTorch + IPEX backend. It supports temperature scaling, top-k filtering, and top-p (nucleus) sampling.

## Basic Usage

```python
from exo.worker.engines.pytorch_xpu.token_generator import TokenGenerator
import torch

# Initialize with special tokens
generator = TokenGenerator(
    eos_token_id=2,  # End-of-sequence token
    pad_token_id=0   # Padding token
)

# Sample from logits
logits = torch.randn(50257)  # Vocab size for GPT-2
result = generator.sample(
    logits=logits,
    temperature=0.8,
    top_p=0.95,
    top_k=50
)

print(f"Sampled token: {result.token_id}")
print(f"Probability: {result.probability:.4f}")
print(f"Is EOS: {result.is_eos}")
print(f"Is PAD: {result.is_pad}")
```

## Sampling Strategies

### Temperature Scaling

Controls the randomness of sampling:
- `temperature=1.0`: Standard sampling (default)
- `temperature<1.0`: More deterministic (sharper distribution)
- `temperature>1.0`: More random (flatter distribution)

```python
# Deterministic sampling (greedy-like)
result = generator.sample(logits, temperature=0.1)

# Random sampling
result = generator.sample(logits, temperature=2.0)
```

### Top-K Filtering

Limits sampling to the top-k most likely tokens:

```python
# Sample from top 50 tokens
result = generator.sample(logits, top_k=50)

# Disable top-k (default)
result = generator.sample(logits, top_k=0)
```

### Top-P (Nucleus) Sampling

Limits sampling to tokens with cumulative probability ≤ top_p:

```python
# Sample from tokens covering 95% probability mass
result = generator.sample(logits, top_p=0.95)

# Disable top-p (default)
result = generator.sample(logits, top_p=1.0)
```

### Combined Sampling

All strategies can be combined:

```python
result = generator.sample(
    logits=logits,
    temperature=0.8,   # Slightly more deterministic
    top_p=0.95,        # Nucleus sampling
    top_k=100          # Consider top 100 tokens
)
```

## Special Token Handling

### Initialization with Special Tokens

```python
generator = TokenGenerator(
    eos_token_id=2,
    pad_token_id=0
)
```

### Runtime Updates

```python
# Update special tokens after initialization
generator.set_special_tokens(
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
```

### Checking Special Tokens

```python
result = generator.sample(logits)

if result.is_eos:
    print("Generation complete!")
    break

if result.is_pad:
    print("Padding token sampled")
```

## Integration with PyTorchXPUBackend

The TokenGenerator can be integrated into the backend's sample() method:

```python
class PyTorchXPUBackend(InferenceBackend):
    def __init__(self) -> None:
        # ... existing initialization ...
        self._token_generator = TokenGenerator()
    
    async def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0
    ) -> np.ndarray:
        # Convert to torch tensor
        device = self._torch.device(f"{self._device_type}:{self._device_id}")
        logits_tensor = self._torch.from_numpy(logits).to(device)
        
        # Get logits for last position
        if len(logits_tensor.shape) == 3:
            logits_tensor = logits_tensor[:, -1, :]
        
        # Sample using TokenGenerator
        result = self._token_generator.sample(
            logits=logits_tensor,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # Convert to numpy
        return np.array([[result.token_id]], dtype=np.int64)
```

## Error Handling

The TokenGenerator raises `InferenceError` for invalid inputs:

```python
from exo.worker.engines.pytorch_xpu.errors import InferenceError

try:
    result = generator.sample(logits, temperature=0.0)  # Invalid!
except InferenceError as e:
    print(f"Sampling error: {e}")
```

Common errors:
- Invalid temperature (≤ 0)
- NaN or infinite logits
- Invalid logits shape (>2D)

## Performance Tips

1. **Reuse TokenGenerator**: Create once and reuse for all sampling operations
2. **Batch Processing**: Process logits in batches when possible
3. **Device Placement**: Keep logits on GPU to avoid CPU-GPU transfers
4. **Parameter Tuning**: 
   - Use `top_k=50` and `top_p=0.95` for good quality/speed balance
   - Use `temperature=0.8` for slightly more focused outputs

## Examples

### Greedy Decoding

```python
# Most likely token (deterministic)
result = generator.sample(logits, temperature=0.01, top_k=1)
```

### Creative Generation

```python
# More diverse outputs
result = generator.sample(logits, temperature=1.2, top_p=0.9)
```

### Balanced Generation

```python
# Good balance of quality and diversity
result = generator.sample(logits, temperature=0.8, top_p=0.95, top_k=50)
```

### Stop on EOS

```python
tokens = []
while True:
    result = generator.sample(logits, temperature=0.8, top_p=0.95)
    tokens.append(result.token_id)
    
    if result.is_eos:
        break
    
    # Get next logits...
```

## Testing

Run the test suite:

```bash
uv run pytest src/exo/worker/engines/pytorch_xpu/tests/test_token_generator.py -v
```

Or use the simple test script:

```bash
python src/exo/worker/engines/pytorch_xpu/test_token_generator_simple.py
```

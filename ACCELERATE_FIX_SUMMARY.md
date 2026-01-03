# Accelerate Dependency Fix Summary

## Issue
The EXO system was failing to load models with the error:
```
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
```

## Root Cause
The PyTorch model loading code in `src/exo/worker/engines/torch/utils_torch.py` was using `device_map="cpu"` parameter, which requires the `accelerate` library. However, while `accelerate` was included in the package's `propagatedBuildInputs` in `flake.nix`, it was missing from the development shell's dependencies.

## Solution
Added `python313Packages.accelerate` to the `devShells.default` packages list in `flake.nix`:

```nix
# PYTHON PACKAGES
python313Packages.torch
python313Packages.transformers
python313Packages.huggingface-hub
python313Packages.loguru
python313Packages.pydantic
python313Packages.anyio
python313Packages.aiofiles
python313Packages.aiohttp
python313Packages.fastapi
python313Packages.rich
python313Packages.psutil
python313Packages.tiktoken
python313Packages.safetensors
python313Packages.tokenizers
python313Packages.accelerate  # <-- Added this line
```

## Impact
- **Model loading now works**: PyTorch models can be loaded successfully with `device_map="cpu"`
- **Proper device management**: The accelerate library provides proper device mapping functionality
- **Development environment consistency**: Dev shell now matches the package dependencies
- **No code changes needed**: The original PyTorch loading code works as intended

## Testing
Verified the fix with a test script that successfully:
1. Loads model configuration
2. Loads PyTorch model with `device_map="cpu"`
3. Moves model to CPU and sets eval mode
4. Performs forward pass with test input

## Files Changed
- `flake.nix`: Added `python313Packages.accelerate` to devShells.default packages

## Related Issues Fixed
This resolves the "loading → loaded → failed → unknown" status sequence that was occurring because:
1. Model loading would start (status: loading)
2. `initialize_engine` would fail due to missing accelerate (exception thrown)
3. Exception handler would set status to failed
4. Supervisor would lose track of status (unknown)

Now the sequence should be: loading → loaded → warming_up → ready → running, with proper model loading at each step.

## Note
The `accelerate` library was already included in the package dependencies (`propagatedBuildInputs`) but was missing from the development shell. This is why the issue only appeared during development/testing but would have been resolved in the final packaged version.
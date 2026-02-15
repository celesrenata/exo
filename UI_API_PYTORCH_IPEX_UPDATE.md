# UI and API Updates for PyTorch+IPEX Backend

## Summary

Updated the exo UI and API to add PyTorch+IPEX as a backend option alongside tinygrad and MLX. This allows users to select PyTorch+IPEX for Intel Arc GPU inference through the dashboard.

## Changes Made

### Backend Infrastructure

#### 1. Factory (`src/exo/worker/engines/factory.py`)

Added PyTorch+IPEX backend to the factory:

```python
elif backend_name == "pytorch_ipex":
    try:
        from exo.worker.engines.pytorch_ipex.pytorch_ipex_backend import (
            PyTorchIPEXBackend,
        )
        return PyTorchIPEXBackend(shard_downloader)
    except ImportError as e:
        raise BackendNotAvailableError(
            "pytorch_ipex", f"PyTorch+IPEX backend not available: {e}"
        ) from e
```

Updated backend registry:
```python
BACKEND_REGISTRY = {
    "mlx": "MLXBackend",
    "tinygrad": "TinygradBackend",
    "pytorch_ipex": "PyTorchIPEXBackend",  # NEW
    "dummy": "DummyBackend",
}
```

#### 2. Backend Selector (`src/exo/worker/engines/backend_selector.py`)

Updated fallback chain to include PyTorch+IPEX:

```python
def get_fallback_chain(preferred: str) -> list[str]:
    if preferred == "pytorch_ipex":
        return ["pytorch_ipex", "tinygrad", "mlx"]
    elif preferred == "tinygrad":
        return ["tinygrad", "mlx"]
    elif preferred == "npu":
        return ["npu", "pytorch_ipex", "tinygrad", "mlx"]
    else:
        return [preferred]
```

Added environment variable support:

```python
# Check for PyTorch+IPEX first (preferred for Intel Arc)
if os.environ.get("EXO_PYTORCH_IPEX_ENABLED", "false").lower() == "true":
    logger.info("EXO_PYTORCH_IPEX_ENABLED=true, using pytorch_ipex backend")
    return "pytorch_ipex"
```

### Dashboard UI

#### 3. Hardware Types (`dashboard/src/lib/types/hardware.ts`)

Updated backend type definition:

```typescript
export type BackendType = "mlx" | "tinygrad" | "pytorch_ipex" | "npu";

export interface BackendInfo {
  /** Type of backend (mlx, tinygrad, pytorch_ipex, npu) */
  type: BackendType;
}
```

#### 4. Main Dashboard (`dashboard/src/routes/+page.svelte`)

Added PyTorch+IPEX to instance types:

```typescript
type InstanceMeta = "MlxRing" | "MlxIbv" | "MlxJaccl" | "TinygradRing" | "PyTorchIPEXRing";
```

Updated instance type matching logic:

```typescript
selectedInstanceType === "MlxRing"
  ? runtime === "MlxRing"
  : selectedInstanceType === "TinygradRing"
    ? runtime === "TinygradRing"
    : selectedInstanceType === "PyTorchIPEXRing"
      ? runtime === "PyTorchIPEXRing"
      : runtime === "MlxIbv" || runtime === "MlxJaccl";
```

Added instance type display:

```typescript
else if (instanceTag === "PyTorchIPEXRingInstance")
  instanceType = "PyTorch+IPEX Ring";
```

Added UI button for PyTorch+IPEX selection:

```svelte
<button
  onclick={() => {
    selectedInstanceType = "PyTorchIPEXRing";
    saveLaunchDefaults();
  }}
  class="flex items-center gap-2 py-2 px-4 text-sm font-mono border rounded transition-all duration-200 cursor-pointer {selectedInstanceType ===
  'PyTorchIPEXRing'
    ? 'bg-transparent text-exo-yellow border-exo-yellow'
    : 'bg-transparent text-white/70 border-exo-medium-gray/50 hover:border-exo-yellow/50'}"
>
  <span
    class="w-4 h-4 rounded-full border-2 flex items-center justify-center {selectedInstanceType ===
    'PyTorchIPEXRing'
      ? 'border-exo-yellow'
      : 'border-exo-medium-gray'}"
  >
    {#if selectedInstanceType === "PyTorchIPEXRing"}
      <span class="w-2 h-2 rounded-full bg-exo-yellow"></span>
    {/if}
  </span>
  PyTorch+IPEX Ring
</button>
```

#### 5. Model Card (`dashboard/src/lib/components/ModelCard.svelte`)

Updated runtime type:

```typescript
runtime?: "MlxRing" | "MlxIbv" | "MlxJaccl" | "TinygradRing" | "PyTorchIPEXRing";
```

Added runtime display:

```svelte
: runtime === "TinygradRing"
  ? "Tinygrad Ring"
  : runtime === "PyTorchIPEXRing"
    ? "PyTorch+IPEX Ring"
    : runtime
```

## Usage

### Environment Variable

Enable PyTorch+IPEX backend by setting:

```bash
export EXO_PYTORCH_IPEX_ENABLED=true
```

### Dashboard

1. Open the exo dashboard
2. Click "Launch Model"
3. Select "PyTorch+IPEX Ring" from the instance type options
4. Configure other settings (model, sharding, nodes)
5. Click "Launch"

### Fallback Behavior

If PyTorch+IPEX is not available, the system will automatically fall back to:
1. tinygrad (if available)
2. MLX (baseline)

## Benefits

1. **User Choice**: Users can now select PyTorch+IPEX for Intel Arc GPU inference
2. **Visibility**: Backend type is displayed in the dashboard
3. **Flexibility**: Environment variable allows easy switching between backends
4. **Fallback**: Automatic fallback ensures inference works even if PyTorch+IPEX unavailable
5. **Consistency**: Follows the same pattern as tinygrad and MLX backends

## Testing

To test the changes:

1. Set `EXO_PYTORCH_IPEX_ENABLED=true`
2. Start exo with the dashboard
3. Verify "PyTorch+IPEX Ring" appears in instance type selector
4. Launch a model with PyTorch+IPEX
5. Verify the backend is used and displayed correctly

## Next Steps

1. Implement `PyTorchIPEXBackend` class (Task 5)
2. Add PyTorchIPEXRingInstance to instance types
3. Test end-to-end with actual models
4. Update documentation

## Files Modified

- `src/exo/worker/engines/factory.py` - Added PyTorch+IPEX backend
- `src/exo/worker/engines/backend_selector.py` - Added fallback chain and env var
- `dashboard/src/lib/types/hardware.ts` - Added backend type
- `dashboard/src/routes/+page.svelte` - Added UI option
- `dashboard/src/lib/components/ModelCard.svelte` - Added runtime display

---

**Date**: 2026-02-15
**Status**: Complete
**Related**: Task 3 (Model Loader), Task 5 (Inference Engine)

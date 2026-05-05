---
inclusion: always
---

# PyTorch XPU Backend — Project Context

## Project Goal

Connect all 4 gremlin nodes' Intel iGPUs and shard large language models across them using pipeline parallelism via PyTorch's native `torch.xpu` and Gloo distributed backend.

## Hardware

Every gremlin node has 8× Intel Arc Graphics (Meteor Lake-P) iGPUs with shared system memory. gremlin-1 additionally has an NVIDIA RTX 4070 Ti SUPER (discrete).

| Node | Intel iGPU | NVIDIA | System RAM | XPU target |
|------|-----------|--------|------------|------------|
| gremlin-1 (10.1.1.12) | 8× Meteor Lake-P Arc | RTX 4070 Ti SUPER | ~94 GiB shared | Yes |
| gremlin-2 (10.1.1.13) | 8× Meteor Lake-P Arc | — | ~94 GiB shared | Yes |
| gremlin-3 (10.1.1.14) | 8× Meteor Lake-P Arc | — | ~94 GiB shared | Yes |
| gremlin-4 (10.1.1.15) | 8× Meteor Lake-P Arc | — | ~94 GiB shared | Yes |

Total cluster memory: ~376 GiB usable across 4 nodes (shared memory architecture).

**The NVIDIA card on gremlin-1 is NOT the target.** The project is about the Intel iGPUs.

## PyTorch Version

- **Required**: PyTorch 2.11+ from the XPU wheel index
- **Nix derivation**: `nix/pytorch-xpu.nix` fetches the XPU wheel from `download.pytorch.org/whl/xpu/`
- **Current version**: 2.11.0+xpu
- **Do NOT use CUDA wheels** (`+cu128`) — they have `torch.xpu` module but `is_available()` returns False
- PyTorch is managed by Nix, not pip/venv. Do not pip install on gremlin nodes.

## Backend Status: GPU Detection Working, Distributed Pipeline Implemented

### Completed
✅ Device Manager (Intel Arc detection and selection)
✅ Model Loader (HuggingFace model loading with XPU optimization)
✅ KV Cache Manager (efficient memory management)
✅ PyTorchXPUBackend (main inference engine)
✅ Token Generator (sampling with temperature, top-k, top-p)
✅ Runner integration with distributed dispatch
✅ Distributed Generation Pipeline (distributed_generator.py)
✅ GPU detection on gremlin nodes (`torch.xpu.is_available() == True`)
✅ Intel oneAPI runtime with proper RPATH (libumf, libhwloc, level-zero)

### Pending
❌ End-to-end generation on GPU (Qwen3.5 forward pass needs testing)
❌ Performance optimization
❌ Multi-node distributed generation end-to-end test

## Intel XPU Runtime Stack (Critical for GPU Detection)

The XPU stack has multiple layers that must all be present and use the SAME glibc:

```
PyTorch (torch._C) → libsycl.so.8 → libur_loader.so.0 → libur_adapter_level_zero.so
    → libumf.so.1 (needs libhwloc.so.15)
    → libze_loader.so.1 → libze_intel_gpu.so.1 (kernel driver: i915)
```

### Key Packages and Their Roles

| Package | Provides | Role |
|---------|----------|------|
| `intel-oneapi-runtime` (nix/intel-oneapi-runtime.nix) | libsycl.so.8, libur_*.so, libumf.so.1 | SYCL runtime + Unified Runtime |
| `intel-compute-runtime` (from pkgsExo) | libze_intel_gpu.so.1 (in .drivers output) | Level Zero GPU driver |
| `level-zero` (from pkgsExo) | libze_loader.so.1 | Level Zero loader |
| `hwloc` (nixpkgs) | libhwloc.so.15 | Hardware locality (needed by libumf) |
| `unified-memory-framework` (PyPI `umf` wheel) | libumf.so.1 | Memory management (needed by UR adapter) |

### Critical Constraints

1. **glibc version must match across ALL libraries loaded into the same process.** The exo Python process uses glibc from the exo flake's nixpkgs. ALL GPU runtime libraries must use the same glibc. This means `intel-compute-runtime` and `level-zero` must come from `pkgsExo` (the exo flake's package set), NOT from the gremlin system's `pkgs`.

2. **The gremlin flake must use exo flake's packages for GPU runtime:**
   ```nix
   intelGpuPackages = [
     exo.packages.${system}.intel-compute-runtime
     exo.packages.${system}.intel-compute-runtime-drivers
     exo.packages.${system}.level-zero
     exo.packages.${system}.intel-oneapi-runtime
   ];
   ```
   Using `pkgs.intel-compute-runtime` from the gremlin system's nixpkgs causes glibc mismatch.

3. **`intel-oneapi-runtime` must bundle libumf.so.1** (from PyPI `umf` wheel) and have `hwloc` in buildInputs for `autoPatchelfHook` to resolve `libhwloc.so.15`.

4. **Level Zero driver discovery**: The Level Zero loader finds `libze_intel_gpu.so.1` via `LD_LIBRARY_PATH`. The NixOS module (`nix/distributed-inference.nix`) puts `/run/opengl-driver/lib` and the drivers path on `LD_LIBRARY_PATH`. The `hardware.graphics.extraPackages` populates `/run/opengl-driver/lib/`.

5. **Do NOT update the exo flake's nixpkgs** without testing the full build chain. The nixpkgs pin determines glibc version, and changing it breaks the `intel-compute-runtime` compatibility and may break `anyio`, `transformers`, etc.

## Key Files

- Engine: `src/exo/worker/engines/pytorch_xpu/`
- Distributed generator: `src/exo/worker/engines/pytorch_xpu/distributed_generator.py`
- Model loader (Qwen3.5 support): `src/exo/worker/engines/pytorch_xpu/model_loader.py`
- Tests: `src/exo/worker/engines/pytorch_xpu/tests/`
- Runner: `src/exo/worker/runner/runner.py`
- GPU detector: `src/exo/worker/engines/pytorch_xpu/gpu_detector.py`
- Distributed comms: `src/exo/worker/engines/pytorch_xpu/distributed.py`
- Intel oneAPI runtime (Nix): `nix/intel-oneapi-runtime.nix`
- NixOS distributed module: `nix/distributed-inference.nix`
- PyTorch XPU wheel: `nix/pytorch-xpu.nix`

## Debugging GPU Detection

If `torch.xpu.is_available()` returns False:

1. Check `LD_LIBRARY_PATH` includes the drivers path and `/run/opengl-driver/lib`
2. Check `ldd` on `libze_intel_gpu.so.1` — all deps must resolve to the same glibc
3. Check `ldd` on `libur_adapter_level_zero.so` — must find `libumf.so.1` and `libze_loader.so.1`
4. Test Level Zero directly: `ctypes.CDLL("libze_loader.so.1"); ze.zeInit(0)` should return 0
5. Enable tracing: `SYCL_UR_TRACE=1` shows which adapters load/fail
6. Check kernel driver: `lsmod | grep i915` and `/dev/dri/renderD128` must exist

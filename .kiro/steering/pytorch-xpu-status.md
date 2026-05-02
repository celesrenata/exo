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
| gremlin-1 (10.1.1.12) | 8× Meteor Lake-P Arc | RTX 4070 Ti SUPER | ~30 GiB usable | Yes |
| gremlin-2 (10.1.1.13) | 8× Meteor Lake-P Arc | — | ~30 GiB usable | Yes |
| gremlin-3 (10.1.1.14) | 8× Meteor Lake-P Arc | — | ~30 GiB usable | Yes |
| gremlin-4 (10.1.1.15) | 8× Meteor Lake-P Arc | — | ~30 GiB usable | Yes |

Total cluster memory: ~105 GiB usable across 4 nodes (shared memory architecture).

**The NVIDIA card on gremlin-1 is NOT the target.** The project is about the Intel iGPUs.

## PyTorch Version

- **Required**: PyTorch 2.11+ from the XPU wheel index
- **Nix derivation**: `nix/pytorch-xpu.nix` fetches the XPU wheel from `download.pytorch.org/whl/xpu/`
- **Current version**: 2.11.0+xpu (updated from 2.9.1+xpu)
- **Do NOT use CUDA wheels** (`+cu128`) — they have `torch.xpu` module but `is_available()` returns False
- PyTorch is managed by Nix, not pip/venv. Do not pip install on gremlin nodes.

## Backend Status: Core Components Complete, Integration Incomplete

### Completed
✅ Device Manager (Intel Arc detection and selection)
✅ Model Loader (HuggingFace model loading with XPU optimization)
✅ KV Cache Manager (efficient memory management)
✅ PyTorchXPUBackend (main inference engine)
✅ Token Generator (sampling with temperature, top-k, top-p)
✅ Basic runner integration
✅ ipex → xpu rename complete

### Pending
❌ Distributed Coordinator (multi-node inference via Gloo)
❌ Full runner integration (model loading, generation loop)
❌ End-to-end testing on actual Intel iGPU hardware
❌ Performance optimization
❌ PyTorch 2.11 XPU wheel deployed to all gremlin nodes

## Key Files

- Engine: `src/exo/worker/engines/pytorch_xpu/`
- Tests: `src/exo/worker/engines/pytorch_xpu/tests/`
- Runner: `src/exo/worker/runner/runner.py`
- Factory: `src/exo/worker/engines/factory.py`
- GPU detector: `src/exo/worker/engines/pytorch_xpu/gpu_detector.py`
- Distributed: `src/exo/worker/engines/pytorch_xpu/distributed.py`

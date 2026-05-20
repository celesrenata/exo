<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/imgs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/imgs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

exo: Distributed AI inference on Intel Arc iGPUs via PyTorch XPU.

Fork of [exo-explore/exo](https://github.com/exo-explore/exo) maintained by [celesrenata](https://github.com/celesrenata).

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0.html" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-Apache2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

</div>

---

This fork extends exo to support Intel Arc Graphics (Meteor Lake-P) iGPUs through a custom PyTorch XPU backend. It runs large language models across a cluster of NixOS machines using pipeline and tensor parallelism, with the entire runtime stack managed by Nix flakes.

**Key additions over upstream:**
- PyTorch XPU backend for Intel Arc iGPUs (torch.xpu + Level Zero + oneAPI)
- Pipeline parallelism across multiple nodes via Gloo distributed backend
- Tensor parallelism for model sharding across devices
- NixOS flake-based deployment (no pip, no venv, no brew)
- Automated cluster deployment via `deploy_cluster.sh`
- Documentation generator tool (`generate-docs`)

---

## Features

| Feature | Description |
|---------|-------------|
| **PyTorch XPU Engine** | Intel Arc iGPU support with device detection, model loading, KV cache management, and token generation |
| **Pipeline Parallelism** | Distribute model layers across nodes for memory-efficient inference |
| **Tensor Parallelism** | Shard model weights across devices for faster inference |
| **NixOS Deployment** | Declarative configuration via nix flakes, one-command cluster rebuild |
| **OpenAI-compatible API** | Drop-in replacement for OpenAI, Claude, and Ollama client libraries |
| **Automatic Device Discovery** | Nodes find each other via libp2p — no manual configuration |
| **Event Sourcing** | Immutable state management with typed pub/sub messaging |
| **Dashboard** | Svelte 5 + TypeScript UI for cluster management |

---

## Supported Backends

| Backend | Platform | Hardware | Status |
|---------|----------|----------|--------|
| PyTorch XPU | Linux (NixOS) | Intel Arc iGPUs | Active development |
| MLX | macOS | Apple Silicon (M-series) | Primary (upstream) |
| Image Models | macOS / Linux | Any | Experimental |

---

## Hardware — The Gremlin Cluster

The reference deployment runs on a 4-node NixOS cluster.

| Node | IP | GPUs | System RAM |
|------|------|------|------------|
| gremlin-1 | 10.1.1.12 | 8× Intel Arc (Meteor Lake-P) + RTX 4070 Ti SUPER | ~94 GiB |
| gremlin-2 | 10.1.1.13 | 8× Intel Arc (Meteor Lake-P) | ~94 GiB |
| gremlin-3 | 10.1.1.14 | 8× Intel Arc (Meteor Lake-P) | ~94 GiB |
| gremlin-4 | 10.1.1.15 | 8× Intel Arc (Meteor Lake-P) | ~94 GiB |

**Total cluster memory:** ~376 GiB usable across 4 nodes (shared memory architecture).

The NVIDIA card on gremlin-1 is not used — all inference targets the Intel Arc iGPUs via `torch.xpu`.

**Verified model:** Qwen/Qwen3.5-4B with pipeline parallelism across all 4 nodes producing coherent output.

---

## Quick Start (Development)

```bash
# Clone
git clone git@github.com:celesrenata/exo.git
cd exo

# Build the Svelte dashboard
cd dashboard && npm install && npm run build && cd ..

# Run exo (starts master + worker + API at http://localhost:52415)
uv run exo

# Run with verbose logging
uv run exo -v
```

### Pre-commit checks

```bash
uv run basedpyright     # type checking (strict mode)
uv run ruff check       # linting
nix fmt                 # formatting
uv run pytest           # tests
```

---

## Deployment (NixOS Cluster)

Each gremlin node runs NixOS. The exo flake is an input to each node's `/etc/nixos/flake.nix`.

### One-command deploy

```bash
bash deploy_cluster.sh
```

This pushes the current branch, updates the flake input on each node, runs `nixos-rebuild switch`, restarts the exo service, and verifies the cluster is running.

### Deploy specific nodes

```bash
bash deploy_cluster.sh gremlin-1              # master only
bash deploy_cluster.sh gremlin-2 gremlin-3    # specific workers
```

### Verify the cluster

```bash
curl -s http://10.1.1.12:52415/state | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'Nodes: {len(d.get(\"topology\",{}).get(\"nodes\",{}))}')"
```

---

## API

The master node (gremlin-1) serves an OpenAI-compatible API at `http://10.1.1.12:52415`.

### Chat completion

```bash
curl http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 50
  }'
```

### Supported endpoints

- `/v1/chat/completions` — OpenAI Chat Completions
- `/v1/messages` — Claude Messages API
- `/v1/responses` — OpenAI Responses API
- `/ollama/api/chat` — Ollama API
- `/models` — List available models
- `/state` — Cluster state and topology

---

## Documentation Generator

A CLI tool that scans source files, identifies undocumented code, and uses local LLM servers to generate documentation.

```bash
uv run generate-docs              # generate docs for changed files
uv run generate-docs --dry-run    # preview without writing
uv run generate-docs --force      # regenerate everything
uv run generate-docs --target src/exo/api/   # scope to a directory
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview changes without writing files |
| `--force` | Regenerate all docs regardless of file changes |
| `--strict` | Exit with code 1 if any validation failures occur |
| `--target <path>` | Only process files under this path |
| `--output <dir>` | Output directory (default: `docs/`) |

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCGEN_MODEL_URL` | Local model server URL | `http://localhost:11434` |
| `DOCGEN_MODEL_MAP` | Model alias mapping | Built-in defaults |

Requires a local model server (e.g. ollama) running. Uses incremental SHA-256 hashing to skip unchanged files.

---

## Architecture

The system follows the same component model as upstream exo:

- **Router** — libp2p pub/sub messaging via Rust bindings (exo_pyo3_bindings)
- **Worker** — Handles inference tasks, downloads models, manages runner processes
- **Master** — Coordinates cluster state, places model instances across nodes
- **Election** — Bully algorithm for master election
- **API** — FastAPI server for OpenAI-compatible endpoints
- **Dashboard** — Svelte 5 + TypeScript frontend

### PyTorch XPU Engine

Located in `src/exo/worker/engines/pytorch_xpu/`:

| Module | Role |
|--------|------|
| `gpu_detector.py` | Intel Arc device detection via torch.xpu |
| `model_loader.py` | HuggingFace model loading with XPU optimization |
| `kv_cache_manager.py` | Efficient KV cache memory management |
| `token_generator.py` | Sampling with temperature, top-k, top-p |
| `distributed_generator.py` | Multi-node pipeline parallelism orchestration |
| `pipeline_parallel_shard.py` | Layer-by-layer model sharding with DynamicCache |

### Intel XPU Runtime Stack

```
PyTorch (torch._C) → libsycl.so.8 → libur_loader.so.0 → libur_adapter_level_zero.so
    → libumf.so.1 (needs libhwloc.so.15)
    → libze_loader.so.1 → libze_intel_gpu.so.1 (kernel driver: i915)
```

All runtime libraries are managed by Nix derivations to ensure glibc version consistency.

---

## Contributing

1. Fork the repository and branch from `twenty-tps-research`.
2. Ensure code passes `uv run basedpyright`, `uv run ruff check`, and `uv run pytest`.
3. Format with `nix fmt` before committing.
4. Open a pull request with a description of what changed and how to test it.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

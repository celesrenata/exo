# Requirements Document

## Introduction

This spec covers two related changes to the exo distributed inference system:

1. **PyTorch-Compatible Model Cards** — Add TOML model cards for standard HuggingFace safetensors models that work with PyTorch on Linux (NVIDIA CUDA and Intel XPU). The existing MLX-quantized model cards only work on Apple Silicon. The gremlin cluster (1× RTX 4070 Ti SUPER with 15.5 GiB VRAM, 3× Intel iGPU nodes with ~30 GiB usable RAM each, ~105 GiB total) needs cards for models ranging from single-node testing to full-cluster distributed inference.

2. **Rename ipex → xpu** — IPEX (Intel Extension for PyTorch) is discontinued. All references to "ipex" in directory names, type names, enum values, variable names, Nix modules, log messages, and documentation must be renamed to "xpu" to reflect that the backend now uses native PyTorch XPU support (2.11+).

## Glossary

- **Model_Card**: A TOML file in `resources/inference_model_cards/` that defines a model's HuggingFace repo ID, storage size, layer count, hidden size, tensor parallelism support, and tasks.
- **Model_Card_Loader**: The Python module `src/exo/shared/models/model_cards.py` that discovers, parses, and caches model cards from the TOML search path.
- **Safetensors**: A safe, fast file format for storing tensors. Standard HuggingFace models use `.safetensors` files; MLX-quantized models use a different weight format.
- **MLX_Card**: A model card referencing an `mlx-community/` HuggingFace repo with MLX-quantized weights (Apple Silicon only).
- **PyTorch_Card**: A model card referencing a HuggingFace repo with standard safetensors weights compatible with PyTorch on Linux (NVIDIA CUDA and Intel XPU).
- **GPTQ_Card**: A model card referencing a GPTQ-quantized HuggingFace model for reduced memory usage on PyTorch.
- **Gremlin_Cluster**: The 4-node Linux cluster: gremlin-1 (RTX 4070 Ti SUPER, 15.5 GiB VRAM), gremlin-2/3/4 (Intel iGPU, ~30 GiB usable RAM each). Total usable memory ~105 GiB.
- **Pipeline_Parallelism**: Distributing a model's layers across multiple nodes, each node running a contiguous subset of layers.
- **Instance_Meta**: The enum `InstanceMeta` in `src/exo/shared/types/worker/instances.py` that identifies backend types (MlxRing, MlxJaccl, TinygradRing, and the value being renamed).
- **PyTorch_XPU_Backend**: The PyTorch inference backend in `src/exo/worker/engines/` that uses native `torch.xpu` for Intel GPU support (formerly called "pytorch_ipex").
- **Runner**: The worker component (`src/exo/worker/runner/runner.py`) that dispatches inference tasks based on instance type.
- **Placement**: The master component (`src/exo/master/placement.py`) that assigns model instances to cluster nodes.
- **NixOS_Module**: The Nix flake module in `flake.nix` that configures the exo service, including backend options and environment variables.

## Requirements

### Requirement 1: Small PyTorch Model Cards (< 4 GiB)

**User Story:** As a cluster operator, I want model cards for small PyTorch-compatible models, so that I can test single-node inference on any gremlin node without distributed sharding.

#### Acceptance Criteria

1. THE Model_Card_Loader SHALL discover and parse PyTorch_Card files from `resources/inference_model_cards/` using the same TOML format as existing cards.
2. WHEN a PyTorch_Card for a model under 4 GiB storage size is loaded, THE Model_Card_Loader SHALL produce a valid ModelCard with `model_id` pointing to a HuggingFace repo containing standard safetensors weights (not MLX-quantized).
3. THE PyTorch_Card files SHALL include cards for at least: Qwen2.5-0.5B-Instruct (~494 MiB), Qwen2.5-1.5B-Instruct (~1.5 GiB), Llama-3.2-1B-Instruct, and Llama-3.2-3B-Instruct (~3.2 GiB).
4. WHEN a small PyTorch_Card is loaded, THE ModelCard SHALL have `quantization` set to an empty string (full-precision safetensors).
5. WHEN a small PyTorch_Card is loaded, THE ModelCard SHALL have `tasks` containing `TextGeneration`.

### Requirement 2: Medium PyTorch Model Cards (4–16 GiB)

**User Story:** As a cluster operator, I want model cards for medium-sized PyTorch-compatible models, so that I can run inference on gremlin-1's NVIDIA GPU (15.5 GiB VRAM) or on a single Intel iGPU node.

#### Acceptance Criteria

1. THE PyTorch_Card files SHALL include cards for medium models in the 4–16 GiB range that fit within gremlin-1's 15.5 GiB VRAM.
2. THE medium PyTorch_Card files SHALL include at least: Qwen2.5-7B-Instruct (~7.6 GiB), Meta-Llama-3.1-8B-Instruct (~8 GiB), and Mistral-7B-Instruct-v0.3 (~7 GiB).
3. WHEN a medium PyTorch_Card references a GPTQ-quantized model, THE ModelCard SHALL have `quantization` set to the quantization method (e.g., `"GPTQ-Int4"`).
4. WHEN a medium PyTorch_Card is loaded, THE ModelCard SHALL have accurate `n_layers`, `hidden_size`, and `storage_size` values matching the HuggingFace model's actual configuration.

### Requirement 3: Large Distributed PyTorch Model Cards (16–100 GiB)

**User Story:** As a cluster operator, I want model cards for large models that require distributed sharding across the Gremlin_Cluster, so that I can run models too large for a single node via Pipeline_Parallelism.

#### Acceptance Criteria

1. THE PyTorch_Card files SHALL include cards for large models in the 16–100 GiB range that require distribution across multiple gremlin nodes.
2. THE large PyTorch_Card files SHALL include at least one model in the 16–30 GiB range (fits on 2 nodes) and at least one model in the 30–100 GiB range (requires 3–4 nodes).
3. WHEN a large PyTorch_Card is loaded, THE ModelCard SHALL have `supports_tensor` set to `true` only if the model architecture supports tensor parallelism (as determined by `ConfigData.supports_tensor` in the Model_Card_Loader).
4. WHEN a large PyTorch_Card has `storage_size` exceeding 105 GiB, THE Model_Card_Loader SHALL still parse the card, but Placement SHALL reject the card if no cycle has sufficient aggregate memory.

### Requirement 4: MLX Card Preservation

**User Story:** As a macOS user, I want the existing MLX model cards to remain available, so that I can continue using exo on Apple Silicon.

#### Acceptance Criteria

1. THE existing MLX_Card files in `resources/inference_model_cards/` SHALL remain unchanged after adding PyTorch_Card files.
2. WHEN the Model_Card_Loader scans `resources/inference_model_cards/`, THE Model_Card_Loader SHALL load both MLX_Card and PyTorch_Card files into the card cache.
3. THE PyTorch_Card files SHALL use distinct filenames from existing MLX_Card files (the HuggingFace repo slug naturally ensures this since PyTorch_Card model IDs differ from `mlx-community/` IDs).

### Requirement 5: Model Card TOML Format Consistency

**User Story:** As a developer, I want all model cards to use the same TOML schema, so that the Model_Card_Loader parses them uniformly without backend-specific logic.

#### Acceptance Criteria

1. THE PyTorch_Card TOML files SHALL contain all required fields: `model_id`, `storage_size.in_bytes`, `n_layers`, `hidden_size`, `supports_tensor`, `tasks`, `family`, `quantization`, `base_model`, and `capabilities`.
2. WHEN a PyTorch_Card TOML file is parsed, THE Model_Card_Loader SHALL produce a valid `ModelCard` Pydantic model without raising `ValidationError`.
3. THE PyTorch_Card TOML files SHALL use the same `[storage_size]` table format with `in_bytes` key as existing cards.
4. FOR ALL PyTorch_Card files, parsing then serializing via `ModelCard.save()` then re-parsing SHALL produce an equivalent ModelCard object (round-trip property).

### Requirement 6: Rename Directory pytorch_ipex → pytorch_xpu

**User Story:** As a developer, I want the backend directory renamed from `pytorch_ipex` to `pytorch_xpu`, so that the directory name reflects the actual technology (native PyTorch XPU, not discontinued IPEX).

#### Acceptance Criteria

1. THE directory `src/exo/worker/engines/pytorch_ipex/` SHALL be renamed to `src/exo/worker/engines/pytorch_xpu/`.
2. WHEN any Python module imports from the PyTorch_XPU_Backend, THE import path SHALL use `exo.worker.engines.pytorch_xpu` (not `pytorch_ipex`).
3. WHEN the Runner imports PyTorch_XPU_Backend modules, THE import statements SHALL reference `exo.worker.engines.pytorch_xpu.generator`, `exo.worker.engines.pytorch_xpu.model_loader`, and `exo.worker.engines.pytorch_xpu.warmup`.
4. IF a Python module still contains an import path referencing `pytorch_ipex`, THEN the test suite SHALL fail with a clear error identifying the stale reference.

### Requirement 7: Rename Instance Types and Enum Values

**User Story:** As a developer, I want the instance type names updated from IPEX to XPU, so that the type system accurately describes the backend.

#### Acceptance Criteria

1. THE enum value `InstanceMeta.PyTorchIPEXRing` SHALL be renamed to `InstanceMeta.PyTorchXPURing`.
2. THE class `PyTorchIPEXRingInstance` SHALL be renamed to `PyTorchXPURingInstance`.
3. WHEN Placement creates an instance for the PyTorch_XPU_Backend, THE Placement module SHALL use `InstanceMeta.PyTorchXPURing` and `PyTorchXPURingInstance`.
4. WHEN the Runner dispatches on instance type, THE Runner SHALL check for `PyTorchXPURingInstance` (not `PyTorchIPEXRingInstance`).
5. THE `Instance` type union in `instances.py` SHALL include `PyTorchXPURingInstance` instead of `PyTorchIPEXRingInstance`.

### Requirement 8: Rename Variables, Log Messages, and Internal References

**User Story:** As a developer, I want all internal variable names, log messages, and code comments updated from "ipex" to "xpu", so that the codebase is consistent and searchable.

#### Acceptance Criteria

1. WHEN the Runner detects a PyTorch_XPU_Backend instance, THE Runner SHALL set `backend_type` to `"pytorch_xpu"` (not `"pytorch_ipex"`).
2. WHEN the Runner logs backend selection, THE log message SHALL reference "PyTorch XPU" (not "PyTorch IPEX" or "PyTorchIPEX").
3. THE variable `is_pytorch_ipex` in Runner SHALL be renamed to `is_pytorch_xpu`.
4. WHEN any Python source file in `src/` contains the string `pytorch_ipex` (excluding test files that verify the rename), THE test suite SHALL flag the occurrence as a stale reference.

### Requirement 9: Rename NixOS Module Configuration Keys

**User Story:** As a NixOS deployer, I want the Nix module configuration keys updated from `pytorch_ipex` to `pytorch_xpu`, so that the NixOS configuration reflects the current backend name.

#### Acceptance Criteria

1. THE NixOS module option `services.exo.intel.pytorch_ipex.enable` SHALL be renamed to `services.exo.intel.pytorch_xpu.enable`.
2. THE NixOS module option `services.exo.intel.pytorch_ipex.preferredBackend` SHALL be renamed to `services.exo.intel.pytorch_xpu.preferredBackend`.
3. THE environment variable `EXO_PYTORCH_IPEX_ENABLED` SHALL be renamed to `EXO_PYTORCH_XPU_ENABLED`.
4. THE environment variable `IPEX_TILE_AS_DEVICE` SHALL be removed (IPEX-specific, not needed for native PyTorch XPU).
5. WHEN the NixOS module is configured with `services.exo.intel.pytorch_xpu.enable = true`, THE module SHALL set `PYTORCH_ENABLE_XPU = "1"` and `EXO_PYTORCH_XPU_ENABLED = "true"`.
6. THE NixOS module export name `exo-distributed` SHALL remain unchanged (the export is not backend-specific).

### Requirement 10: Rename Documentation and Steering Files

**User Story:** As a developer, I want all documentation and steering files updated to reference "xpu" instead of "ipex", so that new contributors find accurate information.

#### Acceptance Criteria

1. THE steering file `pytorch-ipex-status.md` SHALL be renamed to `pytorch-xpu-status.md` with all internal references updated.
2. THE example NixOS config `docs/examples/nixos-pytorch-ipex-config.nix` SHALL be renamed to `docs/examples/nixos-pytorch-xpu-config.nix` with all internal references updated.
3. THE Nix file `nix/ipex-xpu.nix` SHALL be renamed to `nix/xpu.nix` (or removed if the placeholder is no longer needed since IPEX is discontinued).
4. WHEN any markdown file in `.kiro/steering/` references `pytorch_ipex`, THE content SHALL use `pytorch_xpu` instead.

### Requirement 11: Atomic Rename with Test Continuity

**User Story:** As a developer, I want the rename to be atomic so that all references are updated together and the test suite passes without regressions.

#### Acceptance Criteria

1. WHEN the rename is complete, THE existing test suite (128 spec tests from the distributed-gpu-sharding spec) SHALL pass without modification to test logic (only import paths and type names change).
2. IF a test file references `PyTorchIPEXRingInstance` or `InstanceMeta.PyTorchIPEXRing`, THEN the test file SHALL be updated to use `PyTorchXPURingInstance` and `InstanceMeta.PyTorchXPURing`.
3. WHEN `grep -r "PyTorchIPEX" src/` is run after the rename, THE result SHALL be empty (zero matches).
4. WHEN `grep -r "pytorch_ipex" src/` is run after the rename, THE result SHALL be empty (zero matches), excluding test files that explicitly verify the absence of stale references.
5. THE `nix/verify-gpu-on-startup.py` script SHALL import from `exo.worker.engines.pytorch_xpu.gpu_detector` (not `pytorch_ipex`).

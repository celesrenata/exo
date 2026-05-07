# Implementation Plan: PyTorch Model Cards & ipex → xpu Rename

## Overview

This plan has two workstreams executed in order: (1) the codebase-wide rename of `ipex` → `xpu` across Python source, Nix configs, docs, and steering files, then (2) adding new PyTorch-compatible model card TOML files. The rename is done first so that new model cards and tests reference the correct `pytorch_xpu` paths from the start. All code uses Python, matching the existing codebase.

## Tasks

- [x] 1. Rename directory and update Python module imports
  - [x] 1.1 Rename `src/exo/worker/engines/pytorch_ipex/` to `src/exo/worker/engines/pytorch_xpu/`
    - Move the entire directory (all `.py` files, `tests/` subdirectory, `README.md`, `USAGE.md`, `USAGE_TOKEN_GENERATOR.md`)
    - Rename `pytorch_ipex_backend.py` to `pytorch_xpu_backend.py` inside the new directory
    - _Requirements: 6.1_

  - [x] 1.2 Update all import paths from `exo.worker.engines.pytorch_ipex` to `exo.worker.engines.pytorch_xpu`
    - Update `src/exo/worker/runner/runner.py`: change lazy imports to `exo.worker.engines.pytorch_xpu.generator`, `.model_loader`, `.warmup`
    - Update `src/exo/worker/runner/bootstrap.py`: change import of `PyTorchIPEXRingInstance` (handled in task 2) and log messages referencing `pytorch_ipex`
    - Update `src/exo/worker/engines/factory.py`: change `"pytorch_ipex"` backend name to `"pytorch_xpu"`, update import path to `exo.worker.engines.pytorch_xpu.pytorch_xpu_backend`, rename class `PyTorchIPEXBackend` → `PyTorchXPUBackend` in registry
    - Update `src/exo/master/api.py`: change `InstanceMeta.PyTorchIPEXRing` → `InstanceMeta.PyTorchXPURing`
    - Update `nix/verify-gpu-on-startup.py`: change import from `exo.worker.engines.pytorch_ipex.gpu_detector` to `exo.worker.engines.pytorch_xpu.gpu_detector`
    - Update any internal cross-imports within `src/exo/worker/engines/pytorch_xpu/` files that reference the old module path
    - _Requirements: 6.2, 6.3, 11.5_

- [x] 2. Rename instance types, enum values, and variables
  - [x] 2.1 Rename `InstanceMeta.PyTorchIPEXRing` → `InstanceMeta.PyTorchXPURing` in `src/exo/shared/types/worker/instances.py`
    - Rename class `PyTorchIPEXRingInstance` → `PyTorchXPURingInstance`
    - Update the `Instance` type union to use `PyTorchXPURingInstance`
    - _Requirements: 7.1, 7.2, 7.5_

  - [x] 2.2 Update `src/exo/master/placement.py` to use renamed types
    - Change import to `PyTorchXPURingInstance`
    - Change `case InstanceMeta.PyTorchIPEXRing:` → `case InstanceMeta.PyTorchXPURing:`
    - Change `PyTorchIPEXRingInstance(...)` constructor call → `PyTorchXPURingInstance(...)`
    - _Requirements: 7.3_

  - [x] 2.3 Update `src/exo/worker/runner/runner.py` variable names and dispatch logic
    - Rename `is_pytorch_ipex` → `is_pytorch_xpu`
    - Change `backend_type = "pytorch_ipex"` → `backend_type = "pytorch_xpu"`
    - Change `isinstance(instance, PyTorchIPEXRingInstance)` → `isinstance(instance, PyTorchXPURingInstance)`
    - Update all `instance_type_name == "PyTorchIPEXRingInstance"` string checks → `"PyTorchXPURingInstance"`
    - Update log messages from "PyTorchIPEXRingInstance" → "PyTorchXPURingInstance"
    - Update import at top of file: `PyTorchIPEXRingInstance` → `PyTorchXPURingInstance`
    - _Requirements: 7.4, 8.1, 8.2, 8.3_

  - [x] 2.4 Update `src/exo/worker/runner/bootstrap.py` references
    - Change import to `PyTorchXPURingInstance`
    - Change `isinstance(bound_instance.instance, PyTorchIPEXRingInstance)` → `PyTorchXPURingInstance`
    - Change `os.environ["EXO_PYTORCH_IPEX_ENABLED"]` → `os.environ["EXO_PYTORCH_XPU_ENABLED"]`
    - Update all `backend_type="pytorch_ipex"` in log messages → `backend_type="pytorch_xpu"`
    - _Requirements: 8.1, 8.2, 8.3, 9.3_

- [x] 3. Checkpoint — Verify rename compiles and core tests pass
  - Ensure all Python imports resolve correctly after the rename
  - Run: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/shared/types/tests/ src/exo/master/tests/ -v --tb=short --ignore=src/exo/worker/engines/pytorch_xpu/tests/test_device_manager.py --ignore=src/exo/worker/engines/pytorch_xpu/tests/test_model_loader.py`
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Update test files for the rename
  - [x] 4.1 Update `src/exo/worker/engines/pytorch_xpu/tests/test_no_ipex_imports.py`
    - Update any internal paths that reference `pytorch_ipex` to `pytorch_xpu`
    - Ensure the grep-based stale-reference detection still works (it should now search for `pytorch_ipex` as a stale reference)
    - _Requirements: 6.4, 8.4, 11.3, 11.4_

  - [x] 4.2 Update `src/exo/master/tests/test_placement_extensions.py`
    - Change import `PyTorchIPEXRingInstance` → `PyTorchXPURingInstance`
    - Change `InstanceMeta.PyTorchIPEXRing` → `InstanceMeta.PyTorchXPURing`
    - Rename test class `TestPlaceInstancePyTorchIPEXRing` → `TestPlaceInstancePyTorchXPURing`
    - Update all `isinstance` checks and assertion messages
    - _Requirements: 11.1, 11.2_

  - [x] 4.3 Update `src/exo/worker/runner/tests/test_runner_dispatch.py`
    - Change all imports from `PyTorchIPEXRingInstance` → `PyTorchXPURingInstance`
    - Update all `patch()` target strings from `exo.worker.engines.pytorch_ipex.*` → `exo.worker.engines.pytorch_xpu.*`
    - Update `isinstance` checks and docstrings
    - _Requirements: 11.1, 11.2_

  - [x] 4.4 Update `nix/tests/test_distributed_config.py`
    - Change path references from `pytorch_ipex` → `pytorch_xpu` in the excluded file set and assertions
    - _Requirements: 11.1, 11.2_

  - [x] 4.5 Update remaining test files in `src/exo/worker/engines/pytorch_xpu/tests/`
    - Update `test_distributed_properties.py`, `test_distributed.py`, `test_gpu_detector_properties.py`, `test_gpu_detector.py`, `test_kv_cache_manager.py`, `test_token_generator.py`, `test_api_compatibility.py` — change any import paths referencing `pytorch_ipex` to `pytorch_xpu`
    - _Requirements: 11.1, 11.2_

- [x] 5. Checkpoint — Run full test suite after rename
  - Run: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/shared/types/tests/ src/exo/master/tests/ src/exo/worker/runner/tests/ nix/tests/ -v --tb=short --ignore=src/exo/worker/engines/pytorch_xpu/tests/test_device_manager.py --ignore=src/exo/worker/engines/pytorch_xpu/tests/test_model_loader.py`
  - Verify the 128 distributed-gpu-sharding spec tests still pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Update NixOS module, Nix files, and documentation
  - [x] 6.1 Update `flake.nix` NixOS module configuration
    - Rename `services.exo.intel.pytorch_ipex` → `services.exo.intel.pytorch_xpu`
    - Rename `EXO_PYTORCH_IPEX_ENABLED` → `EXO_PYTORCH_XPU_ENABLED`
    - Remove `IPEX_TILE_AS_DEVICE` environment variable entirely
    - Update description strings from "PyTorch+IPEX" → "PyTorch XPU"
    - Update devShell references: remove `IPEX_TILE_AS_DEVICE`, update echo messages
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 6.2 Rename and update Nix helper files
    - Rename `nix/ipex-xpu.nix` → `nix/xpu.nix` (or remove if placeholder)
    - Rename `nix/PYTORCH_IPEX_INSTALLATION.md` → `nix/PYTORCH_XPU_INSTALLATION.md`, update content
    - Rename `nix/README-ipex-xpu.md` → `nix/README-xpu.md`, update content
    - Rename `nix/verify-ipex-xpu.py` → `nix/verify-xpu.py`, update content
    - Rename `nix/verify-pytorch-ipex-xpu.py` → `nix/verify-pytorch-xpu-setup.py`, update content
    - Update any `flake.nix` references to these renamed files
    - _Requirements: 10.3_

  - [x] 6.3 Rename and update documentation files
    - Rename `docs/examples/nixos-pytorch-ipex-config.nix` → `docs/examples/nixos-pytorch-xpu-config.nix`, update all `pytorch_ipex` → `pytorch_xpu` inside
    - _Requirements: 10.2_

  - [x] 6.4 Rename and update steering files
    - Rename `.kiro/steering/pytorch-ipex-status.md` → `.kiro/steering/pytorch-xpu-status.md`
    - Update all internal references from `pytorch_ipex` → `pytorch_xpu`, `IPEX` → `XPU` where appropriate
    - Update file paths in content (e.g., `src/exo/worker/engines/pytorch_ipex/` → `src/exo/worker/engines/pytorch_xpu/`)
    - Update `.kiro/steering/dev-environment.md` to reference `pytorch_xpu` paths instead of `pytorch_ipex`
    - _Requirements: 10.1, 10.4_

- [x] 7. Checkpoint — Verify no stale ipex references remain
  - Run `grep -r "pytorch_ipex" src/` and verify zero matches (excluding the stale-reference test itself)
  - Run `grep -r "PyTorchIPEX" src/` and verify zero matches
  - Run `grep -r "pytorch_ipex" flake.nix` and verify zero matches
  - Run `grep -r "IPEX_TILE_AS_DEVICE" flake.nix` and verify zero matches
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Create new PyTorch model card TOML files — small models
  - [x] 8.1 Create `resources/inference_model_cards/Qwen--Qwen3.5-2B.toml`
    - `model_id = "Qwen/Qwen3.5-2B"`, `n_layers = 36`, `hidden_size = 2560`, `supports_tensor = false`
    - `tasks = ["TextGeneration"]`, `family = "qwen"`, `quantization = ""`, `capabilities = ["text", "thinking"]`
    - `storage_size.in_bytes` ≈ 4000000000
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 5.1, 5.2, 5.3_

  - [x] 8.2 Create `resources/inference_model_cards/meta-llama--Llama-3.2-1B-Instruct.toml`
    - `model_id = "meta-llama/Llama-3.2-1B-Instruct"`, `n_layers = 16`, `hidden_size = 2048`, `supports_tensor = true`
    - `tasks = ["TextGeneration"]`, `family = "llama"`, `quantization = ""`
    - `storage_size.in_bytes` ≈ 2470000000
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3_

- [x] 9. Create new PyTorch model card TOML files — medium models
  - [x] 9.1 Create `resources/inference_model_cards/Qwen--Qwen3.5-4B.toml`
    - `model_id = "Qwen/Qwen3.5-4B"`, `n_layers = 36`, `hidden_size = 3584`, `supports_tensor = false`
    - `tasks = ["TextGeneration"]`, `family = "qwen"`, `quantization = ""`, `capabilities = ["text", "thinking"]`
    - `storage_size.in_bytes` ≈ 8000000000
    - _Requirements: 2.1, 2.4, 5.1, 5.2, 5.3_

- [x] 10. Create new PyTorch model card TOML files — large models
  - [x] 10.1 Create `resources/inference_model_cards/Qwen--Qwen3.6-27B.toml`
    - `model_id = "Qwen/Qwen3.6-27B"`, `n_layers = 64`, `hidden_size = 5120`, `supports_tensor = false`
    - `tasks = ["TextGeneration"]`, `family = "qwen"`, `quantization = ""`, `capabilities = ["text", "thinking"]`
    - `storage_size.in_bytes` ≈ 57982058496
    - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

  - [x] 10.2 Create `resources/inference_model_cards/Qwen--Qwen3.6-35B-A3B.toml`
    - `model_id = "Qwen/Qwen3.6-35B-A3B"`, `n_layers = 40`, `hidden_size = 2048`, `supports_tensor = false`
    - `tasks = ["TextGeneration"]`, `family = "qwen"`, `quantization = ""`, `capabilities = ["text", "thinking"]`
    - `storage_size.in_bytes` ≈ 70000000000
    - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

  - [x] 10.3 Create `resources/inference_model_cards/zai-org--GLM-4.7-Flash.toml`
    - `model_id = "zai-org/GLM-4.7-Flash"`, `n_layers = 40`, `hidden_size = 3584`, `supports_tensor = false`
    - `tasks = ["TextGeneration"]`, `family = "glm"`, `quantization = ""`, `capabilities = ["text", "thinking"]`
    - `storage_size.in_bytes` ≈ 60000000000
    - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

  - [x] 10.4 Create `resources/inference_model_cards/hugging-quants--Meta-Llama-3.3-70B-Instruct-GPTQ-INT4.toml`
    - `model_id = "hugging-quants/Meta-Llama-3.3-70B-Instruct-GPTQ-INT4"`, `n_layers = 80`, `hidden_size = 8192`, `supports_tensor = true`
    - `tasks = ["TextGeneration"]`, `family = "llama"`, `quantization = "GPTQ-Int4"`
    - `storage_size.in_bytes` ≈ 36000000000
    - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

- [x] 11. Write model card validation tests
  - [x] 11.1 Create `src/exo/shared/models/tests/test_pytorch_model_cards.py`
    - Test that each new TOML card file loads without `ValidationError`
    - Test that small cards have `quantization == ""`  and `TextGeneration` in tasks
    - Test that GPTQ cards have `quantization` set correctly (e.g., `"GPTQ-Int4"`)
    - Test that `supports_tensor` is correct per model architecture (true for Llama, false for Qwen)
    - Test that all expected new card files exist in `resources/inference_model_cards/`
    - Test that existing MLX cards still load correctly alongside new PyTorch cards
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 5.1, 5.2_

  - [x] 11.2 Write property test for ModelCard TOML round-trip
    - **Property 1: ModelCard TOML round-trip**
    - Use Hypothesis to generate random valid `ModelCard` instances with arbitrary fields
    - Serialize via `ModelCard.save()` to a temp TOML file, re-parse via `ModelCard.load_from_path()`
    - Assert the re-parsed card equals the original (all fields preserved)
    - Minimum 100 examples per run
    - **Validates: Requirements 1.1, 5.1, 5.3, 5.4**

- [x] 12. Write stale-reference detection tests
  - [x] 12.1 Create `src/exo/shared/tests/test_xpu_rename_completeness.py`
    - Grep `src/` for `"pytorch_ipex"` — assert zero matches (excluding the test itself and `test_no_ipex_imports.py`)
    - Grep `src/` for `"PyTorchIPEX"` — assert zero matches
    - Verify `InstanceMeta.PyTorchXPURing` exists and is importable
    - Verify `PyTorchXPURingInstance` is importable from `exo.shared.types.worker.instances`
    - Verify `runner.py` uses `is_pytorch_xpu` variable name (grep or AST check)
    - Verify `bootstrap.py` sets `EXO_PYTORCH_XPU_ENABLED` (not `EXO_PYTORCH_IPEX_ENABLED`)
    - Grep `flake.nix` for `pytorch_ipex` — assert zero matches
    - Grep `flake.nix` for `IPEX_TILE_AS_DEVICE` — assert zero matches
    - _Requirements: 6.4, 8.4, 11.3, 11.4_

- [x] 13. Final checkpoint — Full test suite passes
  - Run: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/shared/types/tests/ src/exo/shared/models/tests/ src/exo/shared/tests/ src/exo/master/tests/ src/exo/worker/runner/tests/ nix/tests/ -v --tb=short --ignore=src/exo/worker/engines/pytorch_xpu/tests/test_device_manager.py --ignore=src/exo/worker/engines/pytorch_xpu/tests/test_model_loader.py`
  - Verify all new model cards parse correctly
  - Verify all stale-reference tests pass
  - Verify the distributed-gpu-sharding spec tests still pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The rename (tasks 1–7) is done before model cards (tasks 8–10) so new code references `pytorch_xpu` from the start
- Checkpoints at tasks 3, 5, 7, and 13 ensure incremental validation
- Property tests validate the TOML round-trip correctness property from the design
- Unit tests validate specific model card content and stale-reference absence
- Some pre-existing test files (`test_device_manager.py`, `test_model_loader.py`) require real PyTorch and are excluded from CI runs via `--ignore`
- The `nix/verify-gpu-on-startup.py` import path update (task 1.2) is critical for gremlin deployment
- Internal function names like `pytorch_ipex_generate` and `warmup_pytorch_ipex_inference` inside the `pytorch_xpu/` module are lower priority renames — they work correctly with the directory rename but can be cleaned up for consistency

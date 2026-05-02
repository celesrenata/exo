# Development Environment Notes

## Python & uv

- Python 3.12 is pinned in `.python-version`. The workspace has `exo` (requires `>=3.12,<3.13`), `exo-bench`, and `exo-pyo3-bindings` as workspace members.
- `uv run pytest` works after the version alignment fix. Always use `uv run` to invoke pytest.

## NixOS libstdc++ Workaround

On this NixOS workstation, numpy (and other C-extension packages) need `libstdc++.so.6` on `LD_LIBRARY_PATH`. Without it, imports fail with:

```
ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory
```

**Fix**: Prefix commands with:
```bash
LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH"
```

This is needed for any `uv run pytest` invocation that touches numpy, torch, or other C-extension packages.

**Shorthand for running tests:**
```bash
LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest <test_paths> -v --tb=short
```

## Nix Flake Status

The nix flake (`nix develop`) is currently broken due to an argument error in `nix/ipex-xpu.nix`:
```
error: function 'anonymous lambda' called with unexpected argument 'intel-compute-runtime'
```
Do NOT attempt to use `nix develop` or `nix flake check` until this is fixed.

## Test Collection Caveats

Some pre-existing test files fail to collect:
- `test_opencl.py` — calls `sys.exit(1)` at module level
- `src/exo/worker/engines/pytorch_ipex/test_kv_cache_simple.py` — broken relative import
- `src/exo/worker/engines/pytorch_ipex/tests/test_device_manager.py` — needs real torch
- `src/exo/worker/engines/pytorch_ipex/tests/test_model_loader.py` — needs real torch
- `src/exo/worker/tests/unittests/test_mlx/` — needs CUDA libs for mlx

When running the full suite, ignore these with `--ignore=` flags. When running specific test files, target them directly.

## Running Spec Tests

For the distributed-gpu-sharding spec tests specifically:
```bash
LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest \
  src/exo/shared/types/tests/test_profiling.py \
  src/exo/worker/engines/pytorch_ipex/tests/test_gpu_detector_properties.py \
  src/exo/worker/engines/pytorch_ipex/tests/test_gpu_detector.py \
  -v --tb=short
```

Add new test files to this list as they are created.

## No Git Repository

There is no `.git` directory in this workspace. Do not use git commands. The `.githooks/` directory and `.gitignore` exist but are not active.

## Type Checking

`uv run basedpyright` is configured in `pyproject.toml` with strict mode. It targets `pythonPlatform = "Darwin"` and `pythonVersion = "3.12"`. It may need the same `LD_LIBRARY_PATH` fix for numpy stubs.

fmt:
    nix fmt

lint:
    uv run ruff check --fix

test:
    uv run pytest src

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cargo run --bin stub_gen
    just sync-clean

build-dashboard:
    #!/usr/bin/env bash
    cd dashboard
    npm install
    npm run build

package:
    uv run pyinstaller packaging/pyinstaller/exo.spec

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard/node_modules
    rm -rf dashboard/.svelte-kit
    rm -rf dashboard/build

# NixOS specific commands
nix-build-cpu:
    nix build .#exo-cpu

nix-build-debug:
    ./build-debug.sh

nix-shell:
    nix develop

# Troubleshooting commands
debug-rust:
    #!/usr/bin/env bash
    echo "🦀 Rust Debug Info"
    echo "=================="
    rustc --version
    cargo --version
    echo "Cargo home: $CARGO_HOME"
    echo "Rust toolchain: $(rustup show active-toolchain 2>/dev/null || echo 'Not using rustup')"

debug-python:
    #!/usr/bin/env bash
    echo "🐍 Python Debug Info"
    echo "==================="
    python --version
    which python
    python -c "import sys; print('Python path:', sys.path)"
    echo "Pip version: $(pip --version)"

debug-env:
    #!/usr/bin/env bash
    echo "🌍 Environment Debug Info"
    echo "========================"
    echo "NIX_STORE: $NIX_STORE"
    echo "PATH: $PATH"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"

#!/usr/bin/env bash
# Debug script for EXO build issues on NixOS

set -e

echo "🔍 EXO Build Debug Script"
echo "========================="

# Check if we're in a Nix environment
if [ -n "$NIX_STORE" ]; then
    echo "✅ Running in Nix environment"
else
    echo "❌ Not in Nix environment - run with 'nix develop'"
    exit 1
fi

# Check Rust toolchain
echo "🦀 Checking Rust toolchain..."
rustc --version
cargo --version

# Check Python
echo "🐍 Checking Python..."
python --version
which python

# Check Node.js
echo "📦 Checking Node.js..."
node --version
npm --version

# Try building dashboard first
echo "🎨 Building dashboard..."
cd dashboard
if npm ci; then
    echo "✅ Dashboard dependencies installed"
else
    echo "❌ Dashboard dependency installation failed"
    exit 1
fi

if npm run build; then
    echo "✅ Dashboard built successfully"
else
    echo "❌ Dashboard build failed"
    exit 1
fi
cd ..

# Try building Rust bindings
echo "🔧 Building Rust bindings..."
cd rust/exo_pyo3_bindings

# Check if maturin is available
if command -v maturin &> /dev/null; then
    echo "✅ Maturin found"
    if maturin build --release; then
        echo "✅ Rust bindings built successfully"
    else
        echo "❌ Rust bindings build failed"
        exit 1
    fi
else
    echo "⚠️  Maturin not found, trying cargo build..."
    if cargo build --release; then
        echo "✅ Cargo build successful"
    else
        echo "❌ Cargo build failed"
        exit 1
    fi
fi

cd ../..

# Try installing Python package
echo "🐍 Installing Python package..."
if python -m pip install -e .; then
    echo "✅ Python package installed successfully"
else
    echo "❌ Python package installation failed"
    exit 1
fi

echo "🎉 All builds completed successfully!"
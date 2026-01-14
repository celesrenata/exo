{ lib
, rustPlatform
, rustToolchain ? null
, pkg-config
, openssl
, python313
, stdenv
, darwin
, libiconv
}:

rustPlatform.buildRustPackage rec {
  pname = "exo-rust-bindings";
  version = "0.3.0";

  src = lib.cleanSource ../../.;

  cargoLock = {
    lockFile = ../../Cargo.lock;
  };

  # Only build from the rust directory
  sourceRoot = "source/rust";

  # Copy Cargo.lock to the rust directory during patch phase
  postPatch = ''
    cp ../Cargo.lock .
    # Ensure target directory is writable
    chmod -R u+w .
  '';

  nativeBuildInputs = [
    pkg-config
    python313
  ] ++ lib.optionals (rustToolchain != null) [
    rustToolchain
  ];

  # Use nightly Rust for PyO3 features
  RUSTC_BOOTSTRAP = "1";

  buildInputs = [
    openssl
    python313
  ] ++ lib.optionals stdenv.isDarwin [
    darwin.apple_sdk.frameworks.Security
    darwin.apple_sdk.frameworks.SystemConfiguration
    libiconv
  ];

  # Build only the Python bindings
  cargoBuildFlags = [ "--package" "exo_pyo3_bindings" ];

  # Prepare build environment
  preBuild = ''
    export CARGO_TARGET_DIR="$PWD/target"
    mkdir -p "$CARGO_TARGET_DIR"
  '';

  # Set environment variables for PyO3 and cross-compilation
  env = {
    PYO3_PYTHON = "${python313}/bin/python";
    OPENSSL_NO_VENDOR = "1";
    PKG_CONFIG_PATH = "${openssl.dev}/lib/pkgconfig";
  } // lib.optionalAttrs (stdenv.buildPlatform != stdenv.hostPlatform) {
    # Cross-compilation environment variables
    PYO3_CROSS_LIB_DIR = "${python313}/lib";
    PYO3_CROSS_PYTHON_VERSION = "3.13";
  };

  # Configure cross-compilation
  CARGO_BUILD_TARGET = lib.optionalString (stdenv.buildPlatform != stdenv.hostPlatform)
    stdenv.hostPlatform.rust.rustcTargetSpec;

  # Install the shared library for Python
  postInstall = ''
    mkdir -p $out/lib/python3.13/site-packages
    
    # Handle different library extensions based on platform
    if [ -f target/release/libexo_pyo3_bindings.so ]; then
      cp target/release/libexo_pyo3_bindings.so $out/lib/python3.13/site-packages/exo_pyo3_bindings.so
    elif [ -f target/release/libexo_pyo3_bindings.dylib ]; then
      cp target/release/libexo_pyo3_bindings.dylib $out/lib/python3.13/site-packages/exo_pyo3_bindings.so
    elif [ -f target/${stdenv.hostPlatform.rust.rustcTargetSpec or "release"}/release/libexo_pyo3_bindings.so ]; then
      cp target/${stdenv.hostPlatform.rust.rustcTargetSpec}/release/libexo_pyo3_bindings.so $out/lib/python3.13/site-packages/exo_pyo3_bindings.so
    elif [ -f target/${stdenv.hostPlatform.rust.rustcTargetSpec or "release"}/release/libexo_pyo3_bindings.dylib ]; then
      cp target/${stdenv.hostPlatform.rust.rustcTargetSpec}/release/libexo_pyo3_bindings.dylib $out/lib/python3.13/site-packages/exo_pyo3_bindings.so
    else
      echo "Error: Could not find compiled library"
      find target -name "*exo_pyo3_bindings*" -type f
      exit 1
    fi
    
    # Also install stub files if they exist
    if [ -f rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi ]; then
      cp rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi $out/lib/python3.13/site-packages/
    fi
  '';

  meta = with lib; {
    description = "Rust networking bindings for EXO distributed AI inference";
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.unix;
    # Ensure we support the architectures mentioned in requirements
    badPlatforms = [ ];
  };
}

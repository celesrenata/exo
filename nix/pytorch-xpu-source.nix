# PyTorch 2.11.0 built from source with XPU + ibverbs (RDMA) support
#
# The pre-built XPU wheel from Intel does NOT include Gloo ibverbs transport.
# This derivation builds PyTorch from source with:
#   USE_XPU=1        — Intel Arc GPU support via SYCL/Level Zero
#   USE_IBVERBS=1    — Gloo RDMA transport via libibverbs (SIW/RoCE/iWARP)
#   USE_GLOO=1       — Gloo collective communications
#   USE_DISTRIBUTED=1 — torch.distributed
#
# Build time: ~2-4 hours on a 16-core machine.
# Run on gremlin-1 (Intel Core Ultra 9 185H, 16C/22T).
#
# Usage in flake.nix:
#   pytorch-xpu = pkgsExo.python313.pkgs.callPackage ./nix/pytorch-xpu-source.nix { };

{ lib
, buildPythonPackage
, fetchFromGitHub
, python
, cmake
, ninja
, which
, pkg-config
, git
, patchelf
  # Build dependencies
, rdma-core
, libnl
, numactl
  # Intel GPU runtime
, intel-compute-runtime
, level-zero
  # oneAPI
, mkl
, oneDNN
, onetbb
  # System
, stdenv
, glibcLocales
  # Python deps
, filelock
, fsspec
, jinja2
, networkx
, numpy
, pyyaml
, setuptools
, sympy
, typing-extensions
, pybind11
}:

buildPythonPackage rec {
  pname = "torch";
  version = "2.11.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "pytorch";
    repo = "pytorch";
    rev = "v${version}";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # TODO: fill after first build attempt
    fetchSubmodules = true;
  };

  LOCALE_ARCHIVE = "${glibcLocales}/lib/locale/locale-archive";
  LC_ALL = "en_US.UTF-8";

  nativeBuildInputs = [
    cmake
    ninja
    which
    pkg-config
    git
    patchelf
    glibcLocales
    python.pkgs.setuptools
    python.pkgs.wheel
  ];

  buildInputs = [
    # RDMA support
    rdma-core
    libnl
    numactl
    # Intel GPU
    intel-compute-runtime
    level-zero
    # oneAPI math libs
    mkl
    oneDNN
    onetbb
    # System
    stdenv.cc.cc.lib
    pybind11
  ];

  propagatedBuildInputs = [
    filelock
    fsspec
    jinja2
    networkx
    numpy
    pyyaml
    setuptools
    sympy
    typing-extensions
  ];

  # CMake build flags
  CMAKE_BUILD_TYPE = "Release";

  preConfigure = ''
    export MAX_JOBS=$NIX_BUILD_CORES
    export BUILD_TEST=0

    # Core features
    export USE_DISTRIBUTED=1
    export USE_GLOO=1
    export USE_IBVERBS=1
    export USE_XPU=1

    # Disable unused backends to speed up build
    export USE_CUDA=0
    export USE_CUDNN=0
    export USE_NCCL=0
    export USE_ROCM=0
    export USE_MPS=0
    export USE_MKLDNN=1
    export USE_FBGEMM=0
    export USE_NNPACK=0
    export USE_QNNPACK=0
    export USE_XNNPACK=0
    export USE_KINETO=0
    export USE_TENSORPIPE=0

    # Point to rdma-core headers and libs
    export IBVERBS_INCLUDE_DIR="${rdma-core}/include"
    export IBVERBS_LIBRARY="${rdma-core}/lib/libibverbs.so"

    # Intel oneAPI paths
    export INTEL_MKL_DIR="${mkl}"
    export DNNL_DIR="${oneDNN}"
    export TBB_DIR="${onetbb}"
  '';

  # Skip tests — they require GPU hardware
  doCheck = false;
  dontCheck = true;
  pythonImportsCheck = [ ];

  postFixup = ''
    # Add runtime library paths
    find $out -name "*.so*" -type f | while read lib; do
      patchelf --add-rpath ${lib.makeLibraryPath [
        rdma-core
        intel-compute-runtime
        level-zero
        mkl
        oneDNN
        onetbb
        stdenv.cc.cc.lib
      ]} "$lib" 2>/dev/null || true
    done
  '';

  meta = with lib; {
    description = "PyTorch 2.11.0 with Intel XPU + Gloo ibverbs (RDMA) support — built from source";
    homepage = "https://pytorch.org/";
    license = licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}

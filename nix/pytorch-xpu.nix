{ lib
, buildPythonPackage
, fetchurl
, python
, autoPatchelfHook
, addDriverRunpath
, stdenv
, glibcLocales
  # Intel GPU runtime dependencies
, intel-compute-runtime
, level-zero
  # oneAPI dependencies
, mkl
, oneDNN
, onetbb
  # Python dependencies
, filelock
, fsspec
, jinja2
, networkx
, numpy
, pyyaml
, setuptools
, sympy
, typing-extensions
}:

# PyTorch 2.11.0 with Intel XPU (Arc GPU) support — native torch.xpu
buildPythonPackage rec {
  pname = "torch";
  version = "2.11.0+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/xpu/torch-2.11.0%2Bxpu-cp313-cp313-linux_x86_64.whl";
    hash = "sha256-x8KZcJJCc3w3ZGctcQTWwmq36Qbvbr1NhAvUVjl5EvQ=";
  };

  LOCALE_ARCHIVE = "${glibcLocales}/lib/locale/locale-archive";
  LC_ALL = "en_US.UTF-8";

  nativeBuildInputs = [
    autoPatchelfHook
    addDriverRunpath
    glibcLocales
  ];

  # These libraries are provided at runtime by pip-installed Intel oneAPI packages
  # (intel-cmplr-lib-rt, onemkl-sycl-*, oneccl) — not available in the Nix sandbox
  autoPatchelfIgnoreMissingDeps = [
    "libsycl.so.8"
    "libpti_view.so.0"
    "libccl.so.1"
    "libmkl_sycl_blas.so.5"
    "libmkl_sycl_dft.so.5"
    "libmkl_sycl_lapack.so.5"
    "libsvml.so"
    "libirng.so"
    "libimf.so"
    "libintlc.so.5"
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    # Intel GPU runtime
    intel-compute-runtime
    level-zero
    # oneAPI libraries
    mkl
    oneDNN
    onetbb
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

  postFixup = ''
    # Add Intel runtime libraries to RPATH
    find $out -name "*.so*" -type f | while read lib; do
      addDriverRunpath "''${lib}"
      patchelf --add-rpath ${lib.makeLibraryPath [ 
        intel-compute-runtime 
        level-zero 
        mkl 
        oneDNN 
        onetbb 
        stdenv.cc.cc.lib 
      ]} "''${lib}" 2>/dev/null || true
    done
  '';

  # Skip imports check - requires GPU
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "PyTorch 2.11.0 with Intel XPU (Arc GPU) support";
    homepage = "https://pytorch.org/";
    license = licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}

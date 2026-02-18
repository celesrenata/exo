{ lib
, buildPythonPackage
, fetchurl
, python
, autoPatchelfHook
, addDriverRunpath
, stdenv
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

# PyTorch 2.6.0 with Intel XPU (Arc GPU) support - Pre-built wheel from Intel
#
# Using PyTorch 2.6.0+xpu because:
# - Python 3.12 is required (3.13 not supported by Intel yet)
# - PyTorch 2.5.x+xpu wheels are not available for Python 3.12
# - PyTorch 2.6.0+xpu is the earliest stable version with cp312 wheels
# - Matches with IPEX 2.6.10+xpu

buildPythonPackage rec {
  pname = "torch";
  version = "2.6.0+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/xpu/torch-${version}-cp312-cp312-linux_x86_64.whl";
    hash = "sha256-xMXGdiXK88NXZcK5Th/hZuPjP0pUUhshJaWtG+sLD8I=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
    addDriverRunpath
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
    description = "PyTorch 2.6.0 with Intel XPU (Arc GPU) support - pre-built wheel";
    homepage = "https://pytorch.org/";
    license = licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}

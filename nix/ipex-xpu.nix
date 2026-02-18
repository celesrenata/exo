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
  # PyTorch dependency (must be XPU-enabled)
, pytorch-xpu
  # Python dependencies
, numpy
, pyyaml
, typing-extensions
, setuptools
, psutil
}:

# Intel Extension for PyTorch (IPEX) 2.6.10 with XPU support
#
# Using IPEX 2.6.10+xpu because:
# - Matches PyTorch 2.6.0+xpu (major.minor must match)
# - Python 3.12 support (cp312 wheels available)
# - Stable release with Intel Arc GPU optimizations

buildPythonPackage rec {
  pname = "intel-extension-for-pytorch";
  version = "2.6.10+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-${version}-cp312-cp312-linux_x86_64.whl";
    # Hash will be computed on first build
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
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
    pytorch-xpu
    numpy
    pyyaml
    typing-extensions
    setuptools
    psutil
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
    description = "Intel Extension for PyTorch 2.6.10 with XPU (Arc GPU) support";
    homepage = "https://intel.github.io/intel-extension-for-pytorch/";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}

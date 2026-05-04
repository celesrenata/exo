# Intel oneAPI Runtime Libraries
#
# Provides the shared libraries needed by PyTorch 2.11+xpu at runtime:
#   - libsycl.so.8 (from intel-sycl-rt)
#   - libur_loader.so.0 (from intel-cmplr-lib-ur)
#   - libpti_view.so.0 (from intel-pti)
#   - libsvml.so, libirng.so, libimf.so, libintlc.so.5 (from intel-cmplr-lib-rt)
#   - libccl.so.1 (from oneccl)
#   - libmkl_sycl_blas.so.5 (from onemkl-sycl-blas)
#   - libmkl_sycl_dft.so.5 (from onemkl-sycl-dft)
#   - libmkl_sycl_lapack.so.5 (from onemkl-sycl-lapack)
#
# These are extracted from PyPI wheels and combined into a single output
# with $out/lib containing all shared libraries.

{ lib
, stdenv
, fetchurl
, autoPatchelfHook
, unzip
, zlib
}:

let
  # Intel SYCL Runtime — provides libsycl.so.8
  intel-sycl-rt = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/i/intel_sycl_rt/intel_sycl_rt-2025.3.3-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-++9fUWNJK+FPZe2QD2m0U0Mr7D0/P2Tl5VF1i/pM7N8=";
  };

  # Intel Unified Runtime — provides libur_loader.so.0, libur_adapter_level_zero.so
  # Required by PyTorch XPU for device discovery via Level Zero backend
  intel-cmplr-lib-ur = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/i/intel_cmplr_lib_ur/intel_cmplr_lib_ur-2025.3.3-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-oYzq3TNq0vgvR5dyFN3JRWjFoz4mLuO50bNjG3iXtmk=";
  };

  # Intel Compiler Runtime — provides libsvml.so, libirng.so, libimf.so, libintlc.so.5
  intel-cmplr-lib-rt = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/i/intel_cmplr_lib_rt/intel_cmplr_lib_rt-2025.3.3-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-yQTuEgxOak6uPEb5BKMrmPNkC5oI0LLP0LgneDWkzrs=";
  };

  # Intel PTI — provides libpti_view.so.0
  intel-pti = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/i/intel_pti/intel_pti-0.17.0-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-GjMnuGg69y5g4eqPdUFg+xL/uKFeV5QffjZyvMVA4vw=";
  };

  # oneCCL — provides libccl.so.1 (needed for distributed collective communications)
  # Using 2021.17.2 which links against libsycl.so.8 (matching PyTorch 2.11)
  oneccl = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/o/oneccl/oneccl-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-p3wsVu66HHg3mlMKK6K70zQH9kQLREdvgyktFpQQA3M=";
  };

  # oneMKL SYCL BLAS — provides libmkl_sycl_blas.so.5
  onemkl-sycl-blas = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/o/onemkl_sycl_blas/onemkl_sycl_blas-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-uVbBTbRJkFec83brcrpgELDIOuJ/wTL2Z1KxszfZ5yg=";
  };

  # oneMKL SYCL DFT — provides libmkl_sycl_dft.so.5
  onemkl-sycl-dft = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/o/onemkl_sycl_dft/onemkl_sycl_dft-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-OfQ6M00Fq1mJsizrRiQ+rM5FcdAWyfZ5Ixef9OQnuIo=";
  };

  # oneMKL SYCL LAPACK — provides libmkl_sycl_lapack.so.5
  onemkl-sycl-lapack = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/o/onemkl_sycl_lapack/onemkl_sycl_lapack-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Ng/GNrwIMtWi7i08T0QBRexD2UwXCjUdWIqa96hISc4=";
  };

  # MKL core — provides libmkl_core.so.2, libmkl_intel_lp64.so.2, etc.
  # Required by onemkl-sycl-* packages which are just SYCL dispatch wrappers
  mkl-core = fetchurl {
    url = "https://files.pythonhosted.org/packages/py2.py3/m/mkl/mkl-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-2zHln6No3U+kW0lDUfS34OYgSwjX2yeDYRjE4TcOsBE=";
  };

in
stdenv.mkDerivation {
  pname = "intel-oneapi-runtime";
  version = "2025.3.3";

  dontUnpack = true;

  nativeBuildInputs = [ autoPatchelfHook unzip ];

  # Libraries that the Intel runtime libs themselves depend on
  buildInputs = [
    stdenv.cc.cc.lib # libstdc++
    zlib
  ];

  # Some Intel libs have circular deps or need runtime-only libs — ignore them
  autoPatchelfIgnoreMissingDeps = [
    "libze_loader.so.1"
    "libze_tracing_layer.so.1"
    "libmpi.so.12"
    "libmpicxx.so.12"
    "libmpifort.so.12"
    "libfabric.so.1"
    "libumf.so.1"
    "libiomp5.so"
    "libOpenCL.so.1"
    "libtbb.so.12"
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib

    # Extract .so files from each wheel's data directory
    for wheel in ${intel-sycl-rt} ${intel-cmplr-lib-ur} ${intel-cmplr-lib-rt} ${intel-pti} ${oneccl} ${onemkl-sycl-blas} ${onemkl-sycl-dft} ${onemkl-sycl-lapack} ${mkl-core}; do
      echo "Extracting from: $wheel"
      ${stdenv.shell} -c "unzip -o -j '$wheel' '*.data/data/lib/*.so*' -d $out/lib/ 2>/dev/null || true"
    done

    # Remove static libraries and other non-essential files
    find $out/lib -name '*.a' -delete
    find $out/lib -name '*.py' -delete
    find $out/lib -name '*.bc' -delete
    find $out/lib -name '*.o' -delete
    find $out/lib -name '*.new.o' -delete

    runHook postInstall
  '';

  meta = with lib; {
    description = "Intel oneAPI runtime libraries for PyTorch XPU support";
    homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html";
    license = licenses.unfree;
    platforms = [ "x86_64-linux" ];
  };
}

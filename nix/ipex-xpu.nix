{ lib
, intel-compute-runtime ? null
, level-zero ? null
, mkl ? null
, oneDNN ? null
, onetbb ? null
, pytorch-xpu ? null
}:

# Intel Extension for PyTorch (IPEX) with XPU Support - Pip Installation Required
#
# IMPORTANT: Intel's IPEX XPU wheels cannot be fetched directly in Nix
# due to authentication/access restrictions on their download server.
#
# This derivation is a placeholder. Users must install IPEX with XPU
# support via pip at runtime.
#
# Installation instructions:
#   pip install intel-extension-for-pytorch==2.6.10+xpu \
#     --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
#
# See nix/PYTORCH_IPEX_INSTALLATION.md for complete instructions.

# Return null as placeholder
# Users who need IPEX XPU support should install via pip as documented
null

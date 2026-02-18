{ lib
, python3Packages
}:

# PyTorch with Intel XPU (Arc GPU) Support - Pip Installation Required
#
# IMPORTANT: Intel's PyTorch XPU wheels cannot be fetched directly in Nix
# due to authentication/access restrictions on their download server.
#
# This derivation returns standard PyTorch from nixpkgs as a placeholder.
# Users must install PyTorch+IPEX with XPU support via pip at runtime.
#
# Installation instructions:
#   pip install torch==2.6.0+xpu torchvision==0.20.1+xpu \
#     --index-url https://download.pytorch.org/whl/xpu
#
# See nix/PYTORCH_IPEX_INSTALLATION.md for complete instructions.

# Return standard PyTorch from nixpkgs
# Users who need XPU support should install via pip as documented
python3Packages.pytorch

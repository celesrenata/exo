{ lib
, intel-compute-runtime ? null
, level-zero ? null
, mkl ? null
, oneDNN ? null
, onetbb ? null
, pytorch-xpu ? null
}:

# PyTorch XPU Support - Native torch.xpu (PyTorch 2.11+)
#
# IMPORTANT: IPEX (Intel Extension for PyTorch) is discontinued.
# PyTorch 2.11+ includes native XPU support via torch.xpu.
#
# This derivation is a placeholder. Users should install PyTorch
# with XPU support via pip at runtime.
#
# Installation instructions:
#   pip install torch --index-url https://download.pytorch.org/whl/xpu
#
# See nix/PYTORCH_XPU_INSTALLATION.md for complete instructions.

# Return null as placeholder
# Users who need PyTorch XPU support should install via pip as documented
null

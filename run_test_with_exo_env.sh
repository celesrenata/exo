#!/usr/bin/env bash
# Run validation test using the exact same environment as exo service

echo "Running Llama transformer validation with exo environment..."
echo ""

# Set the same environment variables as exo service
export PYTHONNOUSERSITE='true'
export EXO_TINYGRAD_ENABLED='true'
export EXO_RESOURCES_DIR='/nix/store/1ia0zhn2i6a2and3kgjb3hqrv2ysahhs-source/resources'
export EXO_DASHBOARD_DIR='/nix/store/1xn9ghlpky7iaf0ny76xi45ck0skdmp9-exo-dashboard'

# Build PYTHONPATH with all the dependencies that exo uses
export PYTHONPATH="/nix/store/jrsxpngcaif0rxflm6bax7pi1cd1hhyd-exo-0.3.0/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/74lsy15mvdbsnn40jjr9w95y17mm8b0v-python3.13-numpy-2.3.5/lib/python3.13/site-packages"
export PYTHONPATH="$PYTHONPATH:/nix/store/xg0j9lib60xlszacmivg00nfw20bdybd-python3.13-tinygrad-0.12.0/lib/python3.13/site-packages"

# Find loguru
LOGURU_PATH=$(find /nix/store -maxdepth 1 -name '*loguru*' -type d 2>/dev/null | head -1)
if [ -n "$LOGURU_PATH" ]; then
  export PYTHONPATH="$PYTHONPATH:$LOGURU_PATH/lib/python3.13/site-packages"
fi

# Use the python that exo uses
PYTHON=/nix/store/qzc04a3npl70cyyy6flnnrb2ig3kayxm-python3-3.13.11/bin/python3.13

echo "PYTHONPATH set, running test..."
echo ""

# Run our test
exec $PYTHON /tmp/test_llama_validation.py

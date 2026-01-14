# Simple test to verify the testing infrastructure works
{ lib, pkgs, system }:

pkgs.runCommand "test-simple"
{
  nativeBuildInputs = [ pkgs.bash pkgs.coreutils ];
} ''
  set -euo pipefail
  
  echo "=== Simple EXO Test ==="
  echo "System: ${system}"
  echo "Date: $(date)"
  echo
  
  # Test basic functionality
  echo "Testing basic shell functionality..."
  
  # Test that we can run basic commands
  if command -v echo >/dev/null 2>&1; then
    echo "✓ echo command available"
  else
    echo "ERROR: echo command not available"
    exit 1
  fi
  
  if command -v date >/dev/null 2>&1; then
    echo "✓ date command available"
  else
    echo "ERROR: date command not available"
    exit 1
  fi
  
  # Test that we can create files
  test_file="/tmp/exo_test_file"
  echo "test content" > "$test_file"
  
  if [ -f "$test_file" ]; then
    echo "✓ File creation successful"
  else
    echo "ERROR: File creation failed"
    exit 1
  fi
  
  # Test that we can read files
  content=$(cat "$test_file")
  if [ "$content" = "test content" ]; then
    echo "✓ File reading successful"
  else
    echo "ERROR: File reading failed"
    exit 1
  fi
  
  # Clean up
  rm -f "$test_file"
  
  echo "✓ Simple test completed successfully"
  touch $out
''

#!/usr/bin/env python3
"""
Debug script to investigate why runner 4fdc56d1-0a9e-49f0-88ea-f119d7df852b keeps terminating
"""

import json
import subprocess
import sys
import time
from datetime import datetime

def check_model_paths():
    """Check if model files exist in expected locations."""
    print("=== Model Path Analysis ===")
    
    # Check both potential locations
    paths_to_check = [
        "/var/lib/exo/models/microsoft--DialoGPT-medium",
        "/var/lib/exo/exo/models/microsoft--DialoGPT-medium",
        "/var/lib/models/microsoft--DialoGPT-medium"
    ]
    
    for path in paths_to_check:
        try:
            result = subprocess.run(['ssh', 'gremlin-1', f'ls -la {path}'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Found model at: {path}")
                print(f"   Contents: {result.stdout.strip()}")
            else:
                print(f"❌ No model at: {path}")
        except Exception as e:
            print(f"❌ Error checking {path}: {e}")

def check_environment_variables():
    """Check EXO environment variables on gremlin-1."""
    print("\n=== Environment Variables ===")
    
    env_vars = [
        'XDG_DATA_HOME',
        'XDG_CACHE_HOME', 
        'EXO_HOME',
        'EXO_MODELS_DIR',
        'HOME'
    ]
    
    for var in env_vars:
        try:
            result = subprocess.run(['ssh', 'gremlin-1', f'systemctl show exo --property=Environment | grep {var}'], 
                                  capture_output=True, text=True, timeout=10)
            if result.stdout.strip():
                print(f"{var}: {result.stdout.strip()}")
            else:
                print(f"{var}: Not set")
        except Exception as e:
            print(f"Error checking {var}: {e}")

def check_system_resources():
    """Check system resources on gremlin-1."""
    print("\n=== System Resources on gremlin-1 ===")
    
    commands = {
        'Memory': 'free -h',
        'Disk Space': 'df -h /var/lib/exo',
        'CPU Load': 'uptime',
        'Process Count': 'ps aux | wc -l',
        'EXO Processes': 'ps aux | grep -E "(exo|python)" | grep -v grep'
    }
    
    for name, cmd in commands.items():
        try:
            result = subprocess.run(['ssh', 'gremlin-1', cmd], 
                                  capture_output=True, text=True, timeout=10)
            print(f"\n{name}:")
            print(result.stdout.strip())
        except Exception as e:
            print(f"Error checking {name}: {e}")

def check_torch_installation():
    """Check if PyTorch is properly installed and working."""
    print("\n=== PyTorch Installation Check ===")
    
    python_check = '''
import sys
print(f"Python version: {sys.version}")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU device: {torch.device('cpu')}")
    # Try to create a simple tensor
    x = torch.randn(2, 3)
    print(f"Tensor creation successful: {x.shape}")
except Exception as e:
    print(f"PyTorch error: {e}")
'''
    
    try:
        result = subprocess.run(['ssh', 'gremlin-1', f'cd /var/lib/exo && python3 -c "{python_check}"'], 
                              capture_output=True, text=True, timeout=30)
        print("PyTorch check output:")
        print(result.stdout)
        if result.stderr:
            print("PyTorch check errors:")
            print(result.stderr)
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

def monitor_runner_startup():
    """Monitor the next runner startup attempt."""
    print("\n=== Monitoring Next Runner Startup ===")
    print("Watching logs for runner startup...")
    
    try:
        # Start monitoring logs
        process = subprocess.Popen(['ssh', 'gremlin-1', 'journalctl -u exo -f --no-pager'], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, bufsize=1)
        
        start_time = time.time()
        timeout = 60  # Monitor for 1 minute
        
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                # Look for relevant log entries
                if any(keyword in line.lower() for keyword in ['runner', '4fdc56d1', 'torch', 'error', 'terminated']):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
            
            # Check if process is still running
            if process.poll() is not None:
                break
                
        process.terminate()
        
    except Exception as e:
        print(f"Error monitoring logs: {e}")

def get_runner_status():
    """Get current runner status from API."""
    print("\n=== Current Runner Status ===")
    
    try:
        result = subprocess.run(['curl', '-s', 'http://10.1.1.12:52415/state'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Find our specific runner
            runner_id = "4fdc56d1-0a9e-49f0-88ea-f119d7df852b"
            runner_status = data.get('runners', {}).get(runner_id, {})
            
            if runner_status:
                status_type = list(runner_status.keys())[0]
                status_data = list(runner_status.values())[0]
                
                print(f"Runner {runner_id}:")
                print(f"  Status: {status_type}")
                
                if isinstance(status_data, dict):
                    for key, value in status_data.items():
                        print(f"  {key}: {value}")
            else:
                print(f"Runner {runner_id} not found in current state")
                
        else:
            print(f"Failed to get API state: {result.stderr}")
            
    except Exception as e:
        print(f"Error getting runner status: {e}")

def suggest_fixes():
    """Suggest potential fixes based on analysis."""
    print("\n=== Suggested Fixes ===")
    
    print("1. Model Path Fix:")
    print("   The issue might be related to model path configuration.")
    print("   Check if XDG_DATA_HOME should be '/var/lib' or '/var/lib/exo'")
    
    print("\n2. Memory/Resource Fix:")
    print("   If the runner is being killed due to resource constraints:")
    print("   - Increase memory limits")
    print("   - Check for memory leaks")
    print("   - Monitor system resources during startup")
    
    print("\n3. PyTorch Fix:")
    print("   If PyTorch initialization is failing:")
    print("   - Verify PyTorch installation")
    print("   - Check for conflicting dependencies")
    print("   - Test PyTorch functionality manually")
    
    print("\n4. Race Condition Fix:")
    print("   If this is a timing issue:")
    print("   - Add delays between startup attempts")
    print("   - Improve error handling in runner supervisor")
    print("   - Check for resource conflicts between runners")

def main():
    """Main debugging function."""
    print("EXO Runner Termination Debug Analysis")
    print("=" * 50)
    print(f"Analysis time: {datetime.now()}")
    print(f"Target runner: 4fdc56d1-0a9e-49f0-88ea-f119d7df852b")
    print(f"Target node: gremlin-1")
    
    get_runner_status()
    check_model_paths()
    check_environment_variables()
    check_system_resources()
    check_torch_installation()
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("Analysis complete.")
    
    # Ask if user wants to monitor next startup
    try:
        response = input("\nMonitor next runner startup? (y/n): ")
        if response.lower() == 'y':
            monitor_runner_startup()
    except KeyboardInterrupt:
        print("\nMonitoring cancelled.")

if __name__ == "__main__":
    main()
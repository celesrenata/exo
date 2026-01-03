#!/usr/bin/env python3
"""
Debug script to understand why bootstrap is failing.
"""

import sys
import os
import json
import subprocess
import time

def check_runner_status():
    """Check current runner status."""
    print("🔍 Checking runner status...")
    
    try:
        result = subprocess.run([
            "curl", "-s", "http://10.1.1.12:52415/state"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"❌ Failed to get state: {result.stderr}")
            return {}, {}
            
        state = json.loads(result.stdout)
        
        runners = state.get("runners", {})
        instances = state.get("instances", {})
        
        print(f"📊 Found {len(runners)} runners, {len(instances)} instances")
        
        for runner_id, runner_status in runners.items():
            print(f"\n🔧 Runner {runner_id[:8]}...")
            if "RunnerFailed" in runner_status:
                error_msg = runner_status["RunnerFailed"]["errorMessage"]
                print(f"   ❌ FAILED: {error_msg}")
            else:
                print(f"   ✅ Status: {list(runner_status.keys())[0]}")
        
        # Check what model is being loaded
        for instance_id, instance_data in instances.items():
            print(f"\n📦 Instance {instance_id[:8]}...")
            if "CpuRingInstance" in instance_data:
                model_id = instance_data["CpuRingInstance"]["shardAssignments"]["modelId"]
                print(f"   📋 Model: {model_id}")
                
                shard_assignments = instance_data["CpuRingInstance"]["shardAssignments"]["runnerToShard"]
                for runner_id, shard_info in shard_assignments.items():
                    device_rank = shard_info["PipelineShardMetadata"]["deviceRank"]
                    start_layer = shard_info["PipelineShardMetadata"]["startLayer"]
                    end_layer = shard_info["PipelineShardMetadata"]["endLayer"]
                    print(f"   🎯 Runner {runner_id[:8]}: rank {device_rank}, layers {start_layer}-{end_layer}")
        
        return runners, instances
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return {}, {}

def check_model_availability():
    """Check if the model files are available."""
    print("\n🔍 Checking model availability...")
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_path = f"/home/celes/.local/share/exo/models/{model_id.replace('/', '--')}"
    
    print(f"📁 Expected model path: {model_path}")
    
    if os.path.exists(model_path):
        print("✅ Model directory exists")
        files = os.listdir(model_path)
        print(f"   📄 Files: {len(files)} files found")
        for f in sorted(files)[:10]:  # Show first 10 files
            print(f"      - {f}")
        if len(files) > 10:
            print(f"      ... and {len(files) - 10} more files")
    else:
        print("❌ Model directory does not exist")
        print("   This could be why bootstrap is failing!")
        
        # Check parent directory
        parent_dir = os.path.dirname(model_path)
        if os.path.exists(parent_dir):
            print(f"   📁 Parent directory exists: {parent_dir}")
            models = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
            print(f"   📋 Available models: {models}")
        else:
            print(f"   ❌ Parent directory doesn't exist: {parent_dir}")

def check_system_resources():
    """Check system resources."""
    print("\n🔍 Checking system resources...")
    
    try:
        # Check memory
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if 'MemTotal:' in line:
                total_mem = int(line.split()[1]) // 1024  # Convert to MB
                print(f"💾 Total Memory: {total_mem} MB ({total_mem/1024:.1f} GB)")
            elif 'MemAvailable:' in line:
                avail_mem = int(line.split()[1]) // 1024  # Convert to MB
                print(f"💾 Available Memory: {avail_mem} MB ({avail_mem/1024:.1f} GB)")
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"💽 Disk Space: {free//1024//1024//1024} GB free of {total//1024//1024//1024} GB total")
        
    except Exception as e:
        print(f"❌ Error checking resources: {e}")

def check_recent_logs():
    """Check recent logs for bootstrap errors."""
    print("\n🔍 Checking recent logs...")
    
    try:
        import subprocess
        
        # Get recent logs with bootstrap-related errors
        result = subprocess.run([
            "journalctl", "-u", "exo", "--since", "5 minutes ago", 
            "--grep", "bootstrap|Bootstrap|failed|error|ERROR|exception|Exception"
        ], capture_output=True, text=True, timeout=10)
        
        if result.stdout:
            print("📋 Recent error logs:")
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:  # Show last 20 lines
                if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'bootstrap']):
                    print(f"   🔴 {line}")
        else:
            print("✅ No recent error logs found")
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")

def suggest_fixes():
    """Suggest potential fixes."""
    print("\n💡 Potential fixes:")
    print("1. 🔄 Restart the EXO service: sudo systemctl restart exo")
    print("2. 📥 Download the model manually: exo download meta-llama/Llama-3.1-8B-Instruct")
    print("3. 🧹 Clear model cache: rm -rf ~/.local/share/exo/models/meta-llama--Llama-3.1-8B-Instruct")
    print("4. 🔍 Check logs in detail: journalctl -u exo -f")
    print("5. 🎯 Try a smaller model: curl -X POST http://10.1.1.12:52415/v1/chat/completions -d '{\"model\":\"microsoft/DialoGPT-medium\",...}'")

def main():
    """Main diagnostic function."""
    print("🔧 EXO Bootstrap Failure Diagnostic")
    print("=" * 50)
    
    # Check runner status
    runners, instances = check_runner_status()
    
    # Check model availability
    check_model_availability()
    
    # Check system resources
    check_system_resources()
    
    # Check recent logs
    check_recent_logs()
    
    # Suggest fixes
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("🎯 Summary:")
    
    failed_runners = sum(1 for status in runners.values() if "RunnerFailed" in status)
    if failed_runners > 0:
        print(f"❌ {failed_runners} runners are failing during bootstrap")
        print("   Most likely cause: Model files not available or corrupted")
        print("   Recommended action: Download the model manually or try a different model")
    else:
        print("✅ No obvious issues found")

if __name__ == "__main__":
    main()
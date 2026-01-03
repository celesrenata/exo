#!/usr/bin/env python3
"""
Debug script for investigating runner failure in instance B6DAE586
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta

def get_instance_details():
    """Get detailed information about the failing instance."""
    print("=== Instance B6DAE586 Details ===")
    
    try:
        result = subprocess.run(['curl', '-s', 'http://10.1.1.12:52415/state'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"Failed to fetch state: {result.stderr}")
            return None
            
        data = json.loads(result.stdout)
        
        # Find the instance
        instance_id = "b6dae586-076c-474e-ac90-f610d5ebd937"
        instance = data.get('instances', {}).get(instance_id)
        
        if not instance:
            print(f"Instance {instance_id} not found")
            return None
            
        print(f"Instance ID: {instance_id}")
        print(f"Type: {list(instance.keys())[0]}")
        
        instance_data = list(instance.values())[0]
        shard_assignments = instance_data.get('shardAssignments', {})
        
        print(f"Model: {shard_assignments.get('modelId', 'Unknown')}")
        print(f"Runners: {len(shard_assignments.get('runnerToShard', {}))}")
        
        # Check runner statuses
        runners = data.get('runners', {})
        node_to_runner = shard_assignments.get('nodeToRunner', {})
        
        print("\n=== Runner Status ===")
        for node_id, runner_id in node_to_runner.items():
            node_name = get_node_name(data, node_id)
            runner_status = runners.get(runner_id, {})
            status_type = list(runner_status.keys())[0] if runner_status else "Unknown"
            status_data = list(runner_status.values())[0] if runner_status else {}
            
            print(f"Node: {node_name} ({node_id[:8]}...)")
            print(f"  Runner ID: {runner_id}")
            print(f"  Status: {status_type}")
            
            if status_type == "RunnerFailed":
                error_msg = status_data.get('errorMessage', 'No error message')
                print(f"  Error: {error_msg}")
            elif isinstance(status_data, dict):
                for key, value in status_data.items():
                    print(f"  {key}: {value}")
                    
        return {
            'instance_id': instance_id,
            'instance_data': instance_data,
            'runners': runners,
            'node_to_runner': node_to_runner,
            'topology': data.get('topology', {}),
            'node_profiles': data.get('nodeProfiles', {})
        }
        
    except Exception as e:
        print(f"Error getting instance details: {e}")
        return None

def get_node_name(data, node_id):
    """Get friendly name for a node."""
    profiles = data.get('nodeProfiles', {})
    profile = profiles.get(node_id, {})
    return profile.get('friendlyName', node_id[:8] + '...')

def check_system_resources():
    """Check system resources on all nodes."""
    print("\n=== System Resources ===")
    
    try:
        result = subprocess.run(['curl', '-s', 'http://10.1.1.12:52415/state'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return
            
        data = json.loads(result.stdout)
        profiles = data.get('nodeProfiles', {})
        
        for node_id, profile in profiles.items():
            node_name = profile.get('friendlyName', node_id[:8] + '...')
            print(f"\nNode: {node_name}")
            
            # Memory info
            memory = profile.get('memory', {})
            ram_total = memory.get('ramTotal', {}).get('inBytes', 0)
            ram_available = memory.get('ramAvailable', {}).get('inBytes', 0)
            ram_used = ram_total - ram_available
            
            if ram_total > 0:
                ram_usage_pct = (ram_used / ram_total) * 100
                print(f"  Memory: {ram_used / (1024**3):.1f}GB / {ram_total / (1024**3):.1f}GB ({ram_usage_pct:.1f}% used)")
            
            # System info
            system = profile.get('system', {})
            cpu_usage = system.get('pcpuUsage', 0)
            temp = system.get('temp', 0)
            
            print(f"  CPU Usage: {cpu_usage}%")
            print(f"  Temperature: {temp}°C")
            
            # Available engines
            engines = profile.get('availableEngines', [])
            selected_engine = profile.get('selectedEngine', 'Unknown')
            print(f"  Engines: {', '.join(engines)} (selected: {selected_engine})")
            
    except Exception as e:
        print(f"Error checking system resources: {e}")

def check_recent_logs():
    """Check recent logs for any relevant errors."""
    print("\n=== Recent Logs Analysis ===")
    
    # Check for runner-related errors
    log_searches = [
        "runner_24e60126",
        "SmolLM2",
        "terminated",
        "exitcode=None",
        "RunnerFailed",
        "ERROR.*runner",
        "CRITICAL.*runner"
    ]
    
    for search_term in log_searches:
        try:
            result = subprocess.run([
                'journalctl', '-u', 'exo', '--since', '2 hours ago', 
                '--grep', search_term, '--no-pager'
            ], capture_output=True, text=True, timeout=30)
            
            if result.stdout.strip():
                print(f"\nLogs matching '{search_term}':")
                lines = result.stdout.strip().split('\n')
                # Show last 5 lines to avoid spam
                for line in lines[-5:]:
                    print(f"  {line}")
                    
        except Exception as e:
            print(f"Error searching logs for '{search_term}': {e}")

def suggest_fixes(details):
    """Suggest potential fixes based on the analysis."""
    print("\n=== Suggested Actions ===")
    
    if not details:
        print("1. Check EXO service status: systemctl status exo")
        print("2. Restart EXO service: systemctl restart exo")
        return
    
    runners = details.get('runners', {})
    failed_runners = [rid for rid, status in runners.items() 
                     if 'RunnerFailed' in status]
    
    if failed_runners:
        print(f"Found {len(failed_runners)} failed runner(s)")
        print("1. Delete and recreate the instance:")
        print(f"   curl -X DELETE http://10.1.1.12:52415/instance/{details['instance_id']}")
        print("   Then recreate the instance through the dashboard")
        
        print("\n2. Check node connectivity:")
        node_to_runner = details.get('node_to_runner', {})
        for node_id, runner_id in node_to_runner.items():
            if runner_id in failed_runners:
                node_name = get_node_name({'nodeProfiles': details.get('node_profiles', {})}, node_id)
                print(f"   - Check if {node_name} is accessible and healthy")
        
        print("\n3. Restart EXO service on affected nodes")
        print("\n4. Check system resources (memory, CPU, disk space)")
        
    else:
        print("No failed runners found. Instance might be in a transitional state.")
        print("1. Wait a few minutes for the system to stabilize")
        print("2. Check if the instance recovers automatically")

def main():
    """Main diagnostic function."""
    print("EXO Instance B6DAE586 Failure Analysis")
    print("=" * 50)
    print(f"Analysis time: {datetime.now()}")
    
    details = get_instance_details()
    check_system_resources()
    check_recent_logs()
    suggest_fixes(details)
    
    print("\n" + "=" * 50)
    print("Analysis complete.")

if __name__ == "__main__":
    main()
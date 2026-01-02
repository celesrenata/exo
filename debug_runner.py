#!/usr/bin/env python3
"""
Debug script to investigate exo runner failures for instance FF8A5194
"""
import requests
import json

def check_instance_status(host="10.1.1.12", port=52415):
    """Check the status of the failing instance"""
    base_url = f"http://{host}:{port}"
    
    # Get state
    state = requests.get(f"{base_url}/state").json()
    
    # Find instance ff8a5194
    instance_id = "ff8a5194-7c1d-4657-939f-db88917bbd69"
    
    if instance_id in state["instances"]:
        instance = state["instances"][instance_id]
        print(f"Instance {instance_id[:8]} found:")
        print(json.dumps(instance, indent=2))
        print("\n" + "="*80 + "\n")
        
        # Get runner IDs from instance
        if "CpuRingInstance" in instance:
            node_to_runner = instance["CpuRingInstance"]["shardAssignments"]["nodeToRunner"]
            runner_ids = list(node_to_runner.values())
            
            print(f"Runners for this instance:")
            for runner_id in runner_ids:
                if runner_id in state["runners"]:
                    runner_status = state["runners"][runner_id]
                    print(f"  Runner {runner_id[:8]}: {list(runner_status.keys())[0]}")
                    if "RunnerFailed" in runner_status:
                        print(f"    Error: {runner_status['RunnerFailed']['errorMessage']}")
            
            print("\n" + "="*80 + "\n")
            
            # Check nodes
            print("Nodes in cluster:")
            for node_id, profile in state["nodeProfiles"].items():
                print(f"  {profile['friendlyName']} ({node_id[:8]})")
                print(f"    CPU: {profile['chipId']}")
                print(f"    Engine: {profile['selectedEngine']}")
                print(f"    Available engines: {profile['availableEngines']}")
    else:
        print(f"Instance {instance_id[:8]} not found")
    
    return state

if __name__ == "__main__":
    state = check_instance_status()

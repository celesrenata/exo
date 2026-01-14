#!/usr/bin/env python3
"""
Test script to validate the distributed state machine progression.
This script tests the planning logic to ensure runners progress through states correctly.
"""

import sys
import time
import requests
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def wait_for_service():
    """Wait for the EXO service to be ready."""
    for i in range(30):
        try:
            response = requests.get("http://10.1.1.12:52415/state", timeout=5)
            if response.status_code == 200:
                print("✅ EXO service is ready")
                return True
        except:
            pass
        print("⏳ Waiting for EXO service... ({i + 1}/30)")
        time.sleep(2)
    return False


def create_distributed_instance():
    """Create a distributed instance."""
    print("🚀 Creating distributed instance...")
    response = requests.post(
        "http://10.1.1.12:52415/place_instance",
        json={
            "model_id": "smollm2-360m-cpu",
            "instance_meta": "CpuRing",
            "min_nodes": 2,
        },
        timeout=10,
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Instance creation initiated: {result['command_id']}")
        return True
    else:
        print("❌ Failed to create instance: {response.status_code}")
        return False


def get_instance_info():
    """Get current instance information."""
    response = requests.get("http://10.1.1.12:52415/state", timeout=5)
    if response.status_code != 200:
        return None

    data = response.json()
    instances = data.get("instances", {})

    for instance_id, instance_data in instances.items():
        if "CpuRingInstance" in instance_data:
            cpu_instance = instance_data["CpuRingInstance"]
            shard_assignments = cpu_instance["shardAssignments"]

            return {
                "instance_id": instance_id,
                "runners": list(shard_assignments["runnerToShard"].keys()),
                "nodes": list(shard_assignments["nodeToRunner"].keys()),
                "world_size": list(shard_assignments["runnerToShard"].values())[0][
                    "PipelineShardMetadata"
                ]["worldSize"],
            }

    return None


def monitor_state_machine(max_wait_time=60):
    """Monitor the state machine progression."""
    print("🔍 Monitoring state machine progression...")

    start_time = time.time()
    last_states = {}

    while time.time() - start_time < max_wait_time:
        instance_info = get_instance_info()
        if not instance_info:
            print("⏳ Waiting for instance creation...")
            time.sleep(2)
            continue

        print("\n📊 Instance: {instance_info['instance_id']}")
        print("   Runners: {instance_info['runners']}")
        print("   World Size: {instance_info['world_size']}")

        # Check if we can test inference
        if len(instance_info["runners"]) == 2:
            print("🧪 Testing distributed inference...")
            success = test_inference()
            if success:
                print("🎉 Distributed inference working!")
                return True
            else:
                print("⚠️  Inference not ready yet, continuing to monitor...")

        time.sleep(5)

    print("⏰ Timeout waiting for state machine completion")
    return False


def test_inference():
    """Test if distributed inference is working."""
    try:
        response = requests.post(
            "http://10.1.1.12:52415/v1/chat/completions",
            json={
                "model": "smollm2-360m-cpu",
                "messages": [
                    {"role": "user", "content": "Hello distributed inference!"}
                ],
                "max_tokens": 10,
                "temperature": 0.7,
            },
            timeout=15,
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print("✅ Inference successful: {content}")
                return True

        print("❌ Inference failed: {response.status_code}")
        return False

    except requests.exceptions.Timeout:
        print("⏰ Inference timeout (expected if not ready)")
        return False
    except Exception as e:
        print("❌ Inference error: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Testing Distributed State Machine")
    print("=" * 50)

    # Wait for service
    if not wait_for_service():
        print("❌ EXO service not available")
        return False

    # Create distributed instance
    if not create_distributed_instance():
        print("❌ Failed to create distributed instance")
        return False

    # Monitor state machine
    success = monitor_state_machine()

    if success:
        print("\n🎉 SUCCESS: Distributed state machine completed successfully!")
        return True
    else:
        print("\n❌ FAILED: Distributed state machine did not complete")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

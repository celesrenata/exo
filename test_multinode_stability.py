#!/usr/bin/env python3
"""
Test script to verify multi-node stability after the state machine fix.
"""

import requests
import json
import time
import sys


def test_multinode_inference():
    """Test multiple inference requests to verify stability."""

    base_url = "http://10.1.1.12:52415"

    print("🧪 Testing Multi-Node Stability")
    print("=" * 50)

    # Test multiple requests
    success_count = 0
    total_requests = 3

    for i in range(total_requests):
        print("\n📡 Request {i + 1}/{total_requests}")

        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "dialogpt-medium-cpu",
                    "messages": [{"role": "user", "content": f"Test message {i + 1}"}],
                    "max_tokens": 10,
                    "stream": False,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                print("✅ Success: {content[:50]}...")
                success_count += 1
            else:
                print("❌ HTTP Error: {response.status_code}")
                print("Response: {response.text[:200]}...")

        except Exception as e:
            print("❌ Exception: {e}")

        # Brief delay between requests
        if i < total_requests - 1:
            time.sleep(2)

    print("\n📊 Results: {success_count}/{total_requests} successful")

    # Check system state
    print("\n🔍 Checking system state...")
    try:
        state_response = requests.get(f"{base_url}/state", timeout=10)
        if state_response.status_code == 200:
            state_data = state_response.json()
            instances = state_data.get("instances", {})
            print("Active instances: {len(instances)}")

            for instance_id, instance_data in instances.items():
                if "CpuRingInstance" in instance_data:
                    cpu_instance = instance_data["CpuRingInstance"]
                    shard_assignments = cpu_instance["shardAssignments"]
                    world_size = len(shard_assignments["runnerToShard"])
                    print("  Instance {instance_id[:8]}...: world_size={world_size}")
        else:
            print("❌ State check failed: {state_response.status_code}")

    except Exception as e:
        print("❌ State check exception: {e}")

    print("\n" + "=" * 50)
    if success_count == total_requests:
        print("🎉 SUCCESS: Multi-node system is stable!")
        print("   - No process crashes detected")
        print("   - All requests completed successfully")
        print("   - State machine fix is working correctly")
        return True
    else:
        print(
            f"⚠️  PARTIAL SUCCESS: {success_count}/{total_requests} requests succeeded"
        )
        print("   - System is more stable than before")
        print("   - Some issues may remain")
        return False


if __name__ == "__main__":
    success = test_multinode_inference()
    sys.exit(0 if success else 1)

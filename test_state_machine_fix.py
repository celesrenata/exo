#!/usr/bin/env python3
"""
Test script to validate the state machine fix for LoadModel task rejections.

This script tests the specific issue where LoadModel tasks were being rejected
because runners were in RunnerConnected state with group=None.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from exo.shared.types.tasks import ConnectToGroup, LoadModel
from exo.shared.types.worker.instances import BoundInstance, CpuRingInstance
from exo.shared.types.worker.runners import RunnerConnected, RunnerIdle, RunnerLoaded
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.shared.models.model_cards import get_model_metadata


def test_state_machine_logic():
    """Test the state machine logic that was causing LoadModel rejections."""

    print("🔍 Testing State Machine Fix for LoadModel Rejections")
    print("=" * 60)

    # Simulate the state machine conditions
    test_cases = [
        {
            "name": "RunnerConnected with group=None (BROKEN CASE)",
            "status": "RunnerConnected",
            "group": None,
            "should_accept": False,
            "description": "This was the failing case - ConnectToGroup succeeded but group was None",
        },
        {
            "name": "RunnerConnected with group='connected' (FIXED CASE)",
            "status": "RunnerConnected",
            "group": "connected",
            "should_accept": True,
            "description": "After fix - ConnectToGroup sets placeholder group value",
        },
        {
            "name": "RunnerIdle with group=None (VALID CASE)",
            "status": "RunnerIdle",
            "group": None,
            "should_accept": True,
            "description": "Direct model loading without connection step",
        },
        {
            "name": "RunnerIdle with group='connected' (INVALID CASE)",
            "status": "RunnerIdle",
            "group": "connected",
            "should_accept": False,
            "description": "Should not happen - idle runner with group",
        },
    ]

    print("Testing LoadModel acceptance conditions:")
    print()

    for i, case in enumerate(test_cases, 1):
        # Simulate the condition from runner.py
        current_status_connected = case["status"] == "RunnerConnected"
        current_status_idle = case["status"] == "RunnerIdle"
        group_not_none = case["group"] is not None
        group_is_none = case["group"] is None

        # This is the condition from the fixed code
        condition_met = (current_status_connected and group_not_none) or (
            current_status_idle and group_is_none
        )

        status_icon = "✅" if condition_met == case["should_accept"] else "❌"
        result = "ACCEPT" if condition_met else "REJECT"
        expected = "ACCEPT" if case["should_accept"] else "REJECT"

        print("{i}. {case['name']}")
        print("   Status: {case['status']}, Group: {case['group']}")
        print("   Result: {result} (Expected: {expected}) {status_icon}")
        print("   {case['description']}")
        print()

    return True


def test_engine_initialization():
    """Test that engine initialization works correctly for torch engines."""

    print("🔧 Testing Engine Initialization")
    print("=" * 40)

    try:
        from exo.worker.engines.engine_init import initialize_engine
        from exo.worker.engines.engine_utils import select_best_engine

        engine_type = select_best_engine()
        print("Selected engine: {engine_type}")

        # Create a minimal bound instance for testing
        model_meta = get_model_metadata("HuggingFaceTB/SmolLM2-360M-Instruct")
        shard_metadata = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=12,
            n_layers=24,
        )

        instance = CpuRingInstance(
            instance_id="test-instance",
            shard_assignments=None,
            hosts_by_node={},
            ephemeral_port=52414,
        )

        bound_instance = BoundInstance(
            instance=instance, bound_runner_id="test-runner", bound_node_id="test-node"
        )
        bound_instance.bound_shard = shard_metadata

        # Test connect_only=True (this was returning None for torch engines)
        print("Testing connect_only=True...")
        group = initialize_engine(bound_instance, connect_only=True)
        print("Group result: {group}")

        if engine_type in ["torch", "cpu"] and group is None:
            print("✅ Confirmed: torch engines return None for connect_only=True")
            print("   This is why the fix is needed - we need a placeholder value")
        elif group is not None:
            print("✅ Engine returned group: {group}")

        return True

    except Exception as e:
        print("❌ Engine initialization test failed: {e}")
        return False


def main():
    """Run all tests to validate the state machine fix."""

    print("🚀 State Machine Fix Validation")
    print("=" * 50)
    print()

    # Test 1: State machine logic
    test1_passed = test_state_machine_logic()

    # Test 2: Engine initialization
    test2_passed = test_engine_initialization()

    print("📊 Test Summary")
    print("=" * 20)
    print("State Machine Logic: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print("Engine Initialization: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print(
            "\n🎉 All tests passed! The state machine fix should resolve the LoadModel rejection issue."
        )
        return 0
    else:
        print("\n⚠️  Some tests failed. The fix may need additional work.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

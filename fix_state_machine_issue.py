#!/usr/bin/env python3
"""
Analysis and fix for the B6DAE586 state machine issue.
The problem is that LoadModel tasks are being sent to runners in RunnerConnected state.
"""

import sys
import subprocess


def analyze_state_machine_issue():
    """Analyze the state machine issue from the logs."""
    print("=== State Machine Issue Analysis ===")
    print()

    print("Problem: Runner receiving LoadModel in RunnerConnected state")
    print("Expected flow: RunnerIdle -> LoadModel -> RunnerLoaded")
    print("Actual flow: RunnerConnected -> LoadModel -> FAIL")
    print()

    print("Root causes:")
    print("1. Task scheduling race condition")
    print("2. State transition not properly synchronized")
    print("3. Multiple LoadModel tasks sent rapidly")
    print()


def check_runner_state_machine():
    """Check the runner state machine implementation."""
    print("=== Checking Runner State Machine ===")

    # Look for state machine logic in the runner
    try:
        result = subprocess.run(
            [
                "grep",
                "-r",
                "-n",
                "RunnerConnected\|LoadModel\|state.*machine",
                "src/exo/worker/runner/",
                "--include=*.py",
            ],
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print("State machine related code:")
            print(result.stdout[:2000])  # Limit output
        else:
            print("No state machine code found in runner directory")

    except Exception as e:
        print("Error checking state machine: {e}")


def suggest_fixes():
    """Suggest potential fixes for the issue."""
    print("\n=== Suggested Fixes ===")
    print()

    fixes = [
        "1. Add state validation before accepting LoadModel tasks",
        "2. Implement proper state transition synchronization",
        "3. Add task queuing when runner is not in correct state",
        "4. Fix the health monitor ResourceState serialization issue",
        "5. Improve resource cleanup to prevent 'already in progress' errors",
    ]

    for fix in fixes:
        print(fix)

    print()
    print("Priority: Fix the state machine validation first, as this is causing")
    print("the cascade of failures we're seeing with instance B6DAE586.")


def main():
    """Main analysis function."""
    print("EXO Instance B6DAE586 Failure Analysis")
    print("=" * 50)

    analyze_state_machine_issue()
    check_runner_state_machine()
    suggest_fixes()


if __name__ == "__main__":
    main()

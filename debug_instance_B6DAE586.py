#!/usr/bin/env python3
"""
Debug script for EXO instance B6DAE586 failure investigation.
"""

import sys
import subprocess
import json
from datetime import datetime


def get_system_logs(instance_id="B6DAE586"):
    """Get system logs related to the specific instance."""
    print("=== System Logs for Instance {instance_id} ===")

    # Get journalctl logs for exo service
    try:
        result = subprocess.run(
            [
                "journalctl",
                "-u",
                "exo",
                "--since",
                "10 minutes ago",
                "--grep",
                instance_id,
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.stdout:
            print("Recent logs containing instance ID:")
            print(result.stdout)
        else:
            print("No recent logs found for this instance ID")

    except Exception as e:
        print("Error getting system logs: {e}")


def check_process_status():
    """Check if EXO processes are running."""
    print("\n=== Process Status ===")

    try:
        result = subprocess.run(["pgrep", "-f", "exo"], capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split("\n")
            print("Found {len(pids)} EXO processes:")
            for pid in pids:
                # Get process info
                ps_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "pid,ppid,cmd"],
                    capture_output=True,
                    text=True,
                )
                print(ps_result.stdout)
        else:
            print("No EXO processes found")

    except Exception as e:
        print("Error checking processes: {e}")


def check_network_connectivity():
    """Check network connectivity for multi-node setup."""
    print("\n=== Network Connectivity ===")

    # Check if we can reach other nodes
    try:
        # Get network interfaces
        result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
        print("Network interfaces:")
        print(result.stdout)

        # Check for listening ports (EXO typically uses various ports)
        result = subprocess.run(["netstat", "-tlnp"], capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.split("\n")
            exo_ports = [line for line in lines if "exo" in line.lower()]
            if exo_ports:
                print("\nEXO-related listening ports:")
                for port in exo_ports:
                    print(port)
            else:
                print("\nNo EXO-related ports found listening")

    except Exception as e:
        print("Error checking network: {e}")


def check_resource_usage():
    """Check system resource usage."""
    print("\n=== Resource Usage ===")

    try:
        # Memory usage
        result = subprocess.run(["free", "-h"], capture_output=True, text=True)
        print("Memory usage:")
        print(result.stdout)

        # CPU usage
        result = subprocess.run(["top", "-bn1"], capture_output=True, text=True)
        lines = result.stdout.split("\n")[:10]  # First 10 lines
        print("\nCPU usage (top processes):")
        for line in lines:
            print(line)

        # Disk usage
        result = subprocess.run(["df", "-h"], capture_output=True, text=True)
        print("\nDisk usage:")
        print(result.stdout)

    except Exception as e:
        print("Error checking resources: {e}")


def check_exo_config():
    """Check EXO configuration and environment."""
    print("\n=== EXO Configuration ===")

    # Check for EXO environment variables
    env_vars = ["EXO_", "CUDA_", "TORCH_"]
    for var_prefix in env_vars:
        matching_vars = {
            k: v for k, v in os.environ.items() if k.startswith(var_prefix)
        }
        if matching_vars:
            print("\n{var_prefix}* environment variables:")
            for k, v in matching_vars.items():
                print("  {k}={v}")


def main():
    """Main diagnostic function."""
    print("EXO Instance B6DAE586 Diagnostic Report")
    print("Generated at: {datetime.now()}")
    print("=" * 60)

    get_system_logs()
    check_process_status()
    check_network_connectivity()
    check_resource_usage()
    check_exo_config()

    print("\n" + "=" * 60)
    print("Diagnostic complete. Please share the output above for analysis.")


if __name__ == "__main__":
    import os

    main()

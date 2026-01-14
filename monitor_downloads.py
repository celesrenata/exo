#!/usr/bin/env python3
"""
Monitor model download progress.
"""

import time
import os
import sys
from pathlib import Path


def get_directory_size(path):
    """Get total size of directory in MB."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total / (1024 * 1024)  # Convert to MB


def monitor_downloads():
    """Monitor download progress in both possible locations."""

    # Possible model directories
    locations = [
        "/var/lib/exo/models",
        "/var/lib/exo/exo/models",
        Path.home() / ".local/share/exo/models",
    ]

    print("🔍 Monitoring model download progress...")
    print("=" * 60)

    previous_sizes = {}

    try:
        while True:
            print("\n⏰ {time.strftime('%H:%M:%S')}")

            found_activity = False

            for location in locations:
                location = Path(location)
                if location.exists():
                    current_size = get_directory_size(location)
                    previous_size = previous_sizes.get(str(location), 0)

                    if current_size > 0:
                        found_activity = True
                        change = current_size - previous_size
                        change_indicator = f"(+{change:.1f}MB)" if change > 0 else ""

                        print("📁 {location}")
                        print("   Size: {current_size:.1f} MB {change_indicator}")

                        # List recent files
                        try:
                            files = []
                            for item in location.iterdir():
                                if item.is_dir():
                                    model_size = get_directory_size(item)
                                    if model_size > 0:
                                        files.append((item.name, model_size))

                            if files:
                                files.sort(key=lambda x: x[1], reverse=True)
                                print("   Models:")
                                for name, size in files[:5]:  # Show top 5
                                    print("     • {name}: {size:.1f} MB")
                                if len(files) > 5:
                                    print("     ... and {len(files) - 5} more")
                        except Exception as e:
                            print("   Error listing files: {e}")

                    previous_sizes[str(location)] = current_size

            if not found_activity:
                print("📭 No model directories found or no downloads in progress")

                # Check if EXO service is running
                try:
                    result = os.system(
                        "systemctl is-active exo.service >/dev/null 2>&1"
                    )
                    if result == 0:
                        print("✅ EXO service is running")
                    else:
                        print("❌ EXO service is not running")
                except:
                    pass

            print("-" * 60)
            time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        return


if __name__ == "__main__":
    monitor_downloads()

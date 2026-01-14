#!/usr/bin/env python3
"""
Check if EXO is running and what its status is.
"""

import requests
import json
import sys


def check_exo_api():
    """Check EXO API status and endpoints."""
    base_url = "http://localhost:52415"

    print("=== Checking EXO API Status ===")

    # Try different endpoints
    endpoints = [
        "/health",
        "/",
        "/v1/models",
        "/v1/chat/completions",
        "/models",
        "/status",
    ]

    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            print("{endpoint}: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print("  Response: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print("  Response: {response.text[:100]}...")
            elif response.status_code == 404:
                print("  Not found")
            else:
                print("  Error: {response.text[:100]}")

        except requests.exceptions.ConnectionError:
            print("{endpoint}: Connection refused (EXO not running)")
            break
        except Exception as e:
            print("{endpoint}: Error - {e}")

    return True


def main():
    try:
        check_exo_api()

        print("\n=== Recommendations ===")
        print("1. Start EXO with: nix develop --command exo")
        print("2. Check if model download is triggered automatically")
        print("3. Look for download progress in the logs")
        print("4. The model should download before loading attempts")

        return 0

    except Exception as e:
        print("Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

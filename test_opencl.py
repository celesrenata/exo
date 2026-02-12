#!/usr/bin/env python3
import os
import sys

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("Python path:", sys.path[:3])

try:
    import pyopencl as cl
    print("✓ pyopencl imported successfully")
    
    platforms = cl.get_platforms()
    print(f"✓ Found {len(platforms)} OpenCL platform(s)")
    
    for i, platform in enumerate(platforms):
        print(f"\nPlatform {i}: {platform.name}")
        print(f"  Vendor: {platform.vendor}")
        print(f"  Version: {platform.version}")
        
        devices = platform.get_devices()
        print(f"  Devices: {len(devices)}")
        
        for j, device in enumerate(devices):
            print(f"    Device {j}: {device.name}")
            print(f"      Type: {cl.device_type.to_string(device.type)}")
            print(f"      Memory: {device.global_mem_size / (1024**3):.2f} GB")
            
except ImportError as e:
    print(f"✗ Failed to import pyopencl: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ OpenCL detection successful!")

#!/usr/bin/env python3
import sys
import subprocess

# Read the original file
with open("/tmp/ops_gpu_orig.py", "r") as f:
    lines = f.readlines()

# Insert the diagnostic code after the imports and before CLCompiler class
insert_after_line = None
for i, line in enumerate(lines):
    if line.strip() == "def checked(ret, status): return (check(status.value), ret)[1]":
        insert_after_line = i
        break

if insert_after_line is None:
    print("Could not find insertion point!")
    sys.exit(1)

diagnostic_code = '''
# Intel Arc GPU diagnostic tracking
_allocation_count = 0
_total_allocated = 0
_largest_allocation = 0
_allocations_over_100mb = []
_device_info_logged = False

def _log_device_info(device):
  """Log device information at startup for diagnostic purposes."""
  global _device_info_logged
  if _device_info_logged:
    return
  _device_info_logged = True
  
  try:
    device_name = device.device_name if hasattr(device, 'device_name') else "Unknown"
    print(f"[INTEL ARC DEBUG] ===== DEVICE INFORMATION =====", flush=True)
    print(f"[INTEL ARC DEBUG] Device: {device_name}", flush=True)
    if hasattr(device, 'driver_version'):
      print(f"[INTEL ARC DEBUG] Driver: {device.driver_version}", flush=True)
    print(f"[INTEL ARC DEBUG] ==============================", flush=True)
  except Exception as e:
    print(f"[INTEL ARC DEBUG] Error logging device info: {e}", flush=True)

def _log_allocation_summary():
  """Log summary of allocations for diagnostic purposes."""
  print(f"[INTEL ARC DEBUG] ===== ALLOCATION SUMMARY =====", flush=True)
  print(f"[INTEL ARC DEBUG] Total allocations: {_allocation_count}", flush=True)
  print(f"[INTEL ARC DEBUG] Total allocated: {_total_allocated / (1024**3):.2f} GB", flush=True)
  print(f"[INTEL ARC DEBUG] Largest allocation: {_largest_allocation / (1024**3):.3f} GB", flush=True)
  print(f"[INTEL ARC DEBUG] Allocations >100MB: {len(_allocations_over_100mb)}", flush=True)
  if _allocations_over_100mb:
    print(f"[INTEL ARC DEBUG] Top 5 largest allocations:", flush=True)
    for i, (size, context) in enumerate(sorted(_allocations_over_100mb, reverse=True)[:5], 1):
      print(f"[INTEL ARC DEBUG]   {i}. {size / (1024**3):.3f} GB - {context}", flush=True)
  print(f"[INTEL ARC DEBUG] ==============================", flush=True)
'''

# Insert after the checked function
lines.insert(insert_after_line + 1, diagnostic_code)

# Now find and replace the return statement in CLAllocator._alloc
# Find the line with "return (checked(cl.clCreateBuffer"
for i, line in enumerate(lines):
    if (
        "return (checked(cl.clCreateBuffer(self.dev.context, cl.CL_MEM_READ_WRITE, size, None, status := ctypes.c_int32()), status), options)"
        in line
    ):
        # Replace this line with our diagnostic code + the return
        indent = "    "
        replacement = f"""{indent}
{indent}# Log device info on first allocation
{indent}_log_device_info(self.dev)
{indent}
{indent}# Track allocation statistics
{indent}global _allocation_count, _total_allocated, _largest_allocation, _allocations_over_100mb
{indent}_allocation_count += 1
{indent}_total_allocated += size
{indent}_largest_allocation = max(_largest_allocation, size)
{indent}
{indent}# Calculate sizes for logging
{indent}size_mb = size / (1024 * 1024)
{indent}size_gb = size / (1024 * 1024 * 1024)
{indent}
{indent}# Capture context information for large allocations
{indent}context_info = ""
{indent}if size_mb > 100:
{indent}  try:
{indent}    # Get stack trace to identify where allocation is coming from
{indent}    stack = traceback.extract_stack()
{indent}    # Find the most relevant frame (skip this file)
{indent}    relevant_frames = [f for f in stack if 'ops_gpu.py' not in f.filename]
{indent}    if relevant_frames:
{indent}      frame = relevant_frames[-1]
{indent}      context_info = f"{{frame.filename}}:{{frame.lineno}} in {{frame.name}}"
{indent}    
{indent}    # Try to get tensor shape from options if available
{indent}    if hasattr(options, 'shape'):
{indent}      context_info += f" shape={{options.shape}}"
{indent}    if hasattr(options, 'dtype'):
{indent}      context_info += f" dtype={{options.dtype}}"
{indent}  except Exception as e:
{indent}    context_info = f"(context capture failed: {{e}})"
{indent}
{indent}# Log allocations >100MB (Requirement 1.1)
{indent}if size_mb > 100:
{indent}  print(f"[INTEL ARC DEBUG] Allocation #{{_allocation_count}}: {{size}} bytes ({{size_mb:.2f}} MB / {{size_gb:.3f}} GB)", flush=True)
{indent}  if context_info:
{indent}    print(f"[INTEL ARC DEBUG]   Context: {{context_info}}", flush=True)
{indent}  _allocations_over_100mb.append((size, context_info))
{indent}
{indent}# Warn on allocations >4GB (Requirement 1.3)
{indent}if size_gb > 4.0:
{indent}  print(f"[INTEL ARC DEBUG] WARNING: Buffer >4GB detected!", flush=True)
{indent}  print(f"[INTEL ARC DEBUG]   Size: {{size_gb:.3f}} GB ({{size}} bytes)", flush=True)
{indent}  print(f"[INTEL ARC DEBUG]   Context: {{context_info}}", flush=True)
{indent}  print(f"[INTEL ARC DEBUG]   This may exceed Intel Arc GPU limits", flush=True)
{indent}
{indent}# Attempt allocation
{indent}flags = cl.CL_MEM_READ_WRITE
{indent}return (checked(cl.clCreateBuffer(self.dev.context, flags, size, None, status := ctypes.c_int32()), status), options)
"""
        lines[i] = replacement
        break

# Also need to add the import at the top
for i, line in enumerate(lines):
    if line.startswith("import ctypes, functools, hashlib"):
        lines[i] = line.rstrip() + "\nimport sys, traceback, time\n"
        break

# Write the modified file
with open("/tmp/ops_gpu_new.py", "w") as f:
    f.writelines(lines)

print("Modified file created successfully")

# Create the patch
subprocess.run(
    ["diff", "-u", "/tmp/ops_gpu_orig.py", "/tmp/ops_gpu_new.py"],
    stdout=open("patches/tinygrad-intel-arc-4gb-fix.patch", "w"),
)
print("Patch file created")

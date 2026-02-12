# Intel Arc iGPU OpenCL Fix for Tinygrad

## Problem Identified

Intel Arc GPUs (including the integrated GPU in Intel Core Ultra 185H) have a **4GB OpenCL buffer allocation limit** by default. Our Llama-3.2-3B model is 6.4GB, which exceeds this limit, causing:

```
OpenCL Error -4: CL_MEM_OBJECT_ALLOCATION_FAILURE
```

## Root Cause

From ChatGPT research with web search:
- Intel Arc GPUs require special flags for allocations >4GB
- Buffer flag: `(1 << 23)` (CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL)
- Compiler option: `-cl-intel-greater-than-4GB-buffer-required`
- These cannot be set via environment variables - require code changes

## Solution: Patch Tinygrad OpenCL Runtime

### Location
File: `/nix/store/.../tinygrad/runtime/ops_cl.py`
Line 85: `clCreateBuffer` call in `CLAllocator._alloc()`

### Current Code
```python
def _alloc(self, size:int, options:BufferSpec) -> tuple[ctypes._CData, BufferSpec]:
    # Recalculate real size for texture
    if options.image is not None: size = options.image.pitch * options.image.shape[0]
    return (checked(cl.clCreateBuffer(self.dev.context, cl.CL_MEM_READ_WRITE, size, None, status := ctypes.c_int32()), status), options)
```

### Fixed Code
```python
def _alloc(self, size:int, options:BufferSpec) -> tuple[ctypes._CData, BufferSpec]:
    # Recalculate real size for texture
    if options.image is not None: size = options.image.pitch * options.image.shape[0]
    
    # Intel Arc GPU fix: Enable >4GB buffer allocations
    # Add CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL flag (1 << 23)
    flags = cl.CL_MEM_READ_WRITE | (1 << 23)
    
    return (checked(cl.clCreateBuffer(self.dev.context, flags, size, None, status := ctypes.c_int32()), status), options)
```

### Compiler Option Fix

Also need to add the compiler flag. In `CLCompiler.compile()` method (line ~25):

**Current:**
```python
build_status: int = cl.clBuildProgram(program, 1, self.dev.device_id, None, cl.clBuildProgram.argtypes[4](), None)
```

**Fixed:**
```python
# Intel Arc GPU fix: Enable >4GB buffer support in kernels
build_options = b"-cl-intel-greater-than-4GB-buffer-required"
build_status: int = cl.clBuildProgram(program, 1, self.dev.device_id, build_options, cl.clBuildProgram.argtypes[4](), None)
```

And in `CLProgram.__init__()` (line ~45):

**Current:**
```python
check(cl.clBuildProgram(self.program, 1, device.device_id, None, cl.clBuildProgram.argtypes[4](), None))
```

**Fixed:**
```python
build_options = b"-cl-intel-greater-than-4GB-buffer-required"
check(cl.clBuildProgram(self.program, 1, device.device_id, build_options, cl.clBuildProgram.argtypes[4](), None))
```

## Implementation Options

### Option 1: Create a Patched Tinygrad Package (Recommended)

Create a NixOS overlay to patch tinygrad:

```nix
# In flake.nix or overlay
tinygrad-intel-arc = python3Packages.tinygrad.overrideAttrs (old: {
  patches = (old.patches or []) ++ [
    ./patches/tinygrad-intel-arc-4gb-fix.patch
  ];
});
```

Create patch file `patches/tinygrad-intel-arc-4gb-fix.patch`:
```patch
diff --git a/tinygrad/runtime/ops_cl.py b/tinygrad/runtime/ops_cl.py
index xxx..yyy 100644
--- a/tinygrad/runtime/ops_cl.py
+++ b/tinygrad/runtime/ops_cl.py
@@ -22,7 +22,9 @@ class CLCompiler(Compiler):
     self.dev = dev
     super().__init__(f"compile_cl_{compile_key}")
   def compile(self, src:str) -> bytes:
     program = checked(cl.clCreateProgramWithSource(self.dev.context, 1, to_char_p_p([src.encode()]), None, status := ctypes.c_int32()), status)
-    build_status: int = cl.clBuildProgram(program, 1, self.dev.device_id, None, cl.clBuildProgram.argtypes[4](), None)
+    # Intel Arc GPU fix: Enable >4GB buffer support
+    build_options = b"-cl-intel-greater-than-4GB-buffer-required"
+    build_status: int = cl.clBuildProgram(program, 1, self.dev.device_id, build_options, cl.clBuildProgram.argtypes[4](), None)
     if build_status != 0:
       cl.clGetProgramBuildInfo(program, self.dev.device_id, cl.CL_PROGRAM_BUILD_LOG, 0, None, log_size := ctypes.c_size_t())
       cl.clGetProgramBuildInfo(program, self.dev.device_id, cl.CL_PROGRAM_BUILD_LOG,
@@ -43,7 +45,9 @@ class CLProgram:
                                                         to_char_p_p([lib], ctypes.c_ubyte), binary_status := ctypes.c_int32(),
                                                         errcode_ret := ctypes.c_int32()), errcode_ret)
     check(binary_status.value)
-    check(cl.clBuildProgram(self.program, 1, device.device_id, None, cl.clBuildProgram.argtypes[4](), None))
+    # Intel Arc GPU fix: Enable >4GB buffer support
+    build_options = b"-cl-intel-greater-than-4GB-buffer-required"
+    check(cl.clBuildProgram(self.program, 1, device.device_id, build_options, cl.clBuildProgram.argtypes[4](), None))
     self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), status := ctypes.c_int32()), status)
 
   def __del__(self):
@@ -82,7 +86,10 @@ class CLAllocator(LRUAllocator['CLDevice']):
   def _alloc(self, size:int, options:BufferSpec) -> tuple[ctypes._CData, BufferSpec]:
     # Recalculate real size for texture
     if options.image is not None: size = options.image.pitch * options.image.shape[0]
-    return (checked(cl.clCreateBuffer(self.dev.context, cl.CL_MEM_READ_WRITE, size, None, status := ctypes.c_int32()), status), options)
+    # Intel Arc GPU fix: Enable >4GB buffer allocations
+    # Add CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL flag (1 << 23)
+    flags = cl.CL_MEM_READ_WRITE | (1 << 23)
+    return (checked(cl.clCreateBuffer(self.dev.context, flags, size, None, status := ctypes.c_int32()), status), options)
   @suppress_finalizing
   def _free(self, opaque:tuple[ctypes._CData, BufferSpec], options:BufferSpec): check(cl.clReleaseMemObject(opaque[0]))
   def _copyin(self, dest:tuple[ctypes._CData, BufferSpec], src:memoryview):
```

### Option 2: Runtime Monkey-Patch (Quick Test)

Create a Python script to patch at runtime:

```python
# patch_tinygrad_intel_arc.py
import ctypes
from tinygrad.runtime import ops_cl
from tinygrad.runtime.autogen import opencl as cl

# Save original methods
original_alloc = ops_cl.CLAllocator._alloc
original_compile = ops_cl.CLCompiler.compile

def patched_alloc(self, size: int, options):
    if options.image is not None:
        size = options.image.pitch * options.image.shape[0]
    # Intel Arc fix: Add >4GB buffer flag
    flags = cl.CL_MEM_READ_WRITE | (1 << 23)
    status = ctypes.c_int32()
    buf = cl.clCreateBuffer(self.dev.context, flags, size, None, ctypes.byref(status))
    if status.value != 0:
        raise RuntimeError(f"OpenCL Error {status.value}")
    return (buf, options)

def patched_compile(self, src: str) -> bytes:
    # ... (similar patching for compile method)
    pass

# Apply patches
ops_cl.CLAllocator._alloc = patched_alloc
# ops_cl.CLCompiler.compile = patched_compile
```

### Option 3: Fork Tinygrad (Long-term)

Submit a PR to tinygrad with Intel Arc support.

## Testing the Fix

After applying the patch:

```bash
# Rebuild with patched tinygrad
bash force_update_gremlin1.sh

# Test generation
curl -X POST http://10.1.1.12:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10
  }'
```

## Current Status

After applying patch (commit 669c899f):
- ✓ Patch applied successfully to tinygrad v0.11.0
- ✓ Build completed without errors
- ✓ Service starts successfully
- ✓ Model loads (6.4GB)
- ✗ Inference fails with: `OpenCL Error -30: CL_INVALID_VALUE`

### Progress
The error changed from `-4 (CL_MEM_OBJECT_ALLOCATION_FAILURE)` to `-30 (CL_INVALID_VALUE)`, which means:
- The >4GB buffer allocation flag is working
- The buffer was created successfully
- The issue is now with the compiler build options

### Next Steps
The `-cl-intel-greater-than-4GB-buffer-required` compiler option may be:
1. Not supported on this Intel Arc iGPU
2. Requires a different format
3. Needs to be passed as NULL with the flag only

Try removing the compiler option and keeping only the buffer flag.

## References

- Reddit: Intel Arc >4GB OpenCL fix
- Intel Community: Level Zero vs OpenCL
- Tinygrad source: `runtime/ops_cl.py`
- ChatGPT research: Intel Arc iGPU compatibility

## Next Steps

1. Create the patch file
2. Update flake.nix to use patched tinygrad
3. Deploy to gremlin-1
4. Test generation
5. Measure performance
6. Consider submitting PR to tinygrad upstream

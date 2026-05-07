# Design Document

## Overview

This design addresses the IPEX Nix build failure by correcting the version specification, providing the proper source hash, and ensuring compatibility with PyTorch XPU. The fix involves updating the `nix/ipex-xpu.nix` file with a valid Git tag and computed hash.

## Root Cause Analysis

The current build failure occurs because:

1. **Invalid Git Tag**: The version `v2.5.10+xpu` doesn't exist in the intel/intel-extension-for-pytorch repository
2. **Placeholder Hash**: The `lib.fakeHash` is a placeholder that must be replaced with an actual SHA256 hash
3. **Version Compatibility**: IPEX versions must match PyTorch versions (e.g., IPEX 2.5.x requires PyTorch 2.5.x)

## Research Findings

### Available IPEX Versions

Intel IPEX releases follow PyTorch versions. For PyTorch 2.5.x, the corresponding IPEX versions are:
- `v2.5.0+xpu` - Initial 2.5 release
- `v2.5.10+xpu` - Does NOT exist
- Latest stable for 2.5 series: `v2.5.0+xpu`

The pytorch-xpu.nix file uses PyTorch 2.5.1, so we should use IPEX v2.5.0+xpu which is compatible.

### Git Tag Format

Intel IPEX uses tags in the format:
- `v{major}.{minor}.{patch}+xpu` for XPU (Arc GPU) builds
- Example: `v2.5.0+xpu`

### Hash Computation Strategy

Nix requires a SHA256 hash of the source. The workflow is:
1. Use `lib.fakeHash` initially to trigger a build
2. Nix will fail and report the expected hash
3. Replace `lib.fakeHash` with the actual hash
4. Rebuild successfully

Alternatively, we can compute the hash directly using:
```bash
nix-prefetch-url --unpack https://github.com/intel/intel-extension-for-pytorch/archive/refs/tags/v2.5.0+xpu.tar.gz
```

## Architecture

### Component Changes

```
nix/ipex-xpu.nix
├── version: "2.5.10+xpu" → "2.5.0+xpu"
├── rev: "v2.5.10+xpu" → "v2.5.0+xpu"
└── hash: lib.fakeHash → "sha256-<actual-hash>"
```

### Version Compatibility Matrix

| PyTorch Version | IPEX Version | Status |
|----------------|--------------|--------|
| 2.5.1 | 2.5.0+xpu | Compatible ✓ |
| 2.5.0 | 2.5.0+xpu | Compatible ✓ |
| 2.4.x | 2.4.x+xpu | Not used |

## Implementation Strategy

### Phase 1: Update Version Specification

Update `nix/ipex-xpu.nix`:
```nix
buildPythonPackage rec {
  pname = "intel-extension-for-pytorch";
  version = "2.5.0+xpu";  # Changed from 2.5.10+xpu
  format = "setuptools";

  src = fetchFromGitHub {
    owner = "intel";
    repo = "intel-extension-for-pytorch";
    rev = "v${version}";  # Will be v2.5.0+xpu
    hash = lib.fakeHash;  # Temporary - will be replaced in Phase 2
    fetchSubmodules = true;
  };
```

### Phase 2: Compute and Set Source Hash

Two approaches:

**Approach A: Let Nix compute it**
1. Build with `lib.fakeHash`
2. Capture the error message with expected hash
3. Update the file with the actual hash

**Approach B: Pre-compute the hash**
```bash
nix-prefetch-url --unpack https://github.com/intel/intel-extension-for-pytorch/archive/refs/tags/v2.5.0+xpu.tar.gz
```

We'll use Approach A as it's more reliable and doesn't require external tools.

### Phase 3: Verify Build

After updating the hash:
1. Run `nix flake check` to verify the derivation evaluates
2. Optionally build the package: `nix build .#ipex-xpu`
3. Verify the package can be imported in Python

## Error Handling

### Build Failures

1. **Hash Mismatch**: If the hash is incorrect, Nix will report the expected hash
2. **Missing Dependencies**: CMake will fail with clear error messages about missing libraries
3. **Version Incompatibility**: Runtime errors when importing IPEX with mismatched PyTorch

### Validation Strategy

1. **Evaluation Check**: `nix flake check` must pass
2. **Build Check**: `nix build .#ipex-xpu` should complete (optional, very slow)
3. **Import Check**: Python should be able to `import intel_extension_for_pytorch` (requires GPU)

## Testing Strategy

### Unit Tests

Not applicable - this is a build system fix.

### Integration Tests

1. **Flake Check**: Verify the Nix flake evaluates without errors
2. **Derivation Evaluation**: Verify the ipex-xpu package can be evaluated
3. **Build Test** (optional): Attempt to build the package on a system with sufficient resources

### Validation Criteria

- `nix flake check` completes without errors
- The ipex-xpu derivation evaluates successfully
- No evaluation errors related to version or hash

## Documentation Updates

### Files to Update

1. `nix/README-ipex-xpu.md`: Document the correct version and compatibility
2. `.kiro/steering/pytorch-ipex-status.md`: Update status if build succeeds

### Version Documentation

Add a section to README-ipex-xpu.md:
```markdown
## Version Compatibility

- IPEX 2.5.0+xpu is compatible with PyTorch 2.5.x
- Always use matching major.minor versions
- Patch versions can differ (e.g., PyTorch 2.5.1 + IPEX 2.5.0)
```

## Rollback Plan

If the fix doesn't work:
1. Revert to the original version specification
2. Try alternative IPEX versions (e.g., v2.4.0+xpu with PyTorch 2.4.x)
3. Consider using pre-built IPEX packages from nixos-mordrag overlay

## Future Considerations

1. **Automated Version Updates**: Script to check for new IPEX releases
2. **Binary Cache**: Consider building and caching IPEX to avoid long build times
3. **Alternative Sources**: Explore using pre-built wheels from PyPI

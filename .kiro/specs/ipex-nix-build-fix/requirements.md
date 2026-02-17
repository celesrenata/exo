# Requirements Document

## Introduction

The Intel Extension for PyTorch (IPEX) Nix package build is failing due to incorrect version specification and missing source hash. This feature will fix the Nix derivation to successfully build IPEX with Intel XPU (Arc GPU) support.

## Glossary

- **IPEX**: Intel Extension for PyTorch - optimizations for Intel hardware
- **XPU**: Intel's term for discrete GPUs (Arc GPUs)
- **Nix Derivation**: A Nix package build specification
- **Source Hash**: Cryptographic hash of source code for reproducible builds
- **Git Tag**: Version identifier in a Git repository

## Requirements

### Requirement 1: Fix IPEX Version Specification

**User Story:** As a developer, I want the IPEX Nix package to use a valid Git tag, so that the source can be fetched successfully.

#### Acceptance Criteria

1. WHEN the Nix build system fetches IPEX source, THE Build System SHALL use a valid Git tag that exists in the intel/intel-extension-for-pytorch repository
2. WHEN specifying the IPEX version, THE Nix Derivation SHALL use the format "v{major}.{minor}.{patch}+xpu" where the tag exists in the upstream repository
3. WHEN the version is updated, THE Nix Derivation SHALL document the PyTorch version compatibility requirement

### Requirement 2: Provide Valid Source Hash

**User Story:** As a developer, I want the IPEX source hash to be computed correctly, so that Nix can verify the source integrity.

#### Acceptance Criteria

1. WHEN building the IPEX package, THE Nix Derivation SHALL provide a valid SHA256 hash for the source code
2. WHEN the source hash is incorrect, THE Build System SHALL provide clear error messages indicating the expected hash
3. WHEN updating the IPEX version, THE Nix Derivation SHALL update the source hash to match the new version

### Requirement 3: Ensure PyTorch Compatibility

**User Story:** As a developer, I want IPEX to be compatible with the PyTorch version, so that the extension loads correctly.

#### Acceptance Criteria

1. WHEN building IPEX, THE Nix Derivation SHALL verify that the IPEX version is compatible with the pytorch-xpu version
2. WHEN there is a version mismatch, THE Build System SHALL fail with a clear error message
3. WHEN both packages are built, THE Nix Derivation SHALL document the version compatibility matrix

### Requirement 4: Handle Build Dependencies

**User Story:** As a developer, I want all IPEX build dependencies to be correctly specified, so that the build succeeds.

#### Acceptance Criteria

1. WHEN building IPEX, THE Nix Derivation SHALL include all required oneAPI libraries (MKL, oneDNN, TBB)
2. WHEN linking IPEX, THE Build System SHALL find the pytorch-xpu installation directory
3. WHEN the build completes, THE Nix Derivation SHALL verify that all shared libraries have correct RPATH entries

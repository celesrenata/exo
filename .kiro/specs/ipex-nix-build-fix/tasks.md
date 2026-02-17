# Implementation Plan

- [x] 1. Update IPEX version specification in nix/ipex-xpu.nix
  - Change version from "2.5.10+xpu" to "2.5.0+xpu"
  - Verify the Git tag exists in the upstream repository
  - _Requirements: 1.1, 1.2_

- [ ] 2. Compute and set the source hash
  - [x] 2.1 Attempt build with lib.fakeHash to get expected hash
    - Run `nix build .#ipex-xpu` or `nix flake check`
    - Capture the hash mismatch error message
    - Extract the expected SHA256 hash from error output
    - _Requirements: 2.1, 2.2_
  
  - [x] 2.2 Update ipex-xpu.nix with the correct hash
    - Replace `lib.fakeHash` with the actual SHA256 hash
    - Ensure the hash is in the format `"sha256-<base64-hash>"`
    - _Requirements: 2.1, 2.3_

- [ ] 3. Verify the fix resolves the build error
  - [ ] 3.1 Run nix flake check to verify evaluation
    - Execute `nix flake check` command
    - Verify no evaluation errors for ipex-xpu package
    - Confirm the derivation can be evaluated successfully
    - _Requirements: 1.1, 2.1, 4.3_
  
  - [ ] 3.2 Verify version compatibility documentation
    - Check that version "2.5.0+xpu" is compatible with pytorch-xpu 2.5.1
    - Document the compatibility in nix/README-ipex-xpu.md
    - _Requirements: 1.3, 3.1, 3.3_

- [ ] 4. Update documentation
  - Add version compatibility section to nix/README-ipex-xpu.md
  - Document the correct version format and Git tag requirements
  - Update .kiro/steering/pytorch-ipex-status.md if build succeeds
  - _Requirements: 1.3, 3.3_

# Implementation Plan: Intel Arc GPU Memory Allocation Fix

## Phase 1: Diagnostic Implementation

- [x] 1. Enhance diagnostic logging in tinygrad patch
  - Update `patches/tinygrad-intel-arc-4gb-fix.patch` to add comprehensive logging
  - Add allocation size tracking with context information
  - Ensure logs use `flush=True` and proper formatting for journalctl
  - Add device information logging at startup
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [-] 2. Deploy and verify diagnostic logging
  - Commit and push diagnostic patch changes
  - Deploy to gremlin-1 using `force_update_gremlin1.sh`
  - Verify service starts successfully
  - Check that diagnostic logs appear in journalctl
  - _Requirements: 1.5, 5.1, 5.5_

- [ ] 3. Trigger model loading and capture diagnostics
  - Send inference request to trigger Llama-3.2-3B-Instruct loading
  - Monitor logs in real-time for allocation messages
  - Capture all `[INTEL ARC DEBUG]` log entries
  - Document allocation sizes and patterns
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [ ] 4. Analyze diagnostic results
  - Identify all allocations >100MB
  - Identify any allocations >4GB
  - Determine which allocation causes CL_INVALID_VALUE
  - Document tensor shapes and contexts for large allocations
  - Create analysis report with findings
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

## Phase 2: Buffer Splitting Implementation

- [ ] 5. Implement device detection
  - Add `_is_intel_arc()` function to detect Intel Arc GPUs
  - Add `_get_max_buffer_size()` function to return safe limits
  - Add device capability detection at initialization
  - Log detected device type and limits at startup
  - _Requirements: 3.1, 5.2, 5.3_

- [ ] 6. Implement buffer size checking
  - Add `_check_buffer_size()` function before allocations
  - Integrate size check into buffer allocation path
  - Log when buffer would exceed limit
  - Return appropriate error/warning messages
  - _Requirements: 3.1, 3.2_

- [ ] 7. Implement SplitBuffer class
  - Create `SplitBuffer` class to manage multiple physical buffers
  - Implement `allocate()` method to create chunks
  - Implement `copy_to_device()` for data transfer
  - Implement `copy_from_device()` for result retrieval
  - Add chunk size calculation logic
  - _Requirements: 3.2, 3.3, 3.4_

- [ ] 8. Integrate buffer splitting into allocation path
  - Modify buffer allocation to use SplitBuffer when needed
  - Add fallback logic: try direct allocation first, split if fails
  - Ensure transparent operation for existing code
  - Add error handling for split allocation failures
  - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [ ] 9. Implement split buffer operations
  - Add kernel execution support for split buffers
  - Implement read operations across chunks
  - Implement write operations across chunks
  - Add synchronization between chunk operations
  - _Requirements: 3.3, 3.4_

## Phase 3: Testing and Verification

- [ ] 10. Deploy buffer splitting implementation
  - Commit buffer splitting code changes
  - Update patch file with complete implementation
  - Deploy to gremlin-1
  - Verify service starts without errors
  - _Requirements: 5.1, 5.2, 5.4_

- [ ] 11. Test basic model loading
  - Trigger Llama-3.2-3B-Instruct loading
  - Verify no CL_INVALID_VALUE errors occur
  - Check logs for successful split buffer creation
  - Verify model loads to "ready" state
  - _Requirements: 4.1, 4.3_

- [ ] 12. Test inference execution
  - Send simple inference request (5-10 tokens)
  - Verify tokens are generated successfully
  - Check output for correctness
  - Verify no errors during generation
  - _Requirements: 4.2, 4.4_

- [ ] 13. Test with longer sequences
  - Send inference request with 50+ tokens
  - Verify generation completes successfully
  - Check for memory leaks or accumulation issues
  - Verify KV cache works correctly with split buffers
  - _Requirements: 4.2, 4.4_

- [ ]* 14. Performance benchmarking
  - Run performance tests from Task 13 test suite
  - Measure tokens per second with split buffers
  - Compare with baseline (if available)
  - Verify performance is within 90% of target
  - Document performance metrics
  - _Requirements: 4.5_

- [ ]* 15. Test with larger model
  - Test with Llama-3.1-8B-Instruct
  - Verify split buffer handling scales appropriately
  - Check for any issues with larger allocations
  - Document results
  - _Requirements: 4.1, 4.2_

## Phase 4: Documentation and Cleanup

- [ ] 16. Document the fix
  - Update `INTEL_ARC_FIX.md` with final solution
  - Document diagnostic findings
  - Document buffer splitting implementation
  - Add troubleshooting guide
  - _Requirements: 2.5, 5.5_

- [ ] 17. Update deployment documentation
  - Update `gremlin-1.md` with Intel Arc specific notes
  - Document patch application process
  - Add verification steps for Intel Arc systems
  - Document rollback procedures
  - _Requirements: 5.1, 5.4, 5.5_

- [ ] 18. Create monitoring and alerting
  - Add metrics for split buffer usage
  - Add alerts for allocation failures
  - Document monitoring procedures
  - Create dashboard for allocation statistics
  - _Requirements: 1.4, 2.4, 5.5_

- [ ]* 19. Prepare upstream contribution
  - Clean up patch for upstream submission
  - Write detailed commit message
  - Create pull request to tinygrad repository
  - Document rationale and testing
  - _Requirements: 3.5, 4.5_

## Phase 5: Validation and Hardening

- [ ] 20. Edge case testing
  - Test with batch size > 1
  - Test with very long contexts (4K+ tokens)
  - Test concurrent inference requests
  - Test rapid model switching
  - _Requirements: 4.1, 4.2, 4.4_

- [ ] 21. Error recovery testing
  - Test behavior when GPU memory is exhausted
  - Test recovery from allocation failures
  - Verify graceful degradation
  - Test fallback to CPU if needed
  - _Requirements: 2.4, 3.2, 5.3_

- [ ] 22. Integration testing
  - Test with exo cluster (multi-node)
  - Test with different model families
  - Test with tinygrad backend selector
  - Verify compatibility with other backends
  - _Requirements: 4.1, 4.2, 5.2, 5.3_

- [ ]* 23. Security audit
  - Review buffer overflow prevention
  - Check resource limit enforcement
  - Verify input validation
  - Test with malicious inputs
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 24. Final validation
  - Run complete test suite
  - Verify all requirements are met
  - Document any known limitations
  - Create final validation report
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

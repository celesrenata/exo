# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** — Multi-Node Instance Deleted Despite All Nodes Recently Seen
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing case — a 2-node `PyTorchXPURingInstance` where both nodes are in `last_seen` (recently seen within 30s) but one node is absent from `topology.list_nodes()`. The unfixed code checks `topology.list_nodes()` and will incorrectly emit `InstanceDeleted`.
  - **Test location**: `src/exo/master/tests/test_plan_health_check_properties.py`
  - **What to test**:
    - Extract the `_plan` loop's instance health check into a testable pure function that takes `(instances, topology, last_seen)` and returns a list of `InstanceId`s to delete
    - Create a `State` with a 2-node `PyTorchXPURingInstance` assigned to `gremlin-1` and `gremlin-2`
    - Set `last_seen` for both nodes to 5 seconds ago (well within the 30s liveness window)
    - Add only `gremlin-1` to the topology (gremlin-2 absent — simulates transient topology gap)
    - Assert that the health check does NOT emit a deletion for this instance
    - On UNFIXED code, the topology-presence check will delete the instance — test FAILS, confirming the bug
  - **Bug Condition from design**: `isBugCondition(input)` where `len(node_to_runner) > 1 AND EXISTS node_id NOT IN topology.list_nodes() AND ALL node_id IN last_seen within 30s`
  - **Expected Behavior from design**: instance is NOT deleted when all nodes are recently seen, regardless of topology presence
  - Use Hypothesis to generate varying `last_seen` timestamps (all within 30s) and varying topology membership, asserting no deletion when all nodes are recently seen
  - Run with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/master/tests/test_plan_health_check_properties.py -v --tb=short -k "bug_condition"`
  - **EXPECTED OUTCOME**: Test FAILS (this is correct — it proves the bug exists)
  - Document counterexamples found to understand root cause
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.3, 2.1, 2.2, 2.3, 2.4_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** — Genuinely Disconnected Nodes and Single-Node Instances
  - **IMPORTANT**: Follow observation-first methodology
  - **Test location**: `src/exo/master/tests/test_plan_health_check_properties.py`
  - **Observe on UNFIXED code**:
    - Observe: a single-node instance where the node is in the topology and `last_seen` within 30s → NOT deleted
    - Observe: a 2-node instance where one node has `last_seen` older than 30s → deleted
    - Observe: a 2-node instance where one node is absent from `last_seen` entirely → deleted
    - Observe: a 2-node instance where both nodes are in topology AND `last_seen` within 30s → NOT deleted
  - **Write property-based tests using Hypothesis**:
    - Generate random `(instance, topology, last_seen)` tuples where at least one node has `last_seen` older than 30s or is absent from `last_seen` — assert instance IS deleted (genuinely disconnected preservation)
    - Generate random single-node instances with the node in topology and `last_seen` within 30s — assert instance is NOT deleted (single-node preservation)
    - Generate random multi-node instances where ALL nodes are in topology AND `last_seen` within 30s — assert instance is NOT deleted (this overlaps with the bug condition but captures the non-buggy subset where topology is also present)
  - **Preservation Requirements from design**: single-node lifecycle unchanged; genuinely disconnected nodes (>30s or absent from `last_seen`) still trigger deletion; `NodeTimedOut` mechanism unchanged; `_kill_runner` logic unchanged
  - Run with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/master/tests/test_plan_health_check_properties.py -v --tb=short -k "preservation"`
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Fix the instance shutdown loop in `Master._plan()`

  - [x] 3.1 Replace topology-presence check with `last_seen`-based liveness check
    - In `src/exo/master/main.py`, in the `Master._plan()` method
    - Remove the `connected_node_ids = set(self.state.topology.list_nodes())` line
    - Replace the `if node_id not in connected_node_ids` check with a `last_seen`-based check:
      ```python
      now = datetime.now(tz=timezone.utc)
      for instance_id, instance in self.state.instances.items():
          for node_id in instance.shard_assignments.node_to_runner:
              last_seen_time = self.state.last_seen.get(node_id)
              if last_seen_time is None or (now - last_seen_time) > timedelta(seconds=30):
                  await self.event_sender.send(InstanceDeleted(instance_id=instance_id))
                  break
      ```
    - This aligns the instance health check with the existing 30-second `NodeTimedOut` threshold
    - The `NodeTimedOut` check (second half of `_plan`) remains unchanged
    - _Bug_Condition: isBugCondition(input) where node_id NOT IN topology.list_nodes() but node_id IN last_seen within 30s_
    - _Expected_Behavior: instance NOT deleted when all nodes are in last_seen within 30s_
    - _Preservation: single-node instances, genuinely disconnected nodes, NodeTimedOut logic, _kill_runner all unchanged_
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** — Multi-Node Instance Kept Alive When All Nodes Recently Seen
    - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied:
      - Multi-node instances are NOT deleted when all assigned nodes have `last_seen` within 30s
      - The topology-presence check no longer causes premature deletion
    - Run with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/master/tests/test_plan_health_check_properties.py -v --tb=short -k "bug_condition"`
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** — Genuinely Disconnected Nodes and Single-Node Instances
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run with: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/master/tests/test_plan_health_check_properties.py -v --tb=short -k "preservation"`
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all preservation tests still pass after fix:
      - Single-node instances not deleted when node is healthy
      - Genuinely disconnected nodes (>30s or absent from `last_seen`) still trigger deletion
      - `NodeTimedOut` mechanism unchanged (second half of `_plan` loop untouched)

- [x] 4. Checkpoint — Ensure all tests pass
  - Run the full property test suite: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/master/tests/test_plan_health_check_properties.py -v --tb=short`
  - Run existing master tests to verify no regressions: `LD_LIBRARY_PATH="/nix/store/cf1a53iqg6ncnygl698c4v0l8qam5a2q-gcc-14.3.0-lib/lib:$LD_LIBRARY_PATH" uv run pytest src/exo/master/tests/ -v --tb=short`
  - Verify type checking passes: `uv run basedpyright`
  - Verify linting passes: `uv run ruff check`
  - Ensure all tests pass, ask the user if questions arise.

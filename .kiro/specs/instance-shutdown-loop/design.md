# Instance Shutdown Loop Bugfix Design

## Overview

Multi-node instances (`world_size > 1`) enter a rapid `CreateRunner → Shutdown` loop because the master's `_plan` loop uses an overly aggressive health check that deletes instances whenever any assigned node is absent from the topology graph. The topology graph (`Topology.list_nodes()`) is a connectivity structure that can temporarily lack a node even when that node is healthy and recently seen. Single-node instances are unaffected because the local node is always present in its own topology.

The fix replaces the topology-presence check with a `last_seen`-based liveness check that aligns with the existing 30-second `NodeTimedOut` logic, ensuring multi-node instances survive transient topology gaps while still being cleaned up when a node genuinely disconnects.

## Glossary

- **Bug_Condition (C)**: The master's `_plan` loop deletes a multi-node instance because at least one assigned node is not in `topology.list_nodes()`, even though the node is healthy and recently seen
- **Property (P)**: Multi-node instances should only be deleted when an assigned node has genuinely disconnected (not seen for ≥30 seconds or confirmed failed)
- **Preservation**: Single-node instance lifecycle, genuine disconnection detection, `RunnerFailed` shutdown logic, and download failure handling must remain unchanged
- **`_plan` loop**: The async loop in `Master` (`src/exo/master/main.py`) that polls every 10 seconds to delete broken instances and time out dead nodes
- **`_kill_runner`**: The function in `src/exo/worker/plan.py` that checks whether a local runner's instance has been deleted or a peer runner has failed, returning `Shutdown` if so
- **`topology.list_nodes()`**: Returns all nodes currently in the `rustworkx` directed graph — nodes are added by `NodeGatheredInfo` events and removed by `NodeTimedOut` events
- **`last_seen`**: A `Mapping[NodeId, datetime]` in `State` tracking the most recent `NodeGatheredInfo` timestamp per node

## Bug Details

### Bug Condition

The bug manifests when the master's `_plan` loop runs its instance health check on a multi-node instance. The check uses `topology.list_nodes()` to determine which nodes are "connected", but this set can be temporarily incomplete — a node may have sent `NodeGatheredInfo` recently (and thus be in `last_seen`) but not yet appear in the topology graph due to event processing order, or the topology graph may have been pruned by an edge deletion without the node itself being removed.

The critical code path is in `Master._plan()` (`src/exo/master/main.py`):

```python
connected_node_ids = set(self.state.topology.list_nodes())
for instance_id, instance in self.state.instances.items():
    for node_id in instance.shard_assignments.node_to_runner:
        if node_id not in connected_node_ids:
            await self.event_sender.send(InstanceDeleted(instance_id=instance_id))
            break
```

For single-node instances, the only assigned node is the local master/worker node, which is always in its own topology. For multi-node instances, the remote node must also be present, and this check has no tolerance for transient absence.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (Instance, TopologyState, LastSeenState)
  OUTPUT: boolean

  LET instance = input.instance
  LET topology_nodes = input.topology.list_nodes()
  LET last_seen = input.last_seen
  LET now = current_time()

  RETURN len(instance.shard_assignments.node_to_runner) > 1
         AND EXISTS node_id IN instance.shard_assignments.node_to_runner
             WHERE node_id NOT IN topology_nodes
         AND ALL node_id IN instance.shard_assignments.node_to_runner
             SATISFY (node_id IN last_seen AND (now - last_seen[node_id]) < 30 seconds)
END FUNCTION
```

The bug condition holds when a multi-node instance has a node missing from the topology graph, but ALL nodes have been seen within the 30-second liveness window. The current code deletes the instance; the correct behavior is to keep it alive.

### Examples

- **Example 1**: 2-node PyTorchXPURingInstance across gremlin-1 and gremlin-2. Both nodes sent `NodeGatheredInfo` 5 seconds ago. gremlin-2 is temporarily absent from `topology.list_nodes()` due to event processing delay. Current behavior: instance deleted. Expected: instance kept alive.
- **Example 2**: 2-node instance where gremlin-2 last sent `NodeGatheredInfo` 45 seconds ago (genuinely disconnected). Current behavior: instance deleted. Expected: instance deleted (correct).
- **Example 3**: 1-node instance on gremlin-1. gremlin-1 is always in its own topology. Current behavior: instance kept alive. Expected: instance kept alive (correct, unchanged).
- **Example 4**: 2-node instance where gremlin-2's runner is in `RunnerFailed` state. Current behavior: worker's `_kill_runner` shuts down local runner. Expected: same (correct, unchanged — this is the worker-side check, not the master-side topology check).

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Single-node instance lifecycle (`CreateRunner → DownloadModel → LoadModel → StartWarmup → Ready`) must continue to work exactly as before
- The worker's `_kill_runner` logic must continue to shut down runners when the instance is deleted from state or when a peer runner enters `RunnerFailed`
- The `NodeTimedOut` mechanism (30-second inactivity threshold) must continue to remove genuinely disconnected nodes
- Download failure handling (`DownloadFailed` state) must remain unchanged
- The `_plan` loop's node timeout check (second half of the loop) must remain unchanged
- Mouse/keyboard/API interactions for placing and deleting instances must remain unchanged

**Scope:**
All inputs that do NOT involve the master's `_plan` loop instance health check should be completely unaffected by this fix. This includes:
- Worker-side plan decisions (`_create_runner`, `_model_needs_download`, `_init_distributed_backend`, `_load_model`, `_ready_to_warmup`, `_pending_tasks`)
- Command processing (`PlaceInstance`, `CreateInstance`, `DeleteInstance`)
- Event application (`apply_instance_created`, `apply_instance_deleted`, `apply_node_timed_out`)
- The `_kill_runner` function in `worker/plan.py` (it reacts to state changes, doesn't cause them)

## Hypothesized Root Cause

Based on the code analysis, the root cause is the master's `_plan` loop using `topology.list_nodes()` as the sole liveness signal for instance health checks.

1. **Overly Aggressive Topology Check**: The `_plan` loop treats topology graph membership as a binary "connected/disconnected" signal. But `topology.list_nodes()` reflects the current state of a `rustworkx` directed graph that is updated asynchronously via events. A node can be healthy and recently seen (`last_seen` within seconds) but temporarily absent from the graph due to event processing order or timing.

2. **No Grace Period for New Instances**: When a multi-node instance is created, the `_plan` loop immediately subjects it to the topology check on its next iteration (within 0–10 seconds). There is no grace period to allow the distributed runner lifecycle to initialize. The runners are shut down before they can progress past `RunnerIdle`.

3. **Asymmetry Between Single-Node and Multi-Node**: Single-node instances are immune because the local node is always in its own topology graph (it adds itself via its own `NodeGatheredInfo`). Multi-node instances require ALL remote nodes to be present, creating a much stricter liveness requirement that the topology graph wasn't designed to guarantee.

4. **Redundancy with `NodeTimedOut`**: The `_plan` loop already has a `NodeTimedOut` mechanism that removes nodes after 30 seconds of inactivity. The topology check is a redundant, more aggressive version of the same concept. When `NodeTimedOut` fires, it removes the node from the topology AND from `last_seen`, which would naturally cause the instance to be cleaned up on the next `_plan` iteration. The topology check short-circuits this by deleting instances before the timeout has elapsed.

## Correctness Properties

Property 1: Bug Condition — Multi-node instances with all nodes recently seen are not deleted

_For any_ multi-node instance where all assigned nodes have a `last_seen` timestamp within the liveness threshold (30 seconds), the master's `_plan` health check SHALL NOT emit an `InstanceDeleted` event for that instance, regardless of whether the nodes are present in `topology.list_nodes()`.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation — Genuinely disconnected nodes still trigger instance deletion

_For any_ multi-node instance where at least one assigned node has NO `last_seen` entry or has a `last_seen` timestamp older than the liveness threshold (30 seconds), the master's `_plan` health check SHALL emit an `InstanceDeleted` event for that instance, preserving the existing cleanup behavior for genuinely disconnected nodes.

**Validates: Requirements 3.2**

Property 3: Preservation — Single-node instance lifecycle unchanged

_For any_ single-node instance, the master's `_plan` health check SHALL NOT emit an `InstanceDeleted` event as long as the single assigned node has a `last_seen` timestamp within the liveness threshold, preserving the existing single-node behavior.

**Validates: Requirements 3.1**

Property 4: Preservation — RunnerFailed shutdown logic unchanged

_For any_ runner whose peer runner in the same instance is in `RunnerFailed` state, the worker's `_kill_runner` function SHALL continue to return `Shutdown`, regardless of changes to the master's health check.

**Validates: Requirements 3.3**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/exo/master/main.py`

**Function**: `Master._plan()`

**Specific Changes**:

1. **Replace topology-presence check with `last_seen`-based liveness check**: Instead of checking `node_id not in connected_node_ids` (where `connected_node_ids` comes from `topology.list_nodes()`), check whether the node has been seen within the liveness threshold using `self.state.last_seen`.

   - Before:
     ```python
     connected_node_ids = set(self.state.topology.list_nodes())
     for instance_id, instance in self.state.instances.items():
         for node_id in instance.shard_assignments.node_to_runner:
             if node_id not in connected_node_ids:
                 await self.event_sender.send(InstanceDeleted(instance_id=instance_id))
                 break
     ```
   - After:
     ```python
     now = datetime.now(tz=timezone.utc)
     for instance_id, instance in self.state.instances.items():
         for node_id in instance.shard_assignments.node_to_runner:
             last_seen_time = self.state.last_seen.get(node_id)
             if last_seen_time is None or (now - last_seen_time) > timedelta(seconds=30):
                 await self.event_sender.send(InstanceDeleted(instance_id=instance_id))
                 break
     ```

2. **Align liveness threshold with `NodeTimedOut`**: Use the same 30-second threshold that the `NodeTimedOut` check uses. This ensures consistency — a node is considered "alive" by the same standard everywhere.

3. **No changes to the `NodeTimedOut` check**: The second half of the `_plan` loop (timing out dead nodes) remains unchanged. It continues to remove nodes from the topology after 30 seconds of inactivity.

4. **No changes to `_kill_runner`**: The worker-side shutdown logic remains unchanged. It continues to react to `InstanceDeleted` events and `RunnerFailed` states.

5. **No changes to `place_instance` or `add_instance_to_placements`**: The placement logic remains unchanged. It continues to use topology cycles for placement decisions.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that construct a `State` with a multi-node instance where all nodes are in `last_seen` (recently seen) but one node is absent from the topology graph. Run the `_plan` loop's health check logic on the UNFIXED code to observe that it incorrectly emits `InstanceDeleted`.

**Test Cases**:
1. **Two-node instance, one node absent from topology**: Create state with 2-node instance, both nodes in `last_seen` (5s ago), but only one node in topology. Assert that the unfixed code emits `InstanceDeleted` (will fail on unfixed code — i.e., the bug is confirmed).
2. **Two-node instance, both nodes in topology**: Create state with 2-node instance, both nodes in topology and `last_seen`. Assert that no `InstanceDeleted` is emitted (should pass on unfixed code).
3. **Two-node instance, one node timed out**: Create state with 2-node instance, one node's `last_seen` is 45 seconds ago. Assert that `InstanceDeleted` is emitted (should pass on unfixed code).
4. **Single-node instance, node in topology**: Create state with 1-node instance, node in topology. Assert no `InstanceDeleted` (should pass on unfixed code).

**Expected Counterexamples**:
- Test case 1 will demonstrate the bug: the unfixed code deletes the instance even though both nodes are healthy
- Root cause confirmed: the topology-presence check is too aggressive

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := plan_health_check_fixed(input)
  ASSERT result does NOT contain InstanceDeleted for input.instance
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT plan_health_check_original(input) = plan_health_check_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain (varying numbers of nodes, instance configurations, `last_seen` timestamps, topology states)
- It catches edge cases that manual unit tests might miss (e.g., nodes with exactly 30-second-old timestamps, empty `last_seen` maps)
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for non-buggy inputs (single-node instances, genuinely disconnected nodes), then write property-based tests capturing that behavior.

**Test Cases**:
1. **Single-node preservation**: Generate random single-node instances with various `last_seen` timestamps. Verify the fixed code produces the same deletion decisions as the original code.
2. **Genuinely disconnected preservation**: Generate random multi-node instances where at least one node has `last_seen` older than 30 seconds or is absent from `last_seen`. Verify the fixed code still deletes these instances.
3. **RunnerFailed preservation**: Generate random runner states with `RunnerFailed` peers. Verify `_kill_runner` behavior is unchanged.

### Unit Tests

- Test the `_plan` health check with a 2-node instance where both nodes are recently seen but one is absent from topology (bug condition — should NOT delete after fix)
- Test the `_plan` health check with a 2-node instance where one node has timed out (should delete)
- Test the `_plan` health check with a 1-node instance (should not delete if node is healthy)
- Test edge case: node in `last_seen` with exactly 30-second-old timestamp (boundary condition)
- Test edge case: node not in `last_seen` at all (should delete)
- Test that `_kill_runner` still returns `Shutdown` when instance is deleted from state
- Test that `_kill_runner` still returns `Shutdown` when peer runner is `RunnerFailed`

### Property-Based Tests

- Generate random `(instance, topology, last_seen)` tuples and verify: if all nodes are in `last_seen` within 30 seconds, no `InstanceDeleted` is emitted
- Generate random `(instance, topology, last_seen)` tuples and verify: if any node is missing from `last_seen` or has `last_seen` > 30 seconds, `InstanceDeleted` IS emitted
- Generate random single-node instances and verify the fixed health check produces identical results to the original

### Integration Tests

- Test full 2-node instance lifecycle: place instance, verify runners progress through `CreateRunner → DownloadModel → ConnectToGroup → LoadModel → StartWarmup → Ready` without premature shutdown
- Test that a genuinely disconnected node (simulated by removing from `last_seen`) causes instance deletion after the timeout
- Test that the `_plan` loop's `NodeTimedOut` logic still works correctly alongside the fixed health check

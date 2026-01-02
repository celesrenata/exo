# Exo Instance FF8A5194 Failure Analysis

## Problem Summary
Instance `ff8a5194-7c1d-4657-939f-db88917bbd69` (TinyLlama/TinyLlama-1.1B-Chat-v1.0) transitions from `unknown` → `failed` state when connecting two CPU hosts.

## Cluster Configuration
- **Node 1 (gremlin-1)**: Intel Core Ultra 9 185H @ 10.1.1.12
  - Engine: torch (CPU)
  - Runner ID: `f26c221e-95b0-47e3-8cf7-fcc2cd49f08e`
  - Shard: Layers 0-11 (device_rank=0)

- **Node 2 (gremlin-3)**: @ 10.1.1.14  
  - Engine: torch (CPU)
  - Runner ID: `3d1300ff-488b-4985-a1f4-0b5db1e2b38c`
  - Shard: Layers 11-22 (device_rank=1)

## Root Cause: Cascading Failure Loop

**Location**: `/src/exo/worker/plan.py` in `_kill_runner()` function

The runners are starting successfully but **immediately shutting down** due to a cascading failure pattern:

```
[ 03:33:11.218 | INFO ] hello from the runner
[ 03:33:11.218 | INFO ] runner created
[ 03:33:11.219 | INFO ] runner shutting down  ← Immediate shutdown!
[ 03:33:11.221 | INFO ] bye from the runner
```

### The Failure Loop

1. **Initial Failure**: One runner fails (likely during distributed initialization)
2. **Cascading Shutdown**: The `_kill_runner()` function detects the failed runner:
   ```python
   for global_runner_id in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
       if isinstance(all_runners.get(global_runner_id, None), RunnerFailed):
           return Shutdown(instance_id=instance_id, runner_id=runner_id)
   ```
3. **All Runners Killed**: When ANY runner in a distributed instance fails, ALL runners are shut down
4. **Restart Attempt**: System tries to recreate runners
5. **Immediate Re-failure**: New runners see the other node's runner as failed and shut down
6. **Infinite Loop**: Steps 2-5 repeat indefinitely

### Why This Happens

The logic assumes: "If one shard fails, the entire distributed instance is invalid, so kill all shards."

This is correct for cleanup, but creates a problem when:
- Runners are being created asynchronously across nodes
- One runner fails during initialization
- The other runner(s) haven't had time to initialize yet
- The failed runner triggers shutdown of healthy runners
- Creates a race condition where no runner can ever fully initialize

## Error Symptoms

### Primary Error
```
RunnerFailed(errorMessage="Terminated (exitcode=1)")
```

### Secondary Errors (cleanup issues)
```python
ValueError: Queue <multiprocessing.queues.Queue object at 0x...> is closed
anyio.ClosedResourceError
```

These are **consequences** of the premature shutdown, not the root cause.

## Possible Causes

### 1. Race Condition in Task Scheduling
The logs show:
```
[ 03:33:11.218 | INFO ] runner created
[ 03:33:11.219 | INFO ] runner shutting down
```

Only **1ms** between creation and shutdown suggests a race condition where:
- A shutdown task is being sent before/during runner initialization
- The runner receives the shutdown command immediately after starting

### 2. Instance State Mismatch
The instance keeps getting recreated and shut down in a loop, suggesting:
- The master/coordinator thinks the instance should be shut down
- But the system keeps trying to recreate it
- This could be a state synchronization issue between nodes

### 3. Network/Communication Issue
The runners are on different nodes communicating over:
- `10.1.1.12:52414` (gremlin-1)
- `10.42.2.1:52414` (gremlin-3 via flannel overlay)

The immediate shutdown might indicate:
- Failure to establish inter-runner communication
- Timeout during distributed initialization
- Network connectivity issues between shards

## Investigation Steps

### 1. Check Runner Main Loop
Look at `/src/exo/worker/runner/runner.py` line 195 to see what triggers "runner shutting down"

### 2. Check Task Queue
Examine why shutdown tasks are being created immediately after runner creation:
```
Worker plan: CreateRunner
Worker plan: Shutdown  ← Why immediately after?
```

### 3. Check Distributed Initialization
For CPU ring instances with world_size=2, verify:
- torch.distributed initialization
- Inter-node communication setup
- Timeout values for distributed operations

### 4. Check Network Connectivity
Test connectivity between:
- `10.1.1.12:52414` ↔ `10.42.2.1:52414`
- Verify no firewall/routing issues

## Recommended Fixes

### Fix 1: Add Grace Period for Distributed Initialization (Recommended)

Don't immediately kill runners if a peer fails during the initialization phase. Add a grace period:

```python
def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)

        for global_runner_id in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
            if runner_id == global_runner_id:
                continue

            peer_status = all_runners.get(global_runner_id, None)
            
            # Only kill if peer is failed AND we're past initialization
            if isinstance(peer_status, RunnerFailed):
                # Don't cascade shutdown if we're still initializing
                if isinstance(runner.status, (RunnerIdle, RunnerConnecting)):
                    continue  # Give initialization time to complete
                    
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )
```

### Fix 2: Add Retry Logic with Backoff

Instead of immediately recreating failed runners, add exponential backoff:

```python
# In worker/main.py
case CreateRunner():
    # Check if this runner recently failed
    if self._should_backoff(task.bound_instance.bound_runner_id):
        continue  # Skip creation, wait for backoff period
        
    self._create_supervisor(task)
```

### Fix 3: Better Distributed Initialization

Ensure all runners reach a "connected" state before allowing failures to cascade:

```python
# Only kill if ALL runners were previously healthy
all_runners_were_healthy = all(
    isinstance(all_runners.get(rid), (RunnerConnected, RunnerLoaded, RunnerReady))
    for rid in runner.bound_instance.instance.shard_assignments.node_to_runner.values()
)

if isinstance(peer_status, RunnerFailed) and all_runners_were_healthy:
    return Shutdown(instance_id=instance_id, runner_id=runner_id)
```

### Fix 4: Root Cause - Why Initial Failure?

The logs don't show WHY the first runner fails. Need to investigate:

1. **Torch Distributed Init Timeout**: Check if `torch.distributed.init_process_group()` is timing out
2. **Network Connectivity**: Verify nodes can reach each other on port 52414
3. **CPU Backend Issues**: The CPU torch backend might have specific requirements

Add debug logging in `runner.py` before distributed init:
```python
logger.info(f"Attempting distributed init: rank={device_rank}, world_size={world_size}, backend={backend}")
try:
    torch.distributed.init_process_group(...)
    logger.info("Distributed init successful")
except Exception as e:
    logger.error(f"Distributed init failed: {e}")
    raise
```

## Next Steps

1. Examine `runner.py:main()` to understand shutdown trigger
2. Check if this is specific to CPU torch backend
3. Test with single-node instance to isolate distributed vs local issues
4. Add debug logging to capture the shutdown reason

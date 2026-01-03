# Runner Status Update Fix Summary

## Issue
The EXO system was showing a status sequence of "loading" → "loaded" → "failed" → "unknown" during runner lifecycle. This indicated that status transitions were not being properly communicated to the supervisor.

## Root Cause
The runner state machine in `src/exo/worker/runner/runner.py` was updating internal status variables but failing to send status update events to the supervisor in several critical transitions:

1. **After LoadModel completion**: Status was set to `RunnerLoaded()` but no event was sent
2. **After StartWarmup completion**: Status was set to `RunnerReady()` but no event was sent  
3. **After ChatCompletion completion**: Status was set back to `RunnerReady()` but no event was sent

## Solution
Added missing `event_sender.send(RunnerStatusUpdated(...))` calls for all status transitions:

### Fix 1: LoadModel Completion
```python
# Before: Missing status notification
model, tokenizer, sampler = initialize_engine(bound_instance)
current_status = RunnerLoaded()
logger.info("runner loaded")

# After: Added status notification
model, tokenizer, sampler = initialize_engine(bound_instance)
current_status = RunnerLoaded()
logger.info("runner loaded")
event_sender.send(
    RunnerStatusUpdated(
        runner_id=runner_id, runner_status=current_status
    )
)
```

### Fix 2: StartWarmup Completion
```python
# Before: Missing status notification
current_status = RunnerReady()
logger.info("runner ready")

# After: Added status notification
current_status = RunnerReady()
logger.info("runner ready")
event_sender.send(
    RunnerStatusUpdated(
        runner_id=runner_id, runner_status=current_status
    )
)
```

### Fix 3: ChatCompletion Completion
```python
# Before: Missing status notification after generation
current_status = RunnerReady()
logger.info("runner ready")

# After: Added status notification
current_status = RunnerReady()
logger.info("runner ready")
event_sender.send(
    RunnerStatusUpdated(
        runner_id=runner_id, runner_status=current_status
    )
)
```

## Impact
- **Consistent status reporting**: The supervisor now receives all status transitions
- **Eliminates "unknown" status**: Proper status notifications prevent the supervisor from losing track of runner state
- **Better debugging**: Status transitions are now properly logged and tracked
- **Prevents false failures**: Missing status updates were likely causing the supervisor to think runners had failed

## Expected Behavior After Fix
The status sequence should now be:
1. `loading` (RunnerLoading) - properly notified
2. `loaded` (RunnerLoaded) - **now properly notified**
3. `warming_up` (RunnerWarmingUp) - already properly notified
4. `ready` (RunnerReady) - **now properly notified**
5. `running` (RunnerRunning) - already properly notified
6. `ready` (RunnerReady) - **now properly notified** after completion

## Related Requirements
This fix addresses requirements from the multinode race condition fix spec:
- **Requirement 3.2**: "WHEN debugging is needed, THE Runner_Supervisor SHALL log lifecycle events clearly"
- **Requirement 5.4**: "WHILE recovering from errors, THE Runner_Supervisor SHALL maintain system state consistency"

The missing status notifications were causing state inconsistencies between the runner and supervisor, leading to the "failed" → "unknown" status transitions.
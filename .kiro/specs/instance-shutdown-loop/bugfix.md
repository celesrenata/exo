# Bugfix Requirements Document

## Introduction

When starting a model on 2 nodes (distributed inference with `world_size=2`), the master creates a `PyTorchXPURingInstance` and assigns it to workers, but the workers immediately receive a Shutdown decision before they can progress through the runner lifecycle (download → connect → load → warmup → ready). This creates a rapid `CreateRunner → Shutdown` loop repeating every ~2 seconds, preventing multi-node inference from ever succeeding. Single-node inference (`world_size=1`) works correctly.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a multi-node instance (`world_size > 1`) is placed across two or more nodes THEN the worker's `plan_step` emits `CreateRunner` followed by `Shutdown` within ~100ms, repeating every ~2 seconds indefinitely

1.2 WHEN a multi-node instance is created and runners are assigned to workers THEN the runners never progress past the `RunnerIdle` state to `LoadModel` because they are shut down before the download/connect/load lifecycle can complete

1.3 WHEN the master's `_plan` loop checks instance health for a multi-node instance THEN it deletes the instance (emitting `InstanceDeleted`) because one or more assigned nodes are not yet present in the topology or the instance is deemed broken, even though the nodes are connected and healthy

1.4 WHEN the instance is deleted and then re-placed by the orchestration layer THEN the same instance_id and runner_id cycle through creation and shutdown repeatedly, with the same pattern visible in logs: `Worker plan: CreateRunner` → `Worker plan: Shutdown`

### Expected Behavior (Correct)

2.1 WHEN a multi-node instance (`world_size > 1`) is placed across two or more nodes THEN the workers SHALL create runners and progress through the full lifecycle (`CreateRunner → DownloadModel → ConnectToGroup → LoadModel → StartWarmup → Ready`) without premature shutdown

2.2 WHEN a multi-node instance is created and runners are assigned to workers THEN the runners SHALL remain alive long enough to complete model download, distributed backend initialization, and model loading before any shutdown decision is evaluated

2.3 WHEN the master's `_plan` loop checks instance health for a multi-node instance THEN it SHALL NOT delete the instance if all assigned nodes are reachable in the topology and no runner has entered a `RunnerFailed` state

2.4 WHEN a multi-node instance is placed and the runner lifecycle is in progress (downloading, connecting, loading) THEN the system SHALL NOT trigger instance deletion or runner shutdown due to transient state synchronization delays between nodes

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a single-node instance (`world_size=1`) is placed THEN the system SHALL CONTINUE TO create the runner, download the model, load it, warm up, and reach the `RunnerReady` state successfully

3.2 WHEN a node genuinely disconnects from the cluster topology (e.g., network failure, node crash) THEN the system SHALL CONTINUE TO detect the disconnection and delete instances that include the disconnected node

3.3 WHEN a runner enters the `RunnerFailed` state on any node in a multi-node instance THEN the system SHALL CONTINUE TO shut down the other runners in that instance via the `_kill_runner` logic

3.4 WHEN a model download fails or the download state reports `DownloadFailed` THEN the system SHALL CONTINUE TO handle the failure appropriately without masking real errors

3.5 WHEN the background model card scanner encounters a `FileNotFoundError` for an unrelated model THEN the system SHALL CONTINUE TO report the error in logs without affecting the download state of the model currently being loaded
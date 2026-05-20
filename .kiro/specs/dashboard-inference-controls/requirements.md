# Requirements Document

## Introduction

This feature adds inference control settings and real-time hardware telemetry to the exo dashboard. The inference controls allow users to manage thinking mode, thinking token budget, and output token budget from the UI — addressing the immediate problem of Qwen3.5 generating unwanted `<think>...</think>` reasoning tokens. The telemetry panels provide real-time GPU utilization and network bandwidth monitoring for all nodes in the gremlin cluster.

## Glossary

- **Dashboard**: The Svelte 5 + TypeScript frontend served from `dashboard/build/` by the FastAPI backend at port 52415
- **Generation_Settings**: A cluster-wide configuration object containing inference parameters (thinking mode, thinking token budget, output token budget) stored in event-sourced state
- **Thinking_Mode**: A boolean toggle controlling whether the model generates `<think>...</think>` reasoning tokens before visible output (maps to `enable_thinking` parameter)
- **Thinking_Token_Budget**: The maximum number of tokens the model allocates for internal reasoning within `<think>...</think>` blocks
- **Output_Token_Budget**: The maximum number of visible output tokens the model generates (maps to `max_tokens` / `max_output_tokens` parameter)
- **Settings_API**: The FastAPI endpoints (`GET /api/generation/settings`, `PATCH /api/generation/settings`) for reading and updating Generation_Settings
- **Telemetry_API**: The FastAPI endpoints (`GET /api/telemetry/cluster`, `GET /api/telemetry/stream`) for reading cluster hardware metrics
- **Telemetry_Collector**: A per-node background service that gathers GPU and network metrics at regular intervals
- **GPU_Metrics**: A data structure containing GPU frequency, utilization percentage, render/compute engine busy percentage, and memory bandwidth usage for a single node
- **Network_Metrics**: A data structure containing bytes sent/received, throughput, and latency measurements for inter-node communication
- **SSE_Stream**: A Server-Sent Events connection from the dashboard to `GET /api/telemetry/stream` for receiving real-time metric updates
- **Stale_Metric**: A metric whose timestamp exceeds a staleness threshold (no update received within the expected collection interval), visually distinguished from zero-value metrics
- **Node**: One of the four gremlin machines (gremlin-1 through gremlin-4) in the cluster, each running an exo process
- **Cluster_Interface**: The network interface used for inter-node communication (10.1.1.0/24 subnet), excluding loopback and Docker interfaces

## Requirements

### Requirement 1: Generation Settings State Management

**User Story:** As a cluster operator, I want generation settings stored in event-sourced state and broadcast to all nodes, so that all nodes use consistent inference parameters.

#### Acceptance Criteria

1. THE Generation_Settings SHALL contain fields for thinking_mode (boolean, default: true), thinking_token_budget (integer in the range 1 to 1,048,576 or null, default: null), and output_token_budget (integer in the range 1 to 1,048,576 or null, default: null)
2. WHEN a Generation_Settings change event is applied, THE State SHALL update the generation_settings field with the new values and broadcast the updated state to all nodes via the event-sourcing mechanism
3. WHEN a new inference request is received by the Worker (task created and acknowledged), THE Worker SHALL snapshot the current Generation_Settings and use the snapshotted values for the entire duration of that request
4. WHEN Generation_Settings change during an in-flight request, THE Worker SHALL continue using the snapshotted values captured at request start without interruption
5. WHEN no Generation_Settings change event has been applied, THE State SHALL use the default Generation_Settings values (thinking_mode: true, thinking_token_budget: null, output_token_budget: null)

### Requirement 2: Generation Settings API

**User Story:** As a dashboard user, I want API endpoints to read and update generation settings, so that the dashboard can control inference parameters.

#### Acceptance Criteria

1. WHEN a GET request is sent to `/api/generation/settings`, THE Settings_API SHALL return the current Generation_Settings as JSON
2. WHEN a PATCH request is sent to `/api/generation/settings` with valid partial settings, THE Settings_API SHALL emit a settings-changed event and return the updated Generation_Settings
3. WHEN a PATCH request contains an invalid value (negative token budget, non-boolean thinking_mode), THE Settings_API SHALL return HTTP 422 with a descriptive error message
4. WHEN a PATCH request sets thinking_mode to false, THE Settings_API SHALL accept the request regardless of thinking_token_budget value

### Requirement 3: Thinking Mode Toggle

**User Story:** As a user chatting with Qwen3.5, I want to disable thinking mode from the dashboard, so that the model produces visible output instead of consuming the token budget on internal reasoning.

#### Acceptance Criteria

1. THE Dashboard SHALL display a thinking mode toggle control in the inference settings panel that visually indicates the current state (on or off)
2. WHEN the user toggles thinking mode off, THE Dashboard SHALL send a PATCH to `/api/generation/settings` with `thinking_mode: false` within 5 seconds
3. IF the PATCH request to update thinking_mode fails or times out, THEN THE Dashboard SHALL revert the toggle to its previous state and display an error message indicating the setting was not saved
4. IF thinking_mode is false in the current Generation_Settings, THEN THE Worker SHALL pass `enable_thinking=False` to the model's chat template, suppressing `<think>...</think>` token generation
5. IF thinking_mode is true in the current Generation_Settings, THEN THE Worker SHALL pass `enable_thinking=True` to the model's chat template, allowing reasoning token generation
6. WHEN the Dashboard page loads, THE Dashboard SHALL read the current thinking mode state from the Settings_API and set the toggle to match the returned value

### Requirement 4: Thinking Token Budget Control

**User Story:** As a user who enables thinking mode, I want to set a maximum thinking token budget, so that the model does not spend excessive tokens on internal reasoning.

#### Acceptance Criteria

1. THE Dashboard SHALL display a numeric input for thinking token budget in the inference settings panel
2. WHILE thinking_mode is disabled, THE Dashboard SHALL hide or disable the thinking token budget input
3. WHEN the user sets a thinking token budget value, THE Dashboard SHALL send a PATCH to `/api/generation/settings` with the new thinking_token_budget
4. WHEN thinking_token_budget is set, THE Worker SHALL limit the model's reasoning token generation to the specified budget
5. WHEN thinking_token_budget is null, THE Worker SHALL allow unlimited reasoning tokens (model default behavior)

### Requirement 5: Output Token Budget Control

**User Story:** As a user, I want to set the maximum number of visible output tokens, so that I can control response length.

#### Acceptance Criteria

1. THE Dashboard SHALL display a numeric input for output token budget in the inference settings panel
2. WHEN the user sets an output token budget value, THE Dashboard SHALL send a PATCH to `/api/generation/settings` with the new output_token_budget
3. WHEN output_token_budget is set, THE Worker SHALL pass the value as `max_tokens` to the inference engine
4. WHEN output_token_budget is null, THE Worker SHALL use the model's default maximum output length

### Requirement 6: GPU Performance Telemetry Collection

**User Story:** As a cluster operator, I want each node to collect GPU utilization metrics, so that I can monitor hardware performance during inference.

#### Acceptance Criteria

1. THE Telemetry_Collector SHALL collect GPU_Metrics from each node at a rate of 1 sample per 800ms to 1200ms
2. THE Telemetry_Collector SHALL read GPU frequency from sysfs (`/sys/class/drm/card0/gt_cur_freq_mhz`) or `intel_gpu_top -J` output
3. THE Telemetry_Collector SHALL read GPU utilization percentage, render/compute engine busy percentage, and memory bandwidth usage
4. THE Telemetry_Collector SHALL use `intel_gpu_top -J` as the primary data source with sysfs as a fallback
5. IF `intel_gpu_top` is unavailable or returns an error, THEN THE Telemetry_Collector SHALL fall back to sysfs metrics and include a coverage field in the GPU_Metrics indicating which metrics are unavailable from the degraded source
6. THE Telemetry_Collector SHALL consume less than 2% CPU overhead averaged over any 60-second window on each node during metric collection
7. IF both `intel_gpu_top` and sysfs are unavailable on a node, THEN THE Telemetry_Collector SHALL emit a GPU_Metrics entry with all values absent and a status indicating total collection failure
8. THE Telemetry_Collector SHALL include a UTC timestamp in each GPU_Metrics sample indicating when the measurement was taken

### Requirement 7: Network Bandwidth Telemetry Collection

**User Story:** As a cluster operator, I want to monitor inter-node network bandwidth, so that I can identify communication bottlenecks.

#### Acceptance Criteria

1. THE Telemetry_Collector SHALL collect Network_Metrics for the Cluster_Interface on each node at approximately 1 Hz
2. THE Telemetry_Collector SHALL read bytes sent and bytes received from `/proc/net/dev` for the Cluster_Interface
3. THE Telemetry_Collector SHALL compute current throughput (bytes/second) between consecutive samples
4. THE Telemetry_Collector SHALL measure latency between nodes using libp2p connection statistics or ICMP probes
5. THE Telemetry_Collector SHALL exclude loopback, Docker bridge, and virtual interfaces from network metrics

### Requirement 8: Telemetry Streaming API

**User Story:** As a dashboard developer, I want a streaming API for telemetry data, so that the dashboard receives real-time metric updates without polling.

#### Acceptance Criteria

1. WHEN a client connects to `GET /api/telemetry/stream`, THE Telemetry_API SHALL establish an SSE connection and stream metric updates
2. THE Telemetry_API SHALL send GPU_Metrics and Network_Metrics for all reporting nodes through the SSE_Stream
3. WHEN a node stops reporting metrics, THE Telemetry_API SHALL include a staleness indicator for that node's metrics after the staleness threshold (3 seconds without update)
4. WHEN a client connects to `GET /api/telemetry/cluster`, THE Telemetry_API SHALL return the latest snapshot of all node metrics as JSON
5. IF a node is offline, THEN THE Telemetry_API SHALL omit that node from the metrics response rather than returning zero values

### Requirement 9: GPU Telemetry Dashboard Display

**User Story:** As a cluster operator, I want to see GPU utilization for each node in the dashboard, so that I can monitor inference workload distribution.

#### Acceptance Criteria

1. THE Dashboard SHALL display a GPU telemetry panel showing metrics for each node in the cluster
2. THE Dashboard SHALL show GPU frequency (MHz), GPU utilization (%), render/compute engine busy (%), and memory bandwidth usage for each node
3. WHEN a metric is stale (no update within 3 seconds), THE Dashboard SHALL visually distinguish the stale value from a live zero value
4. WHEN a node is offline, THE Dashboard SHALL indicate the node is unavailable rather than showing zero metrics
5. THE Dashboard SHALL update GPU metrics in real-time via the SSE_Stream connection

### Requirement 10: Network Telemetry Dashboard Display

**User Story:** As a cluster operator, I want to see network bandwidth between nodes in the dashboard, so that I can identify communication issues.

#### Acceptance Criteria

1. THE Dashboard SHALL display a network telemetry panel showing metrics for each node
2. THE Dashboard SHALL show bytes sent/received, current throughput (formatted in human-readable units), and latency between nodes
3. WHEN a metric is stale, THE Dashboard SHALL visually distinguish the stale value from a live zero value
4. WHEN a node is offline, THE Dashboard SHALL indicate the node is unavailable rather than showing zero metrics
5. THE Dashboard SHALL update network metrics in real-time via the SSE_Stream connection

### Requirement 11: Graceful Degradation

**User Story:** As a user, I want the dashboard to work with partial data, so that individual node failures do not break the entire monitoring view.

#### Acceptance Criteria

1. WHEN some nodes are offline, THE Dashboard SHALL display available metrics for online nodes and indicate offline status for unreachable nodes
2. WHEN the SSE_Stream connection drops, THE Dashboard SHALL attempt reconnection with exponential backoff and display a connection status indicator
3. WHEN the Settings_API is unreachable, THE Dashboard SHALL display the last known settings and indicate the connection issue
4. IF a node reports partial metrics (GPU available but network unavailable), THEN THE Dashboard SHALL display the available metrics and indicate missing data for unavailable metrics

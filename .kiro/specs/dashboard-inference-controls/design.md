# Design Document: Dashboard Inference Controls

## Overview

This feature adds two capabilities to the exo dashboard:

1. **Inference Settings Panel** — UI controls for thinking mode toggle, thinking token budget, and output token budget. These settings are stored in event-sourced `State`, broadcast to all nodes via the existing pub/sub mechanism, and applied to inference requests at the worker level.

2. **Hardware Telemetry Panels** — Real-time GPU utilization and network bandwidth monitoring for all gremlin cluster nodes, streamed via SSE from a per-node telemetry collector.

The primary motivation is controlling Qwen3.5's `<think>...</think>` reasoning token generation from the dashboard, while the telemetry panels provide operational visibility into the cluster during inference.

## Architecture

```mermaid
graph TD
    subgraph Dashboard [Dashboard - Svelte 5]
        ISP[InferenceSettingsPanel]
        GTP[GpuTelemetryPanel]
        NTP[NetworkTelemetryPanel]
        TS[TelemetryStore]
        GS[GenerationSettingsStore]
    end

    subgraph API [FastAPI - gremlin-1]
        SA[GET/PATCH /api/generation/settings]
        TE[GET /api/telemetry/stream]
        TC[GET /api/telemetry/cluster]
        AGG[TelemetryAggregator]
    end

    subgraph EventSourcing [Event Sourcing]
        STATE[State.generation_settings]
        APPLY[apply - GenerationSettingsUpdated]
        PUB[GLOBAL_EVENTS pub/sub]
    end

    subgraph Nodes [Per-Node]
        COL[TelemetryCollector]
        IGT[intel_gpu_top -J]
        SYS[sysfs fallback]
        PND[/proc/net/dev]
        WORKER[Worker - inference]
    end

    ISP -->|PATCH| SA
    SA -->|emit event| APPLY
    APPLY --> STATE
    STATE -->|broadcast| PUB
    PUB --> WORKER

    GS -->|GET on load| SA
    ISP --> GS

    TS -->|SSE| TE
    GTP --> TS
    NTP --> TS

    COL --> IGT
    COL --> SYS
    COL --> PND
    COL -->|report| AGG
    AGG --> TE
    AGG --> TC
```

### Data Flow

1. **Settings write path**: Dashboard → PATCH `/api/generation/settings` → API emits `GenerationSettingsUpdated` event → master indexes and broadcasts → all workers apply to local state.
2. **Settings read path**: Dashboard → GET `/api/generation/settings` → API reads from `State.generation_settings`.
3. **Inference consumption**: Worker snapshots `State.generation_settings` at task acknowledgment → maps fields to `TextGenerationTaskParams` overrides (`enable_thinking`, `max_output_tokens`).
4. **Telemetry collection**: Each node runs a `TelemetryCollector` background task (~1 Hz) → reports metrics to the master's `TelemetryAggregator` via internal pub/sub.
5. **Telemetry streaming**: Dashboard opens SSE to `/api/telemetry/stream` → aggregator pushes updates as they arrive from nodes.

## Components and Interfaces

### Backend Components

#### `GenerationSettings` (Pydantic model)

Location: `src/exo/shared/types/generation_settings.py`

```python
from pydantic import Field
from exo.utils.pydantic_ext import FrozenModel

class GenerationSettings(FrozenModel):
    thinking_mode: bool = True
    thinking_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
    output_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
```

#### `GenerationSettingsUpdated` (Event)

Location: `src/exo/shared/types/events.py` (added to the `Event` union)

```python
class GenerationSettingsUpdated(BaseEvent):
    generation_settings: GenerationSettings
```

#### `State.generation_settings` field

Location: `src/exo/shared/types/state.py`

```python
class State(FrozenModel):
    # ... existing fields ...
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
```

#### `apply` handler

Location: `src/exo/shared/apply.py`

```python
def apply_generation_settings_updated(event: GenerationSettingsUpdated, state: State) -> State:
    return state.model_copy(update={"generation_settings": event.generation_settings})
```

#### Settings API endpoints

Location: `src/exo/api/generation_settings.py` (new module, registered as a FastAPI router)

- `GET /api/generation/settings` → returns `State.generation_settings` as JSON
- `PATCH /api/generation/settings` → validates partial input, merges with current settings, emits `GenerationSettingsUpdated` event via command sender, returns updated settings

#### `UpdateGenerationSettings` (Command)

Location: `src/exo/shared/types/commands.py`

```python
class UpdateGenerationSettings(BaseCommand):
    generation_settings: GenerationSettings
```

The master receives this command, creates a `GenerationSettingsUpdated` event, indexes it, and broadcasts.

#### Worker settings resolution

Location: `src/exo/worker/runner/runner.py` (modification to task handling)

When a `TextGeneration` task is acknowledged, the worker snapshots `State.generation_settings` and applies overrides to `TextGenerationTaskParams`:

- `thinking_mode=False` → set `enable_thinking=False` (unless the request explicitly set it)
- `thinking_token_budget` → set as thinking budget limit (passed to chat template)
- `output_token_budget` → set as `max_output_tokens` (unless the request explicitly set it)

Request-level parameters take precedence over cluster-wide settings.

#### `TelemetryCollector`

Location: `src/exo/telemetry/collector.py`

Background asyncio task running on each node:
- Spawns `intel_gpu_top -J -s 900` subprocess (900ms interval)
- Parses JSON output for GPU frequency, utilization, render/compute busy, memory bandwidth
- Falls back to sysfs reads if `intel_gpu_top` is unavailable
- Reads `/proc/net/dev` for the cluster interface (identified by matching 10.1.1.x subnet)
- Computes throughput from consecutive byte count deltas
- Measures inter-node latency via libp2p ping or ICMP
- Reports `NodeTelemetry` to the master via internal pub/sub at ~1 Hz

#### `TelemetryAggregator`

Location: `src/exo/api/telemetry.py`

Runs on the master (API) node:
- Receives telemetry reports from all nodes
- Maintains latest snapshot per node with timestamps
- Marks nodes as stale after 3 seconds without update
- Serves SSE stream and snapshot endpoint

### Frontend Components

#### `GenerationSettingsStore`

Location: `dashboard/src/lib/stores/generationSettings.svelte.ts`

Svelte 5 reactive store:
- Fetches current settings on initialization via GET
- Exposes reactive state for thinking_mode, thinking_token_budget, output_token_budget
- Provides `update(patch)` method that sends PATCH and handles optimistic update with rollback on failure

#### `TelemetryStore`

Location: `dashboard/src/lib/stores/telemetry.svelte.ts`

Svelte 5 reactive store:
- Manages SSE connection to `/api/telemetry/stream`
- Handles reconnection with exponential backoff (base 1s, max 30s, factor 2)
- Exposes per-node GPU and network metrics with staleness indicators
- Tracks connection status (connected, reconnecting, disconnected)

#### `InferenceSettingsPanel`

Location: `dashboard/src/lib/components/InferenceSettingsPanel.svelte`

- Thinking mode toggle (switch component)
- Thinking token budget numeric input (disabled when thinking_mode is off)
- Output token budget numeric input
- Debounced PATCH on value change (300ms debounce)
- Error state display with auto-dismiss toast

#### `GpuTelemetryPanel`

Location: `dashboard/src/lib/components/GpuTelemetryPanel.svelte`

- Per-node card showing: GPU frequency (MHz), utilization (%), render/compute busy (%), memory bandwidth
- Stale metrics shown with dimmed opacity and "stale" badge
- Offline nodes shown with "offline" indicator
- Compact layout suitable for sidebar placement

#### `NetworkTelemetryPanel`

Location: `dashboard/src/lib/components/NetworkTelemetryPanel.svelte`

- Per-node card showing: bytes sent/received, throughput (human-readable: KB/s, MB/s, GB/s), latency (ms)
- Same staleness and offline indicators as GPU panel
- Human-readable byte formatting utility

### API Interface Summary

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/generation/settings` | GET | — | `GenerationSettings` JSON |
| `/api/generation/settings` | PATCH | Partial `GenerationSettings` | Updated `GenerationSettings` JSON |
| `/api/telemetry/cluster` | GET | — | `ClusterTelemetry` JSON snapshot |
| `/api/telemetry/stream` | GET | — | SSE stream of `TelemetryEvent` |

## Data Models

### GenerationSettings

```python
class GenerationSettings(FrozenModel):
    """Cluster-wide inference parameter defaults."""
    thinking_mode: bool = True
    thinking_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
    output_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
```

### GpuMetrics

```python
class GpuMetrics(FrozenModel):
    """GPU telemetry for a single node."""
    node_id: NodeId
    timestamp: datetime  # UTC
    frequency_mhz: int | None = None
    utilization_percent: float | None = None
    render_busy_percent: float | None = None
    memory_bandwidth_percent: float | None = None
    source: Literal["intel_gpu_top", "sysfs", "unavailable"] = "intel_gpu_top"
```

### NetworkMetrics

```python
class NetworkMetrics(FrozenModel):
    """Network telemetry for a single node's cluster interface."""
    node_id: NodeId
    timestamp: datetime  # UTC
    interface_name: str
    bytes_sent: int
    bytes_received: int
    throughput_sent_bytes_per_sec: float
    throughput_received_bytes_per_sec: float
    latency_ms: float | None = None
```

### NodeTelemetry

```python
class NodeTelemetry(FrozenModel):
    """Combined telemetry report from a single node."""
    node_id: NodeId
    gpu: GpuMetrics
    network: NetworkMetrics
```

### ClusterTelemetry

```python
class ClusterTelemetry(FrozenModel):
    """Snapshot of all node telemetry."""
    nodes: Mapping[NodeId, NodeTelemetry]
    stale_nodes: Sequence[NodeId]  # nodes with metrics older than 3s
```

### TelemetryEvent (SSE)

```json
{
  "event": "telemetry",
  "data": {
    "node_id": "...",
    "gpu": { ... },
    "network": { ... },
    "is_stale": false
  }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: GenerationSettings validation accepts valid values and rejects invalid values

*For any* integer value in the range [1, 1_048_576], constructing a `GenerationSettings` with that value as `thinking_token_budget` or `output_token_budget` SHALL succeed. *For any* integer value outside that range (≤ 0 or > 1_048_576), construction SHALL raise a validation error.

**Validates: Requirements 1.1, 2.3**

### Property 2: Event apply correctness for GenerationSettingsUpdated

*For any* valid `State` and *any* valid `GenerationSettingsUpdated` event, applying the event to the state SHALL produce a new state where `generation_settings` equals the event's `generation_settings` field, and all other state fields remain unchanged.

**Validates: Requirements 1.2**

### Property 3: Partial patch merge preserves unpatched fields

*For any* valid `GenerationSettings` current state and *any* valid partial patch (a subset of fields), merging the patch into the current settings SHALL produce a result where patched fields equal the patch values and unpatched fields equal the original values.

**Validates: Requirements 2.2**

### Property 4: Settings-to-inference parameter resolution

*For any* valid `GenerationSettings`, resolving to inference parameters SHALL map `thinking_mode` to `enable_thinking` (true→True, false→False), `thinking_token_budget` to the thinking budget constraint (null→unlimited), and `output_token_budget` to `max_output_tokens` (null→model default).

**Validates: Requirements 3.4, 3.5, 4.4, 4.5, 5.3, 5.4**

### Property 5: intel_gpu_top JSON parsing correctness

*For any* valid `intel_gpu_top -J` JSON output containing engine and frequency data, the parser SHALL extract `frequency_mhz` as a non-negative integer, `utilization_percent` as a float in [0, 100], `render_busy_percent` as a float in [0, 100], and a valid UTC timestamp.

**Validates: Requirements 6.4, 6.8**

### Property 6: /proc/net/dev parsing correctness

*For any* valid `/proc/net/dev` line containing an interface name and numeric columns, the parser SHALL extract `bytes_received` (column 1) and `bytes_sent` (column 9) as non-negative integers.

**Validates: Requirements 7.2**

### Property 7: Network throughput computation

*For any* two consecutive network samples where the second has byte counts ≥ the first and a positive time delta, the computed throughput SHALL equal `(bytes_delta / time_delta)` and be non-negative.

**Validates: Requirements 7.3**

### Property 8: Interface name filtering

*For any* interface name, the cluster interface filter SHALL exclude names matching `lo`, `docker*`, `veth*`, `br-*`, and `virbr*`, and SHALL include interfaces whose IP address falls within the 10.1.1.0/24 subnet.

**Validates: Requirements 7.5**

### Property 9: Metric staleness detection

*For any* metric timestamp and current UTC time, the metric SHALL be marked stale if and only if `(current_time - timestamp) > 3 seconds`.

**Validates: Requirements 8.3**

### Property 10: Offline node omission from telemetry response

*For any* cluster telemetry snapshot, the response SHALL contain entries only for nodes that have reported at least one metric. Nodes that have never reported or have been removed from topology SHALL be omitted entirely.

**Validates: Requirements 8.5**

### Property 11: Byte count formatting to human-readable units

*For any* non-negative integer byte count, the formatting function SHALL produce a string with the appropriate unit (B, KB, MB, GB, TB) where the numeric portion is in [0, 1024) for all units except the largest applicable, and `format(bytes) → parse(format(bytes))` round-trips to within 1% of the original value.

**Validates: Requirements 10.2**

### Property 12: Exponential backoff delay calculation

*For any* number of consecutive failures `n ≥ 0`, the backoff delay SHALL equal `min(base_delay * 2^n, max_delay)` where `base_delay = 1000ms` and `max_delay = 30000ms`, and the result is always a positive integer.

**Validates: Requirements 11.2**

## Error Handling

### Settings API Errors

| Scenario | HTTP Status | Behavior |
|----------|-------------|----------|
| Invalid field type (e.g., string for thinking_mode) | 422 | Pydantic validation error with field details |
| Out-of-range budget (≤0 or >1,048,576) | 422 | Pydantic validation error with constraint details |
| API unreachable from dashboard | — | Dashboard shows last known settings + connection error indicator |
| Event broadcast failure | 500 | API returns error; dashboard reverts optimistic update |

### Telemetry Errors

| Scenario | Behavior |
|----------|----------|
| `intel_gpu_top` unavailable | Fall back to sysfs; set `source="sysfs"` in metrics |
| Both `intel_gpu_top` and sysfs unavailable | Emit metrics with all values `None`, `source="unavailable"` |
| `/proc/net/dev` unreadable | Emit network metrics with zero throughput, log warning |
| Node stops reporting (>3s) | Mark as stale in aggregator; dashboard shows stale indicator |
| Node removed from topology | Remove from aggregator; dashboard shows offline |
| SSE connection drops | Dashboard reconnects with exponential backoff (1s base, 30s max) |
| Collector exceeds 2% CPU | Reduce collection frequency; log warning |

### Worker Errors

| Scenario | Behavior |
|----------|----------|
| Settings change during inference | No effect — worker uses snapshotted values from task start |
| Invalid settings in state (corruption) | Worker falls back to defaults (thinking_mode=true, no budget limits) |

## Testing Strategy

### Property-Based Tests (pytest + Hypothesis)

Each correctness property is implemented as a property-based test with minimum 100 iterations. The property-based testing library is **Hypothesis** (already available in the project's test dependencies).

Test files:
- `src/exo/shared/types/tests/test_generation_settings_properties.py` — Properties 1, 2, 3, 4
- `src/exo/telemetry/tests/test_telemetry_parsing_properties.py` — Properties 5, 6, 7, 8, 9, 10
- `dashboard/src/lib/utils/tests/format.test.ts` — Properties 11, 12 (using fast-check)

Each test is tagged with:
```python
# Feature: dashboard-inference-controls, Property {N}: {property_text}
```

Configuration:
```python
@settings(max_examples=200)
```

### Unit Tests (Example-Based)

- `src/exo/api/tests/test_generation_settings_api.py` — API endpoint integration tests (GET, PATCH, validation errors)
- `src/exo/telemetry/tests/test_collector.py` — Collector fallback behavior, subprocess management
- `dashboard/src/lib/components/tests/InferenceSettingsPanel.test.ts` — Component rendering, toggle behavior, error rollback

### Integration Tests

- `src/exo/telemetry/tests/test_telemetry_integration.py` — End-to-end SSE streaming, staleness detection with real timing
- `src/exo/worker/tests/test_settings_snapshot.py` — Worker snapshots settings at task start, ignores mid-flight changes

### Dashboard Tests

- Component tests using Svelte testing library + vitest
- SSE reconnection behavior with mock EventSource
- Optimistic update rollback on PATCH failure

# Implementation Plan: Dashboard Inference Controls

## Overview

This plan implements inference settings controls (thinking mode toggle, thinking token budget, output token budget) and hardware telemetry panels (GPU utilization, network bandwidth) for the exo dashboard. The inference controls follow the existing event-sourcing pattern: Pydantic model → event → apply → state → broadcast. The telemetry system adds a per-node collector and master-side aggregator with SSE streaming.

Priority order: inference controls first (requirements 1–5), then telemetry (requirements 6–11).

## Tasks

- [x] 1. Define GenerationSettings model and event-sourcing integration
  - [x] 1.1 Create `src/exo/shared/types/generation_settings.py` with the `GenerationSettings` FrozenModel
    - Fields: `thinking_mode: bool = True`, `thinking_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)`, `output_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)`
    - _Requirements: 1.1_
  - [x] 1.2 Add `GenerationSettingsUpdated` event to `src/exo/shared/types/events.py`
    - Extend `BaseEvent` with `generation_settings: GenerationSettings` field
    - Add to the `Event` union type
    - _Requirements: 1.2_
  - [x] 1.3 Add `generation_settings` field to `State` in `src/exo/shared/types/state.py`
    - `generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)`
    - _Requirements: 1.5_
  - [x] 1.4 Add `apply_generation_settings_updated` handler in `src/exo/shared/apply.py`
    - Match `GenerationSettingsUpdated` in `event_apply`, return state with updated `generation_settings`
    - Import the new event type
    - _Requirements: 1.2_
  - [x] 1.5 Add `UpdateGenerationSettings` command to `src/exo/shared/types/commands.py`
    - Extend `BaseCommand` with `generation_settings: GenerationSettings` field
    - Add to the `Command` union type
    - _Requirements: 2.2_
  - [x] 1.6 Write property test for GenerationSettings validation (Property 1)
    - **Property 1: GenerationSettings validation accepts valid values and rejects invalid values**
    - **Validates: Requirements 1.1, 2.3**
    - File: `src/exo/shared/types/tests/test_generation_settings_properties.py`
  - [x] 1.7 Write property test for event apply correctness (Property 2)
    - **Property 2: Event apply correctness for GenerationSettingsUpdated**
    - **Validates: Requirements 1.2**
    - File: `src/exo/shared/types/tests/test_generation_settings_properties.py`

- [x] 2. Implement Generation Settings API
  - [x] 2.1 Create `src/exo/api/generation_settings.py` with FastAPI router
    - `GET /api/generation/settings` → reads from `State.generation_settings`
    - `PATCH /api/generation/settings` → validates partial input, merges with current, emits `UpdateGenerationSettings` command, returns updated settings
    - Register router in the API app
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 2.2 Write property test for partial patch merge (Property 3)
    - **Property 3: Partial patch merge preserves unpatched fields**
    - **Validates: Requirements 2.2**
    - File: `src/exo/shared/types/tests/test_generation_settings_properties.py`
  - [x] 2.3 Write unit tests for generation settings API endpoints
    - Test GET returns current settings, PATCH updates correctly, invalid values return 422
    - File: `src/exo/api/tests/test_generation_settings_api.py`
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Implement worker settings resolution
  - [x] 3.1 Modify `src/exo/worker/runner/runner.py` to snapshot `GenerationSettings` at task acknowledgment
    - When `TextGeneration` task is acknowledged, read `State.generation_settings`
    - Map `thinking_mode` → `enable_thinking`, `thinking_token_budget` → thinking budget, `output_token_budget` → `max_output_tokens`
    - Request-level parameters take precedence over cluster-wide settings
    - _Requirements: 1.3, 1.4, 3.4, 3.5, 4.4, 4.5, 5.3, 5.4_
  - [x] 3.2 Write property test for settings-to-inference parameter resolution (Property 4)
    - **Property 4: Settings-to-inference parameter resolution**
    - **Validates: Requirements 3.4, 3.5, 4.4, 4.5, 5.3, 5.4**
    - File: `src/exo/shared/types/tests/test_generation_settings_properties.py`

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement dashboard GenerationSettingsStore and InferenceSettingsPanel
  - [x] 5.1 Create `dashboard/src/lib/stores/generationSettings.svelte.ts`
    - Svelte 5 reactive store using `$state`
    - Fetch current settings on initialization via GET `/api/generation/settings`
    - Expose reactive state: `thinking_mode`, `thinking_token_budget`, `output_token_budget`
    - Provide `update(patch)` method: sends PATCH, handles optimistic update with rollback on failure
    - 300ms debounce on value changes
    - _Requirements: 3.1, 3.2, 3.3, 3.6, 4.1, 4.2, 4.3, 5.1, 5.2_
  - [x] 5.2 Create `dashboard/src/lib/components/InferenceSettingsPanel.svelte`
    - Thinking mode toggle (switch component) with visual on/off state
    - Thinking token budget numeric input (disabled when thinking_mode is off)
    - Output token budget numeric input
    - Error state display with auto-dismiss toast on PATCH failure
    - Revert toggle on failure
    - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 5.1, 5.2, 11.3_
  - [x] 5.3 Integrate InferenceSettingsPanel into the dashboard layout
    - Add panel to appropriate location in the dashboard (sidebar or settings area)
    - Wire up to GenerationSettingsStore
    - _Requirements: 3.1, 4.1, 5.1_

- [x] 6. Checkpoint - Ensure inference controls work end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement telemetry data models and parsers
  - [x] 7.1 Create `src/exo/telemetry/__init__.py` and `src/exo/telemetry/models.py`
    - Define `GpuMetrics`, `NetworkMetrics`, `NodeTelemetry`, `ClusterTelemetry` Pydantic models
    - _Requirements: 6.4, 6.8, 7.2, 7.3_
  - [x] 7.2 Create `src/exo/telemetry/parsers.py` with `intel_gpu_top` JSON parser and `/proc/net/dev` parser
    - Parse `intel_gpu_top -J` output for frequency, utilization, render/compute busy, memory bandwidth
    - Parse `/proc/net/dev` lines for bytes_received (column 1) and bytes_sent (column 9)
    - Compute throughput from consecutive samples
    - Filter cluster interface (10.1.1.0/24 subnet), exclude lo, docker*, veth*, br-*, virbr*
    - _Requirements: 6.2, 6.3, 6.4, 7.2, 7.3, 7.5_
  - [x] 7.3 Write property test for intel_gpu_top JSON parsing (Property 5)
    - **Property 5: intel_gpu_top JSON parsing correctness**
    - **Validates: Requirements 6.4, 6.8**
    - File: `src/exo/telemetry/tests/test_telemetry_parsing_properties.py`
  - [x] 7.4 Write property test for /proc/net/dev parsing (Property 6)
    - **Property 6: /proc/net/dev parsing correctness**
    - **Validates: Requirements 7.2**
    - File: `src/exo/telemetry/tests/test_telemetry_parsing_properties.py`
  - [x] 7.5 Write property test for network throughput computation (Property 7)
    - **Property 7: Network throughput computation**
    - **Validates: Requirements 7.3**
    - File: `src/exo/telemetry/tests/test_telemetry_parsing_properties.py`
  - [x] 7.6 Write property test for interface name filtering (Property 8)
    - **Property 8: Interface name filtering**
    - **Validates: Requirements 7.5**
    - File: `src/exo/telemetry/tests/test_telemetry_parsing_properties.py`

- [x] 8. Implement TelemetryCollector
  - [x] 8.1 Create `src/exo/telemetry/collector.py`
    - Background asyncio task running on each node at ~1 Hz
    - Spawn `intel_gpu_top -J -s 900` subprocess, parse JSON output
    - Fall back to sysfs reads if `intel_gpu_top` unavailable
    - Read `/proc/net/dev` for cluster interface
    - Compute throughput from consecutive byte count deltas
    - Report `NodeTelemetry` to master via internal pub/sub
    - CPU overhead < 2% constraint
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 7.1, 7.2, 7.3, 7.4, 7.5_
  - [x] 8.2 Write property test for metric staleness detection (Property 9)
    - **Property 9: Metric staleness detection**
    - **Validates: Requirements 8.3**
    - File: `src/exo/telemetry/tests/test_telemetry_parsing_properties.py`
  - [x] 8.3 Write property test for offline node omission (Property 10)
    - **Property 10: Offline node omission from telemetry response**
    - **Validates: Requirements 8.5**
    - File: `src/exo/telemetry/tests/test_telemetry_parsing_properties.py`

- [x] 9. Implement TelemetryAggregator and streaming API
  - [x] 9.1 Create `src/exo/api/telemetry.py` with FastAPI router
    - `TelemetryAggregator` class: receives reports from nodes, maintains latest snapshot per node, marks stale after 3s
    - `GET /api/telemetry/cluster` → returns `ClusterTelemetry` JSON snapshot
    - `GET /api/telemetry/stream` → SSE stream of `TelemetryEvent` updates
    - Register router in the API app
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 10. Checkpoint - Ensure telemetry backend tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement dashboard telemetry stores and panels
  - [x] 11.1 Create `dashboard/src/lib/stores/telemetry.svelte.ts`
    - Svelte 5 reactive store managing SSE connection to `/api/telemetry/stream`
    - Exponential backoff reconnection (base 1s, max 30s, factor 2)
    - Expose per-node GPU and network metrics with staleness indicators
    - Track connection status (connected, reconnecting, disconnected)
    - _Requirements: 8.1, 9.5, 10.5, 11.2_
  - [x] 11.2 Create `dashboard/src/lib/utils/format.ts` with byte formatting utility
    - Format non-negative integer byte counts to human-readable units (B, KB, MB, GB, TB)
    - Numeric portion in [0, 1024) for all units except the largest applicable
    - _Requirements: 10.2_
  - [x] 11.3 Create `dashboard/src/lib/components/GpuTelemetryPanel.svelte`
    - Per-node card: GPU frequency (MHz), utilization (%), render/compute busy (%), memory bandwidth
    - Stale metrics shown with dimmed opacity and "stale" badge
    - Offline nodes shown with "offline" indicator
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [x] 11.4 Create `dashboard/src/lib/components/NetworkTelemetryPanel.svelte`
    - Per-node card: bytes sent/received, throughput (human-readable), latency (ms)
    - Same staleness and offline indicators as GPU panel
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  - [x] 11.5 Integrate telemetry panels into the dashboard layout
    - Add GPU and network panels to appropriate location
    - Wire up to TelemetryStore
    - _Requirements: 9.1, 10.1, 11.1_
  - [x] 11.6 Write property test for byte count formatting (Property 11)
    - **Property 11: Byte count formatting to human-readable units**
    - **Validates: Requirements 10.2**
    - File: `dashboard/src/lib/utils/tests/format.test.ts` (using fast-check)
  - [x] 11.7 Write property test for exponential backoff delay (Property 12)
    - **Property 12: Exponential backoff delay calculation**
    - **Validates: Requirements 11.2**
    - File: `dashboard/src/lib/utils/tests/format.test.ts` (using fast-check)

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Inference controls (tasks 1–6) are highest priority — implement before telemetry
- The thinking mode toggle is the single most important feature (users are getting unwanted thinking tokens)
- Follow existing event-sourcing patterns in `apply.py` and `events.py`
- Dashboard components should follow existing Svelte 5 patterns (see `app.svelte.ts` for store pattern)
- Backend Python tests use pytest + hypothesis; dashboard tests use vitest + fast-check

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.5"] },
    { "id": 2, "tasks": ["1.4", "1.6", "1.7"] },
    { "id": 3, "tasks": ["2.1"] },
    { "id": 4, "tasks": ["2.2", "2.3", "3.1"] },
    { "id": 5, "tasks": ["3.2", "5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3"] },
    { "id": 7, "tasks": ["7.1"] },
    { "id": 8, "tasks": ["7.2"] },
    { "id": 9, "tasks": ["7.3", "7.4", "7.5", "7.6", "8.1"] },
    { "id": 10, "tasks": ["8.2", "8.3", "9.1"] },
    { "id": 11, "tasks": ["11.1", "11.2"] },
    { "id": 12, "tasks": ["11.3", "11.4", "11.6", "11.7"] },
    { "id": 13, "tasks": ["11.5"] }
  ]
}
```

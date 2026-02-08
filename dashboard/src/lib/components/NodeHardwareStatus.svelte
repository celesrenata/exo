<script lang="ts">
  import type { NodeHardware } from "$lib/types/hardware";
  import {
    formatBackendInfo,
    getBackendStatusColor,
    formatMemoryUsage,
    formatTokensPerSecond,
  } from "$lib/types/hardware";

  interface Props {
    hardware: NodeHardware;
  }

  let { hardware }: Props = $props();

  // Get status color for the active backend
  const statusColor = $derived(getBackendStatusColor(hardware.activeBackend));

  // Format backend display string
  const backendDisplay = $derived(formatBackendInfo(hardware.activeBackend));

  // Format memory usage
  const memoryDisplay = $derived(
    formatMemoryUsage(
      hardware.activeBackend.memoryUsedMB,
      hardware.activeBackend.memoryTotalMB,
    ),
  );

  // Check if we have performance metrics
  const hasMetrics = $derived(hardware.metrics !== undefined);

  // Format performance metrics
  const avgThroughput = $derived(
    hardware.metrics
      ? formatTokensPerSecond(hardware.metrics.avgTokensPerSecond)
      : "N/A",
  );

  const recentThroughput = $derived(
    hardware.metrics
      ? formatTokensPerSecond(hardware.metrics.recentTokensPerSecond)
      : "N/A",
  );
</script>

<div class="hardware-status">
  <div class="header">
    <h3 class="node-id">Node {hardware.nodeId}</h3>
    <div
      class="status-indicator"
      class:green={statusColor === "green"}
      class:yellow={statusColor === "yellow"}
      class:gray={statusColor === "gray"}
      class:blue={statusColor === "blue"}
    ></div>
  </div>

  <div class="backend-info">
    <div class="info-row">
      <span class="label">Backend:</span>
      <span class="value backend-type">{hardware.activeBackend.type}</span>
    </div>

    <div class="info-row">
      <span class="label">Device:</span>
      <span class="value">{hardware.activeBackend.device}</span>
      {#if hardware.activeBackend.runtime}
        <span class="runtime">({hardware.activeBackend.runtime})</span>
      {/if}
    </div>

    <div class="info-row">
      <span class="label">Hardware:</span>
      <span class="value device-name">{hardware.activeBackend.deviceName}</span>
    </div>
  </div>

  <div class="resource-info">
    <div class="info-row">
      <span class="label">Memory:</span>
      <span class="value">{memoryDisplay}</span>
    </div>

    {#if hardware.activeBackend.utilizationPercent !== undefined}
      <div class="info-row">
        <span class="label">GPU Utilization:</span>
        <div class="utilization-bar">
          <div
            class="utilization-fill"
            style="width: {hardware.activeBackend.utilizationPercent}%"
          ></div>
          <span class="utilization-text"
            >{hardware.activeBackend.utilizationPercent.toFixed(0)}%</span
          >
        </div>
      </div>
    {/if}
  </div>

  {#if hasMetrics}
    <div class="metrics-info">
      <h4>Performance Metrics</h4>

      <div class="info-row">
        <span class="label">Avg Throughput:</span>
        <span class="value metric">{avgThroughput}</span>
      </div>

      <div class="info-row">
        <span class="label">Recent Throughput:</span>
        <span class="value metric">{recentThroughput}</span>
      </div>

      <div class="info-row">
        <span class="label">Inferences:</span>
        <span class="value">{hardware.metrics.inferenceCount}</span>
      </div>

      {#if hardware.metrics.avgGpuUtilization !== undefined}
        <div class="info-row">
          <span class="label">Avg GPU Usage:</span>
          <span class="value"
            >{hardware.metrics.avgGpuUtilization.toFixed(1)}%</span
          >
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .hardware-status {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--color-border);
  }

  .node-id {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--color-gray);
  }

  .status-indicator.green {
    background: var(--color-success);
    box-shadow: 0 0 8px var(--color-success);
  }

  .status-indicator.yellow {
    background: var(--color-warning);
    box-shadow: 0 0 8px var(--color-warning);
  }

  .status-indicator.blue {
    background: var(--color-info);
    box-shadow: 0 0 8px var(--color-info);
  }

  .status-indicator.gray {
    background: var(--color-gray);
  }

  .backend-info,
  .resource-info,
  .metrics-info {
    margin-bottom: 1rem;
  }

  .metrics-info h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .info-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
    gap: 0.5rem;
  }

  .label {
    font-size: 0.875rem;
    color: var(--color-text-secondary);
    min-width: 120px;
  }

  .value {
    font-size: 0.875rem;
    color: var(--color-text-primary);
    font-weight: 500;
  }

  .backend-type {
    text-transform: uppercase;
    font-weight: 600;
    color: var(--color-primary);
  }

  .device-name {
    font-family: monospace;
    font-size: 0.8rem;
  }

  .runtime {
    font-size: 0.75rem;
    color: var(--color-text-tertiary);
    font-style: italic;
  }

  .metric {
    font-family: monospace;
    color: var(--color-success);
  }

  .utilization-bar {
    position: relative;
    flex: 1;
    height: 20px;
    background: var(--color-background);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    overflow: hidden;
  }

  .utilization-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(
      90deg,
      var(--color-success),
      var(--color-primary)
    );
    transition: width 0.3s ease;
  }

  .utilization-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-text-primary);
    text-shadow: 0 0 2px rgba(0, 0, 0, 0.5);
  }

  /* CSS variables (should be defined in global styles) */
  :root {
    --color-surface: #ffffff;
    --color-background: #f5f5f5;
    --color-border: #e0e0e0;
    --color-text-primary: #212121;
    --color-text-secondary: #757575;
    --color-text-tertiary: #9e9e9e;
    --color-primary: #2196f3;
    --color-success: #4caf50;
    --color-warning: #ff9800;
    --color-info: #00bcd4;
    --color-gray: #9e9e9e;
  }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    :root {
      --color-surface: #1e1e1e;
      --color-background: #121212;
      --color-border: #333333;
      --color-text-primary: #ffffff;
      --color-text-secondary: #b0b0b0;
      --color-text-tertiary: #808080;
    }
  }
</style>

<script lang="ts">
  import { telemetryStore } from "$lib/stores/telemetry.svelte";

  function formatPercent(value: number | null): string {
    if (value === null || value === undefined) return "—";
    return value.toFixed(1);
  }

  function formatFrequency(value: number | null): string {
    if (value === null || value === undefined) return "—";
    return Math.round(value).toString();
  }
</script>

<div class="border-b border-exo-medium-gray/30 px-3 py-2 space-y-2">
  <div class="flex items-center justify-between">
    <h2 class="text-xs text-exo-light-gray uppercase tracking-wider font-mono">
      GPU Telemetry
    </h2>
    {#if telemetryStore.connectionStatus === "disconnected"}
      <span class="inline-block w-2 h-2 rounded-full bg-red-400"></span>
    {:else if telemetryStore.connectionStatus === "reconnecting"}
      <span class="inline-block w-2 h-2 rounded-full bg-exo-yellow animate-pulse"></span>
    {:else}
      <span class="inline-block w-2 h-2 rounded-full bg-green-400"></span>
    {/if}
  </div>

  {#if telemetryStore.connectionStatus === "disconnected"}
    <div class="flex items-center gap-1.5 py-1">
      <span class="text-xs text-exo-light-gray font-mono">Disconnected</span>
    </div>
  {:else if telemetryStore.nodes.size === 0}
    <div class="flex items-center gap-1.5 py-1">
      <span class="text-xs text-exo-light-gray font-mono">No nodes reporting</span>
    </div>
  {:else}
    {#each [...telemetryStore.nodes.entries()] as [nodeId, node] (nodeId)}
      <div
        class="rounded border border-exo-medium-gray/20 px-2 py-1.5 space-y-1 transition-opacity duration-200 {node.is_stale ? 'opacity-50' : ''}"
      >
        <div class="flex items-center justify-between">
          <span class="text-xs text-exo-light-gray font-mono uppercase tracking-wider">
            {nodeId}
          </span>
          {#if node.is_stale}
            <span class="text-[10px] text-exo-yellow font-mono uppercase px-1 py-0.5 rounded bg-exo-yellow/10 border border-exo-yellow/30">
              stale
            </span>
          {/if}
        </div>

        <div class="grid grid-cols-2 gap-x-3 gap-y-0.5">
          <div class="flex justify-between text-xs font-mono">
            <span class="text-exo-light-gray/70">Freq</span>
            <span class="text-exo-yellow">{formatFrequency(node.gpu.frequency_mhz)} MHz</span>
          </div>
          <div class="flex justify-between text-xs font-mono">
            <span class="text-exo-light-gray/70">Util</span>
            <span class="text-exo-yellow">{formatPercent(node.gpu.utilization_percent)}%</span>
          </div>
          <div class="flex justify-between text-xs font-mono">
            <span class="text-exo-light-gray/70">Render</span>
            <span class="text-exo-yellow">{formatPercent(node.gpu.render_busy_percent)}%</span>
          </div>
          <div class="flex justify-between text-xs font-mono">
            <span class="text-exo-light-gray/70">Mem BW</span>
            <span class="text-exo-yellow">{formatPercent(node.gpu.memory_bandwidth_percent)}%</span>
          </div>
        </div>
      </div>
    {/each}
  {/if}
</div>

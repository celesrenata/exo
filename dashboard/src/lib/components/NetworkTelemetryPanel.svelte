<script lang="ts">
  import { telemetryStore } from "$lib/stores/telemetry.svelte";
  import { formatBytes } from "$lib/utils/format";
</script>

<div class="border-b border-exo-medium-gray/30 px-3 py-2 space-y-2">
  <h2 class="text-xs text-exo-light-gray uppercase tracking-wider font-mono">
    Network Telemetry
  </h2>

  {#if telemetryStore.connectionStatus === "disconnected" && telemetryStore.nodes.size === 0}
    <div
      class="flex items-center gap-1.5 py-1"
      role="status"
    >
      <span class="inline-block w-2 h-2 rounded-full bg-red-400"></span>
      <span class="text-xs text-exo-light-gray font-mono">Offline</span>
    </div>
  {:else if telemetryStore.connectionStatus === "reconnecting" && telemetryStore.nodes.size === 0}
    <div class="flex items-center gap-1.5 py-1">
      <span class="inline-block w-2 h-2 rounded-full bg-exo-yellow animate-pulse"></span>
      <span class="text-xs text-exo-light-gray font-mono">Reconnecting...</span>
    </div>
  {:else if telemetryStore.nodes.size === 0}
    <div class="text-xs text-exo-light-gray font-mono px-1 py-0.5">
      No nodes reporting
    </div>
  {:else}
    {#if telemetryStore.connectionStatus === "disconnected"}
      <div
        class="text-xs text-red-400 font-mono px-1 py-0.5 rounded bg-red-400/10 border border-red-400/20 mb-1"
        role="alert"
      >
        Connection lost
      </div>
    {/if}

    <div class="space-y-2">
      {#each [...telemetryStore.nodes] as [nodeId, node]}
        <div
          class="bg-exo-black rounded border border-exo-medium-gray/30 p-2 space-y-1 transition-opacity duration-200 {node.is_stale ? 'opacity-50' : ''}"
        >
          <div class="flex items-center justify-between">
            <span class="text-xs text-exo-yellow font-mono truncate">{nodeId}</span>
            {#if node.is_stale}
              <span class="text-[10px] text-exo-yellow bg-exo-yellow/20 rounded px-1 font-mono">stale</span>
            {/if}
          </div>

          {#if node.network}
            <div class="text-xs text-exo-light-gray font-mono space-y-0.5">
              <div class="flex justify-between">
                <span class="text-exo-light-gray/60">Interface</span>
                <span>{node.network.interface_name}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-exo-light-gray/60">Sent</span>
                <span>{formatBytes(node.network.bytes_sent)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-exo-light-gray/60">Received</span>
                <span>{formatBytes(node.network.bytes_received)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-exo-light-gray/60">TX Throughput</span>
                <span>{formatBytes(node.network.throughput_sent_bytes_per_sec)}/s</span>
              </div>
              <div class="flex justify-between">
                <span class="text-exo-light-gray/60">RX Throughput</span>
                <span>{formatBytes(node.network.throughput_received_bytes_per_sec)}/s</span>
              </div>
              <div class="flex justify-between">
                <span class="text-exo-light-gray/60">Latency</span>
                {#if node.network.latency_ms != null}
                  <span>{node.network.latency_ms.toFixed(1)} ms</span>
                {:else}
                  <span class="text-exo-light-gray/50">—</span>
                {/if}
              </div>
            </div>
          {:else}
            <div class="text-xs text-exo-light-gray/50 font-mono">
              No network data
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

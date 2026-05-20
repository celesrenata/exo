<script lang="ts">
  import { generationSettingsStore } from "$lib/stores/generationSettings.svelte";

  function handleToggleThinkingMode(): void {
    generationSettingsStore.update({
      thinking_mode: !generationSettingsStore.thinking_mode,
    });
  }

  function handleThinkingTokenChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    const raw = input.value.trim();
    if (raw === "") {
      generationSettingsStore.update({ thinking_token_budget: null });
    } else {
      const num = parseInt(raw, 10);
      if (!isNaN(num) && num >= 1) {
        const clamped = Math.min(num, 1_048_576);
        generationSettingsStore.update({ thinking_token_budget: clamped });
      }
    }
  }

  function handleOutputTokenChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    const raw = input.value.trim();
    if (raw === "") {
      generationSettingsStore.update({ output_token_budget: null });
    } else {
      const num = parseInt(raw, 10);
      if (!isNaN(num) && num >= 1) {
        const clamped = Math.min(num, 1_048_576);
        generationSettingsStore.update({ output_token_budget: clamped });
      }
    }
  }
</script>

<div class="border-b border-exo-medium-gray/30 px-3 py-2 space-y-2">
  <h2 class="text-xs text-exo-light-gray uppercase tracking-wider font-mono">
    Inference Settings
  </h2>

  {#if generationSettingsStore.loading}
    <div class="flex items-center gap-1.5 py-1">
      <span class="inline-block w-2 h-2 rounded-full bg-exo-yellow animate-pulse"></span>
      <span class="text-xs text-exo-light-gray font-mono">Loading...</span>
    </div>
  {:else}
    <!-- Thinking Mode Toggle -->
    <div class="flex items-center justify-between py-1">
      <span class="text-xs text-exo-light-gray uppercase tracking-wider">Thinking Mode</span>
      <button
        type="button"
        role="switch"
        aria-checked={generationSettingsStore.thinking_mode}
        aria-label="Toggle thinking mode"
        onclick={handleToggleThinkingMode}
        class="relative w-8 h-4 rounded-full cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-exo-yellow/50 {generationSettingsStore.thinking_mode ? 'bg-exo-yellow' : 'bg-exo-medium-gray/50 border border-exo-yellow/30'}"
      >
        <div
          class="absolute top-0.5 w-3 h-3 rounded-full transition-all duration-200 {generationSettingsStore.thinking_mode ? 'right-0.5 bg-exo-black' : 'left-0.5 bg-exo-light-gray'}"
        ></div>
      </button>
    </div>

    <!-- Thinking Token Budget -->
    <div class="py-1 transition-opacity duration-200 {generationSettingsStore.thinking_mode ? '' : 'opacity-50'}">
      <label for="thinking-token-budget" class="block text-xs text-exo-light-gray uppercase tracking-wider mb-1">Thinking Token Budget</label>
      <input
        id="thinking-token-budget"
        type="number"
        min="1"
        max="1048576"
        step="1"
        placeholder="Unlimited"
        aria-label="Thinking token budget"
        value={generationSettingsStore.thinking_token_budget ?? ""}
        disabled={!generationSettingsStore.thinking_mode}
        oninput={handleThinkingTokenChange}
        class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1 text-xs font-mono text-exo-yellow placeholder:text-exo-light-gray/50 transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70 disabled:cursor-not-allowed disabled:hover:border-exo-yellow/30"
      />
    </div>

    <!-- Output Token Budget -->
    <div class="py-1">
      <label for="output-token-budget" class="block text-xs text-exo-light-gray uppercase tracking-wider mb-1">Output Token Budget</label>
      <input
        id="output-token-budget"
        type="number"
        min="1"
        max="1048576"
        step="1"
        placeholder="Default"
        aria-label="Output token budget"
        value={generationSettingsStore.output_token_budget ?? ""}
        oninput={handleOutputTokenChange}
        class="w-full bg-exo-medium-gray/50 border border-exo-yellow/30 rounded px-2 py-1 text-xs font-mono text-exo-yellow placeholder:text-exo-light-gray/50 transition-all duration-200 hover:border-exo-yellow/50 focus:outline-none focus:border-exo-yellow/70"
      />
    </div>

    <!-- Error Display -->
    {#if generationSettingsStore.error}
      <div class="text-xs text-red-400 font-mono px-1 py-0.5 rounded bg-red-400/10 border border-red-400/20" role="alert">
        {generationSettingsStore.error}
      </div>
    {/if}
  {/if}
</div>

<style>
  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }

  input[type="number"] {
    -moz-appearance: textfield;
  }
</style>

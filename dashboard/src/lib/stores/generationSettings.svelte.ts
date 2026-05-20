/**
 * GenerationSettingsStore - Manages cluster-wide inference parameter defaults.
 *
 * Fetches current settings on initialization, exposes reactive state,
 * and provides an update() method with optimistic updates, rollback on failure,
 * and 300ms debounce for rapid value changes (e.g., slider input).
 */

import { browser } from "$app/environment";
import { addToast } from "$lib/stores/toast.svelte";

interface GenerationSettingsResponse {
  thinkingMode: boolean;
  thinkingTokenBudget: number | null;
  outputTokenBudget: number | null;
}

export interface GenerationSettingsPatch {
  thinking_mode?: boolean;
  thinking_token_budget?: number | null;
  output_token_budget?: number | null;
}

interface SettingsSnapshot {
  thinking_mode: boolean;
  thinking_token_budget: number | null;
  output_token_budget: number | null;
}

class GenerationSettingsStore {
  thinking_mode = $state<boolean>(false);
  thinking_token_budget = $state<number | null>(null);
  output_token_budget = $state<number | null>(null);
  error = $state<string | null>(null);
  loading = $state<boolean>(true);

  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingPatch: GenerationSettingsPatch = {};
  private snapshotBeforeDebounce: SettingsSnapshot | null = null;

  constructor() {
    if (browser) {
      this.fetchSettings();
    } else {
      this.loading = false;
    }
  }

  private async fetchSettings(): Promise<void> {
    try {
      const response = await fetch("/api/generation/settings");
      if (!response.ok) {
        throw new Error(`Failed to fetch settings: ${response.statusText}`);
      }
      const data: GenerationSettingsResponse = await response.json();
      this.applyResponse(data);
      this.error = null;

      // Re-apply saved settings if server reset to defaults after restart
      const savedRaw = localStorage.getItem("exo-generation-settings");
      if (savedRaw) {
        const savedPatch: GenerationSettingsPatch = JSON.parse(savedRaw);
        const needsReapply =
          (savedPatch.thinking_mode !== undefined && savedPatch.thinking_mode !== data.thinkingMode) ||
          (savedPatch.thinking_token_budget !== undefined && savedPatch.thinking_token_budget !== data.thinkingTokenBudget) ||
          (savedPatch.output_token_budget !== undefined && savedPatch.output_token_budget !== data.outputTokenBudget);
        if (needsReapply) {
          this.update(savedPatch);
        }
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch generation settings";
      this.error = message;
      addToast({ type: "error", message });
    } finally {
      this.loading = false;
    }
  }

  private applyResponse(data: GenerationSettingsResponse): void {
    this.thinking_mode = data.thinkingMode;
    this.thinking_token_budget = data.thinkingTokenBudget;
    this.output_token_budget = data.outputTokenBudget;
  }

  private takeSnapshot(): SettingsSnapshot {
    return {
      thinking_mode: this.thinking_mode,
      thinking_token_budget: this.thinking_token_budget,
      output_token_budget: this.output_token_budget,
    };
  }

  private revertToSnapshot(snapshot: SettingsSnapshot): void {
    this.thinking_mode = snapshot.thinking_mode;
    this.thinking_token_budget = snapshot.thinking_token_budget;
    this.output_token_budget = snapshot.output_token_budget;
  }

  update(patch: GenerationSettingsPatch): void {
    // On the first call in a debounce window, save the current state for rollback
    if (!this.snapshotBeforeDebounce) {
      this.snapshotBeforeDebounce = this.takeSnapshot();
    }

    // Accumulate patches within the debounce window
    this.pendingPatch = { ...this.pendingPatch, ...patch };

    // Apply optimistic update immediately for UI responsiveness
    if (patch.thinking_mode !== undefined) {
      this.thinking_mode = patch.thinking_mode;
    }
    if (patch.thinking_token_budget !== undefined) {
      this.thinking_token_budget = patch.thinking_token_budget;
    }
    if (patch.output_token_budget !== undefined) {
      this.output_token_budget = patch.output_token_budget;
    }

    // Reset debounce timer
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.sendPatch();
    }, 300);
  }

  private async sendPatch(): Promise<void> {
    const patch = this.pendingPatch;
    const snapshot = this.snapshotBeforeDebounce;

    // Clear debounce state
    this.pendingPatch = {};
    this.snapshotBeforeDebounce = null;

    try {
      const response = await fetch("/api/generation/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });

      if (!response.ok) {
        throw new Error(`Failed to update settings: ${response.statusText}`);
      }

      const data: GenerationSettingsResponse = await response.json();
      this.applyResponse(data);
      this.error = null;
      this.saveToLocalStorage();
    } catch (err) {
      // Revert to the state before the debounce window started
      if (snapshot) {
        this.revertToSnapshot(snapshot);
      }

      const message =
        err instanceof Error ? err.message : "Failed to update generation settings";
      this.error = message;
      addToast({ type: "error", message });
    }
  }

  private saveToLocalStorage(): void {
    const patch: GenerationSettingsPatch = {
      thinking_mode: this.thinking_mode,
      thinking_token_budget: this.thinking_token_budget,
      output_token_budget: this.output_token_budget,
    };
    localStorage.setItem("exo-generation-settings", JSON.stringify(patch));
  }
}

export const generationSettingsStore = new GenerationSettingsStore();

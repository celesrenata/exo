/**
 * TelemetryStore - Manages SSE connection to /api/telemetry/stream for real-time
 * per-node GPU and network metrics.
 *
 * Features:
 * - Exponential backoff reconnection (base 1s, max 30s, factor 2)
 * - Per-node GPU and network metrics with staleness indicators
 * - Connection status tracking (connected, reconnecting, disconnected)
 */

import { browser } from "$app/environment";

export interface GpuMetrics {
  node_id: string;
  timestamp: string;
  frequency_mhz: number | null;
  utilization_percent: number | null;
  render_busy_percent: number | null;
  memory_bandwidth_percent: number | null;
  source: string;
}

export interface NetworkMetrics {
  node_id: string;
  timestamp: string;
  interface_name: string;
  bytes_sent: number;
  bytes_received: number;
  throughput_sent_bytes_per_sec: number;
  throughput_received_bytes_per_sec: number;
  latency_ms: number | null;
}

export interface NodeTelemetry {
  node_id: string;
  gpu: GpuMetrics;
  network: NetworkMetrics;
  is_stale: boolean;
}

export type ConnectionStatus = "connected" | "reconnecting" | "disconnected";

/**
 * Calculate exponential backoff delay for reconnection attempts.
 *
 * For any number of consecutive failures n >= 0, the delay equals
 * min(baseDelay * 2^n, maxDelay) and is always a positive integer.
 */
export function calculateBackoffDelay(
  attempt: number,
  baseDelay: number = 1000,
  maxDelay: number = 30000,
): number {
  const delay = Math.floor(baseDelay * Math.pow(2, attempt));
  return Math.max(1, Math.min(delay, maxDelay));
}

class TelemetryStore {
  nodes = $state<Map<string, NodeTelemetry>>(new Map());
  connectionStatus = $state<ConnectionStatus>("disconnected");

  private eventSource: EventSource | null = null;
  private consecutiveFailures = 0;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    if (browser) {
      this.connect();
    }
  }

  connect(): void {
    if (!browser) return;
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }

    this.connectionStatus = "reconnecting";

    const es = new EventSource("/api/telemetry/stream");
    this.eventSource = es;

    es.addEventListener("telemetry", (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as NodeTelemetry;
        const updated = new Map(this.nodes);
        updated.set(data.node_id, data);
        this.nodes = updated;
      } catch (err) {
        console.error("Failed to parse telemetry event:", err);
      }
    });

    es.onopen = () => {
      this.consecutiveFailures = 0;
      this.connectionStatus = "connected";
    };

    es.onerror = () => {
      es.close();
      this.eventSource = null;
      this.scheduleReconnect();
    };
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.consecutiveFailures = 0;
    this.connectionStatus = "disconnected";
  }

  private scheduleReconnect(): void {
    this.connectionStatus = "reconnecting";
    const delay = calculateBackoffDelay(this.consecutiveFailures);
    this.consecutiveFailures++;
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }
}

export const telemetryStore = new TelemetryStore();

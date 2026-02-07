/**
 * Hardware and backend information types for dashboard visibility.
 *
 * These types represent hardware capabilities and backend status
 * for nodes in the exo cluster.
 */

/**
 * Backend type identifier
 */
export type BackendType = 'mlx' | 'tinygrad' | 'npu';

/**
 * Device type identifier
 */
export type DeviceType = 'CPU' | 'GPU' | 'NPU' | 'METAL';

/**
 * GPU runtime identifier
 */
export type RuntimeType = 'LEVEL_ZERO' | 'OPENCL' | 'METAL' | 'CUDA';

/**
 * Information about an active inference backend
 */
export interface BackendInfo {
  /** Type of backend (mlx, tinygrad, npu) */
  type: BackendType;

  /** Device being used (CPU, GPU, NPU, METAL) */
  device: DeviceType;

  /** GPU runtime if applicable (LEVEL_ZERO, OPENCL, etc.) */
  runtime?: RuntimeType;

  /** Human-readable device name */
  deviceName: string;

  /** Memory currently used in MB */
  memoryUsedMB: number;

  /** Total available memory in MB */
  memoryTotalMB: number;

  /** GPU utilization percentage (0-100) if available */
  utilizationPercent?: number;

  /** Timestamp of last update */
  timestamp?: number;
}

/**
 * Hardware information for a node in the cluster
 */
export interface NodeHardware {
  /** Node identifier */
  nodeId: string;

  /** All available backends on this node */
  backends: BackendInfo[];

  /** Currently active backend */
  activeBackend: BackendInfo;

  /** Performance metrics */
  metrics?: {
    /** Average tokens per second */
    avgTokensPerSecond: number;

    /** Recent tokens per second (last 10 inferences) */
    recentTokensPerSecond: number;

    /** Total inferences performed */
    inferenceCount: number;

    /** Average memory usage in MB */
    avgMemoryUsedMB: number;

    /** Average GPU utilization if available */
    avgGpuUtilization?: number;
  };
}

/**
 * Backend initialization event data
 */
export interface BackendInitializedEvent {
  runnerId: string;
  backendType: BackendType;
  deviceType: DeviceType;
  deviceName: string;
  runtime?: RuntimeType;
}

/**
 * Backend failure event data
 */
export interface BackendFailedEvent {
  runnerId: string;
  backendType: BackendType;
  errorMessage: string;
  fallbackBackend?: BackendType;
}

/**
 * GPU metrics event data
 */
export interface GPUMetricsEvent {
  runnerId: string;
  deviceName: string;
  runtime: string;
  memoryUsedMB: number;
  memoryTotalMB: number;
  utilizationPercent?: number;
  timestamp: number;
}

/**
 * Helper function to format backend info for display
 */
export function formatBackendInfo(backend: BackendInfo): string {
  const parts: string[] = [backend.type, backend.device];

  if (backend.runtime) {
    parts.push(`(${backend.runtime})`);
  }

  return parts.join(' ');
}

/**
 * Helper function to get backend status color
 */
export function getBackendStatusColor(backend: BackendInfo): string {
  // GPU with Level Zero = green (optimal)
  if (backend.device === 'GPU' && backend.runtime === 'LEVEL_ZERO') {
    return 'green';
  }

  // GPU with other runtime = yellow (good but not optimal)
  if (backend.device === 'GPU') {
    return 'yellow';
  }

  // Metal = green (optimal for macOS)
  if (backend.device === 'METAL') {
    return 'green';
  }

  // CPU = gray (fallback)
  if (backend.device === 'CPU') {
    return 'gray';
  }

  // NPU = blue (experimental)
  if (backend.device === 'NPU') {
    return 'blue';
  }

  return 'gray';
}

/**
 * Helper function to format memory usage
 */
export function formatMemoryUsage(usedMB: number, totalMB: number): string {
  const usedGB = (usedMB / 1024).toFixed(1);
  const totalGB = (totalMB / 1024).toFixed(1);

  if (totalMB > 0) {
    const percent = ((usedMB / totalMB) * 100).toFixed(0);
    return `${usedGB} / ${totalGB} GB (${percent}%)`;
  }

  return `${usedGB} GB`;
}

/**
 * Helper function to format tokens per second
 */
export function formatTokensPerSecond(tokensPerSec: number): string {
  if (tokensPerSec >= 1000) {
    return `${(tokensPerSec / 1000).toFixed(1)}K tok/s`;
  }

  return `${tokensPerSec.toFixed(1)} tok/s`;
}

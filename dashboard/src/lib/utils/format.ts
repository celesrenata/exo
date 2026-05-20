/**
 * Byte formatting utilities for human-readable display of byte counts.
 *
 * Used by the NetworkTelemetryPanel to display throughput in human-readable format.
 */

/** Supported byte units in ascending order of magnitude. */
const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

/** Divisor between consecutive byte units. */
const BYTES_PER_UNIT = 1024;

/**
 * Format a non-negative integer byte count into a human-readable string.
 *
 * The function converts bytes to larger units (KB, MB, GB, TB) using 1024 as the divisor.
 * The numeric portion is rounded to at most 2 decimal places. For all units except TB,
 * the numeric part is in [0, 1024). Very large values stay in TB.
 *
 * @param bytes - Non-negative integer byte count.
 * @returns Formatted byte string (e.g., "0 B", "1 KB", "1.5 MB").
 *
 * @example
 * formatBytes(0)             // "0 B"
 * formatBytes(1023)          // "1023 B"
 * formatBytes(1024)          // "1 KB"
 * formatBytes(1536)          // "1.5 KB"
 * formatBytes(1048576)       // "1 MB"
 * formatBytes(1073741824)    // "1 GB"
 * formatBytes(1099511627776) // "1 TB"
 */
export function formatBytes(bytes: number): string {
  let value = bytes;
  let unitIndex = 0;

  while (value >= BYTES_PER_UNIT && unitIndex < BYTE_UNITS.length - 1) {
    value /= BYTES_PER_UNIT;
    unitIndex++;
  }

  // Round to at most 2 decimal places and remove trailing zeros
  const rounded = Math.round(value * 100) / 100;
  const formatted = rounded.toFixed(2).replace(/\.?0+$/, "");

  return `${formatted} ${BYTE_UNITS[unitIndex]}`;
}

/**
 * Parse a formatted byte string back to a number of bytes.
 *
 * Supports the units B, KB, MB, GB, TB (case-insensitive).
 * The round-trip `parseFormattedBytes(formatBytes(n))` is within 1% of `n`
 * for any non-negative integer.
 *
 * @param formatted - A byte string produced by `formatBytes`.
 * @returns The number of bytes represented by the formatted string.
 *
 * @example
 * parseFormattedBytes("0 B")    // 0
 * parseFormattedBytes("1023 B") // 1023
 * parseFormattedBytes("1 KB")   // 1024
 * parseFormattedBytes("1.5 KB") // 1536
 * parseFormattedBytes("1 MB")   // 1048576
 */
export function parseFormattedBytes(formatted: string): number {
  const pattern = /^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)$/i;
  const match = formatted.match(pattern);

  if (!match) {
    throw new Error(`Invalid formatted byte string: "${formatted}"`);
  }

  const value = parseFloat(match[1]);
  const unit = match[2].toUpperCase();

  const unitMultiplier: Record<string, number> = {
    B: 1,
    KB: BYTES_PER_UNIT,
    MB: BYTES_PER_UNIT ** 2,
    GB: BYTES_PER_UNIT ** 3,
    TB: BYTES_PER_UNIT ** 4,
  };

  const multiplier = unitMultiplier[unit];
  if (multiplier === undefined) {
    throw new Error(`Unknown unit: "${unit}"`);
  }

  return value * multiplier;
}

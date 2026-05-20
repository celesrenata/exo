// Feature: dashboard-inference-controls, Property 12: Exponential backoff delay calculation
// **Validates: Requirements 11.2**

import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import { calculateBackoffDelay } from "$lib/stores/telemetry.svelte.ts";

describe("Property 12: Exponential backoff delay calculation", () => {
  it("result equals min(1000 * 2^n, 30000) for default parameters", () => {
    fc.assert(
      fc.property(fc.nat(30), (attempt) => {
        const result = calculateBackoffDelay(attempt);
        const expected = Math.max(1, Math.min(Math.floor(1000 * Math.pow(2, attempt)), 30000));
        expect(result).toBe(expected);
      }),
      { numRuns: 200 },
    );
  });

  it("result is always a positive integer", () => {
    fc.assert(
      fc.property(fc.nat(30), (attempt) => {
        const result = calculateBackoffDelay(attempt);
        expect(result).toBeGreaterThan(0);
        expect(Number.isInteger(result)).toBe(true);
      }),
      { numRuns: 200 },
    );
  });

  it("result never exceeds maxDelay", () => {
    fc.assert(
      fc.property(
        fc.nat(30),
        fc.integer({ min: 1, max: 5000 }),
        fc.integer({ min: 1000, max: 60000 }),
        (attempt, baseDelay, maxDelay) => {
          const result = calculateBackoffDelay(attempt, baseDelay, maxDelay);
          expect(result).toBeLessThanOrEqual(maxDelay);
        },
      ),
      { numRuns: 200 },
    );
  });

  it("result is monotonically non-decreasing as attempt increases", () => {
    fc.assert(
      fc.property(fc.nat(29), fc.integer({ min: 1, max: 29 }), (a, offset) => {
        const b = a + offset;
        const delayA = calculateBackoffDelay(a);
        const delayB = calculateBackoffDelay(b);
        expect(delayB).toBeGreaterThanOrEqual(delayA);
      }),
      { numRuns: 200 },
    );
  });
});

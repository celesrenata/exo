#!/usr/bin/env bash
# =============================================================================
# Stage Balancing Empirical Comparison
# =============================================================================
#
# Runs bench_xpu.py with three layer distributions on the gremlin cluster
# (Qwen3.5-27B, 4 ranks, 64 layers, Intel Arc Meteor Lake-P iGPUs):
#
#   1. Baseline:  [16, 16, 16, 16]  (uniform)
#   2. Candidate: [17, 17, 16, 14]  (front-heavy, lighter final rank)
#   3. Timing-recommended: auto-computed from per-layer profiling
#
# Run this script on gremlin-1 after deploying the latest exo build.
# Results are saved as timestamped JSON files in ./stage_balancing_results/.
#
# Usage:
#   bash run_stage_balancing_comparison.sh
#   bash run_stage_balancing_comparison.sh --iterations 5
#   bash run_stage_balancing_comparison.sh --gen-tokens 256
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID="Qwen/Qwen3.5-27B"
DEVICE="xpu:0"
DTYPE="bf16"
PROMPT_TOKENS=256
GEN_TOKENS=128
WARMUP=2
ITERATIONS=3
SEED=42

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --gen-tokens)
            GEN_TOKENS="$2"
            shift 2
            ;;
        --prompt-tokens)
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--iterations N] [--gen-tokens N] [--prompt-tokens N] [--warmup N] [--seed N]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
RESULTS_DIR="./stage_balancing_results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "Stage Balancing Empirical Comparison"
echo "============================================================"
echo ""
echo "Model:         ${MODEL_ID}"
echo "Device:        ${DEVICE}"
echo "Dtype:         ${DTYPE}"
echo "Prompt tokens: ${PROMPT_TOKENS}"
echo "Gen tokens:    ${GEN_TOKENS}"
echo "Warmup:        ${WARMUP}"
echo "Iterations:    ${ITERATIONS}"
echo "Seed:          ${SEED}"
echo "Results dir:   ${RESULTS_DIR}"
echo ""
echo "Distributions to test:"
echo "  1. Baseline:           [16, 16, 16, 16]"
echo "  2. Front-heavy:        [17, 17, 16, 14]"
echo "  3. Timing-recommended: (auto from --recommend-layer-distribution)"
echo ""
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Common benchmark arguments
# ---------------------------------------------------------------------------

COMMON_ARGS=(
    --model_id "${MODEL_ID}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --prompt_tokens "${PROMPT_TOKENS}"
    --gen_tokens "${GEN_TOKENS}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --seed "${SEED}"
    --benchmark-mode single-request-decode
)

# ---------------------------------------------------------------------------
# Run 1: Baseline [16, 16, 16, 16]
# ---------------------------------------------------------------------------

echo ">>> Run 1/3: Baseline distribution [16, 16, 16, 16]"
echo ""

BASELINE_JSON="${RESULTS_DIR}/baseline_16_16_16_16.json"

python -m exo.worker.engines.pytorch_xpu.bench_xpu \
    "${COMMON_ARGS[@]}" \
    --pipeline-layer-distribution "16,16,16,16" \
    --json-output-path "${BASELINE_JSON}"

echo ""
echo "    Saved: ${BASELINE_JSON}"
echo ""

# ---------------------------------------------------------------------------
# Run 2: Front-heavy [17, 17, 16, 14]
# ---------------------------------------------------------------------------

echo ">>> Run 2/3: Front-heavy distribution [17, 17, 16, 14]"
echo ""

FRONTHEAVY_JSON="${RESULTS_DIR}/frontheavy_17_17_16_14.json"

python -m exo.worker.engines.pytorch_xpu.bench_xpu \
    "${COMMON_ARGS[@]}" \
    --pipeline-layer-distribution "17,17,16,14" \
    --json-output-path "${FRONTHEAVY_JSON}"

echo ""
echo "    Saved: ${FRONTHEAVY_JSON}"
echo ""

# ---------------------------------------------------------------------------
# Run 3: Timing-recommended distribution
# ---------------------------------------------------------------------------

echo ">>> Run 3/3: Timing-recommended distribution"
echo ""

RECOMMENDED_JSON="${RESULTS_DIR}/timing_recommended.json"

python -m exo.worker.engines.pytorch_xpu.bench_xpu \
    "${COMMON_ARGS[@]}" \
    --pipeline-layer-distribution "16,16,16,16" \
    --recommend-layer-distribution \
    --json-output-path "${RECOMMENDED_JSON}"

echo ""
echo "    Saved: ${RECOMMENDED_JSON}"
echo ""

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

echo "============================================================"
echo "COMPARISON SUMMARY"
echo "============================================================"
echo ""

# Extract key metrics from each JSON file using Python
python3 << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

results_dir = sys.argv[1] if len(sys.argv) > 1 else "."

files = {
    "Baseline [16,16,16,16]": Path(results_dir) / "baseline_16_16_16_16.json",
    "Front-heavy [17,17,16,14]": Path(results_dir) / "frontheavy_17_17_16_14.json",
    "Timing-recommended": Path(results_dir) / "timing_recommended.json",
}

results = {}
for label, path in files.items():
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        continue
    with open(path) as f:
        data = json.load(f)
    results[label] = data.get("metrics", {})

if not results:
    print("  No results found. Check that benchmark runs completed.")
    sys.exit(1)

# Print comparison table
print(f"{'Distribution':<30} {'Decode TPS':>12} {'TTFT (s)':>10} {'Total (s)':>10} {'Bubble %':>10}")
print("-" * 76)

for label, metrics in results.items():
    decode_tps = metrics.get("decode_tps_mean", 0.0)
    ttft = metrics.get("ttft_mean_seconds", 0.0)
    total = metrics.get("total_time_mean_seconds", 0.0)
    bubble = metrics.get("pipeline_bubble_estimate", 0.0) * 100

    print(f"{label:<30} {decode_tps:>12.3f} {ttft:>10.3f} {total:>10.3f} {bubble:>9.1f}%")

print("")

# Determine best distribution
if len(results) >= 2:
    best_label = max(results.keys(), key=lambda k: results[k].get("decode_tps_mean", 0.0))
    best_tps = results[best_label].get("decode_tps_mean", 0.0)
    baseline_tps = results.get("Baseline [16,16,16,16]", {}).get("decode_tps_mean", 0.0)

    if baseline_tps > 0:
        improvement = ((best_tps - baseline_tps) / baseline_tps) * 100
        print(f"Best distribution: {best_label}")
        print(f"  Decode TPS: {best_tps:.3f} ({improvement:+.1f}% vs baseline)")
    else:
        print(f"Best distribution: {best_label} ({best_tps:.3f} tok/s)")

print("")

# Per-rank utilization comparison (if available)
has_utilization = any("per_rank_utilization" in m for m in results.values())
if has_utilization:
    print("Per-Rank Utilization:")
    print(f"{'Distribution':<30} {'Rank 0':>8} {'Rank 1':>8} {'Rank 2':>8} {'Rank 3':>8}")
    print("-" * 66)
    for label, metrics in results.items():
        util = metrics.get("per_rank_utilization", {})
        r0 = util.get("0", util.get(0, 0.0)) * 100
        r1 = util.get("1", util.get(1, 0.0)) * 100
        r2 = util.get("2", util.get(2, 0.0)) * 100
        r3 = util.get("3", util.get(3, 0.0)) * 100
        print(f"{label:<30} {r0:>7.1f}% {r1:>7.1f}% {r2:>7.1f}% {r3:>7.1f}%")
    print("")

# Write combined summary JSON
summary = {
    "timestamp": str(Path(results_dir).name),
    "distributions_tested": list(results.keys()),
    "metrics_by_distribution": results,
}
summary_path = Path(results_dir) / "comparison_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Combined summary written to: {summary_path}")
PYTHON_SCRIPT
"${RESULTS_DIR}"

echo ""
echo "============================================================"
echo "All results saved in: ${RESULTS_DIR}"
echo "============================================================"

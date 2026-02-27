#!/bin/bash
#
# Batch Optimal Configuration Comparison (Shell Loop Alternative)
# ================================================================
# Runs the optimal config comparison for each model individually.
#
# PREFERRED: Use the built-in batch mode (auto-discovers models):
#   python scripts/visualization/visualize_trial_results.py \
#     --optimal --compare --batch --results_root results/trials/ \
#     --output_dir visualizations/optimal_comparisons/
#
# This script is an alternative when you want to control the model list manually.
# Edit the MODELS array below to match your trial directories.
#
# Usage:
#   ./batch_optimal_comparison.sh [results_root] [output_dir]
#
# Examples:
#   ./batch_optimal_comparison.sh
#   ./batch_optimal_comparison.sh results/trials/ visualizations/optimal_comparisons/
#
# Defaults:
#   results_root: results/trials/
#   output_dir:   visualizations/optimal_comparisons/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="${1:-$PROJECT_ROOT/results/trials}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/visualizations/optimal_comparisons}"

# Models to compare (customize this list to match your trial directories)
MODELS=(
    eqcct
    phasenet_original
    phasenetlight_stead
    eqtransformer_original
    eqtransformer_nonconservative
)

echo "=============================================="
echo "Batch Optimal Configuration Comparison"
echo "=============================================="
echo "Results root: $RESULTS_ROOT"
echo "Output dir:  $OUTPUT_DIR"
echo "Models:      ${MODELS[*]}"
echo ""

cd "$PROJECT_ROOT"

for model in "${MODELS[@]}"; do
    echo "Comparing optimal configs for: $model"
    python scripts/visualization/visualize_trial_results.py \
        --optimal --compare --model "$model" \
        --results_root "$RESULTS_ROOT" \
        --output_dir "$OUTPUT_DIR/$model/" || true
done

echo ""
echo "Batch comparison complete. Output saved to: $OUTPUT_DIR"

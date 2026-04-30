#!/usr/bin/env bash
# =============================================================================
# CMU Seasons: DA3-LARGE-1.1 vs Reloc3r-512 (self-eval on DB split)
#
# Step 1: Convert dataset (self_eval mode, c0 camera)
# Step 2: Run DA3 + Reloc3r on each slice
#
# Usage:
#   cd LoopAnything/.worktrees/unified_pipeline1.1
#   bash run_cmu_seasons_ablation.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="src:$(cd ../.. && pwd)"

CMU_ROOT="/data/datasets/CMUSeasons"
DATA_DIR="data/cmu_visloc"
DA3_EVAL="ablation/eval_training_free_visloc.py"
DA3_CONFIG="configs/unified_pipeline_large.yaml"
RELOC3R_EVAL="scripts/eval_reloc3r_visloc.py"
CSV_FILE="cmu_seasons_results.csv"

# Step 1: Prepare dataset
echo "========== Preparing CMU Seasons =========="
if [ ! -d "$DATA_DIR" ]; then
    python scripts/prepare_cmu_seasons_visloc.py \
        --cmu-root "$CMU_ROOT" \
        --output-dir "$DATA_DIR" \
        --mode self_eval --camera c0
else
    echo "  $DATA_DIR already exists, skipping."
fi

# Discover scenes
SCENES=()
for d in "$DATA_DIR"/*/; do
    scene=$(basename "$d")
    [ -f "$DATA_DIR/$scene/TestSplit.txt" ] && SCENES+=("$scene")
done

echo "Scenes: ${SCENES[*]:-none}"

# CSV
write_csv_header() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "model,dataset,scene,median_trans_m,median_rot_deg,status,timestamp" > "$CSV_FILE"
    fi
}

parse_metrics() {
    local output="$1"
    local trans rot
    trans=$(echo "$output" | grep -oP "median pose error: \K[0-9]+\.[0-9]+" | head -1)
    rot=$(echo "$output" | grep -oP "[0-9]+\.[0-9]+ deg" | head -1 | grep -oP "^[0-9]+\.[0-9]+")
    [ -z "$trans" ] || [ -z "$rot" ] && { echo ",,,"; return 1; }
    echo "${trans},${rot}"
}

run_one() {
    local model="$1" dataset="$2" scene="$3" cmd="$4"
    local timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo ""
    echo "--- ${model} | ${dataset}/${scene} (${timestamp}) ---"

    set +e
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    set -e

    echo "$output" | grep -E "median pose error|ERROR" | tail -2
    if [ $exit_code -eq 0 ]; then
        metrics=$(parse_metrics "$output")
        if [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
            echo "${model},${dataset},${scene},${metrics},ok,${timestamp}" >> "$CSV_FILE"
        else
            echo "${model},${dataset},${scene},,,parse_error,${timestamp}" >> "$CSV_FILE"
        fi
    else
        echo "  [ERROR] exit ${exit_code}"
        echo "${model},${dataset},${scene},,,error_${exit_code},${timestamp}" >> "$CSV_FILE"
    fi
}

write_csv_header

total=$(( ${#SCENES[@]} * 2 ))
echo ""
echo "========== Running ${total} experiments =========="

for scene in "${SCENES[@]}"; do
    run_one "DA3-LARGE-1.1" "cmu" "$scene" \
        "python $DA3_EVAL --retriever-backend netvlad --unified-config $DA3_CONFIG --dataset cmu --scene $scene --data-root $DATA_DIR --top-k 10 --pose-path cam_dec --anchor-mode reloc3r_motion_averaging --device cuda"

    run_one "Reloc3r-512" "cmu" "$scene" \
        "python $RELOC3R_EVAL --data-root $DATA_DIR --scene $scene --topk 10 --device cuda"
done

echo ""
echo "========== Done. Results in ${CSV_FILE} =========="

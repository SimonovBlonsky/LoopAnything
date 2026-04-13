#!/usr/bin/env bash
# =============================================================================
# Evaluate trained RelPoseHead on all 7Scenes (+ optional Cambridge).
#
# Runs 3 retrieval backends x 7 scenes = 21 experiments.
# Results saved to relpose_head_7scenes_results.csv (incremental).
#
# Usage:
#   cd /path/to/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
#   bash scripts/eval_relpose_head_7scenes.sh <relpose_checkpoint>
#
# Example:
#   bash scripts/eval_relpose_head_7scenes.sh checkpoints/relpose_head/last.ckpt
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <relpose_checkpoint> [output_csv]"
    exit 1
fi

RELPOSE_CKPT="$1"
CSV_FILE="${2:-relpose_head_7scenes_results.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="src:$(cd ../.. && pwd):$(cd ../../reloc3r && pwd)"

# --------------- Config ---------------
EVAL_SCRIPT="ablation/eval_training_free_visloc.py"
UNIFIED_CONFIG="configs/unified_pipeline.yaml"
DA3_SALAD_CKPT="checkpoints/image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt"
DINO_SALAD_CKPT="da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt"

DATASET="7scenes"
SCENES=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
BACKENDS=("netvlad" "da3_salad" "dino_salad")
POSE_PATH="relpose_head"
ANCHOR_MODE="reloc3r_motion_averaging"
TOP_K=10
DEVICE="cuda"
OUTPUT_DIR="workspace/relpose_head_results"

# --------------- CSV helpers ---------------
write_csv_header() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "backend,pose_path,scene,anchor_mode,top_k,median_trans_m,median_rot_deg,status,timestamp" > "$CSV_FILE"
    fi
}

append_csv_row() {
    echo "$1" >> "$CSV_FILE"
}

parse_metrics() {
    local output="$1"
    local trans rot
    trans=$(echo "$output" | grep -oP "median pose error: \K[0-9]+\.[0-9]+" | head -1)
    rot=$(echo "$output" | grep -oP "[0-9]+\.[0-9]+ deg" | head -1 | grep -oP "^[0-9]+\.[0-9]+")
    if [ -z "$trans" ] || [ -z "$rot" ]; then
        echo ",,,"
        return 1
    fi
    echo "${trans},${rot}"
    return 0
}

# --------------- Build command ---------------
build_cmd() {
    local backend="$1"
    local scene="$2"

    local cmd="python ${EVAL_SCRIPT}"
    cmd+=" --retriever-backend ${backend}"
    cmd+=" --unified-config ${UNIFIED_CONFIG}"
    cmd+=" --dataset ${DATASET}"
    cmd+=" --scene ${scene}"
    cmd+=" --top-k ${TOP_K}"
    cmd+=" --top-m 3"
    cmd+=" --pose-path ${POSE_PATH}"
    cmd+=" --relpose-checkpoint ${RELPOSE_CKPT}"
    cmd+=" --anchor-mode ${ANCHOR_MODE}"
    cmd+=" --device ${DEVICE}"
    cmd+=" --output-dir ${OUTPUT_DIR}"

    if [ "$backend" = "da3_salad" ]; then
        cmd+=" --salad-checkpoint ${DA3_SALAD_CKPT}"
    elif [ "$backend" = "dino_salad" ]; then
        cmd+=" --salad-checkpoint ${DINO_SALAD_CKPT}"
    fi

    echo "$cmd"
}

# --------------- Main loop ---------------
write_csv_header

total=$((${#BACKENDS[@]} * ${#SCENES[@]}))
count=0

echo "============================================================"
echo "Evaluating RelPoseHead: ${total} experiments"
echo "  Checkpoint: ${RELPOSE_CKPT}"
echo "  Results: ${CSV_FILE}"
echo "============================================================"

for backend in "${BACKENDS[@]}"; do
    for scene in "${SCENES[@]}"; do
        count=$((count + 1))
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo ""
        echo "[${count}/${total}] backend=${backend}  scene=${scene}  (${timestamp})"

        cmd=$(build_cmd "$backend" "$scene")

        set +e
        output=$(eval "$cmd" 2>&1)
        exit_code=$?
        set -e

        echo "$output" | tail -5

        if [ $exit_code -eq 0 ]; then
            metrics=$(parse_metrics "$output")
            if [ $? -eq 0 ] && [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
                append_csv_row "${backend},${POSE_PATH},${scene},${ANCHOR_MODE},${TOP_K},${metrics},ok,${timestamp}"
                echo "  => ${metrics}"
            else
                append_csv_row "${backend},${POSE_PATH},${scene},${ANCHOR_MODE},${TOP_K},,,parse_error,${timestamp}"
                echo "  => [WARN] Could not parse metrics"
            fi
        else
            append_csv_row "${backend},${POSE_PATH},${scene},${ANCHOR_MODE},${TOP_K},,,error_${exit_code},${timestamp}"
            echo "  => [ERROR] Exit code ${exit_code}"
        fi
    done
done

echo ""
echo "============================================================"
echo "Done. ${total} experiments. Results in ${CSV_FILE}"
echo "============================================================"

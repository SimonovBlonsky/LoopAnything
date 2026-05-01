#!/usr/bin/env bash
# =============================================================================
# DA3-LARGE-1.1 training-free baseline: NetVLAD + DINO-SALAD, cam_dec pose
# 2 backends × 12 scenes (7 7Scenes + 5 Cambridge) = 24 experiments
#
# Usage:
#   cd LoopAnything/.worktrees/unified_pipeline1.1
#   bash run_da3_large_ablation.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EVAL_SCRIPT="../ablation/eval_training_free_visloc.py"
UNIFIED_CONFIG="../configs/unified_pipeline_large.yaml"
DINO_SALAD_CKPT="../da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt"

BACKENDS=("dino_salad")
POSE_PATH="cam_dec"
ANCHOR_MODE="multiview_motion_averaging"
TOP_K=10
TOP_M=3
DEVICE="cuda"
OUTPUT_DIR="../workspace/da3_cambridge_multiview"

CSV_FILE="da3_large_results.csv"

write_csv_header() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "backend,pose_path,dataset,scene,anchor_mode,top_k,median_trans_m,median_rot_deg,status,timestamp" > "$CSV_FILE"
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

build_cmd() {
    local backend="$1"
    local dataset="$2"
    local scene="$3"

    local cmd="PYTHONPATH=src python ${EVAL_SCRIPT}"
    cmd+=" --retriever-backend ${backend}"
    cmd+=" --unified-config ${UNIFIED_CONFIG}"
    cmd+=" --dataset ${dataset}"
    cmd+=" --scene ${scene}"
    cmd+=" --top-k ${TOP_K}"
    cmd+=" --top-m ${TOP_M}"
    cmd+=" --pose-path ${POSE_PATH}"
    cmd+=" --anchor-mode ${ANCHOR_MODE}"
    cmd+=" --device ${DEVICE}"
    cmd+=" --output-dir ${OUTPUT_DIR}"

    if [ "$backend" = "dino_salad" ]; then
        cmd+=" --salad-checkpoint ${DINO_SALAD_CKPT}"
    fi

    echo "$cmd"
}

# All scenes: 7Scenes (7) + Cambridge (5) = 12
SCENES_7=(chess fire heads office pumpkin redkitchen stairs)
SCENES_CAM=(GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch)

write_csv_header

total=$(( ${#BACKENDS[@]} * (${#SCENES_7[@]} + ${#SCENES_CAM[@]}) ))
count=0

echo "============================================================"
echo "DA3-LARGE-1.1 ablation: ${total} experiments"
echo "  Config: ${UNIFIED_CONFIG}"
echo "  Backends: ${BACKENDS[*]}"
echo "  Results: ${CSV_FILE}"
echo "============================================================"

for backend in "${BACKENDS[@]}"; do
    # 7Scenes
    for scene in "${SCENES_7[@]}"; do
        count=$((count + 1))
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo ""
        echo "[${count}/${total}] ${backend} | 7scenes/${scene} (${timestamp})"

        cmd=$(build_cmd "$backend" "7scenes" "$scene")

        set +e
        output=$(eval "$cmd" 2>&1)
        exit_code=$?
        set -e

        echo "$output" | tail -3

        if [ $exit_code -eq 0 ]; then
            metrics=$(parse_metrics "$output")
            if [ $? -eq 0 ] && [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
                append_csv_row "${backend},${POSE_PATH},7scenes,${scene},${ANCHOR_MODE},${TOP_K},${metrics},ok,${timestamp}"
            else
                append_csv_row "${backend},${POSE_PATH},7scenes,${scene},${ANCHOR_MODE},${TOP_K},,,parse_error,${timestamp}"
            fi
        else
            append_csv_row "${backend},${POSE_PATH},7scenes,${scene},${ANCHOR_MODE},${TOP_K},,,error_${exit_code},${timestamp}"
            echo "  [ERROR] exit ${exit_code}"
        fi
    done

    # Cambridge
    for scene in "${SCENES_CAM[@]}"; do
        count=$((count + 1))
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo ""
        echo "[${count}/${total}] ${backend} | cambridge/${scene} (${timestamp})"

        cmd=$(build_cmd "$backend" "cambridge" "$scene")

        set +e
        output=$(eval "$cmd" 2>&1)
        exit_code=$?
        set -e

        echo "$output" | tail -3

        if [ $exit_code -eq 0 ]; then
            metrics=$(parse_metrics "$output")
            if [ $? -eq 0 ] && [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
                append_csv_row "${backend},${POSE_PATH},cambridge,${scene},${ANCHOR_MODE},${TOP_K},${metrics},ok,${timestamp}"
            else
                append_csv_row "${backend},${POSE_PATH},cambridge,${scene},${ANCHOR_MODE},${TOP_K},,,parse_error,${timestamp}"
            fi
        else
            append_csv_row "${backend},${POSE_PATH},cambridge,${scene},${ANCHOR_MODE},${TOP_K},,,error_${exit_code},${timestamp}"
            echo "  [ERROR] exit ${exit_code}"
        fi
    done
done

echo ""
echo "============================================================"
echo "Done. ${count}/${total} experiments. Results in ${CSV_FILE}"
echo "============================================================"

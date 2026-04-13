#!/usr/bin/env bash
# =============================================================================
# Automated 7Scenes training-free baseline ablation
#
# 3 retrieval backends × 2 pose paths × 7 scenes = 42 experiments
# Results are appended to CSV after each experiment completes.
#
# Usage:
#   cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
#   nohup bash run_all_7scenes_ablation.sh > ablation_run.log 2>&1 &
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------- Config ---------------
EVAL_SCRIPT="ablation/eval_training_free_visloc.py"
UNIFIED_CONFIG="configs/unified_pipeline.yaml"
DA3_SALAD_CKPT="checkpoints/image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt"
DINO_SALAD_CKPT="da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt"

DATASET="7scenes"
SCENES=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
BACKENDS=("da3_salad" "dino_salad" "netvlad")
POSE_PATHS=("cam_dec" "ray")
ANCHOR_MODE="reloc3r_motion_averaging"
TOP_K=10
TOP_M=3
DEVICE="cuda"
OUTPUT_DIR="workspace/ablation_results"

CSV_FILE="baseline_7scenes_results.csv"

# --------------- CSV helpers ---------------
write_csv_header() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "backend,pose_path,scene,anchor_mode,top_k,median_trans_m,median_rot_deg,status,timestamp" > "$CSV_FILE"
    fi
}

append_csv_row() {
    echo "$1" >> "$CSV_FILE"
}

# --------------- Build command ---------------
build_cmd() {
    local backend="$1"
    local pose_path="$2"
    local scene="$3"

    local cmd="PYTHONPATH=src python ${EVAL_SCRIPT}"
    cmd+=" --retriever-backend ${backend}"
    cmd+=" --unified-config ${UNIFIED_CONFIG}"
    cmd+=" --dataset ${DATASET}"
    cmd+=" --scene ${scene}"
    cmd+=" --top-k ${TOP_K}"
    cmd+=" --top-m ${TOP_M}"
    cmd+=" --pose-path ${pose_path}"
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

# --------------- Parse metrics from stdout ---------------
parse_metrics() {
    # Expected line format:
    # [Training-Free][backend][branch] Scene X median pose error: 0.12 m  3.45 deg
    local output="$1"
    local pose_path="$2"

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

# --------------- Main loop ---------------
write_csv_header

total=$((${#BACKENDS[@]} * ${#POSE_PATHS[@]} * ${#SCENES[@]}))
count=0

for backend in "${BACKENDS[@]}"; do
    for pose_path in "${POSE_PATHS[@]}"; do
        for scene in "${SCENES[@]}"; do
            count=$((count + 1))
            timestamp=$(date "+%Y-%m-%d %H:%M:%S")
            echo ""
            echo "============================================================"
            echo "[${count}/${total}] backend=${backend}  pose=${pose_path}  scene=${scene}"
            echo "Started: ${timestamp}"
            echo "============================================================"

            cmd=$(build_cmd "$backend" "$pose_path" "$scene")
            echo "CMD: ${cmd}"

            set +e
            output=$(eval "$cmd" 2>&1)
            exit_code=$?
            set -e

            echo "$output"

            if [ $exit_code -eq 0 ]; then
                metrics=$(parse_metrics "$output" "$pose_path")
                if [ $? -eq 0 ] && [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
                    append_csv_row "${backend},${pose_path},${scene},${ANCHOR_MODE},${TOP_K},${metrics},ok,${timestamp}"
                    echo "[RESULT] ${backend} | ${pose_path} | ${scene} => ${metrics}"
                else
                    append_csv_row "${backend},${pose_path},${scene},${ANCHOR_MODE},${TOP_K},,,parse_error,${timestamp}"
                    echo "[WARN] Finished but could not parse metrics."
                fi
            else
                append_csv_row "${backend},${pose_path},${scene},${ANCHOR_MODE},${TOP_K},,,error_${exit_code},${timestamp}"
                echo "[ERROR] Exit code ${exit_code}"
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "All ${total} experiments completed. Results in ${CSV_FILE}"
echo "============================================================"

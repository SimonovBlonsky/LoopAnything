#!/usr/bin/env bash
# =============================================================================
# EuRoC + ETH3D: DA3-LARGE-1.1 vs Reloc3r-512 (NetVLAD retrieval)
#
# Step 1: Convert datasets (if not already done)
# Step 2: Run DA3 + Reloc3r on all scenes, save results incrementally
#
# Usage:
#   cd LoopAnything/.worktrees/unified_pipeline1.1
#   bash run_euroc_eth3d_ablation.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="src:$(cd ../.. && pwd)"

DA3_EVAL="ablation/eval_training_free_visloc.py"
DA3_CONFIG="configs/unified_pipeline_large.yaml"
RELOC3R_EVAL="scripts/eval_reloc3r_visloc.py"
CSV_FILE="euroc_eth3d_results.csv"

# ---- Step 1: Prepare datasets ----
echo "========== Preparing EuRoC =========="
if [ ! -d "data/euroc_visloc" ]; then
    python scripts/prepare_euroc_visloc.py \
        --euroc-root /data/datasets/EuRoC_mav \
        --output-dir data/euroc_visloc
else
    echo "  data/euroc_visloc/ already exists, skipping."
fi

echo ""
echo "========== Preparing ETH3D =========="
if [ ! -d "data/eth3d_visloc" ]; then
    python scripts/prepare_eth3d_visloc.py \
        --eth3d-root /data/datasets/ETH3DSLAM \
        --output-dir data/eth3d_visloc
else
    echo "  data/eth3d_visloc/ already exists, skipping."
fi

# ---- Discover scenes ----
EUROC_SCENES=()
if [ -d "data/euroc_visloc" ]; then
    for d in data/euroc_visloc/*/; do
        scene=$(basename "$d")
        [ -f "data/euroc_visloc/$scene/TestSplit.txt" ] && EUROC_SCENES+=("$scene")
    done
fi

ETH3D_SCENES=()
if [ -d "data/eth3d_visloc" ]; then
    for d in data/eth3d_visloc/*/; do
        scene=$(basename "$d")
        [ -f "data/eth3d_visloc/$scene/TestSplit.txt" ] && ETH3D_SCENES+=("$scene")
    done
fi

echo ""
echo "EuRoC scenes: ${EUROC_SCENES[*]:-none}"
echo "ETH3D scenes: ${ETH3D_SCENES[*]:-none}"

# ---- CSV ----
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

run_experiment() {
    local model="$1" dataset="$2" scene="$3" data_root="$4" cmd="$5"
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

total=$(( (${#EUROC_SCENES[@]} + ${#ETH3D_SCENES[@]}) * 2 ))
echo ""
echo "========== Running ${total} experiments =========="

# ---- EuRoC ----
for scene in "${EUROC_SCENES[@]}"; do
    run_experiment "DA3-LARGE-1.1" "euroc" "$scene" "data/euroc_visloc" \
        "python $DA3_EVAL --retriever-backend netvlad --unified-config $DA3_CONFIG --dataset euroc --scene $scene --data-root data/euroc_visloc --top-k 10 --pose-path cam_dec --anchor-mode reloc3r_motion_averaging --device cuda"

    run_experiment "Reloc3r-512" "euroc" "$scene" "data/euroc_visloc" \
        "python $RELOC3R_EVAL --data-root data/euroc_visloc --scene $scene --topk 10 --device cuda"
done

# ---- ETH3D ----
for scene in "${ETH3D_SCENES[@]}"; do
    run_experiment "DA3-LARGE-1.1" "eth3d" "$scene" "data/eth3d_visloc" \
        "python $DA3_EVAL --retriever-backend netvlad --unified-config $DA3_CONFIG --dataset eth3d --scene $scene --data-root data/eth3d_visloc --top-k 10 --pose-path cam_dec --anchor-mode reloc3r_motion_averaging --device cuda"

    run_experiment "Reloc3r-512" "eth3d" "$scene" "data/eth3d_visloc" \
        "python $RELOC3R_EVAL --data-root data/eth3d_visloc --scene $scene --topk 10 --device cuda"
done

echo ""
echo "========== Done. Results in ${CSV_FILE} =========="

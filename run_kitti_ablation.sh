#!/usr/bin/env bash
# =============================================================================
# KITTI visual localization: DA3-LARGE-1.1 vs Reloc3r-512
#
# Runs both models on all KITTI sequences with loop closures.
# Uses NetVLAD retrieval for both (fair comparison).
# Results appended incrementally to CSV.
#
# Usage:
#   cd LoopAnything/.worktrees/unified_pipeline1.1
#   bash run_kitti_ablation.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="src:$(cd ../.. && pwd)"

DATA_ROOT="data/kitti_visloc"
SEQUENCES=("00" "05" "06" "07")
CSV_FILE="kitti_results.csv"

# DA3 config
DA3_EVAL="ablation/eval_training_free_visloc.py"
DA3_CONFIG="configs/unified_pipeline_large.yaml"

# Reloc3r config
RELOC3R_EVAL="scripts/eval_reloc3r_kitti.py"

write_csv_header() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "model,backend,scene,median_trans_m,median_rot_deg,status,timestamp" > "$CSV_FILE"
    fi
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
}

write_csv_header

total=$(( ${#SEQUENCES[@]} * 2 ))
count=0

echo "============================================================"
echo "KITTI Visual Localization: DA3-LARGE-1.1 vs Reloc3r-512"
echo "  Sequences: ${SEQUENCES[*]}"
echo "  Results: ${CSV_FILE}"
echo "============================================================"

for scene in "${SEQUENCES[@]}"; do
    # --- DA3-LARGE-1.1 ---
    count=$((count + 1))
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo ""
    echo "[${count}/${total}] DA3-LARGE-1.1 | kitti/${scene} (${timestamp})"

    set +e
    output=$(python "$DA3_EVAL" \
        --retriever-backend netvlad \
        --unified-config "$DA3_CONFIG" \
        --dataset kitti --scene "$scene" \
        --data-root "$DATA_ROOT" \
        --top-k 10 --pose-path cam_dec \
        --anchor-mode reloc3r_motion_averaging \
        --device cuda 2>&1)
    exit_code=$?
    set -e

    echo "$output" | tail -3
    if [ $exit_code -eq 0 ]; then
        metrics=$(parse_metrics "$output")
        if [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
            echo "$metrics" | sed "s/^/  => /"
            echo "DA3-LARGE-1.1,netvlad,${scene},${metrics},ok,${timestamp}" >> "$CSV_FILE"
        else
            echo "DA3-LARGE-1.1,netvlad,${scene},,,parse_error,${timestamp}" >> "$CSV_FILE"
        fi
    else
        echo "  [ERROR] exit ${exit_code}"
        echo "DA3-LARGE-1.1,netvlad,${scene},,,error_${exit_code},${timestamp}" >> "$CSV_FILE"
    fi

    # --- Reloc3r-512 ---
    count=$((count + 1))
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo ""
    echo "[${count}/${total}] Reloc3r-512 | kitti/${scene} (${timestamp})"

    set +e
    output=$(python "$RELOC3R_EVAL" \
        --data-root "$DATA_ROOT" \
        --scene "$scene" \
        --topk 10 \
        --device cuda 2>&1)
    exit_code=$?
    set -e

    echo "$output" | tail -3
    if [ $exit_code -eq 0 ]; then
        metrics=$(parse_metrics "$output")
        if [ -n "$metrics" ] && [ "$metrics" != ",,," ]; then
            echo "$metrics" | sed "s/^/  => /"
            echo "Reloc3r-512,netvlad,${scene},${metrics},ok,${timestamp}" >> "$CSV_FILE"
        else
            echo "Reloc3r-512,netvlad,${scene},,,parse_error,${timestamp}" >> "$CSV_FILE"
        fi
    else
        echo "  [ERROR] exit ${exit_code}"
        echo "Reloc3r-512,netvlad,${scene},,,error_${exit_code},${timestamp}" >> "$CSV_FILE"
    fi
done

echo ""
echo "============================================================"
echo "Done. ${count}/${total} experiments. Results in ${CSV_FILE}"
echo "============================================================"

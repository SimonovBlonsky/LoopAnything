#!/usr/bin/env bash
# =============================================================================
# Train RelPoseHead on DA3 frozen backbone (4x RTX3090)
#
# Prerequisites:
#   - reloc3r training data available under reloc3r/data/
#   - DA3 pretrained weights cached (auto-downloaded on first run)
#
# Usage:
#   cd /path/to/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
#   bash scripts/train_relpose_head.sh
#
# Or with nohup for unattended training:
#   nohup bash scripts/train_relpose_head.sh > train_relpose.log 2>&1 &
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# reloc3r datasets use relative ./data/ paths resolved from reloc3r/ dir.
# Create symlink if not already present so datasets are accessible from here.
RELOC3R_DATA="$(cd ../../reloc3r/data 2>/dev/null && pwd)" || true
if [ -n "$RELOC3R_DATA" ] && [ ! -e "data" ]; then
    ln -sf "$RELOC3R_DATA" data
    echo "[INFO] Symlinked data -> $RELOC3R_DATA"
fi

export PYTHONPATH="src:$(cd ../.. && pwd):$(cd ../../reloc3r && pwd)"

echo "============================================================"
echo "Training RelPoseHead on DA3 backbone"
echo "  Project dir: $PROJECT_DIR"
echo "  GPUs: 4"
echo "  Config: configs/train_relpose_head.yaml"
echo "============================================================"

torchrun --nproc_per_node=4 \
    train/train_relpose_head.py \
    --config configs/train_relpose_head.yaml \
    --devices 4 \
    --seed 42

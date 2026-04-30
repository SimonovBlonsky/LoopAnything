for seq in 00 05 06 07; do
  echo "=== Reloc3r-512 | kitti/${seq} ==="
  PYTHONPATH="src:/home/chenguyuan/code/NeurIPS26:/home/chenguyuan/code/NeurIPS26/reloc3r" \
    python scripts/eval_reloc3r_kitti.py \
    --data-root data/kitti_visloc --scene $seq --topk 10 --device cuda
  echo ""
done

python ablation/eval_training_free_visloc.py \
    --retriever-backend netvlad \
    --unified-config configs/unified_pipeline_large.yaml \
    --dataset 7scenes --scene heads \
    --top-k 10 --pose-path cam_dec \
    --anchor-mode reloc3r_motion_averaging \
    --device cuda \
    --scale-diagnostics
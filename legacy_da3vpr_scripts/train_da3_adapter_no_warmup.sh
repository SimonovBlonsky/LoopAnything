CUDA_VISIBLE_DEVICES=2 \
  conda run --no-capture-output -n da3 \
  python train/train_da3_adapter_salad.py \
    --seed 0 \
    --feature-adapter-arch identity \
    --aggregator-ckpt-path none \
    ${COMMON_ARGS}
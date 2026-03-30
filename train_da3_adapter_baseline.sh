CUDA_VISIBLE_DEVICES=2 \
  conda run --no-capture-output -n da3 \
  python train/train_da3_adapter_salad.py \
    --seed 0 \
    --feature-adapter-arch identity \
    --aggregator-ckpt-path da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt \
    ${COMMON_ARGS}
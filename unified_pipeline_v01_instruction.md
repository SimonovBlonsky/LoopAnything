# Unified Pipeline v0.1 运行文档

所有命令均在以下工作目录执行：

```bash
cd ~/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1
conda activate da3
```

## 目录结构

```
configs/
  unified_pipeline.yaml          # 模型配置（推理/评估用）
  train_unified_stage1.yaml      # Stage 1 训练配置
eval_unified_visloc.py           # 主评估脚本
train/train_unified_pipeline.py  # 训练脚本
ablation/
  eval_ourvpr_plus_reloc3r.py    # 消融1: 我们的VPR + reloc3r位姿
  eval_dinosalad_plus_ourpose.py # 消融2: DINO SALAD + 我们的位姿
checkpoints/
  image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt  # 预训练VPR权重
```

## 数据集

7Scenes 数据位于 `~/code/NeurIPS26/reloc3r/data/7scenes/`，已有全部 7 个场景：
chess, fire, heads, office, pumpkin, redkitchen, stairs

每个场景通过 `TrainSplit.txt` / `TestSplit.txt` 按 sequence 划分 train/test。

---

## 0. Pipeline 架构

### 0.1 完整前向流程（`forward()`）

```
Query Image [B,1,3,H,W]
        |
        v
+------------------+
|  DA3 Backbone    |  (frozen, DINOv2-based)
|  (single pass)   |
+------------------+
        |
        +---> feats (final layer)          aux_feats (intermediate layer 5)
        |     [B,1,P,1536]                 [B,1,P,768]
        |           |                            |
        |           |                            v
        |           |                  +--------------------+
        |           |                  | extract_aux_patch  |
        |           |                  | _tokens            |
        |           |                  +--------------------+
        |           |                            |
        |           |                       [B,P,768]
        |           |                            v
        |           |                  +--------------------+
        |           |                  | Feature Adapter    |  (PatchOnly, trainable)
        |           |                  | (bottleneck 640)   |
        |           |                  +--------------------+
        |           |                            |
        |           |                    feature_map + global_token
        |           |                            v
        |           |                  +--------------------+
        |           |                  | SALAD Aggregator   |  (trainable)
        |           |                  | (16 clusters)      |
        |           |                  +--------------------+
        |           |                            |
        |           |                    query_descriptor [B,D]
        |           |                            |
        |           |         +------------------+
        |           |         |
        |           |         v
        |           |  +------------------+     candidate_descriptors [K,D]
        |           |  | Retrieval        |  <--- (from database)
        |           |  | Strategy         |
        |           |  | (SoftAttention)  |
        |           |  +------------------+
        |           |         |
        |           |    weights [B,K]
        |           |         |
        |           v         v
        |  +---------------------------------+    candidate_patch_tokens [B,K,P,C]
        |  |   Cross-View Fusion             | <--- candidate_camera_tokens [B,K,C]
        |  |   (cross-attention, 2 layers)   |      (from database)
        |  +---------------------------------+
        |           |
        |   enhanced_patch [B,P,1536]
        |   enhanced_cam   [B,1536]
        |           |
        |           v
        |  +------------------+
        |  |  DA3 Head        |  (DualDPT, trainable)
        |  |  (depth/3DGS)   |
        |  +------------------+
        |           |
        |   enhanced_cam
        |           |
        |           v
        |  +------------------+
        |  |  CamDec          |  (trainable)
        |  +------------------+
        |           |
        |       pose_enc [B,1,9]
        |           |
        |           v
        |  +------------------+
        |  | pose_encoding_   |  (caller-side decoding)
        |  | to_extri_intri() |
        |  +------------------+
        |           |
        |       c2w [B,1,4,4]  (camera-to-world)
```

### 0.2 调用模式总览

```
                            UnifiedPipeline
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   retrieval_only()        forward()              │
    │   ┌──────────┐     ┌─────────────────────┐       │
    │   │ Backbone │     │ Backbone            │       │
    │   │    +     │     │    +                │       │
    │   │ Adapter  │     │ Adapter + Retrieval │       │
    │   │    +     │     │    +                │       │
    │   │ SALAD    │     │ Fusion + Head       │       │
    │   └──┬───────┘     └────────┬────────────┘       │
    │      │                      │                    │
    │  descriptor            pose_enc                  │
    │                                                  │
    │   pose_only()      extract_database_features()   │
    │   ┌──────────┐     ┌─────────────────────┐       │
    │   │ Backbone │     │ Backbone            │       │
    │   │ (q + K)  │     │    +                │       │
    │   │    +     │     │ Adapter + SALAD     │       │
    │   │ Fusion   │     │    +                │       │
    │   │    +     │     │ extract final feats │       │
    │   │ Head     │     └────────┬────────────┘       │
    │   └──┬───────┘              │                    │
    │      │               descriptor [B,D]            │
    │  pose_enc            patch_tokens [B,P,C]        │
    │                      camera_tokens [B,C]         │
    └──────────────────────────────────────────────────┘
```

### 0.3 训练阶段冻结策略

```
                        Stage 1 (Pose)           VPR Training
                        ─────────────            ────────────
  DA3 Backbone          frozen                   frozen
  Feature Adapter       frozen                   TRAINABLE
  SALAD Aggregator      frozen                   TRAINABLE
  Retrieval Strategy    --                       --
  Cross-View Fusion     TRAINABLE                frozen
  DA3 Head + CamDec     TRAINABLE                frozen
                        ─────────────            ────────────
  Loss                  Geodesic + L2 trans      MultiSimilarity
  Data                  7scenes (pose pairs)     GSVCities (places)
  Config                train_unified_stage1     train_unified_vpr
```

### 0.4 评估 / 部署流程

```
                    ┌───────────────────────────────────────────┐
  Offline (建库)    │  for each db_frame:                       │
                    │    extract_database_features(frame)       │
                    │      → descriptor   → 存盘 (memmap)       │
                    │      → patch_tokens → 存盘 (memmap)       │
                    │      → camera_token → 存盘 (memmap)       │
                    └───────────────────────────────────────────┘
                                        |
                                        v
                    ┌───────────────────────────────────────────┐
  Online (查询)     │  1. retrieval_only(query)                 │
                    │      → query_descriptor                   │
                    │  2. cosine_similarity → top-K indices     │
                    │  3. load top-K patch/cam tokens (memmap)  │
                    │  4. forward(query, cand_patch, cand_cam,  │
                    │              cand_desc)                   │
                    │      → pose_enc → decode → c2w            │
                    └───────────────────────────────────────────┘
```

### 0.5 关键张量形状

| 张量 | 形状 | 说明 |
|------|------|------|
| 输入图像 | `[B, 1, 3, H, W]` | H=W=504 (eval) 或 224 (VPR训练) |
| aux_feats | `[B, 1, P, 768]` | P = (H/14)*(W/14), layer 5 输出 |
| final feats patch | `[B, 1, P, 1536]` | 768*2 (cat_token) |
| final feats cam | `[B, 1, 1536]` | camera token |
| descriptor | `[B, D]` | D 取决于 SALAD 配置 |
| candidate patch | `[B, K, P, C]` | K=top_k, C=1536 |
| candidate cam | `[B, K, C]` | C=1536 |
| weights | `[B, K]` | 检索权重，和为 1 |
| pose_enc | `[B, 1, 9]` | 编码后的位姿 |
| c2w | `[B, 1, 4, 4]` | camera-to-world 矩阵 |

---

## 1. 评估（Evaluation）

### 1.1 单场景评估

```bash
python eval_unified_visloc.py \
    --model-config configs/unified_pipeline.yaml \
    --dataset 7scenes \
    --scene fire \
    --top-k 10 \
    --batch-size 16 \
    --device cuda \
    --cache-dir workspace/db_cache/fire
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-config` | 模型配置 YAML | 必填 |
| `--checkpoint` | 训练后的 pipeline checkpoint | 无（用预训练权重） |
| `--dataset` | 数据集名 | `7scenes` 或 `cambridge` |
| `--scene` | 场景名 | 必填 |
| `--top-k` | 检索返回的候选数量 | 10 |
| `--batch-size` | 构建 database 时的 batch size | 16 |
| `--device` | 计算设备 | cuda |
| `--image-size` | 输入图像尺寸 | 504 504 |
| `--cache-dir` | database 特征缓存目录（memmap） | 临时目录 |
| `--data-root` | 覆盖默认数据集路径 | 无 |
| `--output-dir` | 结果保存目录 | workspace/eval_results |

### 1.2 全场景批量评估

```bash
for scene in chess fire heads office pumpkin redkitchen stairs; do
    echo "=== Evaluating $scene ==="
    python eval_unified_visloc.py \
        --model-config configs/unified_pipeline.yaml \
        --dataset 7scenes \
        --scene $scene \
        --top-k 10 \
        --batch-size 16 \
        --device cuda \
        --cache-dir workspace/db_cache/$scene \
        --output-dir workspace/eval_results
done
```

### 1.3 使用训练后 checkpoint 评估

```bash
python eval_unified_visloc.py \
    --model-config configs/unified_pipeline.yaml \
    --checkpoint logs/unified_pipeline/checkpoints/last.ckpt \
    --dataset 7scenes \
    --scene fire \
    --device cuda \
    --cache-dir workspace/db_cache/fire
```

### 1.4 输出格式

- 终端打印：`Scene fire median pose error: X.XX m  X.XX deg`
- 文件保存：`workspace/eval_results/7scenes_fire_results.npz`
  - `rotation_errors`: 每个 query 的旋转误差 (deg)
  - `translation_errors`: 每个 query 的平移误差 (m)

---

## 2. 训练（Training）

### 2.1 Stage 1 训练（冻结 backbone + VPR，训练 fusion + head）

```bash
python train/train_unified_pipeline.py \
    --config configs/train_unified_stage1.yaml \
    --devices 1 \
    --seed 42
```

**关键训练配置（`train_unified_stage1.yaml`）：**
- 冻结：backbone + VPR (adapter + SALAD)
- 可训练：cross_view_fusion + da3_head + cam_dec
- 优化器：AdamW, lr=1e-4, weight_decay=1e-4
- 调度器：CosineAnnealing, T_max=50
- 损失：geodesic rotation loss + L2 translation loss（等权重）
- 候选采样：pos_threshold=1.0, neg_threshold=3.0, pos_ratio=0.7
- 评估：每 5 个 epoch 验证一次

### 2.2 从 checkpoint 恢复训练

```bash
python train/train_unified_pipeline.py \
    --config configs/train_unified_stage1.yaml \
    --resume-from logs/unified_pipeline/checkpoints/last.ckpt \
    --devices 1
```

### 2.3 多 GPU 训练

```bash
python train/train_unified_pipeline.py \
    --config configs/train_unified_stage1.yaml \
    --devices 4
```

### 2.4 训练输出

- Checkpoint 保存在 `logs/unified_pipeline/` 下
- 保留 top-3 best（按 val/trans_err_m）+ last.ckpt
- TensorBoard 日志：`tensorboard --logdir logs/unified_pipeline/`

---

## 3. 消融实验（Ablation）

### 3.1 消融 1：我们的 VPR + reloc3r 位姿回归

用我们的 image retrieval 检索 top-K，然后用 reloc3r 对每个 pair 做相对位姿回归 + motion averaging。

```bash
python ablation/eval_ourvpr_plus_reloc3r.py \
    --unified-config configs/unified_pipeline.yaml \
    --reloc3r-model "Reloc3rRelpose(img_size=512)" \
    --dataset 7scenes \
    --scene fire \
    --top-k 10 \
    --batch-size 16 \
    --device cuda
```

全场景批量：
```bash
for scene in chess fire heads office pumpkin redkitchen stairs; do
    echo "=== Ablation 1: $scene ==="
    python ablation/eval_ourvpr_plus_reloc3r.py \
        --unified-config configs/unified_pipeline.yaml \
        --reloc3r-model "Reloc3rRelpose(img_size=512)" \
        --dataset 7scenes \
        --scene $scene \
        --top-k 10 \
        --device cuda \
        --output-dir workspace/ablation_results
done
```

**参数说明（额外）：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--unified-config` | unified pipeline 配置 | 必填 |
| `--unified-checkpoint` | pipeline checkpoint（仅用 VPR 部分） | 无 |
| `--reloc3r-model` | reloc3r 模型构造字符串 | `Reloc3rRelpose(img_size=512)` |

### 3.2 消融 2：DINO SALAD 检索 + 我们的位姿估计

用独立的 DINO SALAD 模型检索 top-K，然后用我们的 `pose_only()` 做位姿回归。

```bash
python ablation/eval_dinosalad_plus_ourpose.py \
    --unified-config configs/unified_pipeline.yaml \
    --salad-checkpoint <DINO_SALAD_CHECKPOINT_PATH> \
    --dataset 7scenes \
    --scene fire \
    --top-k 10 \
    --batch-size 16 \
    --device cuda
```

全场景批量：
```bash
for scene in chess fire heads office pumpkin redkitchen stairs; do
    echo "=== Ablation 2: $scene ==="
    python ablation/eval_dinosalad_plus_ourpose.py \
        --unified-config configs/unified_pipeline.yaml \
        --salad-checkpoint <DINO_SALAD_CHECKPOINT_PATH> \
        --dataset 7scenes \
        --scene $scene \
        --top-k 10 \
        --device cuda \
        --output-dir workspace/ablation_results
done
```

**参数说明（额外）：**
| 参数 | 说明 |
|------|------|
| `--salad-checkpoint` | DINO SALAD 预训练 checkpoint 路径（必填） |

---

## 4. 单元测试

```bash
conda run -n da3 python -m pytest tests/ -v
```

---

## 5. 注意事项

- **首次运行评估**会自动下载 DA3-BASE 预训练权重（`depth-anything/DA3-BASE`），需要网络连接
- **`--cache-dir`**：建议指定固定目录，避免重复构建 database。同一场景的 cache 可复用（只要模型权重不变）
- **内存**：database 特征全部以 memmap 方式存盘，评估时峰值内存约 500MB，不会 OOM
- **当前状态**：pipeline 的 cross_view_fusion 和 head 尚未经过端到端训练，评估结果为 baseline（DA3 原始权重 + 未训练的 fusion）。训练后用 `--checkpoint` 加载即可
- **指标定义**：与 reloc3r 完全一致 — median rotation error (deg) + median translation error (m)

# Experiment Log

## Directions to explore

- Sweep dropout rate and temperature
- Further decrease learning rate
- Try CNN pyramid architecture
- Increase batch size (with smaller transformer or CNN)
- Address repetitive elements (soft-masked/lowercase in genome): they have strong patterns the model may latch onto but are mostly non-functional. Could confuse the contrastive objective. Ideas:
  - Mask out repeat regions before pooling (only embed non-repetitive content)
  - Weighted mean pooling (downweight soft-masked positions, similar to MLM loss weighting)
  - Augmentation via repeat shuffling (permute repeat regions in positive pairs to teach invariance)

---

## 2026-03-13: SimCSE Small Transformer

**Branch**: `SimCSE`

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_transformer_small \
  trainer.devices=1 trainer.strategy=auto \
  data.dataset_name=songlab/gpn-animal-promoter-dataset \
  data.batch_size=256
```

**Config**:
- Model: SimCSE, 6-layer transformer, 512 hidden, 8 heads (18.7M params)
- Data: songlab/gpn-animal-promoter-dataset (9M train sequences)
- Batch size: 256 (no gradient accumulation)
- Optimizer: AdamW, lr=0.001, weight_decay=0.1
- Scheduler: cosine with warmup (2k warmup, 20k total steps)
- Precision: bf16-mixed
- Dataloader workers: 8 (default from data config)
- GPU: 1x NVIDIA L40S (46GB)

**W&B**: https://wandb.ai/gonzalobenegas/glm-experiments/runs/pwhql8by?nw=nwusergonzalobenegas

**Results**:

| Step | train/simcse_loss | val/simcse_loss | val/traitgym_mendelian_promoter_auprc |
|------|-------------------|-----------------|---------------------------------------|
| 1000 | 0.0069            | 0.0106          | 0.2271                                |
| 2000 | 0.0331            | 0.0316          | 0.0881                                |

Run cancelled at step 2240 (~2 it/s).

**Observations**:
- Train loss collapses fast (4.57 → 0.007 by 1k) then rebounds at 2k — sign of training instability
- Val loss lags behind train loss, tripling between 1k and 2k (0.011 → 0.032)
- AUPRC peaks at 1k (0.23) then drops below random baseline by 2k (0.09) — the learned representations degrade
- Still in warmup phase at 2k/2k warmup steps — lr is still ramping up, which may explain the destabilization
- Batch size 256 is small for contrastive learning — fewer in-batch negatives means noisier gradients

**Next steps**:
- Lower peak lr (0.001 seems too aggressive for this model size)
- Shorten warmup or reduce total steps to avoid late-warmup instability
- Log more frequently (every 100-200 steps) to see dynamics more clearly
- Consider larger effective batch size via gradient accumulation (more in-batch negatives help contrastive objectives)
- Try temperature sweep (current 0.05 is quite sharp)

---

## 2026-03-13: SimCSE Small Transformer — lr=1e-4

**Hypothesis**: Previous run destabilized as lr ramped to 1e-3 peak. Reducing peak lr by 10x should stabilize training and preserve AUPRC gains.

**Changes from previous run**:
- lr: 0.001 → 0.0001

**Config**: `configs/experiment/simcse_small_lr1e4.yaml`

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_small_lr1e4
```

**W&B**: https://wandb.ai/gonzalobenegas/glm-experiments/runs/zxatgqpv

**Results**:

| Step | train/simcse_loss | val/simcse_loss | val/traitgym_mendelian_promoter_auprc |
|------|-------------------|-----------------|---------------------------------------|
| 100  | 1.660             | 1.700           | 0.1468                                |
| 200  | 0.467             | 1.700           | 0.1230                                |
| 300  | 0.146             | 0.589           | 0.1305                                |
| 400  | 0.073             | 0.297           | 0.1892                                |
| 500  | 0.028             | 0.100           | 0.2132                                |
| 600  | 0.023             | 0.060           | 0.2860                                |
| 700  | 0.008             | 0.027           | 0.2494                                |
| 800  | 0.008             | 0.023           | 0.2177                                |
| 900  | 0.003             | 0.013           | 0.2734                                |
| 1000 | 0.002             | 0.011           | 0.1709                                |
| 1100 | 0.002             | 0.007           | 0.1499                                |
| 1200 | 0.002             | 0.007           | 0.1783                                |
| 1300 | 0.001             | 0.005           | 0.1825                                |
| 1400 | 0.001             | 0.006           | 0.1647                                |
| 1500 | 0.001             | 0.004           | 0.1606                                |
| 1600 | 0.000             | 0.004           | 0.1698                                |
| 1700 | 0.000             | 0.004           | 0.1502                                |
| 1800 | 0.001             | 0.003           | 0.1164                                |
| 1900 | 0.000             | 0.004           | 0.1661                                |
| 2000 | 0.001             | 0.003           | 0.1382                                |
| 2100 | 0.000             | 0.003           | 0.1249                                |
| 2200 | 0.000             | 0.003           | 0.1154                                |
| 2300 | 0.000             | 0.003           | 0.1222                                |

Run cancelled at step ~2355. Peak AUPRC: 0.2860 at step 600.

**Observations**:
- Much more stable than lr=1e-3 — loss decreases monotonically, no rebound
- AUPRC peaks early (0.286 at step 600) then gradually degrades back to baseline by step 2k+
- Loss keeps decreasing while AUPRC gets worse — classic sign that the contrastive objective is not aligned with the downstream task
- The model gets better at matching dropout-augmented views but the learned representations lose variant-discriminative information

**Next steps**:
- Add MLP projection head — contrastive learning literature (SimCLR, SimCSE) shows a projection head lets the encoder preserve richer representations while the head learns the contrastive-specific mapping

---

## 2026-03-13: SimCSE Small Transformer — lr=1e-4 + MLP projection head

**Hypothesis**: Adding an MLP projection head (512 → GELU → 512) after mean pooling will decouple the encoder representations from the contrastive objective, allowing AUPRC to improve for longer before degrading.

**Changes from previous run**:
- Added 2-layer MLP projection head (512 → GELU → 512, +0.5M params)

**Config**: `configs/experiment/simcse_small_lr1e4_mlp.yaml`

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_small_lr1e4_mlp
```

**W&B**: https://wandb.ai/gonzalobenegas/glm-experiments/runs/5cata0gf

**Results**:

| Step | train/simcse_loss | val/simcse_loss | val/traitgym_mendelian_promoter_auprc |
|------|-------------------|-----------------|---------------------------------------|
| 100  | 1.370             | 1.470           | 0.1556                                |
| 200  | 0.321             | 1.470           | 0.1406                                |
| 300  | 0.083             | 0.335           | 0.1679                                |
| 400  | 0.061             | 0.151           | 0.1429                                |
| 500  | 0.019             | 0.082           | 0.1852                                |
| 600  | 0.013             | 0.039           | 0.2417                                |
| 700  | 0.005             | 0.019           | 0.1832                                |
| 800  | 0.007             | 0.011           | 0.2584                                |
| 900  | 0.006             | 0.011           | 0.2088                                |
| 1000 | 0.002             | 0.007           | 0.1748                                |
| 1100 | 0.003             | 0.007           | 0.1904                                |
| 1200 | 0.002             | 0.007           | 0.2076                                |
| 1300 | 0.003             | 0.006           | 0.1619                                |
| 1400 | 0.001             | 0.005           | 0.2150                                |
| 1500 | 0.000             | 0.004           | 0.1611                                |
| 1600 | 0.001             | 0.004           | 0.1588                                |
| 1700 | 0.001             | 0.006           | 0.1296                                |
| 1800 | 0.000             | 0.004           | 0.1546                                |
| 1900 | 0.001             | 0.004           | 0.1057                                |
| 2000 | 0.001             | 0.004           | 0.1206                                |

Completed at step 2000. Peak AUPRC: 0.2584 at step 800.

**Observations**:
- MLP head doesn't clearly help — peak AUPRC 0.258 vs 0.286 without MLP
- Similar pattern: AUPRC peaks early then degrades, though degradation is slightly slower
- Loss curves similar to no-MLP run

**Post-hoc analysis** (see `experiments/visualize_scores.py`, `experiments/visualize_embeddings.py`):

UMAP of validation embeddings colored by repeat fraction and GC content (`experiments/figures/umap_comparison.png`). Score histograms of cosine similarity (ref vs alt) for TraitGym Mendelian promoter variants (`experiments/figures/score_histograms.png`), and benign variants stratified by repeat vs non-repeat (`experiments/figures/score_histograms_by_region.png`).

Results:
- All cosine similarities are in the range 0.998–1.000 (mean pooling over 512 positions dilutes the single-nucleotide change)
- At step 800 (peak AUPRC): benign mean=0.9988, pathogenic mean=0.9983 — a gap of ~0.0005
- At step 2000: gap shrinks to ~0.0002 (0.9982 vs 0.9980)
- Benign variants in repeat regions have slightly higher similarity (0.9990) than non-repeat (0.9988) at step 800

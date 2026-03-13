# Experiment Log

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
- _TBD_

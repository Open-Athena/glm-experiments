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
- log_every_n_steps: 1000 → 100
- num_workers: 8 → 4

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_transformer_small \
  trainer.devices=1 trainer.strategy=auto \
  data.dataset_name=songlab/gpn-animal-promoter-dataset \
  data.batch_size=256 \
  data.num_workers=4 \
  model.optimizer.lr=0.0001 \
  trainer.log_every_n_steps=100
```

**Results**: _pending_

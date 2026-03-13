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

Also tested position-level VEP scoring (comparing hidden states at variant position instead of mean-pooled embeddings, see `experiments/position_level_vep.py`):
- Much wider score range (cosine sim ~0.73) but worse discrimination (AUPRC 0.131 vs 0.258)
- Both benign and pathogenic see similar spread at the variant position — noise, not signal

---

## 2026-03-13: SimCSE Small Transformer — dropout=0.2

**Hypothesis**: Higher dropout (0.2 vs 0.1) creates more diverse positive pairs in SimCSE, forcing the model to learn more abstract features that may be more discriminative for VEP.

**Changes from lr1e4+MLP run**:
- Dropout increased from 0.1 to 0.2 (both encoder internal and post-encoder)

**Config**: `configs/experiment/simcse_small_dropout02.yaml`

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_small_dropout02
```

**W&B**: https://wandb.ai/gonzalobenegas/glm-experiments/runs/lodnrw1a

**Results**:

| Step | train/simcse_loss | val/simcse_loss | val/traitgym_mendelian_promoter_auprc |
|------|-------------------|-----------------|---------------------------------------|
| 100  | 1.470             | 1.560           | 0.1573                                |
| 200  | 0.417             | 0.522           | 0.1330                                |
| 300  | 0.124             | 0.150           | 0.1576                                |
| 400  | 0.084             | 0.105           | 0.1616                                |
| 500  | 0.041             | 0.063           | 0.1537                                |
| 600  | 0.025             | 0.040           | 0.2317                                |
| 700  | 0.004             | 0.016           | 0.1877                                |
| 800  | 0.010             | 0.012           | 0.1860                                |
| 900  | 0.005             | 0.010           | 0.1920                                |
| 1000 | 0.003             | 0.011           | 0.1879                                |
| 1100 | 0.002             | 0.009           | 0.1965                                |
| 1200 | 0.004             | 0.008           | 0.2405                                |
| 1300 | 0.003             | 0.009           | 0.2413                                |
| 1400 | 0.001             | 0.006           | 0.1586                                |
| 1500 | 0.001             | 0.005           | 0.1559                                |
| 1600 | 0.001             | 0.005           | 0.1598                                |
| 1700 | 0.001             | 0.005           | 0.2076                                |
| 1800 | 0.001             | 0.005           | 0.1429                                |
| 1900 | 0.001             | 0.004           | 0.1532                                |
| 2000 | 0.001             | 0.004           | 0.2031                                |

Completed at step 2000. Peak AUPRC: 0.2413 at step 1300.

**Observations**:
- Worse than dropout=0.1 — peak AUPRC 0.241 vs 0.258
- AUPRC peaks later (step 1300 vs 800) and is more volatile
- Loss converges slightly slower as expected with higher dropout

---

## 2026-03-13: SimCSE Small Transformer — temperature=0.1

**Hypothesis**: Higher temperature (0.1 vs 0.05) softens the contrastive distribution, encouraging more uniform spread of embeddings on the hypersphere. More spread = more room for benign/pathogenic separation.

**Changes from lr1e4+MLP run**:
- Temperature increased from 0.05 to 0.1

**Config**: `configs/experiment/simcse_small_temp01.yaml`

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_small_temp01
```

**W&B**: https://wandb.ai/gonzalobenegas/glm-experiments/runs/asdvvmr5

**Results**:

| Step | train/simcse_loss | val/simcse_loss | val/traitgym_mendelian_promoter_auprc |
|------|-------------------|-----------------|---------------------------------------|
| 100  | 2.220             | 2.320           | 0.1546                                |
| 200  | 1.160             | 1.230           | 0.1527                                |
| 300  | 0.493             | 0.659           | 0.1583                                |
| 400  | 0.319             | 0.382           | 0.1521                                |
| 500  | 0.212             | 0.318           | 0.2074                                |
| 600  | 0.152             | 0.208           | 0.2054                                |
| 700  | 0.101             | 0.142           | 0.2016                                |
| 800  | 0.100             | 0.106           | 0.2133                                |
| 900  | 0.062             | 0.075           | 0.2095                                |
| 1000 | 0.038             | 0.078           | 0.1827                                |
| 1100 | 0.036             | 0.048           | 0.2534                                |
| 1200 | 0.030             | 0.045           | 0.1952                                |
| 1300 | 0.031             | 0.040           | 0.2432                                |
| 1400 | 0.024             | 0.032           | 0.2093                                |
| 1500 | 0.027             | 0.033           | 0.1937                                |
| 1600 | 0.023             | 0.030           | 0.2010                                |
| 1700 | 0.023             | 0.034           | 0.1786                                |
| 1800 | 0.022             | 0.026           | 0.1888                                |
| 1900 | 0.021             | 0.025           | 0.2505                                |
| 2000 | 0.022             | 0.027           | 0.1695                                |

Completed at step 2000. Peak AUPRC: 0.2534 at step 1100.

**Observations**:
- Similar to baseline — peak AUPRC 0.253 vs 0.258
- Loss converges much slower (val=0.027 at step 2000 vs 0.004 with temp=0.05)
- AUPRC volatile throughout, no clear degradation pattern

---

## 2026-03-13: SimCSE CNN Pyramid Tiny — bs=4096

**Hypothesis**: A CNN pyramid encoder (inspired by AlphaGenome's sequence encoder) with progressive downsampling may capture TF binding site motifs hierarchically. Tiny model (3.5M params) allows batch size 4096 (16x more in-batch negatives).

**Architecture**: CNN pyramid with 9 stages of max-pool(2), d_model=128, channel_growth=16 per stage (128→256 channels), kernel_size=5, initial wide conv kernel=15. See `glm_experiments/models/components/cnn_pyramid.py`.

**Config**: `configs/experiment/simcse_cnn_tiny_bs4096.yaml`

**Command**:
```bash
uv run python glm_experiments/train.py experiment=simcse_cnn_tiny_bs4096
```

**W&B**: https://wandb.ai/gonzalobenegas/glm-experiments/runs/6vu4o5rj

**Results**:

| Step | train/simcse_loss | val/simcse_loss | val/traitgym_mendelian_promoter_auprc |
|------|-------------------|-----------------|---------------------------------------|
| 100  | 7.170             | 4.990           | 0.1375                                |
| 200  | 1.100             | 0.023           | 0.1742                                |
| 300  | 0.072             | 0.011           | 0.1547                                |
| 400  | 0.027             | 0.011           | 0.1445                                |
| 500  | —                 | —               | 0.1425                                |
| 600  | —                 | —               | 0.1460                                |
| 700  | 0.006             | 0.011           | 0.1530                                |
| 800  | —                 | —               | 0.1515                                |

Cancelled at step ~800. Peak AUPRC: 0.1742 at step 200.

**Observations**:
- Much worse than transformer baseline (peak 0.174 vs 0.258)
- Loss collapses very fast (near zero by step 200) but AUPRC stays flat ~0.15
- CNN pyramid pools down to 1 position — may lose too much spatial information for VEP

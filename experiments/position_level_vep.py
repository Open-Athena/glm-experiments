"""Position-level VEP scoring using SimCSE encoder hidden states.

Instead of comparing mean-pooled sequence embeddings, compares the encoder's
hidden state at the variant position (256) between ref and alt sequences.
This avoids diluting the single-nucleotide signal across 512 positions.

Usage:
    python experiments/position_level_vep.py \
        --ckpt_path logs/.../800.ckpt

    # Compare with mean-pooled baseline
    python experiments/position_level_vep.py \
        --ckpt_path logs/.../800.ckpt --compare
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from sklearn.metrics import average_precision_score

VARIANT_POS = 256  # Center of 512-bp window


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--experiment", type=str, default="simcse_small_lr1e4_mlp",
        help="Experiment config for model instantiation",
    )
    parser.add_argument(
        "--eval_cache", type=str,
        default="data/evals_cache/songlab_TraitGym_mendelian_traits_test_traitgym_promoter_window512_simcse",
    )
    parser.add_argument("--compare", action="store_true", help="Also compute mean-pooled baseline")
    return parser.parse_args()


def load_model(experiment, ckpt_path):
    """Instantiate model and load checkpoint."""
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent.parent / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])

    model = instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.cuda()
    return model


def get_hidden_states(model, input_ids):
    """Get encoder hidden states before mean pooling.

    Returns shape (batch, seq_len, hidden_dim).
    """
    input_ids = input_ids.long()
    x = model.net.embedder(input_ids)
    x = model.net.encoder(x)
    x = model.net.layer_norm(x)
    return x


def compute_position_scores(model, dataset, batch_size=128):
    """Compute cosine similarity at the variant position only."""
    scores = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            input_ids = batch["input_ids"].cuda()  # (B, 2, L)

            ref_hidden = get_hidden_states(model, input_ids[:, 0])  # (B, L, D)
            alt_hidden = get_hidden_states(model, input_ids[:, 1])  # (B, L, D)

            ref_at_pos = ref_hidden[:, VARIANT_POS, :]  # (B, D)
            alt_at_pos = alt_hidden[:, VARIANT_POS, :]  # (B, D)

            cos_sim = F.cosine_similarity(ref_at_pos, alt_at_pos)
            scores.append(cos_sim.cpu().numpy())
    return np.concatenate(scores)


def compute_meanpool_scores(model, dataset, batch_size=128):
    """Compute cosine similarity with mean-pooled embeddings (baseline)."""
    scores = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            input_ids = batch["input_ids"].cuda()
            ref_emb = model.net.get_embeddings(input_ids[:, 0])
            alt_emb = model.net.get_embeddings(input_ids[:, 1])
            cos_sim = F.cosine_similarity(ref_emb, alt_emb)
            scores.append(cos_sim.cpu().numpy())
    return np.concatenate(scores)


def main():
    args = parse_args()

    print(f"Loading model from {args.ckpt_path}...")
    model = load_model(args.experiment, args.ckpt_path)

    print(f"Loading eval dataset...")
    dataset = load_from_disk(args.eval_cache)
    dataset.set_format("torch")
    labels = np.array(dataset["label"])
    print(f"  {len(dataset)} variants ({labels.sum()} pathogenic, {len(labels) - labels.sum()} benign)")

    # Position-level scoring
    print("\nComputing position-level scores...")
    pos_scores = compute_position_scores(model, dataset)
    # Negate for AUPRC (lower similarity = more pathogenic)
    pos_auprc = average_precision_score(labels, -pos_scores)

    print(f"\n=== Position-level (pos={VARIANT_POS}) ===")
    print(f"  Cosine sim — benign mean:     {pos_scores[~labels.astype(bool)].mean():.6f}  "
          f"std: {pos_scores[~labels.astype(bool)].std():.6f}")
    print(f"  Cosine sim — pathogenic mean:  {pos_scores[labels.astype(bool)].mean():.6f}  "
          f"std: {pos_scores[labels.astype(bool)].std():.6f}")
    print(f"  AUPRC: {pos_auprc:.4f}")

    if args.compare:
        print("\nComputing mean-pooled scores (baseline)...")
        mp_scores = compute_meanpool_scores(model, dataset)
        mp_auprc = average_precision_score(labels, -mp_scores)

        print(f"\n=== Mean-pooled (baseline) ===")
        print(f"  Cosine sim — benign mean:     {mp_scores[~labels.astype(bool)].mean():.6f}  "
              f"std: {mp_scores[~labels.astype(bool)].std():.6f}")
        print(f"  Cosine sim — pathogenic mean:  {mp_scores[labels.astype(bool)].mean():.6f}  "
              f"std: {mp_scores[labels.astype(bool)].std():.6f}")
        print(f"  AUPRC: {mp_auprc:.4f}")

        print(f"\n=== Comparison ===")
        print(f"  Position-level AUPRC: {pos_auprc:.4f}")
        print(f"  Mean-pooled AUPRC:    {mp_auprc:.4f}")
        print(f"  Delta:                {pos_auprc - mp_auprc:+.4f}")


if __name__ == "__main__":
    main()

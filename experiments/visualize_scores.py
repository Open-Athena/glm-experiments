"""Visualize distribution of SimCSE variant effect scores by label.

Loads the TraitGym Mendelian promoter eval dataset, computes cosine similarity
scores between ref/alt embeddings for each variant, and plots score distributions
separated by label (pathogenic vs benign).

Supports comparing multiple checkpoints side by side. Caches scores to disk.
Can stratify by repeat vs non-repeat regions.

Usage:
    # Random init only
    python experiments/visualize_scores.py

    # Compare checkpoints
    python experiments/visualize_scores.py \
        --ckpt_paths none logs/.../800.ckpt logs/.../2000.ckpt \
        --labels "Random init" "Step 800 (peak)" "Step 2000 (end)"

    # Force recompute
    python experiments/visualize_scores.py --no_cache
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_paths",
        type=str,
        nargs="+",
        default=None,
        help='Checkpoint path(s). Use "none" for random init.',
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each checkpoint",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="simcse_small_lr1e4_mlp",
        help="Experiment config for model instantiation",
    )
    parser.add_argument(
        "--eval_cache",
        type=str,
        default="data/evals_cache/songlab_TraitGym_mendelian_traits_test_traitgym_promoter_window512_simcse",
        help="Path to cached eval dataset",
    )
    parser.add_argument("--output_dir", type=str, default="experiments/figures", help="Output directory")
    parser.add_argument("--no_cache", action="store_true", help="Force recompute scores")
    return parser.parse_args()


def _cache_key(ckpt_path, experiment):
    ckpt_str = "random" if ckpt_path is None else str(Path(ckpt_path).resolve())
    raw = f"{ckpt_str}|{experiment}|scores"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_model(experiment, ckpt_path=None):
    """Instantiate model from config, optionally loading checkpoint weights."""
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent.parent / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])

    model = instantiate(cfg.model)

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"  Loaded checkpoint from {ckpt_path}")
    else:
        print("  Using random initialization")

    model.eval()
    model.cuda()
    return model


def compute_scores(model, dataset, batch_size=128):
    """Compute cosine similarity scores between ref and alt embeddings."""
    scores = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            input_ids = batch["input_ids"].cuda()  # (B, 2, L)
            ref_emb = model.net.get_embeddings(input_ids[:, 0])
            alt_emb = model.net.get_embeddings(input_ids[:, 1])
            cos_sim = F.cosine_similarity(ref_emb, alt_emb)  # (B,)
            scores.append(cos_sim.cpu().numpy())
    return np.concatenate(scores)


def get_or_compute_scores(ckpt_path, experiment, dataset, cache_dir, use_cache):
    """Load scores from cache or compute them."""
    key = _cache_key(ckpt_path, experiment)
    cache_path = cache_dir / f"scores_{key}.npy"

    if use_cache and cache_path.exists():
        print(f"  Loading cached scores from {cache_path}")
        return np.load(cache_path)

    model = load_model(experiment, ckpt_path)
    print("  Computing scores...")
    scores = compute_scores(model, dataset)
    np.save(cache_path, scores)
    print(f"  Cached scores to {cache_path}")
    del model
    torch.cuda.empty_cache()
    return scores


def get_repeat_status(cache_dir, use_cache):
    """Get repeat/non-repeat status for each variant in the filtered eval set.

    Loads the raw TraitGym dataset, applies the same filter as the eval pipeline,
    then checks the genome at each variant position.
    """
    cache_path = cache_dir / "repeat_status.npy"
    if use_cache and cache_path.exists():
        print("  Loading cached repeat status")
        return np.load(cache_path)

    from biofoundation.data import Genome
    from datasets import load_dataset

    from glm_experiments.data.evals import filter_traitgym_promoter

    print("  Loading filtered TraitGym variants...")
    ds = load_dataset("songlab/TraitGym", "mendelian_traits", split="test")
    ds = filter_traitgym_promoter(ds)

    print("  Checking repeat status from genome...")
    genome = Genome("data/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz")

    is_repeat = np.array([
        genome(v["chrom"], v["pos"] - 1, v["pos"]).islower() for v in ds
    ])

    np.save(cache_path, is_repeat)
    print(f"  Repeat: {is_repeat.sum()}, Non-repeat: {(~is_repeat).sum()}")
    return is_repeat


def build_dataframe(all_scores, variant_labels, ckpt_labels, is_repeat=None):
    """Build a tidy dataframe for seaborn plotting."""
    rows = []
    for scores, ckpt_label in zip(all_scores, ckpt_labels):
        for j, (score, label) in enumerate(zip(scores, variant_labels)):
            row = {
                "checkpoint": ckpt_label,
                "score": float(score),
                "label": "Pathogenic" if label else "Benign",
            }
            if is_repeat is not None:
                row["region"] = "Repeat" if is_repeat[j] else "Non-repeat"
            rows.append(row)
    return pd.DataFrame(rows)


def plot_score_histograms(df, ckpt_labels, output_path):
    """Plot score distributions as overlapping histograms, one row per checkpoint."""
    n_ckpts = len(ckpt_labels)

    fig, axes = plt.subplots(
        n_ckpts, 1, figsize=(8, 3.5 * n_ckpts), sharex=True, squeeze=False
    )

    for i, ckpt_label in enumerate(ckpt_labels):
        ax = axes[i, 0]
        subset = df[df["checkpoint"] == ckpt_label]

        sns.histplot(
            data=subset,
            x="score",
            hue="label",
            hue_order=["Benign", "Pathogenic"],
            stat="density",
            common_norm=False,
            kde=True,
            alpha=0.4,
            ax=ax,
        )
        ax.set_title(ckpt_label)
        ax.set_ylabel("Density")
        if i < n_ckpts - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Cosine similarity (ref vs alt)")

    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_score_histograms_by_region(df, ckpt_labels, output_path):
    """Plot benign variant score distributions with hue=repeat status."""
    n_ckpts = len(ckpt_labels)
    benign = df[df["label"] == "Benign"]

    fig, axes = plt.subplots(
        n_ckpts, 1, figsize=(8, 3.5 * n_ckpts), sharex=True, squeeze=False
    )

    for i, ckpt_label in enumerate(ckpt_labels):
        ax = axes[i, 0]
        subset = benign[benign["checkpoint"] == ckpt_label]

        sns.histplot(
            data=subset,
            x="score",
            hue="region",
            hue_order=["Non-repeat", "Repeat"],
            stat="density",
            common_norm=False,
            kde=True,
            alpha=0.4,
            ax=ax,
        )
        ax.set_title(ckpt_label)
        ax.set_ylabel("Density")
        if i < n_ckpts - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Cosine similarity (ref vs alt)")

    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    # Resolve checkpoint paths
    if args.ckpt_paths is None:
        ckpt_paths = [None]
    else:
        ckpt_paths = [None if p.lower() == "none" else p for p in args.ckpt_paths]

    # Resolve labels
    if args.labels is not None:
        if len(args.labels) != len(ckpt_paths):
            raise ValueError(
                f"--labels ({len(args.labels)}) must match --ckpt_paths ({len(ckpt_paths)})"
            )
        labels = args.labels
    else:
        labels = [
            "Random init" if p is None else Path(p).stem for p in ckpt_paths
        ]

    # Load eval dataset
    print(f"Loading eval dataset from {args.eval_cache}...")
    dataset = load_from_disk(args.eval_cache)
    dataset.set_format("torch")
    variant_labels = np.array(dataset["label"])
    print(f"  {len(dataset)} variants ({variant_labels.sum()} pathogenic, "
          f"{len(variant_labels) - variant_labels.sum()} benign)")

    # Get repeat status
    print("Loading repeat status...")
    is_repeat = get_repeat_status(cache_dir, use_cache)

    # Compute scores for each checkpoint
    all_scores = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[{i + 1}/{len(ckpt_paths)}] {labels[i]}:")
        scores = get_or_compute_scores(ckpt_path, args.experiment, dataset, cache_dir, use_cache)
        all_scores.append(scores)

    # Build tidy dataframe
    df = build_dataframe(all_scores, variant_labels, labels, is_repeat)

    # Plot both views
    plot_score_histograms(df, labels, output_dir / "score_histograms.png")
    plot_score_histograms_by_region(df, labels, output_dir / "score_histograms_by_region.png")


if __name__ == "__main__":
    main()

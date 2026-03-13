"""Visualize SimCSE embedding space colored by sequence properties.

Samples sequences from the validation set, computes embeddings with trained
(or random) SimCSE models, projects to 2D with UMAP, and colors by:
- Fraction of soft-masked (repetitive) positions
- GC content

Supports comparing multiple checkpoints side by side (e.g., random init,
peak AUPRC, end of training). Each checkpoint gets its own UMAP projection
so the embedding structure is faithfully represented.

Caches embeddings and UMAP projections to disk so re-runs (e.g., to tweak
plotting) skip the expensive computation.

Usage:
    # Random init (no checkpoint)
    python experiments/visualize_embeddings.py

    # Single checkpoint
    python experiments/visualize_embeddings.py --ckpt_paths logs/.../step=600.ckpt

    # Compare multiple checkpoints (independent UMAP per checkpoint)
    python experiments/visualize_embeddings.py \\
        --ckpt_paths none logs/.../step=600.ckpt logs/.../step=2000.ckpt \\
        --labels "Random init" "Step 600 (peak)" "Step 2000 (end)"

    # Quick test with fewer samples
    python experiments/visualize_embeddings.py --n_samples 200

    # Force recompute (ignore cache)
    python experiments/visualize_embeddings.py --no_cache
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_paths",
        type=str,
        nargs="+",
        default=None,
        help='Checkpoint path(s). Use "none" for random init. If omitted, runs random init only.',
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each checkpoint (must match length of --ckpt_paths)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="simcse_small_lr1e4_mlp",
        help="Experiment config to use for model instantiation",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None, help="Number of validation samples (default: all)"
    )
    parser.add_argument("--output_dir", type=str, default="experiments/figures", help="Output directory")
    parser.add_argument("--no_cache", action="store_true", help="Force recompute, ignore cached data")
    return parser.parse_args()


def _cache_key(ckpt_path, experiment, n_samples):
    """Generate a cache key from checkpoint path, experiment, and sample count."""
    ckpt_str = "random" if ckpt_path is None else str(Path(ckpt_path).resolve())
    raw = f"{ckpt_str}|{experiment}|{n_samples}"
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
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Using random initialization")

    model.eval()
    model.cuda()
    return model


def load_validation_data(n_samples):
    """Load validation sequences with soft mask info."""
    tokenizer = AutoTokenizer.from_pretrained("gonzalobenegas/tokenizer-dna-mlm")
    dataset = load_dataset("songlab/gpn-animal-promoter-dataset", split="validation", streaming=True)

    input_ids_list = []
    soft_masked_fractions = []
    gc_contents = []

    for i, example in enumerate(dataset):
        if n_samples is not None and i >= n_samples:
            break

        seq = example["seq"]

        ids = tokenizer(
            seq,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]
        input_ids_list.append(ids)

        lowercase_frac = sum(1 for c in seq if c.islower()) / len(seq)
        soft_masked_fractions.append(lowercase_frac)

        seq_upper = seq.upper()
        gc = sum(1 for c in seq_upper if c in "GC") / len(seq_upper)
        gc_contents.append(gc)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "soft_masked_fraction": np.array(soft_masked_fractions),
        "gc_content": np.array(gc_contents),
    }


def compute_embeddings(model, input_ids, batch_size=128):
    """Compute embeddings in batches."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch = input_ids[i : i + batch_size].cuda()
            emb = model.net.get_embeddings(batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def get_or_compute_embeddings(ckpt_path, experiment, data, cache_dir, use_cache):
    """Load embeddings from cache or compute them."""
    key = _cache_key(ckpt_path, experiment, len(data["input_ids"]))
    emb_path = cache_dir / f"emb_{key}.npy"
    umap_path = cache_dir / f"umap_{key}.npy"

    if use_cache and emb_path.exists():
        print(f"  Loading cached embeddings from {emb_path}")
        emb = np.load(emb_path)
    else:
        model = load_model(experiment, ckpt_path)
        print("  Computing embeddings...")
        emb = compute_embeddings(model, data["input_ids"])
        np.save(emb_path, emb)
        print(f"  Cached embeddings to {emb_path}")
        del model
        torch.cuda.empty_cache()

    if use_cache and umap_path.exists():
        print(f"  Loading cached UMAP from {umap_path}")
        emb_2d = np.load(umap_path)
    else:
        print("  Running UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        emb_2d = reducer.fit_transform(emb)
        np.save(umap_path, emb_2d)
        print(f"  Cached UMAP to {umap_path}")

    print(f"  Embeddings shape: {emb.shape}")
    return emb_2d


def plot_single(embeddings_2d, colors, title, label, output_path):
    """Create a single UMAP scatter plot colored by a property."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        cmap="viridis",
        s=5,
        alpha=0.6,
    )
    cbar = plt.colorbar(scatter, ax=ax, label=label)
    cbar.solids.set_rasterized(True)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_comparison_grid(all_embeddings_2d, labels, data, output_path):
    """Create a grid of UMAP plots: rows = checkpoints, columns = properties.

    Each checkpoint has its own independent UMAP projection so the embedding
    structure is faithfully represented. Color scales are shared across rows
    within each property column.
    """
    n_ckpts = len(all_embeddings_2d)
    properties = [
        ("soft_masked_fraction", "Soft-masked fraction"),
        ("gc_content", "GC content"),
    ]
    n_props = len(properties)

    fig, axes = plt.subplots(
        n_ckpts, n_props, figsize=(7 * n_props, 5.5 * n_ckpts), squeeze=False
    )

    for col, (prop_key, prop_label) in enumerate(properties):
        colors = data[prop_key]
        vmin, vmax = colors.min(), colors.max()

        for row, (emb_2d, label) in enumerate(zip(all_embeddings_2d, labels)):
            ax = axes[row, col]
            scatter = ax.scatter(
                emb_2d[:, 0],
                emb_2d[:, 1],
                c=colors,
                cmap="viridis",
                s=5,
                alpha=0.6,
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(scatter, ax=ax, label=prop_label)
            cbar.solids.set_rasterized(True)
            ax.set_title(f"{label} — {prop_label}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

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

    # Resolve checkpoint paths: None means random init
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
        labels = []
        for p in ckpt_paths:
            if p is None:
                labels.append("Random init")
            else:
                labels.append(Path(p).stem)

    # Load data once (shared across all checkpoints)
    n_desc = args.n_samples if args.n_samples is not None else "all"
    print(f"Loading {n_desc} validation samples...")
    data = load_validation_data(args.n_samples)

    # Compute embeddings + UMAP for each checkpoint (with caching)
    all_embeddings_2d = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[{i + 1}/{len(ckpt_paths)}] {labels[i]}:")
        emb_2d = get_or_compute_embeddings(ckpt_path, args.experiment, data, cache_dir, use_cache)
        all_embeddings_2d.append(emb_2d)

    if len(ckpt_paths) == 1:
        suffix = labels[0].replace(" ", "_").lower()

        plot_single(
            all_embeddings_2d[0],
            data["soft_masked_fraction"],
            f"SimCSE Embeddings — Repeat Fraction ({labels[0]})",
            "Soft-masked fraction",
            output_dir / f"umap_repeat_fraction_{suffix}.png",
        )
        plot_single(
            all_embeddings_2d[0],
            data["gc_content"],
            f"SimCSE Embeddings — GC Content ({labels[0]})",
            "GC content",
            output_dir / f"umap_gc_content_{suffix}.png",
        )
    else:
        plot_comparison_grid(
            all_embeddings_2d,
            labels,
            data,
            output_dir / "umap_comparison.png",
        )

    # Print summary stats
    print("\n--- Summary ---")
    print(f"Samples: {len(data['input_ids'])}")
    print(
        f"Repeat fraction: mean={data['soft_masked_fraction'].mean():.3f}, "
        f"std={data['soft_masked_fraction'].std():.3f}"
    )
    print(
        f"GC content: mean={data['gc_content'].mean():.3f}, "
        f"std={data['gc_content'].std():.3f}"
    )


if __name__ == "__main__":
    main()

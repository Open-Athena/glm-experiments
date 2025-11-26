"""TraitGym Mendelian Promoter variant dataset loading for evaluation.

This module provides functions to load and transform the TraitGym dataset
(filtered to non-exonic proximal promoter variants) for variant effect
prediction evaluation during training.
"""

import logging
import urllib.request
from functools import partial
from pathlib import Path

import pandas as pd
from biofoundation.data import Genome, transform_llr_clm, transform_llr_mlm
from biofoundation.model.base import Tokenizer
from datasets import Dataset, load_dataset

log = logging.getLogger(__name__)


def download_genome(url: str, path: str | Path) -> Path:
    """Download reference genome if not already present.

    Args:
        url: URL to download genome from (e.g., Ensembl FTP)
        path: Local path to save the genome file

    Returns:
        Path to the downloaded genome file
    """
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)  # nosec B310
    return path


def load_traitgym_mendelian_promoter_dataset(
    tokenizer: Tokenizer,
    genome_path: str | Path,
    dataset_name: str = "songlab/TraitGym",
    dataset_config: str = "mendelian_traits",
    window_size: int = 512,
    cache_dir: str | Path = "data/traitgym_cache",
    objective: str = "mlm",
) -> Dataset:
    """Load and transform TraitGym Mendelian Promoter dataset for variant effect prediction.

    Loads the TraitGym dataset from HuggingFace, filters to non-exonic proximal
    promoter variants, and applies the appropriate transform (transform_llr_mlm or
    transform_llr_clm) based on the objective.

    The Genome is only loaded if the transformed dataset is not cached.

    Args:
        tokenizer: Tokenizer implementing the biofoundation Tokenizer protocol
        genome_path: Path to the reference genome file
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., "mendelian_traits")
        window_size: Size of the window around variants (must be even)
        cache_dir: Directory to cache transformed dataset
        objective: Training objective ("mlm" or "clm") - determines transform function

    Returns:
        Transformed dataset with columns: input_ids, pos, ref, alt, label
    """
    from datasets import load_from_disk

    # Create cache path based on config and objective (different transforms need separate caches)
    cache_path = (
        Path(cache_dir) / f"{dataset_config}_mendelian_promoter_window{window_size}_{objective}"
    )

    # Check if cached transformed dataset exists
    if cache_path.exists():
        log.info(f"Loading cached TraitGym Mendelian Promoter dataset from {cache_path}")
        dataset = load_from_disk(str(cache_path))
        dataset.set_format(type="torch")
        return dataset

    # Not cached - need to transform with genome
    log.info(f"Loading TraitGym dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="test")  # nosec B615

    # Filter to non-exonic proximal promoter variants
    log.info("Filtering to non-exonic proximal promoter variants...")
    subset_url = (
        "https://huggingface.co/datasets/songlab/TraitGym/resolve/main/"
        "mendelian_traits_matched_9/subset/nonexonic_AND_proximal.parquet"
    )
    V = dataset.to_pandas()
    subset = pd.read_parquet(subset_url)
    V = V.merge(subset, on=["chrom", "pos", "ref", "alt"], how="inner")
    log.info(f"Filtered dataset size: {len(V)} variants (from {len(dataset)})")
    dataset = Dataset.from_pandas(V, preserve_index=False)

    log.info(f"Loading reference genome from {genome_path} (this may take a minute)...")
    genome = Genome(genome_path)

    # We need to keep the label column for evaluation
    original_columns = dataset.column_names

    # Select transform function based on objective
    if objective == "mlm":
        transform_func = transform_llr_mlm
    elif objective == "clm":
        transform_func = transform_llr_clm
    else:
        raise ValueError(f"Unknown objective: {objective}. Must be 'mlm' or 'clm'.")

    transform_fn = partial(
        transform_func,
        tokenizer=tokenizer,
        genome=genome,
        window_size=window_size,
    )

    # Transform the dataset
    log.info("Transforming TraitGym Mendelian Promoter dataset...")
    dataset = dataset.map(
        transform_fn,
        remove_columns=[c for c in original_columns if c != "label"],
    )

    # Save to cache
    log.info(f"Saving transformed dataset to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))

    # Set format to PyTorch tensors for proper DataLoader collation
    dataset.set_format(type="torch")

    return dataset

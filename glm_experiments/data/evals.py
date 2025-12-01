"""Evaluation dataset loading for variant effect prediction.

This module provides functions to load and transform evaluation datasets
(TraitGym, PlantCAD, etc.) for variant effect prediction during training.

Supports configurable dataset-specific filtering via a registry pattern.
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


def filter_traitgym_promoter(dataset: Dataset) -> Dataset:
    """Filter TraitGym dataset to non-exonic proximal promoter variants.

    Args:
        dataset: TraitGym dataset from HuggingFace

    Returns:
        Filtered dataset containing only non-exonic proximal promoter variants
    """
    log.info("Filtering to non-exonic proximal promoter variants...")
    subset_url = (
        "https://huggingface.co/datasets/songlab/TraitGym/resolve/main/"
        "mendelian_traits_matched_9/subset/nonexonic_AND_proximal.parquet"
    )
    V = dataset.to_pandas()
    subset = pd.read_parquet(subset_url)
    V = V.merge(subset, on=["chrom", "pos", "ref", "alt"], how="inner")
    log.info(f"Filtered dataset size: {len(V)} variants (from {len(dataset)})")
    return Dataset.from_pandas(V, preserve_index=False)


# Registry mapping filter names to filter functions
EVAL_FILTERS = {
    "traitgym_promoter": filter_traitgym_promoter,
    "none": lambda dataset: dataset,  # No-op filter
}


def download_genome(url: str, data_dir: str | Path = "data") -> Path:
    """Download reference genome if not already present.

    Path is auto-derived from URL basename (e.g., genome.fa.gz).

    Args:
        url: URL to download genome from (e.g., Ensembl FTP)
        data_dir: Directory to save genome (default: "data")

    Returns:
        Path to the downloaded genome file
    """
    path = Path(data_dir) / Path(url).name
    if path.exists():
        log.info(f"Genome already exists at {path}, skipping download")
        return path

    log.info(f"Downloading genome from {url} to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)  # nosec B310
    log.info(f"Genome download complete: {path}")
    return path


def load_eval_dataset(
    tokenizer: Tokenizer,
    dataset_name: str,
    genome_url: str,
    filter_name: str = "none",
    dataset_config: str | None = None,
    window_size: int = 512,
    cache_dir: str | Path = "data/evals_cache",
    objective: str = "mlm",
    data_dir: str | Path = "data",
    label_column: str = "label",
) -> Dataset:
    """Load and transform evaluation dataset with optional filtering.

    Loads a variant dataset from HuggingFace, applies optional dataset-specific
    filtering, and transforms it using the appropriate objective (MLM or CLM).

    The Genome is only loaded if the transformed dataset is not cached.

    Args:
        tokenizer: Tokenizer implementing the biofoundation Tokenizer protocol
        dataset_name: HuggingFace dataset name (e.g., "songlab/TraitGym")
        genome_url: URL to reference genome (path auto-derived from basename)
        filter_name: Name of filter function in EVAL_FILTERS registry (default: "none")
        dataset_config: Dataset configuration (e.g., "mendelian_traits")
        window_size: Size of the window around variants (must be even)
        cache_dir: Directory to cache transformed dataset
        objective: Training objective ("mlm" or "clm") - determines transform function
        data_dir: Directory for genome downloads (default: "data")
        label_column: Name of the label column to preserve (default: "label")

    Returns:
        Transformed dataset with columns: input_ids, pos, ref, alt, {label_column}

    Raises:
        ValueError: If filter_name not in EVAL_FILTERS or objective not mlm/clm
    """
    from datasets import load_from_disk

    # Validate filter name
    if filter_name not in EVAL_FILTERS:
        raise ValueError(
            f"Unknown filter_name: {filter_name}. " f"Must be one of {list(EVAL_FILTERS.keys())}"
        )

    # Create cache path based on config, filter, and objective
    cache_name_parts = [dataset_name.replace("/", "_")]
    if dataset_config:
        cache_name_parts.append(dataset_config)
    cache_name_parts.extend([filter_name, f"window{window_size}", objective])
    cache_name = "_".join(cache_name_parts)
    cache_path = Path(cache_dir) / cache_name

    # Check if cached transformed dataset exists
    if cache_path.exists():
        log.info(f"Loading cached evaluation dataset from {cache_path}")
        dataset = load_from_disk(str(cache_path))
        dataset.set_format(type="torch")
        return dataset

    # Not cached - need to transform with genome
    log.info(f"Loading evaluation dataset: {dataset_name}")
    if dataset_config:
        log.info(f"  Dataset config: {dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="test")  # nosec B615

    # Apply dataset-specific filtering
    if filter_name != "none":
        log.info(f"  Applying filter: {filter_name}")
    dataset = EVAL_FILTERS[filter_name](dataset)

    # Download genome (auto-derives path from URL)
    genome_path = download_genome(genome_url, data_dir)
    log.info(f"Loading reference genome from {genome_path} (this may take a minute)...")
    genome = Genome(genome_path)

    # Keep original columns for evaluation
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
    log.info("Transforming evaluation dataset...")
    dataset = dataset.map(
        transform_fn,
        remove_columns=[c for c in original_columns if c != label_column],
    )

    # Save to cache
    log.info(f"Saving transformed dataset to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))

    # Set format to PyTorch tensors for proper DataLoader collation
    dataset.set_format(type="torch")

    return dataset


def load_traitgym_mendelian_promoter_dataset(
    tokenizer: Tokenizer,
    genome_path: str | Path,
    dataset_name: str = "songlab/TraitGym",
    dataset_config: str = "mendelian_traits",
    window_size: int = 512,
    cache_dir: str | Path = "data/traitgym_cache",
    objective: str = "mlm",
) -> Dataset:
    """Load TraitGym Mendelian Promoter dataset for variant effect prediction.

    DEPRECATED: Use load_eval_dataset() with filter_name="traitgym_promoter" instead.

    This function is kept for backward compatibility with existing code.

    Args:
        tokenizer: Tokenizer implementing the biofoundation Tokenizer protocol
        genome_path: Path to the reference genome file
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., "mendelian_traits")
        window_size: Size of the window around variants (must be even)
        cache_dir: Directory to cache transformed dataset
        objective: Training objective (training objective ("mlm" or "clm")

    Returns:
        Transformed dataset with columns: input_ids, pos, ref, alt, label
    """
    import warnings

    warnings.warn(
        "load_traitgym_mendelian_promoter_dataset is deprecated. "
        "Use load_eval_dataset() with filter_name='traitgym_promoter' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility, we need to handle the case where the genome
    # is already downloaded at genome_path. We'll construct a fake file:// URL
    # and ensure the genome file is in the expected location.
    genome_path = Path(genome_path)

    if not genome_path.exists():
        raise FileNotFoundError(
            f"Genome file not found at {genome_path}. "
            "Please download it first or use load_eval_dataset() with genome_url."
        )

    # Use load_eval_dataset with the new API
    # We'll pass data_dir as the parent of genome_path to ensure it's found
    return load_eval_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        genome_url=f"file:///{genome_path.name}",  # Fake URL with just filename
        filter_name="traitgym_promoter",
        dataset_config=dataset_config,
        window_size=window_size,
        cache_dir=cache_dir,
        objective=objective,
        data_dir=genome_path.parent,  # Ensure download_genome finds existing file
    )

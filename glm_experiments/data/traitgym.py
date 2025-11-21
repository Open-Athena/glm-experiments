"""TraitGym variant dataset loading for evaluation.

This module provides functions to load and transform the TraitGym dataset
for variant effect prediction evaluation during training.
"""

import logging
import urllib.request
from functools import partial
from pathlib import Path

from biofoundation.data import Genome, transform_llr_mlm
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


def load_traitgym_dataset(
    tokenizer: Tokenizer,
    genome_path: str | Path,
    dataset_name: str = "songlab/TraitGym",
    dataset_config: str = "mendelian_traits",
    window_size: int = 512,
    cache_dir: str | Path = "data/traitgym_cache",
) -> Dataset:
    """Load and transform TraitGym dataset for variant effect prediction.

    Loads the TraitGym dataset from HuggingFace and applies the transform_llr_mlm
    transformation to prepare examples for masked language model evaluation.

    The Genome is only loaded if the transformed dataset is not cached.

    Args:
        tokenizer: Tokenizer implementing the biofoundation Tokenizer protocol
        genome_path: Path to the reference genome file
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., "mendelian_traits")
        window_size: Size of the window around variants (must be even)
        cache_dir: Directory to cache transformed dataset

    Returns:
        Transformed dataset with columns: input_ids, pos, ref, alt, label
    """
    from datasets import load_from_disk

    # Create cache path based on config
    cache_path = Path(cache_dir) / f"{dataset_config}_window{window_size}"

    # Check if cached transformed dataset exists
    if cache_path.exists():
        log.info(f"Loading cached TraitGym dataset from {cache_path}")
        dataset = load_from_disk(str(cache_path))
        dataset.set_format(type="torch")
        return dataset

    # Not cached - need to transform with genome
    log.info(f"Loading TraitGym dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="test")  # nosec B615

    log.info(f"Loading reference genome from {genome_path} (this may take a minute)...")
    genome = Genome(genome_path)

    # We need to keep the label column for evaluation
    original_columns = dataset.column_names

    transform_fn = partial(
        transform_llr_mlm,
        tokenizer=tokenizer,
        genome=genome,
        window_size=window_size,
    )

    # Transform the dataset
    log.info("Transforming TraitGym dataset...")
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

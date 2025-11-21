"""TraitGym variant dataset loading for evaluation.

This module provides functions to load and transform the TraitGym dataset
for variant effect prediction evaluation during training.
"""

import urllib.request
from functools import partial
from pathlib import Path

from biofoundation.data import Genome, transform_llr_mlm
from biofoundation.model.base import Tokenizer
from datasets import Dataset, load_dataset


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
    genome: Genome,
    tokenizer: Tokenizer,
    dataset_name: str = "songlab/TraitGym",
    dataset_config: str = "mendelian_traits",
    window_size: int = 512,
) -> Dataset:
    """Load and transform TraitGym dataset for variant effect prediction.

    Loads the TraitGym dataset from HuggingFace and applies the transform_llr_mlm
    transformation to prepare examples for masked language model evaluation.

    Args:
        genome: Genome object for sequence extraction
        tokenizer: Tokenizer implementing the biofoundation Tokenizer protocol
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., "mendelian_traits")
        window_size: Size of the window around variants (must be even)

    Returns:
        Transformed dataset with columns: input_ids, pos, ref, alt, label
    """
    # Load raw dataset
    dataset = load_dataset(dataset_name, dataset_config, split="test")  # nosec B615

    # Apply transform_llr_mlm to each example
    # This extracts a window around each variant and masks the reference position
    transform_fn = partial(
        transform_llr_mlm,
        tokenizer=tokenizer,
        genome=genome,
        window_size=window_size,
    )

    # We need to keep the label column for evaluation
    original_columns = dataset.column_names

    # Transform the dataset
    dataset = dataset.map(
        transform_fn,
        remove_columns=[c for c in original_columns if c != "label"],
    )

    return dataset

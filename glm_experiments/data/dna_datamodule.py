"""Generic DataModule for DNA masked language modeling."""

from typing import Optional

import numpy as np
import torch
from Bio.Seq import Seq
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from glm_experiments.data.components.mlm_collator import (
    DataCollatorForLanguageModelingSimplified,
)


class DNADataModule(LightningDataModule):
    """Generic DataModule for DNA masked language modeling.

    Loads any HuggingFace DNA dataset (streaming) and applies tokenization with
    optional reverse complement augmentation and soft masking.

    Based on GPN's implementation in gpn/ss/run_mlm.py.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Total batch size across all devices (will be divided by world_size)
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        mlm_probability: Probability of masking tokens (default: 0.15)
        soft_masked_loss_weight_train: Loss weight for soft-masked regions during training
        soft_masked_loss_weight_eval: Loss weight for soft-masked regions during evaluation
        data_augmentation: Whether to apply reverse complement augmentation (training only)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset_name: str = "songlab/gpn-animal-promoter-dataset",
        tokenizer_name: str = "gonzalobenegas/tokenizer-dna-mlm",
        batch_size: int = 2048,  # Total batch size
        num_workers: int = 8,
        pin_memory: bool = True,
        mlm_probability: float = 0.15,
        soft_masked_loss_weight_train: float = 0.01,
        soft_masked_loss_weight_eval: float = 0.0,
        data_augmentation: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Will be set in setup() based on world_size
        self.batch_size_per_device = batch_size

        # Will be initialized in prepare_data/setup
        self.tokenizer = None
        self.data_train = None
        self.data_val = None
        self.data_collator = None

    def prepare_data(self) -> None:
        """Download data and tokenizer (runs on single GPU/process)."""
        # For streaming datasets, no pre-download needed
        # Download tokenizer only
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)  # nosec B615

    def setup(self, stage: str | None = None) -> None:
        """Load data and create datasets.

        Args:
            stage: Either 'fit' or 'validate'
        """
        # Calculate per-device batch size for DDP
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)  # nosec B615

        # Create data collator for MLM
        self.data_collator = DataCollatorForLanguageModelingSimplified(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.hparams.mlm_probability,
        )

        # Load raw dataset with streaming
        raw_datasets = load_dataset(self.hparams.dataset_name, streaming=True)  # nosec B615

        # Tokenization function
        def tokenize_function(examples, soft_masked_weight, data_aug=False):
            """Tokenize sequences with optional reverse complement augmentation.

            Args:
                examples: Batch of examples with 'seq' field
                soft_masked_weight: Loss weight for lowercase nucleotides
                data_aug: Whether to apply reverse complement augmentation

            Returns:
                Dictionary with input_ids (torch.uint8), special_tokens_mask, and
                loss_weight (torch.float16)
            """
            seq = examples["seq"]

            # Apply reverse complement augmentation
            if data_aug:
                n = len(seq)
                strand = np.random.choice(["+", "-"], n)
                seq = [
                    seq[i] if strand[i] == "+" else str(Seq(seq[i]).reverse_complement())
                    for i in range(n)
                ]

            # Tokenize (returns dict with 'input_ids' as list of lists)
            tokenized = self.tokenizer(
                seq,
                return_special_tokens_mask=True,
                padding=False,
                truncation=False,
            )

            # Convert to tensors
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.uint8)
            special_tokens_mask = torch.tensor(tokenized["special_tokens_mask"], dtype=torch.uint8)

            # Create loss weights (lower weight for soft-masked lowercase regions)
            loss_weight = torch.ones_like(input_ids, dtype=torch.float16)
            for i, s in enumerate(seq):
                lowercase_mask = np.array([c.islower() for c in s])
                loss_weight[i][lowercase_mask] = soft_masked_weight

            return {
                "input_ids": input_ids,
                "special_tokens_mask": special_tokens_mask,
                "loss_weight": loss_weight,
            }

        # Process splits (train and val only)
        if stage == "fit" or stage is None:
            # Training dataset with augmentation and shuffling
            train_dataset = raw_datasets["train"].shuffle(seed=self.hparams.seed)
            train_dataset = train_dataset.map(
                lambda ex: tokenize_function(
                    ex,
                    self.hparams.soft_masked_loss_weight_train,
                    data_aug=self.hparams.data_augmentation,
                ),
                batched=True,
                remove_columns=list(list(raw_datasets["train"].take(1))[0].keys()),
                # drop_last_batch needed for torch.compile to avoid variable batch sizes
                drop_last_batch=True,
                batch_size=self.hparams.batch_size,
            )

            # Validation dataset (no augmentation, no shuffling)
            val_dataset = raw_datasets["validation"].map(
                lambda ex: tokenize_function(
                    ex,
                    self.hparams.soft_masked_loss_weight_eval,
                    data_aug=False,
                ),
                batched=True,
                remove_columns=list(list(raw_datasets["validation"].take(1))[0].keys()),
            )

            # Split datasets by node for DDP
            if self.trainer is not None and self.trainer.world_size > 1:
                train_dataset = split_dataset_by_node(
                    train_dataset,
                    rank=self.trainer.global_rank,
                    world_size=self.trainer.world_size,
                )
                val_dataset = split_dataset_by_node(
                    val_dataset,
                    rank=self.trainer.global_rank,
                    world_size=self.trainer.world_size,
                )

            self.data_train = train_dataset
            self.data_val = val_dataset

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,  # Shuffling handled by dataset
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_collator,
        )

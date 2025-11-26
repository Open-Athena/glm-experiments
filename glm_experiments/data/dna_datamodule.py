"""Generic DataModule for DNA masked language modeling."""

from typing import Any

import numpy as np
import torch
from Bio.Seq import Seq
from biofoundation.model.adapters.hf import HFTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer

from glm_experiments.data.traitgym import (
    download_genome,
    load_traitgym_mendelian_promoter_dataset,
)
from glm_experiments.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def apply_reverse_complement(sequences: list[str]) -> list[str]:
    """Apply random reverse complement augmentation to sequences.

    Each sequence is independently randomly assigned to forward or reverse
    complement strand with equal probability. Uses torch random for proper
    seeding in DataLoader workers.

    Args:
        sequences: List of DNA sequences

    Returns:
        List of sequences, each randomly on forward or reverse complement strand
    """
    n = len(sequences)
    # Use torch random (0=forward, 1=reverse) - properly seeded per DataLoader worker
    reverse_mask = torch.randint(0, 2, (n,))
    return [
        str(Seq(seq).reverse_complement()) if reverse_mask[i] else seq
        for i, seq in enumerate(sequences)
    ]


def apply_mlm_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply masked language modeling to input tokens.

    Uses standard BERT masking strategy:
    - 15% of tokens are selected for masking
    - Of those: 80% replaced with [MASK], 10% random token, 10% unchanged

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        mask_token_id: Token ID for [MASK]
        vocab_size: Vocabulary size for random replacement
        mlm_probability: Probability of selecting a token for masking

    Returns:
        Tuple of (masked_input_ids, labels) both as int8.
        Labels has -100 for non-masked positions (standard PyTorch ignore_index).
    """
    input_ids = input_ids.clone().to(torch.int8)
    labels = input_ids.clone()

    # Select tokens for masking
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # standard PyTorch ignore_index

    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% -> random token (0.5 of remaining 20%)
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.int8)
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> unchanged (implicit)

    return input_ids, labels


def apply_clm_labels(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare CLM labels for next-token prediction.

    Unlike MLM, CLM doesn't need -100 padding or masking. The model will slice
    logits[:, :-1] and labels[:, 1:] to align predictions with targets.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)

    Returns:
        Tuple of (input_ids, labels) both as int8.
        Labels are same as input_ids (slicing happens in model).
    """
    input_ids = input_ids.clone().to(torch.int8)
    labels = input_ids.clone()  # No shifting - model handles slicing

    return input_ids, labels


class LMDataModule(LightningDataModule):
    """Base DataModule for DNA language modeling.

    Loads any HuggingFace DNA dataset (streaming) and applies tokenization with
    optional reverse complement augmentation and soft masking.

    Subclasses override apply_labels() to implement MLM vs CLM label creation.

    Based on GPN's implementation in gpn/ss/run_mlm.py.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Total effective batch size (used for gradient accumulation calculation)
        per_device_batch_size: Batch size per device (what fits in GPU memory)
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        soft_masked_loss_weight_train: Loss weight for soft-masked regions during training
        soft_masked_loss_weight_eval: Loss weight for soft-masked regions during evaluation
        data_augmentation: Whether to apply reverse complement augmentation (training only)
        max_val_lm_samples: Maximum number of samples for LM validation (None = unlimited)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset_name: str = "songlab/gpn-animal-promoter-dataset",
        tokenizer_name: str = "gonzalobenegas/tokenizer-dna-mlm",
        batch_size: int = 2048,  # Total effective batch size
        per_device_batch_size: int = 256,  # Batch size that fits in GPU memory
        num_workers: int = 8,
        pin_memory: bool = True,
        soft_masked_loss_weight_train: float = 0.01,
        soft_masked_loss_weight_eval: float = 0.0,
        data_augmentation: bool = True,
        max_val_lm_samples: int | None = None,
        seed: int = 42,
        evals: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Used in DataLoader
        self.batch_size_per_device = per_device_batch_size

        # Will be initialized in prepare_data/setup
        self.tokenizer = None
        self.data_train = None
        self.data_val = None
        self.data_val_traitgym_mendelian_promoter = None

    def apply_labels(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply objective-specific label creation (override in subclasses).

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len)

        Returns:
            Tuple of (input_ids, labels) for the specific objective
        """
        raise NotImplementedError("Subclasses must implement apply_labels")

    def get_objective(self) -> str:
        """Return the objective type for this data module (override in subclasses).

        Returns:
            Objective string: "mlm" or "clm"
        """
        raise NotImplementedError("Subclasses must implement get_objective")

    def prepare_data(self) -> None:
        """Download data and tokenizer (runs on single GPU/process)."""
        # For streaming datasets, no pre-download needed
        # Download tokenizer only
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)  # nosec B615

        # Download reference genome for TraitGym Mendelian Promoter evaluation if configured
        evals = self.hparams.get("evals") or {}
        traitgym_cfg = evals.get("traitgym_mendelian_promoter")
        if traitgym_cfg:
            download_genome(
                url=traitgym_cfg["genome_url"],
                path=traitgym_cfg["genome_path"],
            )

    def setup(self, stage: str | None = None) -> None:
        """Load data and create datasets.

        Args:
            stage: Either 'fit' or 'validate'
        """
        # Seed torch for reproducibility - this determines the base_seed
        # that PyTorch DataLoader uses to seed each worker
        torch.manual_seed(self.hparams.seed)

        # Calculate and set gradient accumulation for effective batch size
        if self.trainer is not None:
            world_size = self.trainer.world_size
            per_device = self.hparams.per_device_batch_size
            total = self.hparams.batch_size

            # Validate that total batch size is achievable
            if total % (per_device * world_size) != 0:
                raise RuntimeError(
                    f"Total batch size ({total}) must be divisible by "
                    f"(per_device_batch_size * world_size) = ({per_device} * {world_size} = {per_device * world_size})."
                )

            accumulate_grad_batches = total // (per_device * world_size)
            self.trainer.accumulate_grad_batches = accumulate_grad_batches

            # Adjust val_check_interval when using gradient accumulation
            # See: https://github.com/Lightning-AI/pytorch-lightning/issues/17207
            if accumulate_grad_batches > 1 and self.trainer.val_check_interval is not None:
                original_interval = self.trainer.val_check_interval
                adjusted_interval = original_interval * accumulate_grad_batches
                self.trainer.val_check_interval = adjusted_interval

            # Log batch size configuration
            log.info("Batch size configuration:")
            log.info(f"  per_device_batch_size: {per_device}")
            log.info(f"  world_size (num GPUs): {world_size}")
            log.info(f"  accumulate_grad_batches: {accumulate_grad_batches}")
            log.info(
                f"  effective batch_size: {per_device * world_size * accumulate_grad_batches}"
            )

            # Log val_check_interval adjustment
            if accumulate_grad_batches > 1 and self.trainer.val_check_interval is not None:
                log.info(
                    f"  val_check_interval adjusted: {original_interval} → "
                    f"{adjusted_interval} (multiplied by accumulate_grad_batches)"
                )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)  # nosec B615

        def tokenize(seq: list[str]) -> list[list[int]]:
            """Tokenize sequences to input_ids only."""
            return self.tokenizer(
                seq,
                padding=False,
                truncation=False,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_special_tokens_mask=False,
            )["input_ids"]

        def transform_batch(examples: dict, soft_masked_weight: float, data_aug: bool) -> dict:
            """Transform a batch of examples.

            Args:
                examples: Batch of examples with 'seq' field
                soft_masked_weight: Loss weight for lowercase nucleotides
                data_aug: Whether to apply reverse complement augmentation

            Returns:
                Dictionary with input_ids, labels, and loss_weight (all tensors)
            """
            seq = examples["seq"]

            # Apply reverse complement augmentation
            if data_aug:
                seq = apply_reverse_complement(seq)

            # Tokenize
            input_ids = torch.tensor(tokenize(seq), dtype=torch.int8)

            # Create loss weights (lower weight for soft-masked lowercase regions)
            loss_weight = torch.ones(input_ids.shape, dtype=torch.float16)
            for i, s in enumerate(seq):
                lowercase_mask = np.array([c.islower() for c in s])
                loss_weight[i][lowercase_mask] = soft_masked_weight

            # Apply objective-specific label creation (MLM vs CLM)
            input_ids, labels = self.apply_labels(input_ids)

            return {
                "input_ids": input_ids,
                "labels": labels,
                "loss_weight": loss_weight,
            }

        # Load raw dataset with streaming
        raw_datasets = load_dataset(self.hparams.dataset_name, streaming=True)  # nosec B615

        # Process splits (train and val only)
        if stage == "fit" or stage is None:
            # Training dataset with augmentation and shuffling
            train_dataset = raw_datasets["train"].shuffle(seed=self.hparams.seed)
            train_dataset = train_dataset.map(
                lambda ex: transform_batch(
                    ex,
                    soft_masked_weight=self.hparams.soft_masked_loss_weight_train,
                    data_aug=self.hparams.data_augmentation,
                ),
                batched=True,
                remove_columns=list(list(raw_datasets["train"].take(1))[0].keys()),
                # drop_last_batch helpful for many issues, including
                # https://github.com/Lightning-AI/pytorch-lightning/issues/17207
                drop_last_batch=True,
                batch_size=self.hparams.batch_size,
            )

            # Validation dataset (no augmentation, no shuffling)
            val_dataset = raw_datasets["validation"]

            # Limit samples if max_val_lm_samples is set
            if self.hparams.max_val_lm_samples is not None:
                val_dataset = val_dataset.take(self.hparams.max_val_lm_samples)

            val_dataset = val_dataset.map(
                lambda ex: transform_batch(
                    ex,
                    soft_masked_weight=self.hparams.soft_masked_loss_weight_eval,
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

            # Load TraitGym Mendelian Promoter evaluation dataset if configured
            evals = self.hparams.get("evals") or {}
            traitgym_cfg = evals.get("traitgym_mendelian_promoter")
            if traitgym_cfg:
                self.data_val_traitgym_mendelian_promoter = (
                    load_traitgym_mendelian_promoter_dataset(
                        tokenizer=HFTokenizer(self.tokenizer),
                        genome_path=traitgym_cfg["genome_path"],
                        dataset_name=traitgym_cfg.get("dataset_name", "songlab/TraitGym"),
                        dataset_config=traitgym_cfg.get("dataset_config", "mendelian_traits"),
                        window_size=traitgym_cfg.get("window_size", 512),
                        objective=self.get_objective(),
                    )
                )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,  # Shuffling handled by dataset
            collate_fn=default_collate,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Create validation dataloader(s).

        Returns a single dataloader for MLM validation, or a list of dataloaders
        if TraitGym evaluation is configured: [mlm_val_loader, traitgym_loader].
        """
        mlm_val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=default_collate,
        )

        if self.data_val_traitgym_mendelian_promoter is not None:
            # Get batch size from config, default to 128
            evals = self.hparams.get("evals") or {}
            traitgym_cfg = evals.get("traitgym_mendelian_promoter", {})
            traitgym_batch_size = traitgym_cfg.get("batch_size", 128)

            traitgym_loader = DataLoader(
                dataset=self.data_val_traitgym_mendelian_promoter,
                batch_size=traitgym_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=default_collate,
            )
            return [mlm_val_loader, traitgym_loader]

        return mlm_val_loader


class MLMDataModule(LMDataModule):
    """DataModule for Masked Language Modeling.

    Args:
        mlm_probability: Probability of masking tokens (default: 0.15)
        **kwargs: Other arguments passed to LMDataModule
    """

    def __init__(
        self,
        mlm_probability: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlm_probability = mlm_probability

    def get_objective(self) -> str:
        """Return the objective type for MLM."""
        return "mlm"

    def apply_labels(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking to create labels.

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len)

        Returns:
            Tuple of (masked_input_ids, labels) with -100 for non-masked positions
        """
        return apply_mlm_masking(
            input_ids,
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=self.tokenizer.vocab_size,
            mlm_probability=self.mlm_probability,
        )


class CLMDataModule(LMDataModule):
    """DataModule for Causal Language Modeling.

    Args:
        **kwargs: Arguments passed to LMDataModule
    """

    def get_objective(self) -> str:
        """Return the objective type for CLM."""
        return "clm"

    def apply_labels(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare CLM labels (no masking, model handles slicing).

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len)

        Returns:
            Tuple of (input_ids, labels) where labels are same as input_ids
        """
        return apply_clm_labels(input_ids)

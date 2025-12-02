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

from glm_experiments.data.evals import download_genome, load_eval_dataset
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


def apply_dlm_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply diffusion masking to input tokens.

    For each sequence, samples a random masking ratio r ~ Uniform(0, 1),
    then masks each token with probability r. Unlike BERT/MLM, there is
    no token replacement (100% of selected tokens become [MASK]).

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        mask_token_id: Token ID for [MASK]

    Returns:
        Tuple of (masked_input_ids, labels) both as int8.
        Labels has -100 for non-masked positions (standard PyTorch ignore_index).
    """
    input_ids = input_ids.clone().to(torch.int8)
    labels = input_ids.clone()

    batch_size, seq_len = input_ids.shape

    # Sample masking ratio r ~ Uniform(0, 1) for each sequence
    masking_ratios = torch.rand(batch_size, 1)  # Shape: (batch_size, 1)

    # Create probability matrix: each sequence has its own masking ratio
    probability_matrix = masking_ratios.expand(batch_size, seq_len)  # (batch_size, seq_len)

    # Select tokens for masking based on per-sequence ratio
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Standard PyTorch ignore_index

    # Replace ALL masked tokens with [MASK] (no random replacement)
    input_ids[masked_indices] = mask_token_id

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
        soft_masked_weight: Loss weight for soft-masked regions (not used in data module)
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
        soft_masked_weight: float = 0.01,
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
        # Dynamic eval datasets (keyed by eval_name from config)
        self.eval_datasets: dict[str, Any] = {}

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

        # Download genomes for all configured eval datasets
        evals = self.hparams.get("evals") or []
        for eval_cfg in evals:
            download_genome(
                url=eval_cfg["genome_url"],
                data_dir=eval_cfg.get("data_dir", "data"),
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

        def transform_batch(examples: dict, data_aug: bool) -> dict:
            """Transform a batch of examples.

            Args:
                examples: Batch of examples with 'seq' field
                data_aug: Whether to apply reverse complement augmentation

            Returns:
                Dictionary with input_ids, labels, and soft_masked (all tensors)
            """
            seq = examples["seq"]

            # Apply reverse complement augmentation
            if data_aug:
                seq = apply_reverse_complement(seq)

            # Tokenize
            input_ids = torch.tensor(tokenize(seq), dtype=torch.int8)

            # Create soft_masked boolean tensor (True for lowercase nucleotides)
            soft_masked = torch.zeros(input_ids.shape, dtype=torch.bool)
            for i, s in enumerate(seq):
                lowercase_mask = np.array([c.islower() for c in s])
                soft_masked[i][lowercase_mask] = True

            # Apply objective-specific label creation (MLM vs CLM)
            input_ids, labels = self.apply_labels(input_ids)

            return {
                "input_ids": input_ids,
                "labels": labels,
                "soft_masked": soft_masked,
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

            # Load all configured eval datasets dynamically
            evals = self.hparams.get("evals") or []
            for eval_cfg in evals:
                eval_name = eval_cfg["name"]
                log.info(f"Loading eval dataset: {eval_name}")
                self.eval_datasets[eval_name] = load_eval_dataset(
                    tokenizer=HFTokenizer(self.tokenizer),
                    dataset_name=eval_cfg["dataset_name"],
                    genome_url=eval_cfg["genome_url"],
                    filter_name=eval_cfg.get("filter_name", "none"),
                    dataset_config=eval_cfg.get("dataset_config"),
                    split=eval_cfg.get("split", "test"),
                    window_size=eval_cfg.get("window_size", 512),
                    objective=self.get_objective(),
                    data_dir=eval_cfg.get("data_dir", "data"),
                    label_column=eval_cfg.get("label_column", "label"),
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

        Returns a single dataloader for LM validation, or a list of dataloaders
        if eval datasets are configured: [lm_val_loader, eval_loader_1, eval_loader_2, ...].
        """
        lm_val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=default_collate,
        )

        if not self.eval_datasets:
            return lm_val_loader

        # Create dataloaders for all eval datasets
        eval_loaders = [lm_val_loader]
        evals = self.hparams.get("evals") or []
        eval_dict = {e["name"]: e for e in evals}

        for eval_name, eval_dataset in self.eval_datasets.items():
            eval_cfg = eval_dict[eval_name]
            eval_loaders.append(
                DataLoader(
                    dataset=eval_dataset,
                    batch_size=eval_cfg.get("batch_size", 128),
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    collate_fn=default_collate,
                )
            )

        return eval_loaders


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


class DLMDataModule(LMDataModule):
    """DataModule for Diffusion Language Modeling.

    Uses per-sequence variable masking ratio r ~ Uniform(0, 1).
    No token replacement (100% [MASK]).

    Args:
        **kwargs: Arguments passed to LMDataModule
    """

    def get_objective(self) -> str:
        """Return the objective type for DLM."""
        return "dlm"

    def apply_labels(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply DLM masking to create labels.

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len)

        Returns:
            Tuple of (masked_input_ids, labels) with -100 for non-masked positions
        """
        return apply_dlm_masking(
            input_ids,
            mask_token_id=self.tokenizer.mask_token_id,
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

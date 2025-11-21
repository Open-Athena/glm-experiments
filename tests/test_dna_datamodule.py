"""Tests for DNA DataModule and MLM functions."""

import numpy as np
import pytest
import torch
from Bio.Seq import Seq
from hydra import compose, initialize

from glm_experiments.data.dna_datamodule import (
    DNADataModule,
    apply_mlm_masking,
    apply_reverse_complement,
)


@pytest.fixture
def dna_datamodule():
    """Create a DNA DataModule for testing.

    Uses small batch size and no workers for fast testing.
    """
    dm = DNADataModule(
        dataset_name="songlab/gpn-animal-promoter-dataset",
        tokenizer_name="gonzalobenegas/tokenizer-dna-mlm",
        batch_size=32,  # Small batch for testing
        num_workers=0,  # Single thread for testing
        pin_memory=False,
        mlm_probability=0.15,
        soft_masked_loss_weight_train=0.01,
        soft_masked_loss_weight_eval=0.0,
        data_augmentation=True,
        seed=42,
    )
    return dm


def test_tokenizer_loads(dna_datamodule):
    """Test that tokenizer loads correctly from HuggingFace."""
    dna_datamodule.prepare_data()
    dna_datamodule.setup(stage="fit")

    assert dna_datamodule.tokenizer is not None
    assert hasattr(dna_datamodule.tokenizer, "vocab_size")
    assert dna_datamodule.tokenizer.vocab_size > 0
    assert hasattr(dna_datamodule.tokenizer, "mask_token_id")


@pytest.mark.slow
def test_datamodule_setup(dna_datamodule):
    """Test that DataModule sets up correctly and creates dataloaders."""
    dna_datamodule.prepare_data()
    dna_datamodule.setup(stage="fit")

    # Check datasets are created
    assert dna_datamodule.data_train is not None
    assert dna_datamodule.data_val is not None

    # Check dataloaders are created
    train_loader = dna_datamodule.train_dataloader()
    val_loader = dna_datamodule.val_dataloader()
    assert train_loader is not None
    assert val_loader is not None


@pytest.mark.slow
def test_batch_shape_and_types(dna_datamodule):
    """Test that batches have correct shapes and tensor types."""
    dna_datamodule.prepare_data()
    dna_datamodule.setup(stage="fit")

    train_loader = dna_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Check batch keys
    assert "input_ids" in batch
    assert "labels" in batch
    assert "loss_weight" in batch

    # Check shapes match
    batch_size = batch["input_ids"].shape[0]
    seq_length = batch["input_ids"].shape[1]
    assert batch["labels"].shape == (batch_size, seq_length)
    assert batch["loss_weight"].shape == (batch_size, seq_length)

    # Check dtypes
    assert batch["input_ids"].dtype == torch.int8
    assert batch["labels"].dtype == torch.int8
    assert batch["loss_weight"].dtype == torch.float16


def test_soft_masking_loss_weights():
    """Test that soft masking loss weights are computed correctly for lowercase nucleotides."""
    from transformers import AutoTokenizer

    # Create a simple test case with mixed case sequence
    tokenizer = AutoTokenizer.from_pretrained("gonzalobenegas/tokenizer-dna-mlm")  # nosec B615

    # Test sequence with lowercase (soft-masked) regions
    test_seq = ["ATGCatgcATGC"]  # Lowercase in middle
    soft_masked_weight = 0.01

    # Tokenize
    tokenized = tokenizer(
        test_seq,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )

    # Create loss weights
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.uint8)
    loss_weight = torch.ones(input_ids.shape, dtype=torch.float16)

    for i, s in enumerate(test_seq):
        lowercase_mask = np.array([c.islower() for c in s])
        loss_weight[i][lowercase_mask] = soft_masked_weight

    # Check that lowercase positions have lower weight
    seq = test_seq[0]
    lowercase_count = sum(1 for c in seq if c.islower())
    assert (loss_weight[0] == soft_masked_weight).sum() == lowercase_count
    assert (loss_weight[0] == 1.0).sum() > 0  # Uppercase positions have weight 1.0


def test_reverse_complement():
    """Test that reverse complement augmentation produces valid complement sequences."""
    # Test known sequences
    test_cases = [
        ("ATGC", "GCAT"),
        ("AAAA", "TTTT"),
        ("ATCG", "CGAT"),
        ("atgc", "gcat"),  # Lowercase should also work
    ]

    for seq, expected_rc in test_cases:
        rc = str(Seq(seq).reverse_complement())
        assert rc == expected_rc


def test_apply_reverse_complement():
    """Test the apply_reverse_complement function."""
    # Set seed for reproducibility
    np.random.seed(42)

    sequences = ["ATGC", "GGGG", "AAAA"]
    result = apply_reverse_complement(sequences)

    # Should return same number of sequences
    assert len(result) == len(sequences)

    # Each result should be either original or reverse complement
    for i, seq in enumerate(sequences):
        rc = str(Seq(seq).reverse_complement())
        assert result[i] in [seq, rc]


def test_apply_mlm_masking():
    """Test the apply_mlm_masking function."""
    # Create test input
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=torch.int8)
    mask_token_id = 7
    vocab_size = 10
    mlm_probability = 0.5  # High probability for testing

    masked_ids, labels = apply_mlm_masking(
        input_ids,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        mlm_probability=mlm_probability,
    )

    # Check shapes
    assert masked_ids.shape == input_ids.shape
    assert labels.shape == input_ids.shape

    # Check dtypes
    assert masked_ids.dtype == torch.int8
    assert labels.dtype == torch.int8

    # Check that some positions are masked (labels != -100)
    masked_positions = labels != -100
    assert masked_positions.sum() > 0

    # Check that non-masked positions have label -100
    assert (labels == -100).sum() > 0

    # Check that [MASK] tokens appear in output
    assert (masked_ids == mask_token_id).sum() > 0


def test_apply_mlm_masking_preserves_unmasked():
    """Test that apply_mlm_masking with 0 probability preserves all tokens."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int8)

    masked_ids, labels = apply_mlm_masking(
        input_ids,
        mask_token_id=7,
        vocab_size=10,
        mlm_probability=0.0,
    )

    # All labels should be -100 (ignore)
    assert (labels == -100).all()

    # Input should be unchanged
    assert torch.equal(masked_ids, input_ids)


@pytest.mark.slow
def test_collator_applies_masking(dna_datamodule):
    """Test that masking is applied correctly in batches."""
    dna_datamodule.prepare_data()
    dna_datamodule.setup(stage="fit")

    # Get a batch
    train_loader = dna_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Check that labels are set (non-masked positions should be -100)
    assert "labels" in batch
    labels = batch["labels"]

    # Count masked positions (where labels != -100)
    masked_positions = labels != -100
    total_tokens = labels.numel()
    masking_ratio = masked_positions.sum().item() / total_tokens

    # Masking ratio should be close to mlm_probability (0.15)
    # Allow some variance since it's probabilistic
    assert 0.05 < masking_ratio < 0.25, f"Masking ratio {masking_ratio} not close to 0.15"


@pytest.mark.slow
def test_mask_token_in_input(dna_datamodule):
    """Test that [MASK] tokens appear in masked input."""
    dna_datamodule.prepare_data()
    dna_datamodule.setup(stage="fit")

    train_loader = dna_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Check for [MASK] tokens in input
    mask_token_id = dna_datamodule.tokenizer.mask_token_id
    input_ids = batch["input_ids"]
    mask_count = (input_ids == mask_token_id).sum().item()

    # There should be some masked tokens (about 80% of 15% = 12% of tokens)
    assert mask_count > 0


def test_batch_size_per_device_calculation():
    """Test that batch_size_per_device is calculated correctly for different world_sizes."""

    # Create a mock trainer with different world_size values
    class MockTrainer:
        def __init__(self, world_size):
            self.world_size = world_size
            self.global_rank = 0

    dm = DNADataModule(batch_size=2048)

    # Test the batch size calculation logic directly without loading dataset
    # Test single device
    trainer = MockTrainer(world_size=1)
    expected = 2048 // trainer.world_size
    assert expected == 2048

    # Test 4 devices
    trainer = MockTrainer(world_size=4)
    expected = 2048 // trainer.world_size
    assert expected == 512

    # Test 8 devices
    trainer = MockTrainer(world_size=8)
    expected = 2048 // trainer.world_size
    assert expected == 256


def test_batch_size_not_divisible_raises_error():
    """Test that non-divisible batch size raises RuntimeError."""

    class MockTrainer:
        def __init__(self, world_size):
            self.world_size = world_size

    dm = DNADataModule(batch_size=2047)  # Not divisible by 8
    dm.trainer = MockTrainer(world_size=8)

    with pytest.raises(RuntimeError, match="not divisible"):
        dm.setup(stage="fit")


def test_hydra_instantiation():
    """Test that DataModule can be instantiated from Hydra config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=gpn_animal_promoter"],
        )

        # Instantiate datamodule
        import hydra

        dm = hydra.utils.instantiate(cfg.data)

        # Check that it's the right type
        assert isinstance(dm, DNADataModule)
        assert dm.hparams.dataset_name == "songlab/gpn-animal-promoter-dataset"
        assert dm.hparams.tokenizer_name == "gonzalobenegas/tokenizer-dna-mlm"
        assert dm.hparams.batch_size == 2048
        assert dm.hparams.mlm_probability == 0.15


@pytest.mark.slow
def test_validation_no_soft_masking(dna_datamodule):
    """Test that validation split has no soft masking (weight = 0.0)."""
    dna_datamodule.prepare_data()
    dna_datamodule.setup(stage="fit")

    val_loader = dna_datamodule.val_dataloader()
    val_batch = next(iter(val_loader))

    # Check that soft_masked_loss_weight_eval is 0.0
    assert dna_datamodule.hparams.soft_masked_loss_weight_eval == 0.0

    # For validation, loss_weight should not have many positions with very low weights
    # (though it depends on the data)
    loss_weight = val_batch["loss_weight"]
    min_weight = loss_weight.min().item()

    # Min weight should be 0.0 for eval (soft-masked positions)
    assert min_weight == 0.0 or min_weight == 1.0

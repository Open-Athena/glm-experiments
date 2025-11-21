"""Tests for BERT Lightning module."""

import pytest
import torch
from hydra import compose, initialize

from glm_experiments.models.bert_lit_module import BERTLitModule, MaskedLMAdapter


@pytest.fixture
def bert_lit_module():
    """Create BERTLitModule from config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    return hydra.utils.instantiate(cfg.model)


def test_bert_lit_module_instantiation():
    """Test that BERTLitModule can be instantiated from config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    assert isinstance(model, BERTLitModule)
    assert hasattr(model, "net")
    assert hasattr(model, "hparams")
    assert "optimizer" in model.hparams
    assert "scheduler" in model.hparams


def test_bert_lit_module_forward():
    """Test forward pass through LightningModule."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "loss_weight": torch.ones(batch_size, seq_len),
    }

    # Forward pass
    loss = model.model_step(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_bert_lit_module_configure_optimizers():
    """Test optimizer and scheduler configuration."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    optim_config = model.configure_optimizers()

    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config
    assert optim_config["lr_scheduler"]["interval"] == "step"


def test_bert_lit_module_training_step():
    """Test training step."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "loss_weight": torch.ones(batch_size, seq_len),
    }

    # Training step
    loss = model.training_step(batch, batch_idx=0)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_bert_lit_module_validation_step():
    """Test validation step."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Create dummy batch
    batch_size = 2
    seq_len = 100
    batch = {
        "input_ids": torch.randint(0, 6, (batch_size, seq_len)),
        "labels": torch.randint(0, 6, (batch_size, seq_len)),
        "loss_weight": torch.ones(batch_size, seq_len),
    }

    # Validation step (returns None)
    result = model.validation_step(batch, batch_idx=0)
    assert result is None


def test_bert_lit_module_save_hyperparameters():
    """Test that hyperparameters are saved correctly."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    import hydra

    model = hydra.utils.instantiate(cfg.model)

    # Check that hparams were saved (excluding net)
    assert hasattr(model, "hparams")
    assert "optimizer" in model.hparams
    assert "scheduler" in model.hparams
    # net should NOT be in hparams
    assert "net" not in model.hparams


def test_masked_lm_adapter(bert_lit_module):
    """Test MaskedLMAdapter returns logits from BERT model."""
    adapter = bert_lit_module.mlm_adapter

    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, 7, (batch_size, seq_len))

    logits = adapter(input_ids)

    assert logits.shape == (batch_size, seq_len, 7)  # vocab_size=7
    assert logits.dtype == torch.float32


def test_validation_step_traitgym_mendelian_promoter(bert_lit_module):
    """Test validation_step with TraitGym Mendelian Promoter batch (dataloader_idx=1)."""
    batch_size = 4
    seq_len = 512

    # Create TraitGym Mendelian Promoter-style batch
    batch = {
        "input_ids": torch.randint(0, 7, (batch_size, seq_len)),
        "pos": torch.full((batch_size,), 256),  # Center position
        "ref": torch.randint(1, 5, (batch_size,)),  # Token IDs for A, C, G, T
        "alt": torch.randint(1, 5, (batch_size,)),
        "label": torch.randint(0, 2, (batch_size,)),  # Binary labels
    }

    # Run validation step with dataloader_idx=1 (TraitGym Mendelian Promoter)
    bert_lit_module.validation_step(batch, batch_idx=0, dataloader_idx=1)

    # Check that metrics were updated
    assert bert_lit_module.traitgym_mendelian_promoter_scores.update_count == 1
    assert bert_lit_module.traitgym_mendelian_promoter_labels.update_count == 1


def test_on_validation_epoch_end_computes_auprc(bert_lit_module):
    """Test on_validation_epoch_end computes AUPRC from accumulated data."""
    # Simulate accumulated data from multiple batches
    scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    labels = torch.tensor([0, 0, 1, 1])

    bert_lit_module.traitgym_mendelian_promoter_scores.update(scores)
    bert_lit_module.traitgym_mendelian_promoter_labels.update(labels)

    # Run epoch end
    bert_lit_module.on_validation_epoch_end()

    # Metrics should be reset after computation
    assert bert_lit_module.traitgym_mendelian_promoter_scores.update_count == 0
    assert bert_lit_module.traitgym_mendelian_promoter_labels.update_count == 0


def test_validation_step_mlm_still_works(bert_lit_module):
    """Test that MLM validation (dataloader_idx=0) still works."""
    batch_size = 2
    seq_len = 100

    batch = {
        "input_ids": torch.randint(0, 7, (batch_size, seq_len)),
        "labels": torch.randint(0, 7, (batch_size, seq_len)),
        "loss_weight": torch.ones(batch_size, seq_len),
    }

    # Should not raise
    result = bert_lit_module.validation_step(batch, batch_idx=0, dataloader_idx=0)
    assert result is None

"""Tests for BERT Lightning module."""

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from glm_experiments.models.bert_lit_module import BERTLitModule


@pytest.fixture
def bert_lit_module():
    """Create BERTLitModule from config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["model=bert_bytenet_small"])

    # Instantiate model
    model = BERTLitModule(
        net=torch.nn.Module(),  # Dummy for now
        optimizer=cfg.model.optimizer,
        scheduler=cfg.model.scheduler,
        compile=False,
    )
    return model


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
    assert "compile" in model.hparams
    # net should NOT be in hparams
    assert "net" not in model.hparams

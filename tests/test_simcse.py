"""Tests for SimCSE contrastive learning model."""

import pytest
import torch

from glm_experiments.data.lm_datamodule import SimCSEDataModule, apply_simcse_labels
from glm_experiments.models.components.lm import SimCSE
from glm_experiments.models.components.transformer import Embedding, Transformer
from glm_experiments.models.lm_lit_module import SimCSELitModule


def _make_simcse(hidden_size=128, n_layers=2, num_heads=4, dropout_p=0.1):
    """Helper to create a small SimCSE model with Transformer encoder."""
    embedder = Embedding(vocab_size=6, d_model=hidden_size)
    encoder = Transformer(
        hidden_size=hidden_size, n_layers=n_layers, num_heads=num_heads, dropout=dropout_p
    )
    layer_norm = torch.nn.RMSNorm(hidden_size)
    return SimCSE(
        embedder=embedder,
        encoder=encoder,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
        temperature=0.05,
    )


@pytest.fixture
def simcse_model():
    """Create a small SimCSE model for testing."""
    return _make_simcse()


def test_simcse_get_embeddings_shape(simcse_model):
    """Test that get_embeddings returns shape (B, D)."""
    input_ids = torch.randint(0, 6, (4, 100))
    embeddings = simcse_model.get_embeddings(input_ids)

    assert embeddings.shape == (4, 128)
    assert embeddings.dtype == torch.float32


def test_simcse_forward_returns_loss(simcse_model):
    """Test that forward returns dict with 'loss' key."""
    batch_size = 4
    seq_len = 100

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = input_ids.clone()
    soft_masked = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    soft_masked_weight = 0.01

    simcse_model.train()
    loss_dict = simcse_model(input_ids, labels, soft_masked, soft_masked_weight)

    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert loss_dict["loss"].dtype == torch.float32
    assert loss_dict["loss"].item() >= 0.0


def test_simcse_dropout_creates_different_embeddings(simcse_model):
    """Test that in train mode, two forward passes produce different embeddings."""
    input_ids = torch.randint(0, 6, (4, 100))

    simcse_model.train()
    emb1 = simcse_model.get_embeddings(input_ids)
    emb2 = simcse_model.get_embeddings(input_ids)

    # Embeddings should differ due to dropout in encoder + post-layer_norm
    assert not torch.allclose(emb1, emb2, atol=1e-6)


def test_simcse_eval_mode_deterministic(simcse_model):
    """Test that in eval mode, two forward passes produce identical embeddings."""
    input_ids = torch.randint(0, 6, (4, 100))

    simcse_model.eval()
    emb1 = simcse_model.get_embeddings(input_ids)
    emb2 = simcse_model.get_embeddings(input_ids)

    assert torch.allclose(emb1, emb2, atol=1e-7)


def test_simcse_rejects_bytenet():
    """Test that SimCSE raises AssertionError when given a ByteNet encoder."""
    from glm_experiments.models.components.bytenet import ByteNet

    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    layer_norm = torch.nn.LayerNorm(128)

    with pytest.raises(AssertionError, match="ByteNet has no dropout support"):
        SimCSE(embedder=embedder, encoder=encoder, layer_norm=layer_norm)


def test_simcse_vep_cosine_similarity():
    """Test that _compute_raw_llr returns values in [-1, 1]."""
    from functools import partial

    net = _make_simcse(hidden_size=64, n_layers=2, num_heads=4)
    optimizer = partial(torch.optim.AdamW, lr=0.001)
    scheduler = partial(torch.optim.lr_scheduler.ConstantLR)

    lit_module = SimCSELitModule(net=net, optimizer=optimizer, scheduler=scheduler)

    # Simulate eval batch with CLM-style format: input_ids shape [B, 2, L]
    batch_size = 4
    seq_len = 50
    input_ids = torch.randint(0, 6, (batch_size, 2, seq_len))
    batch = {"input_ids": input_ids, "label": torch.zeros(batch_size)}

    scores = lit_module._compute_raw_llr(batch)

    assert scores.shape == (batch_size,)
    assert (scores >= -1.0).all()
    assert (scores <= 1.0).all()


def test_simcse_vep_identical_sequences():
    """Test that identical ref/alt sequences produce cosine similarity near 1.0."""
    from functools import partial

    net = _make_simcse(hidden_size=64, n_layers=2, num_heads=4)
    optimizer = partial(torch.optim.AdamW, lr=0.001)
    scheduler = partial(torch.optim.lr_scheduler.ConstantLR)

    lit_module = SimCSELitModule(net=net, optimizer=optimizer, scheduler=scheduler)

    # Create identical ref and alt sequences
    batch_size = 4
    seq_len = 50
    single_seq = torch.randint(0, 6, (batch_size, seq_len))
    input_ids = torch.stack([single_seq, single_seq], dim=1)  # (B, 2, L)
    batch = {"input_ids": input_ids, "label": torch.zeros(batch_size)}

    scores = lit_module._compute_raw_llr(batch)

    # Identical sequences should have cosine similarity very close to 1.0
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-5)


def test_apply_simcse_labels():
    """Test that apply_simcse_labels returns input_ids unchanged."""
    input_ids = torch.randint(0, 6, (4, 100), dtype=torch.int8)

    result_input_ids, result_labels = apply_simcse_labels(input_ids)

    # Both should equal the original
    assert torch.equal(result_input_ids, input_ids)
    assert torch.equal(result_labels, input_ids)
    assert result_input_ids.dtype == torch.int8
    assert result_labels.dtype == torch.int8


def test_simcse_datamodule_objective():
    """Test that SimCSEDataModule returns 'simcse' objective."""
    dm = SimCSEDataModule()
    assert dm.get_objective() == "simcse"


def test_simcse_transformer_small_config():
    """Test that simcse_transformer_small config instantiates correctly."""
    import hydra
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model=simcse_transformer_small"],
        )

    model = hydra.utils.instantiate(cfg.model)
    assert isinstance(model, SimCSELitModule)
    assert isinstance(model.net, SimCSE)

    GlobalHydra.instance().clear()

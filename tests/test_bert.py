"""Tests for BERT model."""

import pytest
import torch

from glm_experiments.models.components.bert import BERT
from glm_experiments.models.components.bytenet import ByteNet


@pytest.fixture
def bert_model():
    """Create a small BERT model for testing."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=4, slim=True)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    return BERT(embedder=embedder, encoder=encoder, decoder=decoder)


def test_bert_embed(bert_model):
    """Test BERT embedding layer."""
    input_ids = torch.randint(0, 6, (2, 100))  # (batch, seq_len)
    embeddings = bert_model.embedder(input_ids)

    assert embeddings.shape == (2, 100, 128)  # (batch, seq_len, hidden)
    assert embeddings.dtype == torch.float32


def test_bert_forward(bert_model):
    """Test BERT forward pass with loss calculation."""
    batch_size = 2
    seq_len = 100

    # Create inputs
    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))

    # Mask 15% of positions
    mask = torch.rand(batch_size, seq_len) < 0.15
    labels[~mask] = -100  # -100 means don't compute loss

    # Loss weights (1.0 for hard-masked, 0.01 for soft-masked)
    loss_weight = torch.ones(batch_size, seq_len)

    # Forward pass
    loss = bert_model(input_ids, labels, loss_weight)

    # Should return scalar loss
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0  # Loss should be non-negative


def test_bert_weighted_loss(bert_model):
    """Test BERT correctly applies loss weights."""
    batch_size = 2
    seq_len = 100

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))

    # Mask all positions
    labels[:, :] = torch.randint(0, 6, (batch_size, seq_len))

    # Two scenarios: all weight 1.0 vs all weight 0.01
    loss_weight_high = torch.ones(batch_size, seq_len)
    loss_weight_low = torch.ones(batch_size, seq_len) * 0.01

    loss_high = bert_model(input_ids, labels, loss_weight_high)
    loss_low = bert_model(input_ids, labels, loss_weight_low)

    # Losses should be similar (normalized by weight sum)
    # But not exactly equal due to floating point
    assert torch.allclose(loss_high, loss_low, rtol=1e-5)


def test_bert_no_masked_positions():
    """Test BERT when no positions are masked (all labels = -100)."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    model = BERT(embedder=embedder, encoder=encoder, decoder=decoder)

    batch_size = 2
    seq_len = 50

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100)  # All ignored
    loss_weight = torch.ones(batch_size, seq_len)

    # Should still work (loss might be NaN or 0)
    loss = model(input_ids, labels, loss_weight)
    assert loss.shape == ()


def test_bert_all_masked_positions():
    """Test BERT when all positions are masked."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    model = BERT(embedder=embedder, encoder=encoder, decoder=decoder)

    batch_size = 2
    seq_len = 50

    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))  # All valid
    loss_weight = torch.ones(batch_size, seq_len)

    loss = model(input_ids, labels, loss_weight)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_bert_batch_size_one():
    """Test BERT with batch size of 1."""
    embedder = torch.nn.Embedding(num_embeddings=6, embedding_dim=128, padding_idx=0)
    encoder = ByteNet(hidden_size=128, n_layers=2, slim=True)
    decoder = torch.nn.Linear(in_features=128, out_features=6)
    model = BERT(embedder=embedder, encoder=encoder, decoder=decoder)

    input_ids = torch.randint(0, 6, (1, 50))
    labels = torch.randint(0, 6, (1, 50))
    loss_weight = torch.ones(1, 50)

    loss = model(input_ids, labels, loss_weight)
    assert loss.shape == ()
    assert loss.item() >= 0.0

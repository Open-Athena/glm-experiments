"""BERT model for masked language modeling with genomic sequences."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted cross-entropy loss.

    Args:
        logits: Logits of shape (batch, seq_len, vocab_size)
        labels: Target labels of shape (batch, seq_len), -100 for ignored positions
        loss_weight: Loss weights of shape (batch, seq_len)

    Returns:
        Scalar loss value
    """
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1).long()
    loss_weight = loss_weight.view(-1)
    # Subset to positions where labels != -100 (ignore index)
    mask = labels != -100
    logits = logits[mask]
    labels = labels[mask]
    loss_weight = loss_weight[mask]
    loss = F.cross_entropy(logits, labels, reduction="none")
    loss = (loss * loss_weight / loss_weight.sum()).sum()
    return loss


class BERT(nn.Module):
    """BERT model for masked language modeling using ByteNet encoder.

    Architecture: input_ids → embed → encode → layer_norm → decode → logits

    Note: Masking is handled by the data collator, not the model.

    Args:
        embedder: Token embedding layer (nn.Embedding)
        encoder: Encoder (e.g., ByteNet)
        decoder: Output projection layer (nn.Linear)
    """

    def __init__(
        self,
        embedder: nn.Embedding,
        encoder: nn.Module,
        decoder: nn.Linear,
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.ln = nn.LayerNorm(embedder.embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with loss calculation.

        Args:
            input_ids: Input token IDs (with masks) of shape (batch, seq_len), int8 or long
            labels: True token IDs of shape (batch, seq_len), -100 for non-masked
            loss_weight: Per-token loss weights of shape (batch, seq_len)

        Returns:
            Weighted cross-entropy loss (scalar)
        """
        # Convert int8 to long for embedding lookup
        input_ids = input_ids.long()

        # Embed
        x = self.embedder(input_ids)  # (batch, seq_len, hidden_dim)

        # Encode
        x = self.encoder(x)  # (batch, seq_len, hidden_dim)

        # Layer norm
        x = self.ln(x)  # (batch, seq_len, hidden_dim)

        # Decode to vocabulary
        logits = self.decoder(x)  # (batch, seq_len, vocab_size)

        # Compute loss
        loss = loss_fn(logits, labels, loss_weight)

        return loss

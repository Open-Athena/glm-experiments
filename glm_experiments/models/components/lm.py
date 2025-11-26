"""Language model base class with MLM and CLM subclasses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LM(nn.Module):
    """Base language model class.

    Architecture: input_ids → embedder → encoder → layer_norm → decoder → logits

    Args:
        embedder: Token embedding layer
        encoder: Encoder module (e.g., ByteNet or Transformer)
        layer_norm: Layer normalization module
        decoder: Output projection layer
    """

    def __init__(
        self,
        embedder: nn.Module,
        encoder: nn.Module,
        layer_norm: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.layer_norm = layer_norm
        self.decoder = decoder

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits from input token IDs.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len), int8 or long

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Convert int8 to long for embedding lookup
        input_ids = input_ids.long()

        # Embed
        x = self.embedder(input_ids)  # (batch, seq_len, hidden_dim)

        # Encode
        x = self.encoder(x)  # (batch, seq_len, hidden_dim)

        # Layer norm
        x = self.layer_norm(x)  # (batch, seq_len, hidden_dim)

        # Decode to vocabulary
        logits = self.decoder(x)  # (batch, seq_len, vocab_size)

        return logits

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare logits, labels, and weights for loss computation.

        Override in subclasses to implement MLM vs CLM-specific slicing/filtering.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size)
            labels: Target labels of shape (batch, seq_len)
            loss_weight: Loss weights of shape (batch, seq_len)

        Returns:
            Tuple of (logits, labels, loss_weight) ready for loss computation
        """
        raise NotImplementedError("Subclasses must implement prepare_for_loss")

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Shared loss computation logic for MLM and CLM.

        Args:
            logits: Logits (1D or 2D)
            labels: Target labels (1D)
            loss_weight: Loss weights (1D)

        Returns:
            Scalar loss value
        """
        loss = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss * loss_weight / loss_weight.sum()).sum()
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with loss calculation.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len), int8 or long
            labels: True token IDs of shape (batch, seq_len)
            loss_weight: Per-token loss weights of shape (batch, seq_len)

        Returns:
            Weighted cross-entropy loss (scalar)
        """
        logits = self.get_logits(input_ids)
        logits, labels, loss_weight = self.prepare_for_loss(logits, labels, loss_weight)
        loss = self.compute_loss(logits, labels, loss_weight)
        return loss


class MLM(LM):
    """Masked language model (bidirectional).

    Predicts tokens only at masked positions (labels != -100).
    """

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter to masked positions only.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size)
            labels: Target labels of shape (batch, seq_len), -100 for ignored positions
            loss_weight: Loss weights of shape (batch, seq_len)

        Returns:
            Filtered (logits, labels, loss_weight) for masked positions only
        """
        # Reshape to 1D
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).long()
        loss_weight = loss_weight.view(-1)

        # Filter to masked positions (labels != -100)
        mask = labels != -100
        logits = logits[mask]
        labels = labels[mask]
        loss_weight = loss_weight[mask]

        return logits, labels, loss_weight


class CLM(LM):
    """Causal language model (autoregressive).

    Predicts next token at all positions using causal attention.
    """

    def __init__(
        self,
        embedder: nn.Module,
        encoder: nn.Module,
        layer_norm: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__(embedder, encoder, layer_norm, decoder)

        # Validate encoder supports causal attention
        if not getattr(encoder, "is_causal", True):
            raise ValueError(
                f"CLM requires causal encoder (is_causal=True). "
                f"Got {type(encoder).__name__} with is_causal={encoder.is_causal}"
            )

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice for next-token prediction.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size)
            labels: Target labels of shape (batch, seq_len) (same as input_ids)
            loss_weight: Loss weights of shape (batch, seq_len)

        Returns:
            Sliced (logits, labels, loss_weight) for next-token prediction
        """
        # Slice: logits[:, :-1] predicts labels[:, 1:]
        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        labels = labels[:, 1:].reshape(-1).long()
        loss_weight = loss_weight[:, 1:].reshape(-1)

        return logits, labels, loss_weight

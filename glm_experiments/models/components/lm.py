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
        soft_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare logits, labels, and soft_masked for loss computation.

        Override in subclasses to implement MLM vs CLM-specific slicing/filtering.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size)
            labels: Target labels of shape (batch, seq_len)
            soft_masked: Boolean mask of shape (batch, seq_len)

        Returns:
            Tuple of (logits, labels, soft_masked) ready for loss computation
        """
        raise NotImplementedError("Subclasses must implement prepare_for_loss")

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
        soft_masked_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Compute weighted cross-entropy loss with three variants.

        Computes three loss values:
        1. loss_full: All tokens weighted equally (baseline)
        2. loss_non_soft_masked: Only non-soft-masked tokens
        3. loss: Training loss with soft_masked_weight applied

        Args:
            logits: Logits (1D or 2D)
            labels: Target labels (1D)
            soft_masked: Boolean mask (1D), True for soft-masked positions
            soft_masked_weight: Weight for soft-masked positions in training loss

        Returns:
            Dictionary with keys: loss, loss_full, loss_non_soft_masked
        """
        # Single cross-entropy computation (efficient)
        loss_per_token = F.cross_entropy(logits, labels, reduction="none")

        # Create three weight masks
        weight_full = torch.ones_like(loss_per_token)
        weight_non_soft_masked = (~soft_masked).float()
        weight_training = torch.where(soft_masked, soft_masked_weight, 1.0)

        # Compute normalized losses
        def normalize_and_sum(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            weight_sum = weight.sum()
            if weight_sum > 0:
                return (loss * weight / weight_sum).sum()
            else:
                # Handle edge case: no tokens with weight
                return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        return {
            "loss": normalize_and_sum(loss_per_token, weight_training),
            "loss_full": normalize_and_sum(loss_per_token, weight_full),
            "loss_non_soft_masked": normalize_and_sum(loss_per_token, weight_non_soft_masked),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
        soft_masked_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with loss calculation.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len), int8 or long
            labels: True token IDs of shape (batch, seq_len)
            soft_masked: Boolean mask of shape (batch, seq_len), True for soft-masked positions
            soft_masked_weight: Weight for soft-masked positions in training loss

        Returns:
            Dictionary with loss components (loss, loss_full, loss_non_soft_masked)
        """
        logits = self.get_logits(input_ids)
        logits, labels, soft_masked = self.prepare_for_loss(logits, labels, soft_masked)
        return self.compute_loss(logits, labels, soft_masked, soft_masked_weight)


class MLM(LM):
    """Masked language model (bidirectional).

    Predicts tokens only at masked positions (labels != -100).
    """

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter to masked positions only.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size)
            labels: Target labels of shape (batch, seq_len), -100 for ignored positions
            soft_masked: Boolean mask of shape (batch, seq_len)

        Returns:
            Filtered (logits, labels, soft_masked) for masked positions only
        """
        # Reshape to 1D
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).long()
        soft_masked = soft_masked.view(-1)

        # Filter to masked positions (labels != -100)
        mask = labels != -100
        logits = logits[mask]
        labels = labels[mask]
        soft_masked = soft_masked[mask]

        return logits, labels, soft_masked


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
        soft_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice for next-token prediction.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size)
            labels: Target labels of shape (batch, seq_len) (same as input_ids)
            soft_masked: Boolean mask of shape (batch, seq_len)

        Returns:
            Sliced (logits, labels, soft_masked) for next-token prediction
        """
        # Slice: logits[:, :-1] predicts labels[:, 1:]
        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        labels = labels[:, 1:].reshape(-1).long()
        soft_masked = soft_masked[:, 1:].reshape(-1)

        return logits, labels, soft_masked

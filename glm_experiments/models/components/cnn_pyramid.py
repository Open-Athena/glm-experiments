"""CNN pyramid encoder inspired by AlphaGenome's sequence encoder.

Progressively downsamples via max pooling with channel growth at each stage.
Uses RMSNorm (no batch norm), GELU activations, and residual connections
with zero-padding for channel dimension mismatches.

Architecture overview:
    input (batch, seq_len, d_model)
    → initial_block: Conv1D(wide kernel) + residual conv_block
    → N downres stages: [conv_block(grow channels) + skip, conv_block + skip, max_pool(2)]
    → output_proj: Linear back to d_model
    → output (batch, seq_len // 2^N, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """RMSNorm → GELU → Conv1D with optional dropout.

    Operates in channels-last format (batch, seq_len, channels) internally,
    transposing for Conv1d which expects (batch, channels, seq_len).

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, channels)

        Returns:
            (batch, seq_len, out_channels)
        """
        x = self.norm(x)
        x = F.gelu(x)
        # Transpose for Conv1d: (B, L, C) -> (B, C, L) -> conv -> (B, C', L) -> (B, L, C')
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        return x


class InitialBlock(nn.Module):
    """Initial embedding block: wide Conv1D + residual conv_block.

    Inspired by AlphaGenome's dna_embedder.

    Args:
        d_model: Channel dimension (input and output)
        initial_kernel_size: Kernel size for the initial wide convolution
        kernel_size: Kernel size for the residual conv_block
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, initial_kernel_size: int = 15, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.wide_conv = nn.Conv1d(d_model, d_model, initial_kernel_size, padding=initial_kernel_size // 2)
        self.res_block = ConvBlock(d_model, d_model, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.wide_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.res_block(x)
        return x


class DownresBlock(nn.Module):
    """Downsampling residual block: two conv_blocks with skip connections and channel growth.

    First conv_block increases channels by channel_growth, using zero-padding for the
    residual connection. Second conv_block preserves channels with a standard residual.

    Inspired by AlphaGenome's downres_block.

    Args:
        in_channels: Input channel dimension
        channel_growth: Number of channels to add
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(self, in_channels: int, channel_growth: int = 128, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        out_channels = in_channels + channel_growth
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, dropout)
        self.channel_growth = channel_growth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, in_channels)

        Returns:
            (batch, seq_len, in_channels + channel_growth)
        """
        # First conv with channel growth + zero-padded skip
        out = self.conv1(x)
        out = out + F.pad(x, (0, self.channel_growth))
        # Second conv with standard residual
        out = out + self.conv2(out)
        return out


class ConvPyramid(nn.Module):
    """CNN pyramid encoder inspired by AlphaGenome's sequence encoder.

    Progressively downsamples the input sequence via max pooling (stride 2) at each stage,
    while growing the channel dimension. A final linear projection maps back to d_model.

    For a 512bp input with n_stages=9, the output is (batch, 1, d_model) — a single
    position per sequence. Mean pooling after this is a no-op, making the full sequence
    information available in one vector.

    Args:
        d_model: Input and output channel dimension (must match embedder)
        n_stages: Number of downsampling stages (seq_len is halved each stage)
        channel_growth: Channels added per downres stage
        kernel_size: Conv kernel size for downres blocks
        initial_kernel_size: Conv kernel size for initial embedding block
        dropout: Dropout probability (required > 0 for SimCSE)
    """

    is_causal = False

    def __init__(
        self,
        d_model: int = 512,
        n_stages: int = 9,
        channel_growth: int = 64,
        kernel_size: int = 5,
        initial_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.initial_block = InitialBlock(d_model, initial_kernel_size, kernel_size, dropout)

        self.stages = nn.ModuleList()
        channels = d_model
        for _ in range(n_stages - 1):
            self.stages.append(DownresBlock(channels, channel_growth, kernel_size, dropout))
            channels += channel_growth

        # Project back to d_model
        self.output_proj = nn.Linear(channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model) from embedder

        Returns:
            (batch, seq_len // 2^n_stages, d_model)
        """
        # Initial block (no pooling)
        x = self.initial_block(x)
        # Pool after initial block
        x = x.transpose(1, 2)  # (B, C, L) for max_pool1d
        x = F.max_pool1d(x, kernel_size=2)
        x = x.transpose(1, 2)  # (B, L//2, C)

        # Downres stages with pooling
        for stage in self.stages:
            x = stage(x)
            x = x.transpose(1, 2)
            x = F.max_pool1d(x, kernel_size=2)
            x = x.transpose(1, 2)

        # Project back to d_model
        x = self.output_proj(x)
        return x

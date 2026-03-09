"""
1-D Convolutional Neural Network anomaly classifier for time-series windows.

Architecture:
  - Stack of Conv1d + BatchNorm + ReLU + MaxPool blocks
  - Global average pooling
  - Fully-connected classification head

The CNN captures local temporal patterns (short-range trends, spikes) that
are complementary to the long-range dependencies modeled by GRU / Transformer.
"""

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """
    1-D CNN sequence autoencoder for unsupervised anomaly detection.

    Encoder: stack of Conv1d + BatchNorm + ReLU (same-padding keeps sequence length).
    Decoder: mirrored Conv1d stack projecting back to input_dim.
    Anomaly score = per-window MSE between input and reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Encoder: input_dim -> num_filters (sequence length preserved via same-padding)
        enc_layers = []
        in_ch = input_dim
        for _ in range(num_layers):
            enc_layers += [
                nn.Conv1d(in_ch, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            ]
            in_ch = num_filters
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: num_filters -> input_dim (mirrored, same-padding)
        dec_layers = []
        in_ch = num_filters
        for i in range(num_layers):
            out_ch = input_dim if i == num_layers - 1 else num_filters
            dec_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if i < num_layers - 1:
                dec_layers += [nn.BatchNorm1d(out_ch), nn.ReLU()]
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, input_dim)

        Returns
        -------
        x_hat : (B, W, input_dim)
        """
        x_perm = x.permute(0, 2, 1)       # (B, input_dim, W)
        enc = self.encoder(x_perm)          # (B, num_filters, W)
        enc = self.dropout(enc)
        dec = self.decoder(enc)             # (B, input_dim, W)
        return dec.permute(0, 2, 1)         # (B, W, input_dim)


# Retained for reference / backward compatibility
class CNNAnomalyClassifier(nn.Module):
    """
    1-D CNN for binary anomaly classification on sliding windows.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    num_filters : int
        Number of convolutional filters in each layer.
    kernel_size : int
        Kernel width for all Conv1d layers.
    num_layers : int
        Number of Conv1d blocks.
    dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            ]
            in_ch = num_filters

        self.conv_blocks = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, W, input_dim)

        Returns
        -------
        logits : torch.Tensor of shape (B,)
        """
        # Conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)          # (B, input_dim, W)
        x = self.conv_blocks(x)         # (B, num_filters, W)
        x = x.mean(dim=-1)              # global average pooling -> (B, num_filters)
        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)  # (B,)
        return logits

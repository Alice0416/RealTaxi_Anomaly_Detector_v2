"""
LSTM-based anomaly classifier for time-series windows.

Architecture:
  - Multi-layer LSTM encoder
  - Last hidden state -> fully-connected classification head

The LSTM explicitly models long-range temporal dependencies through its
gating mechanism, complementing the GRU baseline with a richer memory cell.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM sequence autoencoder for unsupervised anomaly detection.

    Encoder compresses the window into (h_n, c_n); decoder reconstructs
    the full sequence from those states with zero-filled inputs.
    Anomaly score = per-window MSE between input and reconstruction.
    """

    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, input_dim)

        Returns
        -------
        x_hat : (B, W, input_dim)
        """
        B, W, _ = x.shape
        _, (h_n, c_n) = self.encoder(x)                                # h_n, c_n: (layers, B, hidden)
        dec_input = torch.zeros(B, W, self.hidden, device=x.device)
        out, _ = self.decoder(dec_input, (h_n, c_n))                   # (B, W, hidden)
        x_hat = self.output_proj(out)                                   # (B, W, input_dim)
        return x_hat


# Retained for reference / backward compatibility
class LSTMAnomalyClassifier(nn.Module):
    """
    LSTM encoder for binary anomaly classification on sliding windows.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden : int
        Number of hidden units in each LSTM layer.
    layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers (only active when layers > 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
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
        out, _ = self.lstm(x)           # (B, W, hidden)
        h = out[:, -1, :]               # last time step -> (B, hidden)
        logits = self.head(h).squeeze(-1)  # (B,)
        return logits

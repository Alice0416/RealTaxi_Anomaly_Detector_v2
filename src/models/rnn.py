import torch
import torch.nn as nn


class GRUAutoencoder(nn.Module):
    """
    GRU sequence autoencoder for unsupervised anomaly detection.

    Encoder compresses the input window into a fixed-size hidden state.
    Decoder reconstructs the full sequence from that hidden state.
    Anomaly score = per-window MSE between input and reconstruction.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden : int
        Hidden size for both encoder and decoder GRU.
    layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers (only active when layers > 1).
    """

    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        # Decoder input: zeros fed at each step; reconstruction driven by initial hidden state
        self.decoder = nn.GRU(
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
        x_hat : (B, W, input_dim)  — reconstructed sequence
        """
        B, W, _ = x.shape
        _, h_n = self.encoder(x)                                        # h_n: (layers, B, hidden)
        # Feed zeros as decoder input; all reconstruction info is in h_n
        dec_input = torch.zeros(B, W, self.hidden, device=x.device)
        out, _ = self.decoder(dec_input, h_n)                           # (B, W, hidden)
        x_hat = self.output_proj(out)                                   # (B, W, input_dim)
        return x_hat


# Retained for reference / backward compatibility
class GRUAnomalyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
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

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        logits = self.head(h).squeeze(-1)
        return logits
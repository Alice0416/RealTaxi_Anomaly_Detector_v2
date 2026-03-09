"""
Transformer-based anomaly classifier for time-series windows.

Architecture (redesigned)
--------------------------
- Linear input projection -> d_model
- Sinusoidal positional encoding
- Stack of Pre-Norm TransformerEncoder layers
  (LayerNorm before each sub-layer for more stable gradient flow)
- Last-token pooling: uses the representation of the FINAL time step,
  which is causally the most informative for predicting whether the
  current (endpoint) timestep is anomalous.
- LayerNorm + classification head

Why last-token pooling instead of mean pooling
-----------------------------------------------
We train on endpoint labels (y = 1 iff the LAST timestep is anomalous).
Mean-pooling averages over ALL 48 timesteps, diluting the signal from the
final step that we actually want to classify. Last-token pooling focuses
the model's representational capacity on predicting the endpoint state,
which directly matches the training objective and the detection goal.

References
----------
Vaswani et al. (2017) "Attention Is All You Need"
Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture"
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerAnomalyClassifier(nn.Module):
    """
    Pre-norm Transformer encoder for binary anomaly classification.

    Pools the representation of the LAST time step to predict whether
    the window endpoint is anomalous (consistent with endpoint labels).

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    d_model : int
        Internal embedding dimension (default 128).
    nhead : int
        Number of attention heads — must divide d_model (default 8).
    num_layers : int
        Number of TransformerEncoder layers (default 3).
    dim_feedforward : int
        Hidden size of the feedforward sublayer (default 256).
    dropout : float
        Dropout probability applied throughout (default 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm: LayerNorm before attention/FF sublayers
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # final layer norm after encoder stack
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
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
        x = self.input_proj(x)          # (B, W, d_model)
        x = self.pos_enc(x)             # add positional encoding + dropout
        x = self.encoder(x)             # (B, W, d_model)
        h = x[:, -1, :]                 # last-token pooling -> (B, d_model)
        logits = self.head(h).squeeze(-1)  # (B,)
        return logits

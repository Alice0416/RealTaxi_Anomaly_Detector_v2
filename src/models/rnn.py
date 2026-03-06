import torch
import torch.nn as nn

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
        # x: (B, W, D)
        out, _ = self.gru(x)
        h = out[:, -1, :]         # last step
        logits = self.head(h).squeeze(-1)
        return logits             # (B,)
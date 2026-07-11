"""
Shared MLP backbone used by all model variants.

Architecture: Input → 128 → 64 → 32 with BatchNorm, ReLU, and Dropout.
"""

import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    Multi-layer perceptron backbone.

    Parameters:
        input_dim: Number of input features.
        hidden_dims: Tuple of hidden layer sizes (default: (128, 64, 32)).
        dropout_rate: Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (128, 64, 32),
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.output_dim = prev_dim  # last hidden dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final hidden representation."""
        return self.backbone(x)

"""
Standard Softmax baseline classifier.

Trained with weighted cross-entropy loss.
Confidence = max softmax probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.models.base_network import BaseNetwork


class SoftmaxClassifier(nn.Module):
    """Standard softmax classifier (baseline)."""

    def __init__(self, input_dim: int, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = BaseNetwork(input_dim, dropout_rate=dropout_rate)
        self.head = nn.Linear(self.backbone.output_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        h = self.backbone(x)
        return self.head(h)

    def predict_with_uncertainty(self, x: torch.Tensor):
        """
        Predict class probabilities and confidence.

        Returns:
            probs: (N, C) softmax probabilities
            predictions: (N,) predicted classes
            confidence: (N,) max probability as confidence
            uncertainty: (N,) 1 - max_prob as uncertainty proxy
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predictions = probs.max(dim=1)
            uncertainty = 1.0 - confidence

        return {
            "logits": logits,
            "probs": probs,
            "predictions": predictions,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "epistemic": uncertainty,  # No decomposition for baseline
            "aleatoric": torch.zeros_like(uncertainty),
        }


def train_softmax(
    model: SoftmaxClassifier,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
):
    """Train the softmax baseline with weighted cross-entropy."""
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)

        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                n_val += len(y_batch)

        val_loss /= n_val
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [Softmax] Epoch {epoch:3d}/{epochs} — "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model

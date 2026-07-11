"""
MC Dropout model (Gal & Ghahramani, ICML 2016).

Keeps dropout active at inference and runs T stochastic forward passes.
Mean prediction = class probability; variance across passes = epistemic uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.base_network import BaseNetwork


class MCDropoutClassifier(nn.Module):
    """Classifier that uses MC Dropout for uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        num_mc_samples: int = 50,
    ):
        super().__init__()
        self.backbone = BaseNetwork(input_dim, dropout_rate=dropout_rate)
        self.head = nn.Linear(self.backbone.output_dim, num_classes)
        self.num_classes = num_classes
        self.num_mc_samples = num_mc_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass returning logits."""
        h = self.backbone(x)
        return self.head(h)

    def _enable_dropout(self):
        """Enable dropout layers during evaluation for MC sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(self, x: torch.Tensor):
        """
        Run T stochastic forward passes with dropout enabled.

        Returns:
            probs: (N, C) mean softmax probabilities
            predictions: (N,) predicted classes
            confidence: (N,) mean max probability
            uncertainty: (N,) predictive entropy
            epistemic: (N,) mutual information (epistemic uncertainty)
            aleatoric: (N,) expected entropy (aleatoric uncertainty)
        """
        self.eval()
        self._enable_dropout()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Stack: (T, N, C)
        all_probs = torch.stack(all_probs, dim=0)

        # Mean prediction
        mean_probs = all_probs.mean(dim=0)  # (N, C)
        predictions = mean_probs.argmax(dim=1)
        confidence = mean_probs.max(dim=1).values

        # Predictive entropy: H[E[p]] = -sum(p_bar * log(p_bar))
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

        # Expected entropy (aleatoric): E[H[p]] = mean over T of -sum(p * log(p))
        per_sample_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)
        expected_entropy = per_sample_entropy.mean(dim=0)

        # Mutual information (epistemic): I = H[E[p]] - E[H[p]]
        mutual_information = predictive_entropy - expected_entropy

        self.eval()  # Reset to eval mode

        return {
            "logits": None,  # Multiple passes, no single logits
            "probs": mean_probs,
            "predictions": predictions,
            "confidence": confidence,
            "uncertainty": predictive_entropy,
            "epistemic": mutual_information,
            "aleatoric": expected_entropy,
        }


def train_mc_dropout(
    model: MCDropoutClassifier,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
):
    """Train MC Dropout model with weighted cross-entropy (same as softmax)."""
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
                f"  [MC Dropout] Epoch {epoch:3d}/{epochs} — "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model

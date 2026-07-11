"""
Deep Ensemble (Lakshminarayanan et al., NeurIPS 2017).

Trains M independently initialised networks and aggregates predictions.
Variance across ensemble members serves as epistemic uncertainty.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.base_network import BaseNetwork
from src.utils.helpers import set_seed


class SingleEnsembleMember(nn.Module):
    """A single network in the ensemble."""

    def __init__(self, input_dim: int, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = BaseNetwork(input_dim, dropout_rate=dropout_rate)
        self.head = nn.Linear(self.backbone.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


class DeepEnsemble:
    """
    Deep Ensemble wrapper: trains M independent networks.

    Parameters:
        input_dim: Number of input features.
        num_classes: Number of output classes.
        num_members: Number of ensemble members (default: 5).
        dropout_rate: Dropout rate for each member.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_members: int = 5,
        dropout_rate: float = 0.3,
    ):
        self.num_members = num_members
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.members = []

        for i in range(num_members):
            set_seed(42 + i)  # Different init per member
            member = SingleEnsembleMember(input_dim, num_classes, dropout_rate)
            self.members.append(member)

    def to(self, device):
        for m in self.members:
            m.to(device)
        return self

    def eval(self):
        for m in self.members:
            m.eval()

    def train(self):
        for m in self.members:
            m.train()

    def predict_with_uncertainty(self, x: torch.Tensor):
        """
        Aggregate predictions from all ensemble members.

        Returns:
            probs: (N, C) mean softmax probabilities
            predictions: (N,) predicted classes
            confidence: (N,) mean max probability
            uncertainty: (N,) predictive entropy
            epistemic: (N,) mutual information
            aleatoric: (N,) expected entropy
        """
        all_probs = []
        with torch.no_grad():
            for member in self.members:
                member.eval()
                logits = member(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Stack: (M, N, C)
        all_probs = torch.stack(all_probs, dim=0)

        # Mean prediction
        mean_probs = all_probs.mean(dim=0)
        predictions = mean_probs.argmax(dim=1)
        confidence = mean_probs.max(dim=1).values

        # Predictive entropy
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

        # Expected entropy (aleatoric)
        per_member_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2)
        expected_entropy = per_member_entropy.mean(dim=0)

        # Mutual information (epistemic)
        mutual_information = predictive_entropy - expected_entropy

        return {
            "logits": None,
            "probs": mean_probs,
            "predictions": predictions,
            "confidence": confidence,
            "uncertainty": predictive_entropy,
            "epistemic": mutual_information,
            "aleatoric": expected_entropy,
        }


def train_deep_ensemble(
    ensemble: DeepEnsemble,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
):
    """Train each ensemble member independently."""
    class_weights = class_weights.to(device)

    for i, member in enumerate(ensemble.members):
        print(f"\n  Training ensemble member {i + 1}/{ensemble.num_members}")
        set_seed(42 + i)
        member.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(member.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            member.train()
            train_loss = 0.0
            n_train = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = member(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
                n_train += len(y_batch)

            train_loss /= n_train

            member.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = member(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    n_val += len(y_batch)

            val_loss /= n_val
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in member.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"    Member {i+1} Epoch {epoch:3d}/{epochs} — "
                    f"train: {train_loss:.4f}, val: {val_loss:.4f}"
                )

        if best_state is not None:
            member.load_state_dict(best_state)
        member.to(device)

    return ensemble

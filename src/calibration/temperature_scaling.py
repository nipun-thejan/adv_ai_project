"""
Post-hoc Temperature Scaling (Guo et al., ICML 2017).

Learns a single scalar temperature T on validation logits that rescales
softmax outputs to improve calibration, without changing predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for model calibration.

    Given logits z, the calibrated probabilities are:
        p_calibrated = softmax(z / T)

    where T is learned on a held-out validation set to minimise NLL.
    """

    def __init__(self):
        super().__init__()
        # Initialise temperature to 1.0 (no effect)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrate(
        self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100
    ):
        """
        Learn the optimal temperature on validation data.

        Args:
            val_logits: (N, C) raw logits from the model on validation set.
            val_labels: (N,) ground-truth labels.
            lr: Learning rate for LBFGS.
            max_iter: Maximum optimisation iterations.

        Returns:
            Optimal temperature value (float).
        """
        nll_criterion = nn.CrossEntropyLoss()

        # Use LBFGS for 1-parameter optimisation (very stable)
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_temp = self.temperature.item()
        print(f"  [Temperature Scaling] Optimal T = {optimal_temp:.4f}")
        return optimal_temp

    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        with torch.no_grad():
            scaled = self.forward(logits)
            return F.softmax(scaled, dim=1)


def collect_logits_and_labels(model, data_loader, device):
    """
    Collect all logits and labels from a data loader.

    For models without direct logits (MC Dropout, Deep Ensemble),
    we use a single forward pass.

    Returns:
        logits: (N, C) tensor
        labels: (N,) tensor
    """
    all_logits = []
    all_labels = []

    if hasattr(model, 'eval'):
        model.eval()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            if hasattr(model, 'forward'):
                # Direct forward pass for logits
                logits = model(X_batch)
            elif hasattr(model, 'members'):
                # Deep Ensemble: use first member's logits
                logits = model.members[0](X_batch)
            else:
                logits = model(X_batch)

            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    return torch.cat(all_logits), torch.cat(all_labels)

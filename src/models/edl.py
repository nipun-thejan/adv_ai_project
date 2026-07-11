"""
Evidential Deep Learning (Sensoy, Kaplan & Kandemir, NeurIPS 2018).

The network outputs Dirichlet concentration parameters α = evidence + 1.
From a single forward pass, this yields:
  - Belief mass per class: b_k = (α_k - 1) / S
  - Uncertainty mass: u = K / S   (vacuity, epistemic)
  - Aleatoric uncertainty: entropy of the expected Dirichlet distribution

Loss = modified cross-entropy (Type II ML) + KL divergence regulariser (annealed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.base_network import BaseNetwork


class EvidentialClassifier(nn.Module):
    """
    Evidential Deep Learning classifier.

    Outputs Dirichlet parameters α via softplus activation,
    decomposing uncertainty into epistemic (vacuity) and aleatoric components
    in a single forward pass.
    """

    def __init__(self, input_dim: int, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = BaseNetwork(input_dim, dropout_rate=dropout_rate)
        self.evidence_head = nn.Linear(self.backbone.output_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return evidence (non-negative) via softplus.
        evidence_k >= 0, then α_k = evidence_k + 1.
        """
        h = self.backbone(x)
        # Use softplus to ensure non-negative evidence
        evidence = F.softplus(self.evidence_head(h))
        return evidence

    def predict_with_uncertainty(self, x: torch.Tensor):
        """
        Single forward pass uncertainty decomposition.

        Returns:
            probs: (N, C) expected class probabilities = α_k / S
            predictions: (N,) predicted classes
            confidence: (N,) max expected probability
            uncertainty: (N,) vacuity u = K / S (epistemic)
            epistemic: (N,) vacuity
            aleatoric: (N,) entropy of expected distribution
        """
        self.eval()
        with torch.no_grad():
            evidence = self.forward(x)
            alpha = evidence + 1.0  # Dirichlet parameters
            S = alpha.sum(dim=1, keepdim=True)  # Dirichlet strength
            K = self.num_classes

            # Expected class probabilities
            probs = alpha / S  # (N, C)
            predictions = probs.argmax(dim=1)
            confidence = probs.max(dim=1).values

            # Epistemic uncertainty (vacuity): u = K / S
            vacuity = K / S.squeeze(1)

            # Aleatoric uncertainty: entropy of the expected categorical
            aleatoric = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

            # Also compute belief masses for analysis
            # b_k = (alpha_k - 1) / S = evidence_k / S
            belief = evidence / S

        return {
            "logits": evidence,  # Evidence serves as "logits" for temp scaling
            "probs": probs,
            "predictions": predictions,
            "confidence": confidence,
            "uncertainty": vacuity,
            "epistemic": vacuity,
            "aleatoric": aleatoric,
            "alpha": alpha,
            "dirichlet_strength": S.squeeze(1),
            "belief": belief,
        }


# ──────────────────────────── EDL Loss Functions ────────────────────────────


def edl_mse_loss(evidence, target, epoch, num_classes, annealing_step=10):
    """
    EDL loss using MSE formulation with KL-divergence regulariser.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)

    # One-hot encode target
    one_hot = F.one_hot(target, num_classes=num_classes).float()

    # MSE loss term
    pred = alpha / S
    err = (one_hot - pred) ** 2
    var = pred * (1 - pred) / (S + 1)
    mse = (err + var).sum(dim=1).mean()

    # KL divergence regulariser (annealed)
    annealing_coeff = min(1.0, epoch / annealing_step)
    alpha_tilde = one_hot + (1 - one_hot) * (alpha - 1) * (1 - one_hot) + 1
    # Simplified: remove evidence for correct class
    alpha_reg = torch.ones_like(alpha)
    alpha_reg += (1 - one_hot) * (alpha - 1)
    kl = kl_divergence_dirichlet(alpha_reg, num_classes)

    return mse + annealing_coeff * kl


def edl_log_loss(evidence, target, epoch, num_classes, annealing_step=10):
    """
    EDL loss using Type II maximum likelihood (log loss) with KL regulariser.
    This is the preferred loss from the original paper.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)

    # One-hot encode target
    one_hot = F.one_hot(target, num_classes=num_classes).float()

    # Type II ML loss: -sum(y_k * (log(alpha_k) - log(S)))
    log_likelihood = (one_hot * (torch.log(S) - torch.log(alpha + 1e-10))).sum(dim=1)
    loss = log_likelihood.mean()

    # KL divergence regulariser (annealed)
    annealing_coeff = min(1.0, epoch / annealing_step)

    # Remove evidence for correct class before KL
    alpha_reg = torch.ones_like(alpha)
    alpha_reg += (1 - one_hot) * (alpha - 1)
    kl = kl_divergence_dirichlet(alpha_reg, num_classes)

    return loss + annealing_coeff * kl


def kl_divergence_dirichlet(alpha, num_classes):
    """
    KL divergence between Dirichlet(alpha) and Dirichlet(1, 1, ..., 1).

    KL[Dir(α) || Dir(1)] = log Γ(Σα) - Σlog Γ(α_k) - log Γ(K)
                           + Σ(α_k - 1)(ψ(α_k) - ψ(Σα))
    """
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)

    ln_B_alpha = (
        torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(S_alpha)
    )
    ln_B_ones = (
        torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(S_ones)
    )

    dg_term = (alpha - ones) * (
        torch.digamma(alpha) - torch.digamma(S_alpha)
    )

    kl = (ln_B_alpha - ln_B_ones + dg_term.sum(dim=1, keepdim=True)).squeeze(1)
    return kl.mean()


# ──────────────────────────── Training ────────────────────────────


def train_edl(
    model: EvidentialClassifier,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    annealing_step: int = 10,
):
    """
    Train EDL model with Type II ML loss + annealed KL regulariser.

    Uses class weights by scaling the loss per sample.
    """
    model.to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None
    num_classes = model.num_classes

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            evidence = model(X_batch)

            # Weighted EDL loss
            loss = edl_log_loss(
                evidence, y_batch, epoch, num_classes, annealing_step
            )

            # Apply per-sample class weighting
            with torch.no_grad():
                sample_w = class_weights[y_batch]
                sample_w = sample_w / sample_w.mean()

            # Recompute with weighting
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            one_hot = F.one_hot(y_batch, num_classes=num_classes).float()
            log_likelihood = (one_hot * (torch.log(S) - torch.log(alpha + 1e-10))).sum(dim=1)
            weighted_loss = (log_likelihood * sample_w).mean()

            annealing_coeff = min(1.0, epoch / annealing_step)
            alpha_reg = torch.ones_like(alpha)
            alpha_reg += (1 - one_hot) * (alpha - 1)
            kl = kl_divergence_dirichlet(alpha_reg, num_classes)

            total_loss = weighted_loss + annealing_coeff * kl
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * len(y_batch)
            n_train += len(y_batch)

        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                evidence = model(X_batch)
                loss = edl_log_loss(
                    evidence, y_batch, epoch, num_classes, annealing_step
                )
                val_loss += loss.item() * len(y_batch)
                n_val += len(y_batch)

        val_loss /= n_val
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [EDL] Epoch {epoch:3d}/{epochs} — "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model

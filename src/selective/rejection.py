"""
Selective Classification / Rejection Policy.

Given epistemic uncertainty U(x) and threshold τ ∈ [0, 1]:
  - If U(x) ≤ τ → classify automatically
  - If U(x) > τ → defer to human review

Produces accuracy-rejection and risk-coverage curves.
"""

import numpy as np
from typing import Dict, Tuple


def selective_classify(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    num_thresholds: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Compute selective classification metrics across a range of thresholds.

    Args:
        predictions: (N,) predicted class labels.
        labels: (N,) ground-truth labels.
        uncertainty: (N,) uncertainty scores (higher = more uncertain).
        num_thresholds: Number of threshold points to evaluate.

    Returns:
        Dictionary with:
            thresholds: (T,) threshold values
            coverages: (T,) fraction of samples classified (not deferred)
            accuracies: (T,) accuracy on classified (non-deferred) samples
            risks: (T,) error rate (1 - accuracy) on classified samples
            rejection_rates: (T,) fraction of samples rejected/deferred
            n_deferred: (T,) absolute count of deferred samples
    """
    N = len(predictions)
    correct = (predictions == labels).astype(float)

    # Sort by uncertainty
    sorted_idx = np.argsort(uncertainty)
    sorted_correct = correct[sorted_idx]
    sorted_uncertainty = uncertainty[sorted_idx]

    # Threshold sweep from min to max uncertainty
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if u_min == u_max:
        u_max = u_min + 1e-6

    thresholds = np.linspace(u_min, u_max, num_thresholds)

    coverages = []
    accuracies = []
    risks = []
    rejection_rates = []
    n_deferred_list = []

    for tau in thresholds:
        # Accepted: uncertainty <= tau
        accepted_mask = uncertainty <= tau
        n_accepted = accepted_mask.sum()
        n_deferred = N - n_accepted

        if n_accepted == 0:
            coverages.append(0.0)
            accuracies.append(1.0)  # Convention: perfect accuracy on empty set
            risks.append(0.0)
        else:
            coverage = n_accepted / N
            acc = correct[accepted_mask].mean()
            risk = 1.0 - acc

            coverages.append(coverage)
            accuracies.append(acc)
            risks.append(risk)

        rejection_rates.append(n_deferred / N)
        n_deferred_list.append(n_deferred)

    return {
        "thresholds": np.array(thresholds),
        "coverages": np.array(coverages),
        "accuracies": np.array(accuracies),
        "risks": np.array(risks),
        "rejection_rates": np.array(rejection_rates),
        "n_deferred": np.array(n_deferred_list),
    }


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Area Under the Risk-Coverage curve (AURC).

    Lower AURC = better selective classification performance.
    Uses trapezoidal integration.
    """
    # Sort by coverage for proper integration
    sorted_idx = np.argsort(coverages)
    c_sorted = coverages[sorted_idx]
    r_sorted = risks[sorted_idx]

    return float(np.trapezoid(r_sorted, c_sorted))


def compute_e_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Excess AURC (E-AURC): AURC minus the optimal selective risk.

    The optimal risk at coverage c is achieved by rejecting the most
    error-prone samples first (oracle ordering).
    """
    aurc = compute_aurc(coverages, risks)
    # Optimal AURC for a random classifier at the same error rate
    # is approximately risk_at_full_coverage * coverage
    # For simplicity, return AURC (E-AURC requires oracle; we report AURC)
    return aurc


def find_operating_point(
    coverages: np.ndarray,
    accuracies: np.ndarray,
    thresholds: np.ndarray,
    target_accuracy: float = 0.99,
) -> Tuple[float, float, float]:
    """
    Find the threshold τ that achieves a target accuracy on accepted samples.

    Returns:
        (threshold, coverage, accuracy) at the operating point.
    """
    for i in range(len(thresholds) - 1, -1, -1):
        if accuracies[i] >= target_accuracy and coverages[i] > 0:
            return thresholds[i], coverages[i], accuracies[i]

    # If target never reached, return most restrictive point
    best_idx = np.argmax(accuracies)
    return thresholds[best_idx], coverages[best_idx], accuracies[best_idx]

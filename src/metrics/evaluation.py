"""
Evaluation metrics for classification, calibration, and selective prediction.

Classification: AUPRC, F1, Precision, Recall, ROC-AUC
Calibration: ECE (Expected Calibration Error), Brier score, reliability diagrams
Selective: Accuracy-rejection curves, risk-coverage curves
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, Optional
import os


# ──────────────────────────── Classification Metrics ────────────────────────


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels (N,).
        y_pred: Predicted labels (N,).
        y_prob: Predicted probabilities for positive class (N,).

    Returns:
        Dictionary of metric_name -> value.
    """
    metrics = {}
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0

    # Overall accuracy
    metrics["accuracy"] = float((y_true == y_pred).mean())
    return metrics


# ──────────────────────────── Calibration Metrics ────────────────────────


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the average gap between
    confidence and accuracy within each bin, weighted by bin size.

    ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(y_prob, axis=1) if y_prob.ndim == 2 else y_prob
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append((0, 0, 0, 0))
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum()
        bin_ece = abs(bin_acc - bin_conf) * (bin_size / len(y_true))
        ece += bin_ece
        bin_data.append((bin_acc, bin_conf, bin_size, bin_ece))

    return float(ece)


def compute_classwise_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    target_class: int = 1,
) -> float:
    """
    ECE computed on a specific class only (important for imbalanced datasets).
    """
    if y_prob.ndim == 2:
        class_prob = y_prob[:, target_class]
    else:
        class_prob = y_prob

    class_mask = y_true == target_class
    if class_mask.sum() == 0:
        return 0.0

    class_correct = (class_prob >= 0.5) == (y_true == target_class)
    class_conf = np.where(class_prob >= 0.5, class_prob, 1 - class_prob)

    bin_boundaries = np.linspace(0.5, 1, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (class_conf > lo) & (class_conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = class_correct[mask].mean()
        bin_conf = class_conf[mask].mean()
        ece += abs(bin_acc - bin_conf) * (mask.sum() / N)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier score = mean((p_fraud - y_true)^2).

    Lower is better. Measures both calibration and sharpness.
    """
    if y_prob.ndim == 2:
        p_pos = y_prob[:, 1]
    else:
        p_pos = y_prob

    return float(np.mean((p_pos - y_true) ** 2))


# ──────────────────────────── Plotting ────────────────────────────


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
):
    """
    Plot reliability diagram (calibration curve).

    Perfect calibration = diagonal line.
    """
    confidences = np.max(y_prob, axis=1) if y_prob.ndim == 2 else y_prob
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_sizes.append(0)
        else:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_sizes.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability plot
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]
    ax1.bar(bin_centers, bin_accs, width=1 / n_bins, alpha=0.6, color="steelblue",
            edgecolor="navy", label="Model")
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of confidences
    ax2.bar(bin_centers, bin_sizes, width=1 / n_bins, alpha=0.6, color="coral", edgecolor="darkred")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_rejection_curve(
    rejection_rates: np.ndarray,
    accuracies: np.ndarray,
    model_names: list,
    title: str = "Accuracy vs Rejection Rate",
    save_path: Optional[str] = None,
):
    """
    Plot accuracy as a function of rejection rate for multiple models.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (rr, acc, name) in enumerate(zip(rejection_rates, accuracies, model_names)):
        ax.plot(rr, acc, linewidth=2, color=colors[i % len(colors)], label=name)

    ax.set_xlabel("Rejection Rate", fontsize=13)
    ax.set_ylabel("Accuracy on Accepted Samples", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_coverage_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    model_names: list,
    title: str = "Risk-Coverage Curve",
    save_path: Optional[str] = None,
):
    """
    Plot risk (error rate) as a function of coverage for multiple models.

    Good models have low risk even at high coverage.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (cov, risk, name) in enumerate(zip(coverages, risks, model_names)):
        ax.plot(cov, risk, linewidth=2, color=colors[i % len(colors)], label=name)

    ax.set_xlabel("Coverage (fraction classified)", fontsize=13)
    ax.set_ylabel("Risk (error rate)", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_uncertainty_distribution(
    uncertainty_legit: np.ndarray,
    uncertainty_fraud: np.ndarray,
    title: str = "Uncertainty Distribution",
    xlabel: str = "Epistemic Uncertainty",
    save_path: Optional[str] = None,
):
    """
    Plot the distribution of uncertainty scores for legitimate vs fraud transactions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(uncertainty_legit, bins=50, alpha=0.6, color="steelblue",
            label="Legitimate", density=True)
    ax.hist(uncertainty_fraud, bins=50, alpha=0.6, color="crimson",
            label="Fraud", density=True)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

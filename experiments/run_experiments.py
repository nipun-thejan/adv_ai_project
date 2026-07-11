"""
Main Experiment Runner
======================

Calibrated Uncertainty and Selective Classification
in Financial Fraud Detection

For each dataset × each model (Softmax, MC Dropout, Deep Ensemble, EDL):
  1. Train model
  2. Evaluate classification metrics on test set
  3. Apply temperature scaling → measure ECE before/after
  4. Compute uncertainty scores → selective classification curves
  5. Save all results, generate plots and summary tables

Usage:
    python -m experiments.run_experiments
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.helpers import set_seed, get_device, get_logger, ensure_dir
from src.data.credit_card import load_credit_card
from src.data.paysim import load_paysim
from src.data.samld import load_samld
from src.models.softmax_baseline import SoftmaxClassifier, train_softmax
from src.models.mc_dropout import MCDropoutClassifier, train_mc_dropout
from src.models.deep_ensemble import DeepEnsemble, train_deep_ensemble
from src.models.edl import EvidentialClassifier, train_edl
from src.calibration.temperature_scaling import TemperatureScaler, collect_logits_and_labels
from src.selective.rejection import selective_classify, compute_aurc, find_operating_point
from src.metrics.evaluation import (
    compute_classification_metrics,
    compute_ece,
    compute_classwise_ece,
    compute_brier_score,
    plot_reliability_diagram,
    plot_accuracy_rejection_curve,
    plot_risk_coverage_curve,
    plot_uncertainty_distribution,
)

logger = get_logger("experiments")

# ═══════════════════════════ Configuration ═══════════════════════════

SEED = 42
EPOCHS = 30
EDL_EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
MC_SAMPLES = 50
ENSEMBLE_MEMBERS = 5
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def get_dataset_loaders():
    """Return available dataset loaders with their names."""
    loaders = {}

    # Try each dataset; skip if not downloaded
    try:
        result = load_credit_card(DATA_DIR, batch_size=BATCH_SIZE, seed=SEED)
        loaders["CreditCard"] = result
        logger.info("✓ Credit Card dataset loaded")
    except FileNotFoundError as e:
        logger.warning(f"✗ Credit Card dataset not found: {e}")

    try:
        result = load_paysim(DATA_DIR, batch_size=BATCH_SIZE, seed=SEED)
        loaders["PaySim"] = result
        logger.info("✓ PaySim dataset loaded")
    except FileNotFoundError as e:
        logger.warning(f"✗ PaySim dataset not found: {e}")

    try:
        result = load_samld(DATA_DIR, batch_size=BATCH_SIZE, seed=SEED)
        loaders["SAMLD"] = result
        logger.info("✓ SAML-D dataset loaded")
    except FileNotFoundError as e:
        logger.warning(f"✗ SAML-D dataset not found: {e}")

    return loaders


def collect_test_predictions(model, test_loader, device):
    """
    Run model on test set and collect predictions + uncertainty.

    Returns dict with numpy arrays.
    """
    all_results = {
        "probs": [], "predictions": [], "confidence": [],
        "uncertainty": [], "epistemic": [], "aleatoric": [],
        "labels": [],
    }
    logits_list = []

    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            result = model.predict_with_uncertainty(X_batch)

            all_results["probs"].append(result["probs"].cpu().numpy())
            all_results["predictions"].append(result["predictions"].cpu().numpy())
            all_results["confidence"].append(result["confidence"].cpu().numpy())
            all_results["uncertainty"].append(result["uncertainty"].cpu().numpy())
            all_results["epistemic"].append(result["epistemic"].cpu().numpy())
            all_results["aleatoric"].append(result["aleatoric"].cpu().numpy())
            all_results["labels"].append(y_batch.numpy())

            if result.get("logits") is not None:
                logits_list.append(result["logits"].cpu().numpy())

    # Concatenate
    for key in all_results:
        all_results[key] = np.concatenate(all_results[key], axis=0)

    if logits_list:
        all_results["logits"] = np.concatenate(logits_list, axis=0)

    return all_results


def evaluate_model(
    model_name: str,
    model,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    input_dim: int,
    device: torch.device,
    dataset_name: str,
):
    """
    Full evaluation pipeline for a single model.

    Returns:
        metrics_dict: All computed metrics
        test_results: Raw test predictions and uncertainty
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name} on {dataset_name}")
    logger.info(f"{'='*60}")

    # ── Collect test predictions ──
    test_results = collect_test_predictions(model, test_loader, device)

    y_true = test_results["labels"]
    y_pred = test_results["predictions"]
    y_prob = test_results["probs"]
    epistemic = test_results["epistemic"]
    aleatoric = test_results["aleatoric"]

    # ── Classification metrics ──
    p_fraud = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    cls_metrics = compute_classification_metrics(y_true, y_pred, p_fraud)
    logger.info(f"  Classification: F1={cls_metrics['f1']:.4f}, "
                f"AUPRC={cls_metrics['auprc']:.4f}, "
                f"ROC-AUC={cls_metrics['roc_auc']:.4f}")

    # ── Calibration metrics (before temperature scaling) ──
    ece_before = compute_ece(y_true, y_prob, y_pred)
    classwise_ece = compute_classwise_ece(y_true, y_prob, target_class=1)
    brier = compute_brier_score(y_true, y_prob)
    logger.info(f"  Calibration: ECE={ece_before:.4f}, "
                f"Classwise-ECE(fraud)={classwise_ece:.4f}, Brier={brier:.4f}")

    # ── Temperature scaling (post-hoc calibration) ──
    ece_after = ece_before
    optimal_temp = 1.0
    if "logits" in test_results:
        try:
            val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
            temp_scaler = TemperatureScaler()
            optimal_temp = temp_scaler.calibrate(val_logits, val_labels)

            # Re-compute ECE with calibrated probabilities
            test_logits_tensor = torch.tensor(test_results["logits"], dtype=torch.float32)
            calibrated_probs = temp_scaler.get_calibrated_probs(test_logits_tensor).numpy()
            calibrated_pred = calibrated_probs.argmax(axis=1)
            ece_after = compute_ece(y_true, calibrated_probs, calibrated_pred)
            brier_after = compute_brier_score(y_true, calibrated_probs)
            logger.info(f"  After Temp Scaling (T={optimal_temp:.3f}): "
                        f"ECE={ece_after:.4f}, Brier={brier_after:.4f}")
        except Exception as e:
            logger.warning(f"  Temperature scaling failed: {e}")
            brier_after = brier
    else:
        brier_after = brier
        logger.info("  (No logits available for temperature scaling)")

    # ── Selective classification ──
    sel_results = selective_classify(y_pred, y_true, epistemic)
    aurc = compute_aurc(sel_results["coverages"], sel_results["risks"])

    # Find operating point for 99% accuracy
    tau_99, cov_99, acc_99 = find_operating_point(
        sel_results["coverages"], sel_results["accuracies"],
        sel_results["thresholds"], target_accuracy=0.99
    )
    logger.info(f"  Selective: AURC={aurc:.4f}, "
                f"@99%acc → coverage={cov_99:.2%}, threshold={tau_99:.4f}")

    # ── Plots ──
    save_dir = os.path.join(RESULTS_DIR, dataset_name)
    ensure_dir(save_dir)

    # Reliability diagram before calibration
    plot_reliability_diagram(
        y_true, y_prob, y_pred,
        title=f"{model_name} — Reliability Diagram (Before Cal.) [{dataset_name}]",
        save_path=os.path.join(save_dir, f"{model_name}_reliability_before.png"),
    )

    # Reliability diagram after calibration (if available)
    if "logits" in test_results and optimal_temp != 1.0:
        plot_reliability_diagram(
            y_true, calibrated_probs, calibrated_pred,
            title=f"{model_name} — Reliability Diagram (After Cal., T={optimal_temp:.2f}) [{dataset_name}]",
            save_path=os.path.join(save_dir, f"{model_name}_reliability_after.png"),
        )

    # Uncertainty distribution
    fraud_mask = y_true == 1
    if fraud_mask.sum() > 0:
        plot_uncertainty_distribution(
            epistemic[~fraud_mask], epistemic[fraud_mask],
            title=f"{model_name} — Epistemic Uncertainty [{dataset_name}]",
            xlabel="Epistemic Uncertainty",
            save_path=os.path.join(save_dir, f"{model_name}_epistemic_dist.png"),
        )
        plot_uncertainty_distribution(
            aleatoric[~fraud_mask], aleatoric[fraud_mask],
            title=f"{model_name} — Aleatoric Uncertainty [{dataset_name}]",
            xlabel="Aleatoric Uncertainty",
            save_path=os.path.join(save_dir, f"{model_name}_aleatoric_dist.png"),
        )

    # Collect all metrics
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        **cls_metrics,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "classwise_ece_fraud": classwise_ece,
        "brier_before": brier,
        "brier_after": brier_after,
        "optimal_temperature": optimal_temp,
        "aurc": aurc,
        "coverage_at_99acc": cov_99,
        "threshold_at_99acc": tau_99,
        "mean_epistemic": float(epistemic.mean()),
        "mean_aleatoric": float(aleatoric.mean()),
    }

    return metrics, test_results, sel_results


def run_all_models_on_dataset(
    dataset_name: str,
    train_loader,
    val_loader,
    test_loader,
    input_dim: int,
    class_weights: torch.Tensor,
    device: torch.device,
):
    """
    Train and evaluate all four models on a single dataset.

    Returns list of metrics dicts and raw results for plotting.
    """
    all_metrics = []
    all_sel_results = []
    model_names_for_plot = []

    # ── 1. Softmax Baseline ──
    print(f"\n{'─'*50}")
    print(f"Training Softmax Baseline on {dataset_name}")
    print(f"{'─'*50}")
    set_seed(SEED)
    softmax_model = SoftmaxClassifier(input_dim, num_classes=2)
    softmax_model = train_softmax(
        softmax_model, train_loader, val_loader, class_weights, device, epochs=EPOCHS, lr=LR
    )
    metrics, _, sel_res = evaluate_model(
        "Softmax", softmax_model, train_loader, val_loader, test_loader,
        class_weights, input_dim, device, dataset_name
    )
    all_metrics.append(metrics)
    all_sel_results.append(sel_res)
    model_names_for_plot.append("Softmax")

    # ── 2. MC Dropout ──
    print(f"\n{'─'*50}")
    print(f"Training MC Dropout on {dataset_name}")
    print(f"{'─'*50}")
    set_seed(SEED)
    mc_model = MCDropoutClassifier(input_dim, num_classes=2, num_mc_samples=MC_SAMPLES)
    mc_model = train_mc_dropout(
        mc_model, train_loader, val_loader, class_weights, device, epochs=EPOCHS, lr=LR
    )
    metrics, _, sel_res = evaluate_model(
        "MC_Dropout", mc_model, train_loader, val_loader, test_loader,
        class_weights, input_dim, device, dataset_name
    )
    all_metrics.append(metrics)
    all_sel_results.append(sel_res)
    model_names_for_plot.append("MC Dropout")

    # ── 3. Deep Ensemble ──
    print(f"\n{'─'*50}")
    print(f"Training Deep Ensemble on {dataset_name}")
    print(f"{'─'*50}")
    set_seed(SEED)
    ensemble = DeepEnsemble(input_dim, num_classes=2, num_members=ENSEMBLE_MEMBERS)
    ensemble = train_deep_ensemble(
        ensemble, train_loader, val_loader, class_weights, device, epochs=EPOCHS, lr=LR
    )
    metrics, _, sel_res = evaluate_model(
        "Deep_Ensemble", ensemble, train_loader, val_loader, test_loader,
        class_weights, input_dim, device, dataset_name
    )
    all_metrics.append(metrics)
    all_sel_results.append(sel_res)
    model_names_for_plot.append("Deep Ensemble")

    # ── 4. Evidential Deep Learning ──
    print(f"\n{'─'*50}")
    print(f"Training EDL on {dataset_name}")
    print(f"{'─'*50}")
    set_seed(SEED)
    edl_model = EvidentialClassifier(input_dim, num_classes=2)
    edl_model = train_edl(
        edl_model, train_loader, val_loader, class_weights, device,
        epochs=EDL_EPOCHS, lr=LR, annealing_step=10
    )
    metrics, _, sel_res = evaluate_model(
        "EDL", edl_model, train_loader, val_loader, test_loader,
        class_weights, input_dim, device, dataset_name
    )
    all_metrics.append(metrics)
    all_sel_results.append(sel_res)
    model_names_for_plot.append("EDL")

    # ── Comparative plots ──
    save_dir = os.path.join(RESULTS_DIR, dataset_name)

    # Accuracy-Rejection curves
    plot_accuracy_rejection_curve(
        [s["rejection_rates"] for s in all_sel_results],
        [s["accuracies"] for s in all_sel_results],
        model_names_for_plot,
        title=f"Accuracy vs Rejection Rate [{dataset_name}]",
        save_path=os.path.join(save_dir, "comparative_accuracy_rejection.png"),
    )

    # Risk-Coverage curves
    plot_risk_coverage_curve(
        [s["coverages"] for s in all_sel_results],
        [s["risks"] for s in all_sel_results],
        model_names_for_plot,
        title=f"Risk-Coverage Curve [{dataset_name}]",
        save_path=os.path.join(save_dir, "comparative_risk_coverage.png"),
    )

    return all_metrics


def main():
    """Run the full experimental pipeline."""
    set_seed(SEED)
    device = get_device()
    ensure_dir(RESULTS_DIR)
    ensure_dir(DATA_DIR)

    logger.info(f"Device: {device}")
    logger.info(f"Results directory: {RESULTS_DIR}")

    # Load all available datasets
    dataset_loaders = get_dataset_loaders()

    if not dataset_loaders:
        logger.error(
            "No datasets found! Please download at least one dataset.\n"
            "Instructions:\n"
            "  pip install opendatasets\n"
            "  python -c \"import opendatasets as od; "
            "od.download('https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud', data_dir='data')\""
        )
        sys.exit(1)

    all_results = []
    start_time = time.time()

    for dataset_name, (train_loader, val_loader, test_loader, input_dim, class_weights) in dataset_loaders.items():
        logger.info(f"\n{'═'*70}")
        logger.info(f"DATASET: {dataset_name} (input_dim={input_dim})")
        logger.info(f"{'═'*70}")

        metrics_list = run_all_models_on_dataset(
            dataset_name, train_loader, val_loader, test_loader,
            input_dim, class_weights, device
        )
        all_results.extend(metrics_list)

    # ── Save summary table ──
    results_df = pd.DataFrame(all_results)

    # Reorder columns for readability
    col_order = [
        "dataset", "model",
        "accuracy", "f1", "precision", "recall", "auprc", "roc_auc",
        "ece_before", "ece_after", "classwise_ece_fraud",
        "brier_before", "brier_after", "optimal_temperature",
        "aurc", "coverage_at_99acc", "threshold_at_99acc",
        "mean_epistemic", "mean_aleatoric",
    ]
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]

    csv_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"\nResults saved to {csv_path}")

    # Print summary table
    print(f"\n{'═'*100}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'═'*100}")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    elapsed = time.time() - start_time
    logger.info(f"\nTotal experiment time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return results_df


if __name__ == "__main__":
    main()

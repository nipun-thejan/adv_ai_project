# Calibrated Uncertainty and Selective Classification in Financial Fraud Detection

## Overview

This project implements the methodology from the IntelliCore CS5801 proposal: a unified framework combining **Evidential Deep Learning (EDL)** for single-forward-pass epistemic/aleatoric uncertainty decomposition, **post-hoc calibration** (temperature scaling), and **selective classification** (reject/defer policy) — evaluated on three financial fraud datasets.

## Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── credit_card.py       # Credit Card Fraud dataset loader
│   │   ├── paysim.py            # PaySim dataset loader
│   │   └── samld.py             # SAML-D dataset loader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_network.py      # Shared MLP backbone
│   │   ├── softmax_baseline.py  # Standard softmax classifier
│   │   ├── mc_dropout.py        # MC Dropout model
│   │   ├── deep_ensemble.py     # Deep Ensemble wrapper
│   │   └── edl.py               # Evidential Deep Learning model
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── temperature_scaling.py  # Post-hoc temperature scaling
│   ├── selective/
│   │   ├── __init__.py
│   │   └── rejection.py         # Selective classification / reject policy
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluation.py        # ECE, Brier, AUPRC, F1, risk-coverage, acc-rejection
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Seeds, device, logging utilities
├── experiments/
│   └── run_experiments.py       # Main experiment runner
├── notebooks/
│   └── results_analysis.ipynb   # (optional) Interactive analysis
├── results/                     # Output: plots, tables, saved models
├── data/                        # Raw/processed dataset storage
├── requirements.txt
└── README.md
```

## Proposed Changes

### Component 1 — Data Loaders

#### [NEW] [credit_card.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/data/credit_card.py)
- Load ULB Credit Card Fraud dataset (284,807 txns, 492 fraud)
- StandardScaler on Amount/Time, PCA features already provided
- Stratified train/val/test split (70/15/15)
- Return PyTorch DataLoaders with class-weighted sampling for imbalance

#### [NEW] [paysim.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/data/paysim.py)
- Load PaySim mobile-money dataset
- One-hot encode transaction type, engineer delta-balance features
- Subsample for tractability (~500K txns), stratified split

#### [NEW] [samld.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/data/samld.py)
- Load SAML-D synthetic AML dataset (12 features, 28 typologies)
- Encode categorical features, normalise numerics
- Stratified split

---

### Component 2 — Models

#### [NEW] [base_network.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/models/base_network.py)
- Shared MLP backbone: `Input → 128 → 64 → 32` with BatchNorm + ReLU + Dropout
- Configurable input dimension, dropout rate

#### [NEW] [softmax_baseline.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/models/softmax_baseline.py)
- Standard cross-entropy trained classifier
- Returns softmax probabilities (max probability as confidence)

#### [NEW] [mc_dropout.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/models/mc_dropout.py)
- Keeps dropout active at inference, runs T=50 forward passes
- Mean prediction as class probability, variance across passes as epistemic uncertainty

#### [NEW] [deep_ensemble.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/models/deep_ensemble.py)
- Trains M=5 independent networks with different seeds
- Mean prediction, variance as uncertainty

#### [NEW] [edl.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/models/edl.py)
- **Core contribution**: Evidential Deep Learning (Sensoy et al., NeurIPS 2018)
- Network outputs Dirichlet concentration parameters α = evidence + 1
- Loss: Modified cross-entropy + KL divergence regularizer (annealed)
- Single forward pass → belief mass, uncertainty mass u = K/S (K=classes, S=Dirichlet strength)
- Decomposes into aleatoric (entropy of expected distribution) and epistemic (mutual information / vacuity)

---

### Component 3 — Calibration

#### [NEW] [temperature_scaling.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/calibration/temperature_scaling.py)
- Learn a single temperature T on validation logits via NLL minimisation (Guo et al., ICML 2017)
- Also implement focal loss variant as alternative calibration-aware training

---

### Component 4 — Selective Classification

#### [NEW] [rejection.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/selective/rejection.py)
- Given U(x) (epistemic uncertainty) and threshold τ ∈ [0,1]:
  - U(x) ≤ τ → classify automatically
  - U(x) > τ → defer to human
- Compute accuracy-rejection curves (accuracy on accepted samples vs rejection rate)
- Compute risk-coverage curves (risk on accepted vs coverage fraction)
- Area under risk-coverage curve (AURC) as summary metric

---

### Component 5 — Evaluation Metrics

#### [NEW] [evaluation.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/src/metrics/evaluation.py)
- **Classification**: AUPRC, F1, Precision, Recall, ROC-AUC
- **Calibration**: ECE (15 bins), classwise-ECE, Brier score, reliability diagrams
- **Selective**: Accuracy-rejection curve, risk-coverage curve, AURC
- **OOD detection**: AUROC of uncertainty scores for detecting held-out fraud typologies

---

### Component 6 — Experiment Runner

#### [NEW] [run_experiments.py](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/experiments/run_experiments.py)
- For each dataset × each model (Softmax, MC Dropout, Deep Ensemble, EDL):
  1. Train model
  2. Evaluate classification metrics on test set
  3. Apply temperature scaling → measure ECE before/after
  4. Compute uncertainty scores → selective classification curves
  5. (Credit Card & SAML-D) Hold out rare fraud types for OOD evaluation
- Save all results, generate plots and summary tables
- Full reproducibility: seed=42, deterministic PyTorch

---

### Component 7 — Report

#### [NEW] [report.md](file:///Users/nipun.fonseka/Learn/masters/adv%20ai/project/report/CS5801_IntelliCore_Report.md)
The final report containing:
1. **Introduction** — updated from proposal
2. **Literature Review** — updated from proposal  
3. **Method** — EDL formulation, calibration, selective classification
4. **Experiments** — datasets, preprocessing, training details, baselines
5. **Results** — tables and figures from experiments
6. **Conclusions** — findings, limitations, future work

---

## Verification Plan

### Automated Tests
```bash
# Run all experiments and generate results
python experiments/run_experiments.py

# Verify outputs exist
ls results/*.png results/*.csv
```

### Manual Verification
- Confirm reliability diagrams show improved calibration after temperature scaling
- Confirm EDL epistemic uncertainty is higher on OOD/held-out fraud types
- Confirm accuracy-rejection curves show monotonic accuracy increase with rejection
- Compare results tables against literature baselines

## Open Questions

> [!IMPORTANT]
> **Dataset download**: The datasets need to be downloaded from Kaggle. Do you have Kaggle credentials set up, or should I download them programmatically via the Kaggle API? Alternatively, I can write the code to auto-download using `kagglehub` or `opendatasets`.

> [!NOTE]
> **Compute**: Deep Ensembles train 5 separate networks. On CPU this may take significant time for PaySim (~6M rows). I will subsample PaySim to ~500K for tractability. Is that acceptable?

> [!NOTE]
> **Report format**: I will write the report as a Markdown file that can easily be converted to PDF via pandoc. Is that fine, or do you need LaTeX?

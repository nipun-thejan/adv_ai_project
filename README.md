# Calibrated Uncertainty and Selective Classification in Financial Fraud Detection

**Course:** CS5801 — Advanced Topics in AI  
**Group:** IntelliCore

## Overview

This project implements a unified framework combining:
1. **Evidential Deep Learning (EDL)** for single-forward-pass uncertainty estimation
2. **Post-hoc temperature scaling** for probability calibration
3. **Selective classification** (reject/defer policy) based on epistemic uncertainty

Evaluated on three financial fraud datasets against baselines: Softmax, MC Dropout, and Deep Ensembles.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (requires Kaggle API credentials)
python download_datasets.py

# Run all experiments
python -m experiments.run_experiments
```

## Project Structure

```
├── src/
│   ├── data/           # Dataset loaders (Credit Card, PaySim, SAML-D)
│   ├── models/         # Softmax, MC Dropout, Deep Ensemble, EDL
│   ├── calibration/    # Temperature scaling
│   ├── selective/      # Rejection policy
│   ├── metrics/        # ECE, Brier, AUPRC, reliability diagrams
│   └── utils/          # Seeds, device, logging
├── experiments/        # Main experiment runner
├── results/            # Output plots and tables
├── data/               # Downloaded datasets
└── report/             # Final project report
```

## Datasets

| Dataset | Transactions | Fraud Rate | Source |
|---------|-------------|------------|--------|
| Credit Card Fraud (ULB) | 284,807 | 0.17% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| PaySim | ~6.3M (subsampled to 500K) | ~1.3% | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| SAML-D | Synthetic AML | Very low | [Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) |

Inside data/, create the three folders named exactly: creditcardfraud, paysim1, and synthetic-transaction-monitoring-dataset-aml. and place the csv

## Key Results

After running experiments, results are saved to `results/summary_results.csv` with:
- Classification metrics (F1, AUPRC, ROC-AUC)
- Calibration metrics (ECE before/after temperature scaling, Brier score)
- Selective classification (AURC, coverage at 99% accuracy)
- Comparative plots (reliability diagrams, accuracy-rejection, risk-coverage curves)

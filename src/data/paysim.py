"""
PaySim mobile-money fraud dataset.

Synthetic dataset built from real African mobile-money transactions.
~6.3M transactions; we subsample for tractability.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_paysim(
    data_dir: str = "data",
    batch_size: int = 512,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: int = 500_000,
    seed: int = 42,
):
    """
    Load and preprocess the PaySim dataset.

    Returns:
        train_loader, val_loader, test_loader, input_dim, class_weights
    """
    csv_path = os.path.join(data_dir, "paysim1", "PS_20174392719_1491204439457_log.csv")
    if not os.path.exists(csv_path):
        # Try finding any CSV in the paysim directory
        paysim_dir = os.path.join(data_dir, "paysim1")
        if os.path.isdir(paysim_dir):
            csvs = [f for f in os.listdir(paysim_dir) if f.endswith(".csv")]
            if csvs:
                csv_path = os.path.join(paysim_dir, csvs[0])
            else:
                raise FileNotFoundError(f"No CSV found in {paysim_dir}")
        else:
            raise FileNotFoundError(
                f"PaySim dataset not found at {csv_path}. "
                "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1"
            )

    logger.info(f"Loading PaySim dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Full shape: {df.shape}, Fraud rate: {df['isFraud'].mean():.4%}")

    # Subsample while preserving all fraud cases
    if len(df) > max_samples:
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0].sample(
            n=max_samples - len(fraud_df), random_state=seed
        )
        df = pd.concat([fraud_df, legit_df], ignore_index=True)
        logger.info(f"  Subsampled to {len(df)} (all {len(fraud_df)} fraud kept)")

    # Feature engineering
    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    # Delta balance features
    df["origBalanceDelta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["destBalanceDelta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Error flags
    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    ).abs()
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs()

    # Drop non-numeric / identifier columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    y = df["isFraud"].values
    X = df.drop("isFraud", axis=1).values.astype(np.float32)

    # Scale all features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Stratified splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_ratio + test_ratio, random_state=seed, stratify=y
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=seed, stratify=y_temp
    )

    logger.info(
        f"  Splits — train: {len(y_train)} (fraud {y_train.sum()}), "
        f"val: {len(y_val)} (fraud {y_val.sum()}), "
        f"test: {len(y_test)} (fraud {y_test.sum()})"
    )

    # Class weights
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)

    # Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    # Weighted sampler
    sample_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    return train_loader, val_loader, test_loader, input_dim, class_weights

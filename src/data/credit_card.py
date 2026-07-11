"""
Credit Card Fraud Detection dataset (ULB, 2013).

284,807 transactions with 492 fraud cases.
Features V1-V28 are PCA-transformed; Time and Amount are raw.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_credit_card(
    data_dir: str = "data",
    batch_size: int = 512,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Load and preprocess the Credit Card Fraud dataset.

    Returns:
        train_loader, val_loader, test_loader, input_dim, class_weights
    """
    csv_path = os.path.join(data_dir, "creditcardfraud", "creditcard.csv")
    if not os.path.exists(csv_path):
        # Try alternative path structures
        alt_path = os.path.join(data_dir, "creditcard.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Credit Card dataset not found at {csv_path}. "
                "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )

    logger.info(f"Loading Credit Card dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Shape: {df.shape}, Fraud rate: {df['Class'].mean():.4%}")

    # Separate features and target
    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    # Scale Time and Amount (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    X[:, 0] = scaler.fit_transform(X[:, 0].reshape(-1, 1)).ravel()   # Time
    X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).ravel()  # Amount

    # Stratified splits: train / val / test
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

    # Class weights for loss weighting
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    logger.info(f"  Class weights: {class_weights.tolist()}")

    # Convert to tensors
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

    # Weighted sampler for training to handle class imbalance
    sample_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]  # 30
    return train_loader, val_loader, test_loader, input_dim, class_weights

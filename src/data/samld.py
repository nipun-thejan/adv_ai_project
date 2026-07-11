"""
SAML-D: Synthetic Anti-Money Laundering Dataset (Oztas et al., 2023).

12 features, 28 normal + suspicious typologies, very low suspicious rate.
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


def load_samld(
    data_dir: str = "data",
    batch_size: int = 512,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Load and preprocess the SAML-D dataset.

    Returns:
        train_loader, val_loader, test_loader, input_dim, class_weights
    """
    samld_dir = os.path.join(
        data_dir, "synthetic-transaction-monitoring-dataset-aml"
    )
    csv_path = None

    # Find the CSV file
    if os.path.isdir(samld_dir):
        for f in os.listdir(samld_dir):
            if f.endswith(".csv"):
                csv_path = os.path.join(samld_dir, f)
                break

    if csv_path is None or not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"SAML-D dataset not found in {samld_dir}. "
            "Download from: https://www.kaggle.com/datasets/berkanoztas/"
            "synthetic-transaction-monitoring-dataset-aml"
        )

    logger.info(f"Loading SAML-D dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Shape: {df.shape}")

    # Identify the target column (typically 'Is_Laundering' or 'Label')
    target_col = None
    for candidate in ["Is_Laundering", "is_laundering", "Label", "label", "Is Laundering"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        # Try to find binary column
        for col in df.columns:
            if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
                target_col = col
                break

    if target_col is None:
        raise ValueError(f"Cannot identify target column. Columns: {df.columns.tolist()}")

    logger.info(f"  Target column: {target_col}, Fraud rate: {df[target_col].mean():.4%}")

    y = df[target_col].values
    df_features = df.drop(columns=[target_col])

    # Encode categorical columns
    label_encoders = {}
    for col in df_features.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        label_encoders[col] = le

    # Drop any remaining non-numeric
    df_features = df_features.select_dtypes(include=[np.number])

    # Handle missing values
    df_features = df_features.fillna(0)

    X = df_features.values.astype(np.float32)

    # Scale
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

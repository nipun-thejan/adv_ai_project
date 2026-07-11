"""
Download datasets from Kaggle using opendatasets.

Usage:
    python download_datasets.py

You will be prompted for Kaggle credentials (username + API key)
if they are not found in ~/.kaggle/kaggle.json.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def download_all():
    try:
        import opendatasets as od
    except ImportError:
        print("Installing opendatasets...")
        os.system(f"{sys.executable} -m pip install opendatasets")
        import opendatasets as od

    os.makedirs(DATA_DIR, exist_ok=True)

    datasets = [
        ("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "Credit Card Fraud"),
        ("https://www.kaggle.com/datasets/ealaxi/paysim1", "PaySim"),
        (
            "https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml",
            "SAML-D",
        ),
    ]

    for url, name in datasets:
        print(f"\n{'─'*50}")
        print(f"Downloading: {name}")
        print(f"{'─'*50}")
        try:
            od.download(url, data_dir=DATA_DIR)
            print(f"✓ {name} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")

    print(f"\nDatasets saved to: {DATA_DIR}")
    # List what we have
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            print(f"  {item}/: {len(files)} files")


if __name__ == "__main__":
    download_all()

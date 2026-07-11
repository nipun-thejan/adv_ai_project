"""
Download datasets using kagglehub (works without explicit credentials for public datasets).
Then copies them into the project's data/ directory structure.

Usage:
    python download_datasets_v2.py
"""

import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def download_with_kagglehub():
    import kagglehub

    datasets = [
        ("mlg-ulb/creditcardfraud", "creditcardfraud"),
        ("ealaxi/paysim1", "paysim1"),
        ("berkanoztas/synthetic-transaction-monitoring-dataset-aml",
         "synthetic-transaction-monitoring-dataset-aml"),
    ]

    for kaggle_slug, local_name in datasets:
        target_dir = os.path.join(DATA_DIR, local_name)
        if os.path.exists(target_dir) and any(f.endswith('.csv') for f in os.listdir(target_dir)):
            print(f"✓ {local_name} already exists, skipping")
            continue

        print(f"\nDownloading {kaggle_slug}...")
        try:
            path = kagglehub.dataset_download(kaggle_slug)
            print(f"  Downloaded to: {path}")

            # Copy to our data directory
            os.makedirs(target_dir, exist_ok=True)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    src = os.path.join(path, f)
                    dst = os.path.join(target_dir, f)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        print(f"  Copied: {f}")
            print(f"✓ {local_name} ready")
        except Exception as e:
            print(f"✗ Failed to download {kaggle_slug}: {e}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Datasets in {DATA_DIR}:")
    if os.path.exists(DATA_DIR):
        for item in sorted(os.listdir(DATA_DIR)):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                total_size = sum(
                    os.path.getsize(os.path.join(item_path, f))
                    for f in files if os.path.isfile(os.path.join(item_path, f))
                )
                print(f"  {item}/: {len(files)} files, {total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_with_kagglehub()

"""
Step 1 — Load & Preprocess
Load datasets from HuggingFace, encode categoricals, handle missing values.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import load_dataset


def load_har_dataset():
    """Load Human Activity Recognition dataset (561 features, 10K rows, 6 classes)."""
    ds = load_dataset("DiFronzo/Human_Activity_Recognition")
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    # Last column is label
    X = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Handle missing values
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    print(f"[HAR] X shape: {X.shape}, classes: {len(np.unique(y))}")
    return X, y, le.classes_, "HAR"


def load_fraud_dataset():
    """Load Credit Card Fraud dataset (28 features, 284K rows, imbalanced)."""
    ds = load_dataset("liberatoratif/Credit-card-fraud-detection")
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    # 'Class' is the label column
    y_raw = df["Class"].values
    X = df.drop(columns=["Class"]).values.astype(np.float64)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    print(f"[Fraud] X shape: {X.shape}, classes: {len(np.unique(y))}, "
          f"imbalance ratio: {np.sum(y==0)/max(np.sum(y==1),1):.1f}:1")
    return X, y, le.classes_, "Fraud"


def normalize_features(X):
    """Standardize features to zero mean, unit variance."""
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler


def load_dataset_by_name(name):
    """Load dataset by name. Supported: 'HAR', 'Fraud'."""
    loaders = {
        "HAR": load_har_dataset,
        "Fraud": load_fraud_dataset,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loaders[name]()

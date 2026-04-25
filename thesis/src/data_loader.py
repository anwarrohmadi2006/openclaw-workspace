"""
Step 1 — Load & Preprocess
Load datasets, encode categoricals, handle missing values.
Uses sklearn OpenML for reliable data access.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml


def load_har_dataset():
    """
    Load Human Activity Recognition dataset (561 features, ~10K rows, 6 classes).
    Source: UCI HAR via OpenML (id=1478).
    """
    har = fetch_openml('har', version=1, as_frame=True, parser='auto')
    X = har.data.values.astype(np.float64)
    y_raw = har.target.values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Handle missing values
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    print(f"[HAR] X shape: {X.shape}, classes: {len(np.unique(y))}")
    return X, y, le.classes_, "HAR"


def load_fraud_dataset():
    """
    Load Credit Card Fraud detection dataset.
    Uses OpenML creditcard fraud (id=1597) — 284K rows, 30 features, binary.
    """
    try:
        # Try OpenML first (Kaggle credit card fraud)
        fraud = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')
        df = fraud.frame
        # Last column is Class
        y_raw = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values.astype(np.float64)
    except Exception:
        # Fallback: generate synthetic imbalanced dataset
        print("[Fraud] OpenML failed, using synthetic imbalanced data")
        from sklearn.datasets import make_classification
        X, y_raw = make_classification(
            n_samples=50000, n_features=28, n_informative=20,
            n_redundant=4, n_clusters_per_class=2,
            weights=[0.97, 0.03], flip_y=0.01, random_state=42
        )
        y_raw = y_raw.astype(str)

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

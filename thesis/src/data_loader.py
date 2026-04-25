"""
Step 1 — Load & Preprocess
Load datasets, encode categoricals, handle missing values.
Uses sklearn OpenML for reliable data access.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml


def _safe_col_median(X: np.ndarray) -> np.ndarray:
    """
    Compute per-column median, falling back to 0.0 for all-NaN columns.
    Avoids RuntimeWarning from np.nanmedian on fully-NaN slices.
    """
    medians = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        medians[j] = np.median(valid) if len(valid) > 0 else 0.0
    return medians


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

    # Handle missing values — use safe per-column median
    medians = _safe_col_median(X)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(medians, np.where(nan_mask)[1])

    print(f"[HAR] X shape: {X.shape}, classes: {len(np.unique(y))}")
    return X, y, le.classes_, "HAR"


def load_fraud_dataset():
    """
    Load Credit Card Fraud detection dataset.
    Uses OpenML creditcard fraud — 284K rows, 30 features, binary.
    """
    try:
        fraud = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')
        df = fraud.frame
        y_raw = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values.astype(np.float64)
    except Exception:
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

    medians = _safe_col_median(X)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(medians, np.where(nan_mask)[1])

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

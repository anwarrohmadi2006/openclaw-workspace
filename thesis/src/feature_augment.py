"""
Step 6 — Feature Augmentation
Combine normalized features with Theta features.
"""

import numpy as np


def augment_features(X_normalized, theta_features):
    """
    Augment feature matrix with Theta features.

    Args:
        X_normalized: normalized feature matrix (N, d)
        theta_features: theta feature matrix (N, k) — any number of columns

    Returns:
        augmented matrix (N, d + k)

    Raises:
        ValueError: if row counts do not match
    """
    if X_normalized.shape[0] != theta_features.shape[0]:
        raise ValueError(
            f"[Augment] Row mismatch: X has {X_normalized.shape[0]} rows "
            f"but theta has {theta_features.shape[0]} rows. "
            "Ensure theta is computed on the same sample set as X."
        )
    X_aug = np.hstack([X_normalized, theta_features])
    print(
        f"[Augment] {X_normalized.shape[1]}d features + "
        f"{theta_features.shape[1]} theta cols → {X_aug.shape[1]}d augmented"
    )
    return X_aug

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
        theta_features: theta feature matrix (N, k)

    Returns:
        augmented matrix (N, d+k)
    """
    X_aug = np.hstack([X_normalized, theta_features])
    print(f"[Augment] Shape: {X_normalized.shape} + {theta_features.shape} → {X_aug.shape}")
    return X_aug

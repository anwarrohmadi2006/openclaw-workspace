"""
Main Pipeline — run_full_benchmark(dataset_name)
Theta-Augmented Gradient Boosting Embeddings for Tabular Similarity Search
Based on Bar-Natan & van der Veen (2025) — arXiv:2509.18456

Usage:
    python main_pipeline.py              # Run all datasets
    python main_pipeline.py --dataset HAR    # Run specific dataset
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset_by_name, normalize_features
from src.feature_ordering import get_feature_order
from src.braid_word import generate_braid_words, row_to_braid_word
from src.theta_eval import compute_theta_features
from src.sparse_handler import filter_sparse_features
from src.feature_augment import augment_features
from src.model_training import evaluate_model
from src.efficiency_benchmark import run_efficiency_benchmark
from src.sparsity_ablation import ablation_sparsity
from src.visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_scaling_curve,
    plot_theta_distribution,
    plot_memory_comparison,
    save_results_csv,
)

OUTPUT_DIR = Path(__file__).parent / "output"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

# Number of theta columns produced by compute_theta_features() with defaults:
#   t_values=(0.5, 1/3, 2/3) + writhe = 4 columns
N_THETA_COLS = 4


def run_full_benchmark(dataset_name, max_braid_samples=10000):
    """
    Run the complete pipeline for one dataset.

    Steps:
        1. Load & preprocess
        2. Feature ordering (3 strategies)
        3. Braid word generation
        4. Theta evaluation
        5. Sparse filtering
        6. Feature augmentation
        7. Model training & evaluation (4 models)
        8. Efficiency benchmark
        9. Sparsity ablation
        10. Visualization & reporting
    """
    print("=" * 70)
    print(f"  THETA-AUGMENTED GBT PIPELINE — Dataset: {dataset_name}")
    print("=" * 70)
    t0 = time.time()

    # ── STEP 1: Load & Preprocess ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 1: Load & Preprocess")
    print("─" * 50)
    X, y, class_names, ds_name = load_dataset_by_name(dataset_name)
    X_norm, scaler = normalize_features(X)
    n_classes = len(np.unique(y))

    # Stratified split (important for imbalanced datasets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── STEP 2: Feature Ordering ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 2: Feature Ordering (3 strategies)")
    print("─" * 50)
    max_feat = min(50, X.shape[1])
    order_MI, mi_scores = get_feature_order(X_train, y_train, "MI", max_feat)
    order_corr, _ = get_feature_order(X_train, y_train, "CorrCluster", max_feat)
    order_default, _ = get_feature_order(X_train, y_train, "Default", max_feat)

    # Save orderings as artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_MI.npy", order_MI)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_corr.npy", order_corr)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_default.npy", order_default)
    if mi_scores is not None:
        np.save(ARTIFACTS_DIR / f"mi_scores_{ds_name}.npy", mi_scores)

    # ── STEP 3 & 4: Braid Words + Theta ────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 3 & 4: Braid Word Generation + Theta Evaluation")
    print("─" * 50)

    theta_features = {}
    for strategy_name, order in [("MI", order_MI), ("CorrCluster", order_corr), ("Default", order_default)]:
        print(f"\n  Strategy: {strategy_name}")
        braid_words, sample_idx = generate_braid_words(
            X_norm, order, max_samples=max_braid_samples
        )
        theta_matrix = compute_theta_features(braid_words)
        theta_features[strategy_name] = theta_matrix

        # Save artifacts
        np.save(ARTIFACTS_DIR / f"theta_features_{ds_name}_{strategy_name}.npy", theta_matrix)

    # ── STEP 6: Feature Augmentation ───────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 6: Feature Augmentation")
    print("─" * 50)

    # Compute theta for ALL samples (full X_norm) for each strategy.
    # n_theta_cols is derived dynamically from the first computed matrix.
    X_augmented = {}
    n_theta_cols = None  # will be set on first iteration

    for strategy_name in ["MI", "CorrCluster", "Default"]:
        order = {"MI": order_MI, "CorrCluster": order_corr, "Default": order_default}[strategy_name]

        bws_full, sub_idx = generate_braid_words(X_norm, order, max_samples=max_braid_samples)
        theta_full = compute_theta_features(bws_full)

        # Lock in n_theta_cols from first iteration
        if n_theta_cols is None:
            n_theta_cols = theta_full.shape[1]

        # If subsampled, pad remaining rows with zeros
        if theta_full.shape[0] < X_norm.shape[0]:
            theta_padded = np.zeros((X_norm.shape[0], n_theta_cols), dtype=np.float64)
            theta_padded[sub_idx] = theta_full
            theta_full = theta_padded

        X_aug = augment_features(X_norm, theta_full)
        X_augmented[strategy_name] = X_aug

    # ── STEP 7: Model Training & Evaluation ────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 7: Model Training & Evaluation (4 models)")
    print("─" * 50)

    results_classification = []

    # a. Baseline LGBM (no theta)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    r = evaluate_model(X_tr, y_tr, X_te, y_te, n_classes, "LGBM Baseline")
    results_classification.append(r)

    # b, c, d. LGBM + Theta (one per ordering strategy)
    for strategy_name in ["MI", "CorrCluster", "Default"]:
        X_aug = X_augmented[strategy_name]
        X_tr_aug, X_te_aug, y_tr_aug, y_te_aug = train_test_split(
            X_aug, y, test_size=0.2, random_state=42, stratify=y
        )
        r = evaluate_model(
            X_tr_aug, y_tr_aug, X_te_aug, y_te_aug, n_classes,
            f"LGBM + Theta_{strategy_name}"
        )
        results_classification.append(r)

    # ── STEP 8: Efficiency Benchmark ───────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 8: Computational Efficiency Benchmark")
    print("─" * 50)

    efficiency_results = run_efficiency_benchmark(X_norm, order_MI)

    # ── STEP 9: Sparsity Ablation ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 9: Sparsity Ablation")
    print("─" * 50)

    sparse_results = ablation_sparsity(X_norm, order_MI)

    # ── STEP 10: Visualization & Reporting ─────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 10: Visualization & Reporting")
    print("─" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy_comparison(results_classification, OUTPUT_DIR)

    # Best model confusion matrix
    best = max(results_classification, key=lambda r: r["f1_macro"])
    plot_confusion_matrix(
        best["confusion_matrix"],
        class_names if class_names is not None else [str(i) for i in range(n_classes)],
        best["model"], OUTPUT_DIR
    )

    plot_scaling_curve(efficiency_results, OUTPUT_DIR)

    # Theta distribution from MI strategy (first theta col)
    theta_MI = theta_features["MI"]
    plot_theta_distribution(theta_MI[:len(y)], y, class_names, OUTPUT_DIR)

    plot_memory_comparison(sparse_results, OUTPUT_DIR)

    save_results_csv(results_classification, efficiency_results, sparse_results, OUTPUT_DIR)

    # ── SUMMARY ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  COMPLETED: {dataset_name} in {total_time:.1f}s")
    print("=" * 70)
    print("\nResults Summary:")
    summary_df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "F1": f"{r['f1_macro']:.4f}",
        "AUC": f"{r['auc_roc']:.4f}",
        "Recall@10": f"{r['recall_at_10']:.4f}",
    } for r in results_classification])
    print(summary_df.to_string(index=False))

    return results_classification, efficiency_results, sparse_results


def main():
    parser = argparse.ArgumentParser(description="Theta-Augmented GBT Pipeline")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (HAR, Fraud). Default: run all.")
    args = parser.parse_args()

    datasets = ["HAR", "Fraud"] if args.dataset is None else [args.dataset]

    for ds in datasets:
        try:
            run_full_benchmark(ds)
        except Exception as e:
            print(f"\n[ERROR] Dataset {ds} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

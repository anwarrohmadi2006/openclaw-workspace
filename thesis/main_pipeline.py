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
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset_by_name, normalize_features
from src.feature_ordering import get_feature_order
from src.braid_word import generate_braid_words
from src.braid_closure import make_closures_from_braid_words, closure_feature_matrix
from src.theta_eval import compute_theta_features
from src.alexander import compute_exact_theta_features
from src.sparse_handler import filter_sparse_features
from src.feature_augment import augment_features
from src.model_training import evaluate_model
from src.recall_eval import evaluate_all_recall_conditions
from src.ann_baselines import run_ann_baselines
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


def run_full_benchmark(dataset_name, max_braid_samples=10000):
    """
    Run the complete pipeline for one dataset.

    Steps:
        1.  Load & preprocess
        2.  Feature ordering (3 strategies)
        3.  Braid word generation
        3b. Braid closure (trace closure) + closure feature matrix
        4.  Approximate Theta evaluation (linear, 4 cols)
        4b. Exact Theta evaluation (Alexander matrix, 6 cols)
        5.  Sparse filtering
        6.  Feature augmentation (approx + exact Theta)
        7.  Model training & classification evaluation (4 models)
        7b. 3-condition Recall@k evaluation
        7c. ANN baselines (HNSW + Annoy)
        8.  Efficiency benchmark
        9.  Sparsity ablation
        10. Visualization & reporting
    """
    print("=" * 70)
    print(f"  THETA-AUGMENTED GBT PIPELINE — Dataset: {dataset_name}")
    print("=" * 70)
    t0 = time.time()

    # ── STEP 1 ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 1: Load & Preprocess")
    print("─" * 50)
    X, y, class_names, ds_name = load_dataset_by_name(dataset_name)
    X_norm, scaler = normalize_features(X)
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── STEP 2 ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 2: Feature Ordering (3 strategies)")
    print("─" * 50)
    max_feat = min(50, X.shape[1])
    order_MI, mi_scores = get_feature_order(X_train, y_train, "MI", max_feat)
    order_corr, _ = get_feature_order(X_train, y_train, "CorrCluster", max_feat)
    order_default, _ = get_feature_order(X_train, y_train, "Default", max_feat)
    n_strands = len(order_MI)  # number of braid strands = number of features used

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_MI.npy", order_MI)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_corr.npy", order_corr)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_default.npy", order_default)
    if mi_scores is not None:
        np.save(ARTIFACTS_DIR / f"mi_scores_{ds_name}.npy", mi_scores)

    # ── STEP 3 & 3b ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 3 & 3b: Braid Words + Trace Closure")
    print("─" * 50)

    braid_data = {}   # strategy → (braid_words, closures, sample_idx)
    for strat, order in [("MI", order_MI), ("CorrCluster", order_corr), ("Default", order_default)]:
        print(f"\n  Strategy: {strat}")
        bws, idx = generate_braid_words(X_norm, order, max_samples=max_braid_samples)
        closures = make_closures_from_braid_words(bws, n_strands=len(order))
        braid_data[strat] = (bws, closures, idx)

    # ── STEP 4 & 4b ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 4 & 4b: Approx Theta + Exact Theta (Alexander)")
    print("─" * 50)

    theta_approx = {}   # strategy → (N, 4)
    theta_exact = {}    # strategy → (N, 6)

    for strat in ["MI", "CorrCluster", "Default"]:
        bws, closures, idx = braid_data[strat]
        order = {"MI": order_MI, "CorrCluster": order_corr, "Default": order_default}[strat]
        n_strands_strat = len(order)

        print(f"\n  [{strat}] Approx Theta")
        theta_ap = compute_theta_features(bws)
        np.save(ARTIFACTS_DIR / f"theta_approx_{ds_name}_{strat}.npy", theta_ap)

        print(f"  [{strat}] Exact Theta (Alexander matrix)")
        theta_ex = compute_exact_theta_features(bws, n_strands=n_strands_strat)
        np.save(ARTIFACTS_DIR / f"theta_exact_{ds_name}_{strat}.npy", theta_ex)

        # Pad to full dataset size if subsampled
        n_total = X_norm.shape[0]
        for theta_arr, store in [(theta_ap, theta_approx), (theta_ex, theta_exact)]:
            if theta_arr.shape[0] < n_total:
                padded = np.zeros((n_total, theta_arr.shape[1]), dtype=np.float64)
                padded[idx] = theta_arr
                store[strat] = padded
            else:
                store[strat] = theta_arr

    # ── STEP 5 (Sparse filtering — on X_norm) ───────────────────────────
    print("\n" + "─" * 50)
    print("STEP 5: Sparse Feature Filtering")
    print("─" * 50)
    X_filtered, kept_idx = filter_sparse_features(X_norm, threshold=0.9)
    print(f"  Filtered shape: {X_filtered.shape}")

    # ── STEP 6 ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 6: Feature Augmentation")
    print("─" * 50)

    X_augmented = {}  # strategy → augmented matrix (approx + exact theta)
    for strat in ["MI", "CorrCluster", "Default"]:
        theta_combined = np.hstack([theta_approx[strat], theta_exact[strat]])
        X_aug = augment_features(X_norm, theta_combined)
        X_augmented[strat] = X_aug

    # ── STEP 7 ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 7: Model Training & Evaluation (4 models)")
    print("─" * 50)

    results_classification = []

    # a. Baseline LGBM (no theta)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    r_base = evaluate_model(X_tr, y_tr, X_te, y_te, n_classes, "LGBM Baseline")
    results_classification.append(r_base)

    # b, c, d. LGBM + Theta per strategy
    best_aug_model = None
    best_aug_X = None
    best_aug_theta = None
    for strat in ["MI", "CorrCluster", "Default"]:
        X_aug = X_augmented[strat]
        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
            X_aug, y, test_size=0.2, random_state=42, stratify=y
        )
        r = evaluate_model(X_tr_a, y_tr_a, X_te_a, y_te_a, n_classes,
                           f"LGBM + Theta_{strat}")
        results_classification.append(r)
        # Keep best augmented model for recall evaluation
        if best_aug_model is None or r["f1_macro"] > results_classification[-2]["f1_macro"]:
            best_aug_model = r["trained_model"]
            best_aug_X = X_te_a
            best_aug_theta = theta_approx[strat][:len(y_te_a)]
            best_aug_y = y_te_a

    # ── STEP 7b: 3-condition Recall@k ────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 7b: 3-Condition Recall@k Evaluation")
    print("─" * 50)

    # Use raw X_test for Condition B, best_aug_X for A
    X_raw_te = X_te
    recall_results = evaluate_all_recall_conditions(
        model=best_aug_model,
        X=X_raw_te,
        X_aug=best_aug_X,
        theta_features=best_aug_theta,
        y=best_aug_y,
        k_values=(1, 5, 10),
        metrics=("cosine",),
    )

    # ── STEP 7c: ANN Baselines ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 7c: ANN Baselines (HNSW + Annoy)")
    print("─" * 50)

    ann_results = run_ann_baselines(
        X_raw=X_norm,
        X_aug=X_augmented["MI"],
        y=y,
        k_values=(1, 5, 10),
        subsample=5000,
    )

    # ── STEP 8 ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 8: Computational Efficiency Benchmark")
    print("─" * 50)
    efficiency_results = run_efficiency_benchmark(X_norm, order_MI)

    # ── STEP 9 ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 9: Sparsity Ablation")
    print("─" * 50)
    sparse_results = ablation_sparsity(X_norm, order_MI)

    # ── STEP 10 ──────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 10: Visualization & Reporting")
    print("─" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_accuracy_comparison(results_classification, OUTPUT_DIR)

    best = max(results_classification, key=lambda r: r["f1_macro"])
    plot_confusion_matrix(
        best["confusion_matrix"],
        class_names if class_names is not None else [str(i) for i in range(n_classes)],
        best["model"], OUTPUT_DIR
    )
    plot_scaling_curve(efficiency_results, OUTPUT_DIR)
    theta_MI = theta_approx["MI"]
    plot_theta_distribution(theta_MI[:len(y)], y, class_names, OUTPUT_DIR)
    plot_memory_comparison(sparse_results, OUTPUT_DIR)
    save_results_csv(results_classification, efficiency_results, sparse_results, OUTPUT_DIR)

    # ── SUMMARY ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  COMPLETED: {dataset_name} in {total_time:.1f}s")
    print("=" * 70)

    summary_df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "F1": f"{r['f1_macro']:.4f}",
        "AUC": f"{r['auc_roc']:.4f}",
        "Recall@10": f"{r['recall_at_10']:.4f}",
    } for r in results_classification])
    print("\nClassification Results:")
    print(summary_df.to_string(index=False))

    print("\n3-Condition Recall@k Results:")
    recall_df = pd.DataFrame([{
        k: v for k, v in r.items() if k != "metric"
    } for r in recall_results])
    print(recall_df.to_string(index=False))

    print("\nANN Baselines:")
    ann_df = pd.DataFrame([{
        k: (f"{v:.4f}" if isinstance(v, float) else v)
        for k, v in r.items()
    } for r in ann_results])
    print(ann_df.to_string(index=False))

    return results_classification, recall_results, ann_results, efficiency_results, sparse_results


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

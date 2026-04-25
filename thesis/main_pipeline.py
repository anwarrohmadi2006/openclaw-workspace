"""
Main Pipeline — run_full_benchmark(dataset_name)
Theta-Augmented Gradient Boosting Embeddings for Tabular Similarity Search
Based on Bar-Natan & van der Veen (2025), arXiv:2509.18456
"A Fast, Strong, Topologically Meaningful and Fun Knot Invariant"

Usage:
    python main_pipeline.py                      # Run all datasets
    python main_pipeline.py --dataset HAR        # Run specific dataset
    python main_pipeline.py --dataset HAR --exact-theta  # Pakai Theta eksak
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
from src.braid_utils import (
    validate_braid_words_batch,
    n_strands_from_feature_order,
    braid_word_to_closure,
)
from src.theta_eval import compute_theta_features
from src.theta_exact import compute_exact_theta_features
from src.sparse_handler import filter_sparse_features, apply_sparse_filter
from src.feature_augment import augment_features
from src.model_training import evaluate_model
from src.efficiency_benchmark import run_efficiency_benchmark
from src.sparsity_ablation import ablation_sparsity
from src.similarity_search import run_full_similarity_benchmark
from src.visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_scaling_curve,
    plot_theta_distribution,
    plot_memory_comparison,
    plot_similarity_comparison,
    save_results_csv,
)

OUTPUT_DIR = Path(__file__).parent / "output"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"


def run_full_benchmark(dataset_name: str,
                       max_braid_samples: int = 10000,
                       use_exact_theta: bool = False):
    """
    Run the complete pipeline for one dataset.

    Steps:
        1.  Load & preprocess
        2.  Feature ordering (3 strategies)
        3.  Braid word generation
        3b. Braid validation & closure
        4.  Theta evaluation (approx + optionally exact)
        5.  Sparse filtering (AKTIF)
        6.  Feature augmentation
        7.  Model training & evaluation — 4 models, 5-fold CV
        8.  Efficiency benchmark
        9.  Sparsity ablation
        9b. Similarity search benchmark (3 kondisi + 3 baseline)
        10. Visualization & reporting
    """
    print("=" * 70)
    print(f"  THETA-AUGMENTED GBT PIPELINE — Dataset: {dataset_name}")
    if use_exact_theta:
        print("  Mode: THETA EKSAK (Alexander/Burau)")
    print("=" * 70)
    t0 = time.time()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load & Preprocess ──────────────────────────────────────────
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

    # ── STEP 2: Feature Ordering ───────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 2: Feature Ordering (3 strategies)")
    print("─" * 50)
    max_feat = min(50, X.shape[1])
    order_MI, mi_scores = get_feature_order(X_train, y_train, "MI", max_feat)
    order_corr, _ = get_feature_order(X_train, y_train, "CorrCluster", max_feat)
    order_default, _ = get_feature_order(X_train, y_train, "Default", max_feat)

    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_MI.npy", order_MI)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_corr.npy", order_corr)
    np.save(ARTIFACTS_DIR / f"feature_order_{ds_name}_default.npy", order_default)
    if mi_scores is not None:
        np.save(ARTIFACTS_DIR / f"mi_scores_{ds_name}.npy", mi_scores)

    n_strands = n_strands_from_feature_order(order_MI)

    # ── STEP 3: Braid Word Generation ─────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 3: Braid Word Generation")
    print("─" * 50)

    braid_words_MI, _ = generate_braid_words(X_norm, order_MI, max_samples=max_braid_samples)

    # ── STEP 3b: Braid Validation & Closure ───────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 3b: Braid Validation & Closure")
    print("─" * 50)

    val_stats = validate_braid_words_batch(braid_words_MI, n_strands, sample_size=200)
    np.save(ARTIFACTS_DIR / f"braid_validation_{ds_name}.npy", val_stats)

    # Contoh closure untuk 3 sample pertama
    for i in range(min(3, len(braid_words_MI))):
        if braid_words_MI[i]:  # tidak kosong
            cl = braid_word_to_closure(braid_words_MI[i], n_strands)
            print(f"  Sample {i}: {cl}")

    # ── STEP 4: Theta Evaluation ───────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 4: Theta Evaluation")
    print("─" * 50)

    theta_features = {}
    for strategy_name, order in [("MI", order_MI), ("CorrCluster", order_corr), ("Default", order_default)]:
        print(f"\n  Strategy: {strategy_name}")
        bws, _ = generate_braid_words(X_norm, order, max_samples=max_braid_samples)

        # Theta approksimasi (cepat)
        theta_approx = compute_theta_features(bws)

        if use_exact_theta:
            # Theta eksak via Burau/Alexander (lebih lambat)
            print(f"  [Exact] Computing Alexander polynomial features...")
            theta_ex = compute_exact_theta_features(bws, n_strands)
            # Gabungkan approx + exact
            theta_combined = np.hstack([theta_approx, theta_ex])
        else:
            theta_combined = theta_approx

        theta_features[strategy_name] = theta_combined
        np.save(ARTIFACTS_DIR / f"theta_features_{ds_name}_{strategy_name}.npy", theta_combined)

    # ── STEP 5: Sparse Filtering (AKTIF) ──────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 5: Sparse Filtering")
    print("─" * 50)

    X_filtered, order_MI_filtered, kept_idx = apply_sparse_filter(
        X_norm, order_MI, threshold=0.9
    )
    print(f"  X_norm shape: {X_norm.shape} -> X_filtered: {X_filtered.shape}")
    print(f"  Feature order MI filtered: {len(order_MI_filtered)} features")

    # ── STEP 6: Feature Augmentation ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 6: Feature Augmentation")
    print("─" * 50)

    X_augmented = {}
    X_theta_only = {}  # Kondisi B: theta features saja

    for strategy_name in ["MI", "CorrCluster", "Default"]:
        order = {"MI": order_MI, "CorrCluster": order_corr, "Default": order_default}[strategy_name]
        bws_full, _ = generate_braid_words(X_norm, order, max_samples=max_braid_samples)

        theta_full = compute_theta_features(bws_full)
        if use_exact_theta:
            theta_ex_full = compute_exact_theta_features(bws_full, n_strands)
            theta_full = np.hstack([theta_full, theta_ex_full])

        n_theta_cols = theta_full.shape[1]
        if theta_full.shape[0] < X_norm.shape[0]:
            theta_padded = np.zeros((X_norm.shape[0], n_theta_cols))
            theta_padded[:theta_full.shape[0]] = theta_full
            theta_full = theta_padded

        X_augmented[strategy_name] = augment_features(X_norm, theta_full)   # C: raw + theta
        X_theta_only[strategy_name] = theta_full                              # B: theta only

    # ── STEP 7: Model Training & Evaluation (5-fold CV) ───────────────────
    print("\n" + "─" * 50)
    print("STEP 7: Model Training & Evaluation (4 models, 5-fold CV)")
    print("─" * 50)

    results_classification = []
    trained_models = {}  # simpan model untuk similarity search

    # a. Baseline LGBM (raw features)
    r = evaluate_model(X_train, y_train, X_test, y_test, n_classes,
                        "LGBM Baseline", use_cv=True)
    results_classification.append(r)
    trained_models["raw"] = r["trained_model"]

    # b, c, d. LGBM + Theta tiap strategy
    for strategy_name in ["MI", "CorrCluster", "Default"]:
        X_aug = X_augmented[strategy_name]
        X_tr_aug, X_te_aug, y_tr_aug, y_te_aug = train_test_split(
            X_aug, y, test_size=0.2, random_state=42, stratify=y
        )
        r = evaluate_model(
            X_tr_aug, y_tr_aug, X_te_aug, y_te_aug, n_classes,
            f"LGBM + Theta_{strategy_name}", use_cv=True
        )
        results_classification.append(r)
        trained_models[strategy_name] = r["trained_model"]

    # Model untuk theta-only (kondisi B)
    X_theta_mi = X_theta_only["MI"]
    X_tr_th, X_te_th, y_tr_th, y_te_th = train_test_split(
        X_theta_mi, y, test_size=0.2, random_state=42, stratify=y
    )
    r_theta_only = evaluate_model(
        X_tr_th, y_tr_th, X_te_th, y_te_th, n_classes,
        "LGBM Theta-Only", use_cv=False
    )
    trained_models["theta_only"] = r_theta_only["trained_model"]

    # ── STEP 8: Efficiency Benchmark ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 8: Computational Efficiency Benchmark")
    print("─" * 50)
    efficiency_results = run_efficiency_benchmark(X_norm, order_MI)

    # ── STEP 9: Sparsity Ablation ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 9: Sparsity Ablation")
    print("─" * 50)
    sparse_results = ablation_sparsity(X_norm, order_MI)

    # ── STEP 9b: Similarity Search Benchmark ──────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 9b: Similarity Search — 3 Kondisi + 3 Baseline")
    print("─" * 50)

    similarity_results = run_full_similarity_benchmark(
        X_raw=X_norm,
        X_theta_only=X_theta_only["MI"],
        X_augmented=X_augmented["MI"],
        y=y,
        lgbm_raw=trained_models["raw"],
        lgbm_theta=trained_models["theta_only"],
        lgbm_augmented=trained_models["MI"],
        k=10,
        max_samples=5000,
    )

    # ── STEP 10: Visualization & Reporting ────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 10: Visualization & Reporting")
    print("─" * 50)

    plot_accuracy_comparison(results_classification, OUTPUT_DIR)

    best = max(results_classification, key=lambda r: r["f1_macro"])
    plot_confusion_matrix(
        best["confusion_matrix"],
        class_names if class_names is not None else [str(i) for i in range(n_classes)],
        best["model"], OUTPUT_DIR
    )

    plot_scaling_curve(efficiency_results, OUTPUT_DIR)

    theta_MI = theta_features["MI"]
    plot_theta_distribution(theta_MI[:len(y)], y, class_names, OUTPUT_DIR)

    plot_memory_comparison(sparse_results, OUTPUT_DIR)
    plot_similarity_comparison(similarity_results, OUTPUT_DIR)

    save_results_csv(
        results_classification, efficiency_results,
        sparse_results, similarity_results, OUTPUT_DIR
    )

    # ── SUMMARY ────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  COMPLETED: {dataset_name} in {total_time:.1f}s")
    print("=" * 70)
    print("\nClassification Results:")
    summary_df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": f"{r['accuracy']:.4f}",
        "F1": f"{r['f1_macro']:.4f}",
        "AUC": f"{r['auc_roc']:.4f}",
        "Recall@10": f"{r['recall_at_10']:.4f}",
    } for r in results_classification])
    print(summary_df.to_string(index=False))

    print("\nSimilarity Search Results:")
    sim_df = pd.DataFrame([{
        "Method": r["method"],
        "Recall@10": f"{r['recall_at_k']:.4f}",
        "Time(s)": f"{r['build_time_s']:.3f}",
    } for r in similarity_results])
    print(sim_df.to_string(index=False))

    return results_classification, efficiency_results, sparse_results, similarity_results


def main():
    parser = argparse.ArgumentParser(description="Theta-Augmented GBT Pipeline")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (HAR, Fraud). Default: run all.")
    parser.add_argument("--exact-theta", action="store_true",
                        help="Gunakan Theta eksak via Burau/Alexander (lebih lambat tapi lebih akurat).")
    args = parser.parse_args()

    datasets = ["HAR", "Fraud"] if args.dataset is None else [args.dataset]

    for ds in datasets:
        try:
            run_full_benchmark(ds, use_exact_theta=args.exact_theta)
        except Exception as e:
            print(f"\n[ERROR] Dataset {ds} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

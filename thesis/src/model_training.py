"""
Step 7 — Model Training & Evaluation
Train LGBM models dan compute metrics, termasuk 5-fold cross-validation.
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import time


def train_lgbm(X_train, y_train, X_test, y_test, n_classes, random_state=42):
    """
    Train LightGBM classifier.
    Returns: model, predictions, probabilities, train_time
    """
    params = {
        "objective": "multiclass" if n_classes > 2 else "binary",
        "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if n_classes > 2:
        params["num_class"] = n_classes
    else:
        params["objective"] = "binary"

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    start = time.time()
    model = lgb.train(
        params, train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    train_time = time.time() - start

    y_prob = model.predict(X_test)
    if n_classes > 2:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob > 0.5).astype(int)
        y_prob = np.column_stack([1 - y_prob, y_prob])

    return model, y_pred, y_prob, train_time


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute accuracy, F1-macro, AUC-ROC, confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1, "auc_roc": auc, "confusion_matrix": cm}


def compute_recall_at_k(model, X_test, y_test, k=10, metric="cosine"):
    """
    Compute Recall@k using LGBM leaf-index as embedding.
    Self is excluded by position (indices[i][1:k+1]).
    """
    from sklearn.neighbors import NearestNeighbors
    leaf_embed = model.predict(X_test, pred_leaf=True)
    if leaf_embed.ndim == 1:
        leaf_embed = leaf_embed.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(leaf_embed)
    _, indices = nn.kneighbors(leaf_embed)

    recalls = []
    for i in range(len(X_test)):
        neighbors = indices[i][1:k + 1]  # skip self by position
        same_class = np.sum(y_test[neighbors] == y_test[i])
        recalls.append(same_class / k)
    return float(np.mean(recalls))


def evaluate_model_cv(X, y, n_classes, model_name, n_splits=5):
    """
    5-fold stratified cross-validation.
    Returns mean ± std untuk accuracy, F1, AUC.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        _, y_pred, y_prob, _ = train_lgbm(X_tr, y_tr, X_val, y_val, n_classes,
                                           random_state=42 + fold)
        metrics = compute_metrics(y_val, y_pred, y_prob, n_classes)
        cv_results.append(metrics)
        print(f"    Fold {fold + 1}/{n_splits}: "
              f"Acc={metrics['accuracy']:.4f} "
              f"F1={metrics['f1_macro']:.4f} "
              f"AUC={metrics['auc_roc']:.4f}")

    accs = [r["accuracy"] for r in cv_results]
    f1s = [r["f1_macro"] for r in cv_results]
    aucs = [r["auc_roc"] for r in cv_results]

    print(f"    CV Mean: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
          f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} "
          f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def evaluate_model(X_train, y_train, X_test, y_test, n_classes,
                   model_name="model", k=10, use_cv=True):
    """
    Full evaluation: single split + optional 5-fold CV.
    Returns dict dengan semua metrics.
    """
    print(f"\n--- Evaluating: {model_name} ---")
    model, y_pred, y_prob, train_time = train_lgbm(
        X_train, y_train, X_test, y_test, n_classes
    )
    metrics = compute_metrics(y_test, y_pred, y_prob, n_classes)
    recall_k = compute_recall_at_k(model, X_test, y_test, k=k)

    cv_stats = None
    if use_cv:
        print(f"  Running 5-fold CV...")
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        cv_stats = evaluate_model_cv(X_full, y_full, n_classes, model_name)

    result = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "auc_roc": metrics["auc_roc"],
        "recall_at_10": recall_k,
        "train_time_s": train_time,
        "confusion_matrix": metrics["confusion_matrix"],
        "trained_model": model,
        "cv": cv_stats,
    }

    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  F1-macro:  {result['f1_macro']:.4f}")
    print(f"  AUC-ROC:   {result['auc_roc']:.4f}")
    print(f"  Recall@10: {result['recall_at_10']:.4f}")
    print(f"  Train time: {result['train_time_s']:.2f}s")
    if cv_stats:
        print(f"  CV F1: {cv_stats['f1_mean']:.4f} ± {cv_stats['f1_std']:.4f}")

    return result

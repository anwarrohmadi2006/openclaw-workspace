"""
Step 7 — Model Training & Evaluation
Train LGBM models and compute metrics including Recall@10.
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
import time


def train_lgbm(X_train, y_train, X_test, y_test, n_classes, random_state=42):
    """
    Train LightGBM classifier.

    Returns:
        model, predictions, probabilities, train_time
    """
    params = {
        "objective": "multiclass" if n_classes > 2 else "binary",
        "num_class": n_classes if n_classes > 2 else 1,
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

    if n_classes == 2:
        params["objective"] = "binary"
        params.pop("num_class", None)

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

    # Predict
    y_prob = model.predict(X_test)
    if n_classes > 2:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob > 0.5).astype(int)
        y_prob = np.column_stack([1 - y_prob, y_prob])

    return model, y_pred, y_prob, train_time


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute accuracy, F1-macro, AUC-ROC."""
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
    Compute Recall@k for similarity search using leaf indices as embeddings.

    Uses model leaf predictions as embedding vectors, then finds
    k nearest neighbors by cosine similarity.

    Note: NearestNeighbors is fitted with n_neighbors=k+1 to include self,
    then we skip index position 0 (always self) when computing recall.
    """
    # Get leaf indices as embedding
    leaf_embed = model.predict(X_test, pred_leaf=True)
    if leaf_embed.ndim == 1:
        leaf_embed = leaf_embed.reshape(-1, 1)

    # Fit nearest neighbors with k+1 to account for self
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(leaf_embed)
    distances, indices = nn.kneighbors(leaf_embed)

    # Compute recall@k: skip position 0 (self) by slicing, NOT by value filtering
    recalls = []
    for i in range(len(X_test)):
        # indices[i][0] is always self — skip by position, not by value
        neighbors = indices[i][1:k + 1]
        same_class = np.sum(y_test[neighbors] == y_test[i])
        recalls.append(same_class / k)

    return np.mean(recalls)


def evaluate_model(X_train, y_train, X_test, y_test, n_classes, model_name="model", k=10):
    """
    Full evaluation pipeline for one model configuration.

    Returns:
        dict with all metrics
    """
    print(f"\n--- Evaluating: {model_name} ---")
    model, y_pred, y_prob, train_time = train_lgbm(X_train, y_train, X_test, y_test, n_classes)
    metrics = compute_metrics(y_test, y_pred, y_prob, n_classes)
    recall_k = compute_recall_at_k(model, X_test, y_test, k=k)

    result = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "auc_roc": metrics["auc_roc"],
        "recall_at_10": recall_k,
        "train_time_s": train_time,
        "confusion_matrix": metrics["confusion_matrix"],
        "trained_model": model,
    }

    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  F1-macro:  {result['f1_macro']:.4f}")
    print(f"  AUC-ROC:   {result['auc_roc']:.4f}")
    print(f"  Recall@10: {result['recall_at_10']:.4f}")
    print(f"  Train time: {result['train_time_s']:.2f}s")

    return result

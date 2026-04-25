"""
Step 10 — Visualization & Reporting
Generate all plots and CSV outputs for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plot_style():
    """Set publication-quality plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def plot_accuracy_comparison(results_list, output_dir):
    """
    Figure 1: Bar chart — Accuracy/F1/AUC baseline vs all Theta variations.
    """
    setup_plot_style()
    df = pd.DataFrame([
        {"Model": r["model"], "Accuracy": r["accuracy"],
         "F1-macro": r["f1_macro"], "AUC-ROC": r["auc_roc"]}
        for r in results_list
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["Accuracy", "F1-macro", "AUC-ROC"]):
        bars = ax.bar(df["Model"], df[metric], color=sns.color_palette("viridis", len(df)))
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = Path(output_dir) / "plots" / "accuracy_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def plot_confusion_matrix(cm, classes, model_name, output_dir):
    """
    Figure 2: Heatmap — Confusion matrix for best model.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")

    safe_name = model_name.replace(" ", "_").replace("+", "plus")
    path = Path(output_dir) / "plots" / f"confusion_matrix_{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def plot_scaling_curve(efficiency_results, output_dir):
    """
    Figure 3: Line plot — Log-log scaling curve (time vs N).
    """
    setup_plot_style()
    df = pd.DataFrame(efficiency_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time scaling
    for op in df["operation"].unique():
        sub = df[df["operation"] == op]
        axes[0].loglog(sub["N"], sub["time_s"], "o-", label=op, markersize=6)
    axes[0].set_xlabel("N (samples)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Computational Time Scaling")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # Memory scaling
    for op in df["operation"].unique():
        sub = df[df["operation"] == op]
        axes[1].loglog(sub["N"], sub["memory_mb"], "s-", label=op, markersize=6)
    axes[1].set_xlabel("N (samples)")
    axes[1].set_ylabel("Peak Memory (MB)")
    axes[1].set_title("Memory Usage Scaling")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    path = Path(output_dir) / "plots" / "scaling_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def plot_theta_distribution(theta_matrix, y, labels, output_dir):
    """
    Figure 4: Box plot — Theta(t=0.5) distribution per label class.
    """
    setup_plot_style()

    # Safe label resolution: cast y[i] to int to avoid numpy int64 indexing issues
    # and guard against None labels
    data = []
    for i in range(len(y)):
        idx = int(y[i])
        if labels is not None and idx < len(labels):
            class_name = str(labels[idx])
        else:
            class_name = str(idx)
        data.append({
            "Class": class_name,
            "Theta(t=0.5)": float(theta_matrix[i, 0]),
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Class", y="Theta(t=0.5)", ax=ax, palette="Set2")
    ax.set_title("Distribution of Theta(t=0.5) per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Theta(t=0.5)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = Path(output_dir) / "plots" / "theta_distribution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def plot_memory_comparison(sparse_results, output_dir):
    """
    Figure 5: Grouped bar chart — Memory usage with/without sparse optimization.
    """
    setup_plot_style()
    df = pd.DataFrame(sparse_results)

    # Pivot to side-by-side bars
    pivot = df.pivot_table(index="sparsity", columns="filter", values="memory_mb", aggfunc="mean")
    x = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    cols = pivot.columns.tolist()
    colors = sns.color_palette("Set2", len(cols))
    for i, col in enumerate(cols):
        ax.bar(x + i * width, pivot[col].values, width,
               label=col, alpha=0.85, color=colors[i])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{s:.0%}" for s in pivot.index])
    ax.set_xlabel("Sparsity Level")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Usage: With vs Without Sparse Filtering")
    ax.legend(title="Filter")

    plt.tight_layout()
    path = Path(output_dir) / "plots" / "memory_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def save_results_csv(classification_results, efficiency_results, sparse_results, output_dir):
    """Save all results to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Classification results — exclude non-serializable fields
    cls_rows = []
    for r in classification_results:
        cls_rows.append({
            "model": r["model"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "auc_roc": r["auc_roc"],
            "recall_at_10": r["recall_at_10"],
            "train_time_s": r["train_time_s"],
        })
    pd.DataFrame(cls_rows).to_csv(out / "results_classification.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_classification.csv'}")

    # Efficiency results
    pd.DataFrame(efficiency_results).to_csv(out / "results_efficiency.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_efficiency.csv'}")

    # Sparse results — exclude non-scalar fields if any
    sparse_rows = [
        {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}
        for row in sparse_results
    ]
    pd.DataFrame(sparse_rows).to_csv(out / "results_sparse.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_sparse.csv'}")

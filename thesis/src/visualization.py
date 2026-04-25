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
    """Figure 1: Bar chart — Accuracy/F1/AUC comparison."""
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
    """Figure 2: Confusion matrix heatmap."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    safe_name = model_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
    path = Path(output_dir) / "plots" / f"confusion_matrix_{safe_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def plot_scaling_curve(efficiency_results, output_dir):
    """Figure 3: Log-log scaling curve."""
    setup_plot_style()
    df = pd.DataFrame(efficiency_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for op in df["operation"].unique():
        sub = df[df["operation"] == op]
        axes[0].loglog(sub["N"], sub["time_s"], "o-", label=op, markersize=6)
        axes[1].loglog(sub["N"], sub["memory_mb"], "s-", label=op, markersize=6)
    for ax, title, ylabel in zip(axes,
        ["Computational Time Scaling", "Memory Usage Scaling"],
        ["Time (s)", "Peak Memory (MB)"]):
        ax.set_xlabel("N (samples)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    path = Path(output_dir) / "plots" / "scaling_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def plot_theta_distribution(theta_matrix, y, labels, output_dir):
    """Figure 4: Boxplot Theta(t=0.5) distribution per class."""
    setup_plot_style()
    data = []
    for i in range(len(y)):
        idx = int(y[i])
        class_name = str(labels[idx]) if (labels is not None and idx < len(labels)) else str(idx)
        data.append({"Class": class_name, "Theta(t=0.5)": float(theta_matrix[i, 0])})
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
    """Figure 5: Grouped bar chart memory vs sparsity."""
    setup_plot_style()
    df = pd.DataFrame(sparse_results)
    pivot = df.pivot_table(index="sparsity", columns="filter", values="memory_mb", aggfunc="mean")
    x = np.arange(len(pivot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = pivot.columns.tolist()
    colors = sns.color_palette("Set2", len(cols))
    for i, col in enumerate(cols):
        ax.bar(x + i * width, pivot[col].values, width, label=col, alpha=0.85, color=colors[i])
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


def plot_similarity_comparison(similarity_results, output_dir):
    """
    Figure 6: Bar chart — Recall@k untuk 3 kondisi + 3 baseline.
    Ini adalah figure utama paper untuk similarity search.
    """
    setup_plot_style()
    df = pd.DataFrame(similarity_results)

    # Pisahkan kondisi (LGBM-based) dan baseline
    cond_mask = df["method"].str.contains("LGBM")
    df_cond = df[cond_mask]
    df_base = df[~cond_mask]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel kiri: 3 kondisi LGBM
    colors_cond = sns.color_palette("Blues_d", len(df_cond))
    bars = axes[0].bar(df_cond["method"], df_cond["recall_at_k"],
                       color=colors_cond, alpha=0.9)
    axes[0].set_title("Recall@10: 3 Kondisi Metode Kami", fontsize=13)
    axes[0].set_ylabel("Recall@10")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, df_cond["recall_at_k"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel kanan: semua method berdampingan
    all_methods = pd.concat([df_cond, df_base], ignore_index=True)
    colors_all = ["#2196F3" if "LGBM" in m else "#FF9800"
                  for m in all_methods["method"]]
    bars2 = axes[1].bar(all_methods["method"], all_methods["recall_at_k"],
                        color=colors_all, alpha=0.85)
    axes[1].set_title("Recall@10: Metode vs Baseline", fontsize=13)
    axes[1].set_ylabel("Recall@10")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35)
    for bar, val in zip(bars2, all_methods["recall_at_k"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    # Legenda warna
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(facecolor="#2196F3", label="Metode Kami (LGBM+Theta)"),
        Patch(facecolor="#FF9800", label="Baseline ANN"),
    ])

    plt.tight_layout()
    path = Path(output_dir) / "plots" / "similarity_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
    return path


def save_results_csv(classification_results, efficiency_results,
                     sparse_results, similarity_results, output_dir):
    """Save all results to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Classification
    cls_rows = []
    for r in classification_results:
        row = {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "auc_roc": r["auc_roc"],
            "recall_at_10": r["recall_at_10"],
            "train_time_s": r["train_time_s"],
        }
        if r.get("cv"):
            row.update({
                "cv_accuracy_mean": r["cv"]["accuracy_mean"],
                "cv_accuracy_std": r["cv"]["accuracy_std"],
                "cv_f1_mean": r["cv"]["f1_mean"],
                "cv_f1_std": r["cv"]["f1_std"],
                "cv_auc_mean": r["cv"]["auc_mean"],
                "cv_auc_std": r["cv"]["auc_std"],
            })
        cls_rows.append(row)
    pd.DataFrame(cls_rows).to_csv(out / "results_classification.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_classification.csv'}")

    pd.DataFrame(efficiency_results).to_csv(out / "results_efficiency.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_efficiency.csv'}")

    sparse_rows = [
        {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}
        for row in sparse_results
    ]
    pd.DataFrame(sparse_rows).to_csv(out / "results_sparse.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_sparse.csv'}")

    sim_rows = [
        {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        for r in similarity_results
    ]
    pd.DataFrame(sim_rows).to_csv(out / "results_similarity.csv", index=False)
    print(f"[CSV] Saved: {out / 'results_similarity.csv'}")

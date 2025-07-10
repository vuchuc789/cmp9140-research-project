import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def evaluate(
    model_name: str,
):
    model_dir = "model"
    history_path = f"{model_dir}/{model_name}_history.npy"

    print("Loading training result...")
    with open(history_path, "rb") as f:
        batched_train_loss = np.load(f)
        batched_benign_test_loss = np.load(f)
        batched_anomalous_test_loss = np.load(f)
        batched_auc = np.load(f)
        train_loss = np.load(f)
        benign_loss = np.load(f)
        anomalous_loss = np.load(f)

    y_scores = np.concatenate([benign_loss, anomalous_loss])
    y_true = np.concatenate([np.zeros(len(benign_loss)), np.ones(len(anomalous_loss))])

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # f1_scores canbe nan due to dividing by 0
    f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]

    y_pred = (y_scores >= best_threshold).astype(int)
    best_accuracy = np.mean(y_pred == y_true)

    best_roc_idx = np.argmin(np.abs(roc_thresholds - best_threshold))

    train_loss_mean = np.mean(train_loss)
    train_loss_std = np.std(train_loss)

    print("Showing result...\n")

    print(f"Avg Training Loss (from training history): {batched_train_loss[-1]:>7f}")
    print(
        f"Avg Testing  Loss (from training history): {batched_benign_test_loss[-1]:>7f}\n"
    )

    print(f"Mean (μ) of Training Loss (MSE): {train_loss_mean:.4f}")
    print(f"Std  (σ) of Training Loss (MSE): {train_loss_std:.4f}\n")
    print(
        f"Threshold: {best_threshold:.4f} = μ + σ × {(best_threshold - train_loss_mean) / train_loss_std:.4f} (selected based on F1-score)\n"
    )
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall   : {best_recall:.4f}")
    print(f"F1-score : {best_f1:.4f}")
    print(f"Accuracy : {best_accuracy:.4f}")
    print(f"AUC      : {roc_auc:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot training and validation losses
    ax1.plot(
        batched_train_loss,
        label="Training Loss",
        color="royalblue",
        linewidth=2,
    )
    ax1.plot(
        batched_benign_test_loss,
        label="Validation Loss",
        color="tomato",
        linewidth=2,
        linestyle="--",
    )
    ax1.plot(
        batched_anomalous_test_loss,
        label="Anomalous Loss",
        color="crimson",
        linewidth=2,
        linestyle=":",
    )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (MSE)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Second Y-axis for AUC
    ax2 = ax1.twinx()
    ax2.plot(
        batched_auc,
        label="AUC",
        color="mediumseagreen",
        linewidth=2,
        linestyle="-.",
    )
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_ylim(0.0, 1.05)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title(
        "Autoencoder Learning Curve + Anomaly Loss + AUC (from training history)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    # Histogram with threshold line
    plt.figure(figsize=(8, 5))
    plt.hist(
        benign_loss,
        bins=100,
        alpha=0.6,
        label="Benign",
        color="mediumseagreen",
        density=True,
    )
    plt.hist(
        anomalous_loss,
        bins=100,
        alpha=0.6,
        label="Anomaly",
        color="crimson",
        density=True,
    )

    # Highlight best threshold
    plt.axvline(
        best_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({best_threshold:.4f})",
    )

    plt.title("Reconstruction Error Distribution", fontsize=14)
    plt.xlabel("Reconstruction Loss", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ROC Curve (threshold not directly plotted, but let's annotate it)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

    # Optional: Mark the operating point based on best threshold
    plt.scatter(
        fpr[best_roc_idx],
        tpr[best_roc_idx],
        color="red",
        label=f"Threshold ({best_threshold:.4f})",
        zorder=5,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Anomaly Detection)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Threshold vs Precision, Recall, F1-score
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f1_scores[:-1], label="F1-score", linestyle="--")

    # Add vertical line at best threshold
    plt.axvline(
        best_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({best_threshold:.4f})",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision / Recall / F1-score ")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from app.model.train import init_model, test_loop


def evaluate(
    model_name: str,
):
    model_dir = "model"
    history_path = f"{model_dir}/{model_name}_history.npy"

    print("Loading training result...\n")
    with open(history_path, "rb") as f:
        batched_train_loss = np.load(f)
        batched_benign_test_loss = np.load(f)
        batched_anomalous_test_loss = np.load(f)
        batched_auc = np.load(f)

    (
        model,
        loss_fn,
        optimizer,
        last_epoch,
        device,
        train_loader,
        benign_test_loader,
        anomalous_test_loader,
    ) = init_model(
        model_name=model_name,
        model_type="ae",
        loss_type="mae",
        optimizer_type="sgdm",
    )

    print("Calculating loss...\n")
    train_loss = test_loop(train_loader, model, loss_fn, device)
    print()
    benign_loss = test_loop(benign_test_loader, model, loss_fn, device)
    print()
    anomalous_loss = test_loop(anomalous_test_loader, model, loss_fn, device)
    print()

    print("Calculating metrics...\n")
    train_loss_mean = np.mean(train_loss)
    train_loss_std = np.std(train_loss)

    y_scores = np.concatenate([benign_loss, anomalous_loss])
    y_true = np.concatenate([np.zeros(len(benign_loss)), np.ones(len(anomalous_loss))])

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    # f1_scores canbe nan due to dividing by 0
    f1_scores = np.nan_to_num(f1_scores)

    # Shape: (num_thresholds, num_samples)
    # pred_matrix = (y_scores.reshape(1, -1) >= thresholds.reshape(-1, 1)).astype(int)
    # Shape: (num_thresholds,)
    # accuracy = np.mean(pred_matrix == y_true.reshape(1, -1), axis=1)

    # accuracy = np.zeros_like(thresholds)
    # for i, t in enumerate(thresholds):
    #     y_pred = (y_scores >= t).astype(int)
    #     accuracy[i] = np.mean(y_pred == y_true)
    #     if (i + 1) % 1000 == len(thresholds) % 1000:
    #         print(f"[{i + 1:>4d}/{len(thresholds):>4d}] accuracy: {accuracy[i]:>7f}")
    # print()

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    # best_accuracy = accuracy[best_idx]

    # y_pred = pred_matrix[best_idx]
    y_pred = (y_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    best_accuracy = np.mean(y_pred == y_true)

    best_roc_idx = np.argmin(np.abs(roc_thresholds - best_threshold))

    # print("Model information:\n")
    # print(model)
    # print()
    # print(optimizer)
    # print()

    print("Showing result...\n")

    patience = 10
    delta = 1e-3
    if len(batched_benign_test_loss) > patience:
        diffs = batched_benign_test_loss[:-1] - batched_benign_test_loss[1:]
        window_sum = np.sum(diffs[:patience])
        for i in range(len(diffs) - patience):
            if window_sum >= 0 and window_sum < delta:
                print(
                    f"Convergence Epoch (patience={patience}, delta={delta}): {i + 1}"
                )
                break

            window_sum -= diffs[i]
            window_sum += diffs[i + patience]

    print(f"Best AUC epoch: {np.argmax(batched_auc) + 1}")
    print(f"Current epoch: {last_epoch + 1} (round: {last_epoch})\n")

    print(f"Avg Training Loss (from training history): {batched_train_loss[-1]:>7f}")
    print(
        f"Avg Testing  Loss (from training history): {batched_benign_test_loss[-1]:>7f}\n"
    )

    print(f"Mean (μ) of Training Loss (MSE): {train_loss_mean:.4f}")
    print(f"Std  (σ) of Training Loss (MSE): {train_loss_std:.4f}\n")

    print(
        f"Threshold: {best_threshold:.4f} = μ + σ × {(best_threshold - train_loss_mean) / train_loss_std:.4f} (selected based on max F1-score)\n"
    )

    print(f"True Negatives : {tn:>5d}")
    print(f"False Positives: {fp:>5d}")
    print(f"False Negatives: {fn:>5d}")
    print(f"True Positives : {tp:>5d}\n")

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

    ax1.set_xlabel("Epoch or Round", fontsize=12)
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
        linewidth=1,
        label=f"Threshold ({best_threshold:.4f})",
    )

    plt.title("Reconstruction Error Distribution", fontsize=14)
    plt.xlabel("Reconstruction Loss", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Benign", "Anomaly"],
        yticklabels=["Benign", "Anomaly"],
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
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
    plt.plot(thresholds, f1_scores[:-1], label="F1-score")
    # plt.plot(thresholds, accuracy, label="Accuracy", linestyle="--")

    # Add vertical line at best threshold
    plt.axvline(
        best_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Threshold ({best_threshold:.4f})",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision / Recall / F1-score / Accuracy ")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

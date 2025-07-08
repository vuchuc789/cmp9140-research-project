import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from app.data.data import load_data
from app.model.model import Autoencoder


def calculate_loss(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
):
    model = model.to("cpu")
    losses = np.zeros(len(dataloader.dataset))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader.dataset):
            pred = model(x)
            loss = loss_fn(pred, x)
            losses[i] = loss.numpy()
            if (i + 1) % 1000 == len(dataloader.dataset) % 1000:
                print(
                    f"[{i + 1:>3d}/{len(dataloader.dataset):>3d}] loss: {loss.item():>7f}"
                )

    return losses


def evaluate(model_name: str):
    # use a random value for compatibility
    # further calculations won't handle in batches
    batch_size = 1024

    model_dir = "model"
    params_path = f"{model_dir}/{model_name}_params.pth"
    history_path = f"{model_dir}/{model_name}_history.npy"
    loss_path = f"{model_dir}/{model_name}_loss.npy"

    print("Loading data...\n")
    train_loader, test_loader, anomaly_loader = load_data(batch_size)

    model = Autoencoder()
    loss_fn = nn.MSELoss()

    print("Loading model...\n")
    model.load_state_dict(torch.load(params_path, weights_only=True))
    with open(history_path, "rb") as f:
        batched_train_loss = np.load(f)
        batched_test_loss = np.load(f)

    print("Evaluating anomaly detection performace...")

    if os.path.exists(loss_path):
        print("\nLoading loss from cache...")
        with open(loss_path, "rb") as f:
            train_loss = np.load(f)
            benign_loss = np.load(f)
            anomalous_loss = np.load(f)
    else:
        print("\nCalculating training loss (without batching)...")
        train_loss = calculate_loss(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
        )
        print("\nCalculating benign loss (without batching)...")
        benign_loss = calculate_loss(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
        )
        print("\nCalculating anomalous loss (without batching)...")
        anomalous_loss = calculate_loss(
            dataloader=anomaly_loader,
            model=model,
            loss_fn=loss_fn,
        )

        print("\nSaving loss to cache...")
        with open(loss_path, "wb") as f:
            np.save(f, train_loss)
            np.save(f, benign_loss)
            np.save(f, anomalous_loss)

    y_scores = np.concatenate([benign_loss, anomalous_loss])
    y_true = np.concatenate([np.zeros(len(benign_loss)), np.ones(len(anomalous_loss))])

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    # f1_scores = (
    #     2 * (precision * recall) / (precision + recall + 1e-10)  # prevent dividing by 0
    # )

    accuracy = np.zeros_like(thresholds)
    print("\nCalculating accuracy...\n")
    for i, t in enumerate(thresholds):
        y_pred = (y_scores >= t).astype(int)
        accuracy[i] = np.mean(y_pred == y_true)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    best_accuracy = accuracy[best_idx]
    best_roc_idx = np.argmin(np.abs(roc_thresholds - best_threshold))

    train_loss_mean = np.mean(train_loss)
    train_loss_std = np.std(train_loss)

    print("Showing result...\n")

    print(f"Avg training loss: {batched_train_loss[-1]:>7f}")
    print(f"Avg testing loss: {batched_test_loss[-1]:>7f}\n")

    print(f"Training Loss (MSE) Mean (μ): {train_loss_mean:.4f}")
    print(f"Training Loss (MSE) Std (σ): {train_loss_std:.4f}\n")
    print(
        f"Best threshold (based on F1-score): {best_threshold:.4f} = μ + σ × {(best_threshold - train_loss_mean) / train_loss_std:.4f}\n"
    )
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"F1-score: {best_f1:.4f}")
    print(f"Accuracy: {best_accuracy:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        batched_train_loss,
        label="Training Loss",
        color="royalblue",
        linewidth=2,
    )
    plt.plot(
        batched_test_loss,
        label="Validation Loss",
        color="tomato",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Autoencoder Learning Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
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

    # Threshold vs Precision, Recall, F1, Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f1_scores[:-1], label="F1-score")
    plt.plot(thresholds, accuracy, label="Accuracy", linestyle="--")

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
    plt.title("Threshold vs Precision / Recall / F1 / Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

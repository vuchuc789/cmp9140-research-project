import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from app.data.data import load_data
from app.model.model import Autoencoder


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
):
    loss_sum = 0
    num_batches = len(dataloader)

    # turn into train mode
    model.train()
    for batch, (x, _) in enumerate(dataloader):
        # copy to gpu
        x = x.to(device)

        # reset gradients
        optimizer.zero_grad()

        # feed forward
        pred = model(x)
        loss = loss_fn(pred, x)  # predict input itself

        # backpropagation
        loss.backward()
        optimizer.step()

        # aggregate loss
        loss_sum += loss.item()

        if (batch + 1) % 100 == num_batches % 100:
            print(f"[{batch + 1:>3d}/{num_batches:>3d}] loss: {loss.item():>7f}")

    loss_avg = loss_sum / num_batches
    print(f"Train loss: {loss_avg:>8f}")

    return loss_avg


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str = "cpu",
):
    loss_sum = 0
    num_batches = len(dataloader)

    # turn into test mode
    model.eval()
    with torch.no_grad():  # temporarily disable auto gradient
        for x, _ in dataloader:
            x = x.to(device)

            # test
            pred = model(x)
            loss = loss_fn(pred, x)  # predict input itself

            loss_sum += loss.item()

    loss_avg = loss_sum / num_batches
    print(f"Test loss: {loss_avg:>8f}")

    return loss_avg


def cal_loss(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
):
    model = model.to("cpu")
    losses = np.array([])

    model.eval()
    with torch.no_grad():
        for x, _ in dataloader.dataset:
            pred = model(x)
            loss = loss_fn(pred, x)
            losses = np.append(losses, loss.numpy())

    return losses


def train():
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 50
    regularization_rate = 1e-6

    model_dir = "model"
    model_file = "ddos_ae"
    params_path = f"{model_dir}/{model_file}_params.pth"
    history_path = f"{model_dir}/{model_file}_history.npy"

    # Get supported compute device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    print("Loading data...\n")
    train_loader, test_loader, anomaly_loader = load_data(batch_size)

    model = Autoencoder().to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=regularization_rate,
    )

    train_loss = np.array([])
    test_loss = np.array([])

    load_model = os.path.exists(params_path) and os.path.exists(history_path)

    if load_model:
        print("Loading model...\n")
        model.load_state_dict(torch.load(params_path, weights_only=True))
        with open(history_path, "rb") as f:
            train_loss = np.load(f)
            test_loss = np.load(f)

        test_loop(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )
    else:
        print("Training model...")
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}")

            loss = train_loop(
                dataloader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            train_loss = np.append(train_loss, loss)
            loss = test_loop(
                dataloader=test_loader,
                model=model,
                loss_fn=loss_fn,
                device=device,
            )
            test_loss = np.append(test_loss, loss)
            print()

        print("Saving model...\n")
        torch.save(model.state_dict(), params_path)
        with open(history_path, "wb") as f:
            np.save(f, train_loss)
            np.save(f, test_loss)

    print("Showing result...")

    plt.plot(train_loss, label="Training Loss")
    plt.plot(test_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    benign_loss = cal_loss(
        dataloader=test_loader,
        model=model,
        loss_fn=loss_fn,
    )
    anomalous_loss = cal_loss(
        dataloader=anomaly_loader,
        model=model,
        loss_fn=loss_fn,
    )

    plt.hist(benign_loss, bins=100, alpha=0.5)
    plt.hist(anomalous_loss, bins=100, alpha=0.5)
    plt.show()

    y_scores = np.concatenate([benign_loss, anomalous_loss])
    y_true = np.concatenate([np.zeros(len(benign_loss)), np.ones(len(anomalous_loss))])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Anomaly Detection)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Assume y_true and y_scores from earlier
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = np.array([])
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        accuracy = np.append(accuracy, acc)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    best_accuracy = accuracy[best_idx]

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f1_scores[:-1], label="F1-score")
    plt.plot(thresholds, accuracy, label="Accuracy", linestyle="--")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision / Recall / F1 / Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

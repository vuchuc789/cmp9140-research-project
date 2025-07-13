import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from app.data.data import load_data
from app.model.model import Autoencoder, init_weights
from app.utils.print import verbose_print


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    verbose=True,
):
    num_batches = len(dataloader)
    losses = []

    model.train()
    for batch, x in enumerate(dataloader):
        # copy to gpu
        x = x.to(device)

        # reset gradients
        optimizer.zero_grad()

        # feed forward
        pred = model(x)
        loss = loss_fn(pred, x)  # shape: (batch_size, input_dim)
        loss = loss.mean(dim=1)  # shape: (batch_size)

        # aggregate loss
        # the origin loss is still in the graph
        losses.append(loss.detach().cpu().numpy())

        loss = loss.mean()  # scalar

        # backpropagation
        loss.backward()
        optimizer.step()

        if (batch + 1) % 100 == num_batches % 100:
            verbose_print(
                verbose, f"[{batch + 1:>4d}/{num_batches:>4d}] loss: {loss.item():>7f}"
            )

    return np.concat(losses)


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str = "cpu",
    verbose=True,
):
    num_batches = len(dataloader)
    losses = []

    # turn into test mode
    model.eval()
    with torch.no_grad():  # temporarily disable auto gradient
        for batch, x in enumerate(dataloader):
            x = x.to(device)

            # test
            pred = model(x)
            loss = loss_fn(pred, x)  # shape: (batch_size, input_dim)
            loss = loss.mean(dim=1)  # shape: (batch_size)

            losses.append(loss.cpu().numpy())

            if (batch + 1) % 100 == num_batches % 100:
                verbose_print(
                    verbose,
                    f"[{batch + 1:>3d}/{num_batches:>3d}] loss: {loss.mean().item():>7f}",
                )

    return np.concat(losses)


def get_model(model_type="ae"):
    match model_type:
        case "ae":
            return Autoencoder()


def init_model(
    model_name: str = None,
    model_type="ae",
    loss_type="mae",
    batch_size=64,
    learning_rate=1e-3,
    regularization_rate=0,
    partition="none",
    partition_id=0,
    verbose=False,
):
    if model_name is not None:
        model_dir = "model"
        model_path = f"{model_dir}/{model_name}_model.pth"

    # Get supported compute device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    verbose_print(verbose, f"Using device: {device}\n")

    verbose_print(verbose, "Loading data...\n")
    train_loader, benign_test_loader, anomalous_test_loader = load_data(
        batch_size, partition, partition_id
    )

    model = get_model(model_type).to(device)

    match loss_type:
        case "mae":
            loss_fn = nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=regularization_rate,
    )

    current_epoch: int = 0
    if model_name is not None and os.path.exists(model_path):
        verbose_print(verbose, "Loading checkpoint...\n")
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "epoch" in checkpoint:
            current_epoch = checkpoint["epoch"]

    else:
        model.apply(init_weights)  # apply init weights if no checkpoint

    return (
        model,
        loss_fn,
        optimizer,
        current_epoch,
        device,
        train_loader,
        benign_test_loader,
        anomalous_test_loader,
    )


def fit_model(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    current_epoch=0,
    device="cpu",
    train_loader: DataLoader = None,
    benign_test_loader: DataLoader = None,
    anomalous_test_loader: DataLoader = None,
    model_name: str = None,
    epochs=1,
    verbose=False,
):
    model_dir = "model"
    model_path = f"{model_dir}/{model_name}_model.pth"
    history_path = f"{model_dir}/{model_name}_history.npy"

    # default value
    train_loss = -1
    benign_test_loss = -1
    anomalous_test_loss = -1
    auc = -1

    verbose_print(verbose, "Fitting model...\n")
    for i in range(epochs):
        epoch = current_epoch + i
        verbose_print(verbose, f"Epoch: {epoch + 1}")

        if train_loader is not None:
            train_losses = train_loop(
                dataloader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                verbose=verbose,
            )
            train_loss = np.mean(train_losses)
            verbose_print(verbose)

        if benign_test_loader is not None:
            benign_test_losses = test_loop(
                dataloader=benign_test_loader,
                model=model,
                loss_fn=loss_fn,
                device=device,
                verbose=verbose,
            )
            benign_test_loss = np.mean(benign_test_losses)
            verbose_print(verbose)

        if anomalous_test_loader is not None:
            anomalous_test_losses = test_loop(
                dataloader=anomalous_test_loader,
                model=model,
                loss_fn=loss_fn,
                device=device,
                verbose=verbose,
            )
            anomalous_test_loss = np.mean(anomalous_test_losses)
            verbose_print(verbose)

        if train_loader is not None:
            verbose_print(verbose, f"Train Loss          : {train_loss:>8f}")
        if benign_test_loader is not None:
            verbose_print(verbose, f"Benign Test Loss    : {benign_test_loss:>8f}")
        if anomalous_test_loader is not None:
            verbose_print(verbose, f"Anomalous Test Loss : {anomalous_test_loss:>8f}")

        # calculating auc
        if benign_test_loader is not None and anomalous_test_loader is not None:
            y_score = np.concat(
                (
                    benign_test_losses,
                    anomalous_test_losses,
                ),
                axis=0,
            )
            y_true = np.concat(
                (
                    np.zeros_like(benign_test_losses),
                    np.ones_like(anomalous_test_losses),
                ),
                axis=0,
            )
            auc = roc_auc_score(y_true, y_score)
            verbose_print(verbose, f"AUC                 : {auc:>8f}")
        verbose_print(verbose)

        if model_name is not None and train_loader is not None:
            epoch_path = f"{model_dir}/{model_name}_model_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                epoch_path,
            )
            save_history(
                history_path,
                np.array([train_loss]),
                np.array([benign_test_loss]),
                np.array([anomalous_test_loss]),
                np.array([auc]),
            )

    if model_name is not None and train_loader is not None:
        torch.save(
            {
                "epoch": current_epoch + epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_path,
        )

    return (
        float(train_loss),
        float(benign_test_loss),
        float(anomalous_test_loss),
        float(auc),
    )


def save_history(
    history_path: str,
    train_loss: np.ndarray = -np.ones(1),
    benign_test_loss: np.ndarray = -np.ones(1),
    anomalous_test_loss: np.ndarray = -np.ones(1),
    auc: np.ndarray = -np.ones(1),
):
    history = [
        train_loss,
        benign_test_loss,
        anomalous_test_loss,
        auc,
    ]

    previous_history = []
    for i in range(len(history)):
        previous_history.append(-np.ones(1))

    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            # Retrieve history
            for i in range(len(previous_history)):
                previous_history[i] = np.load(f)

    # Write order is matter
    with open(history_path, "wb") as f:
        # Append results to the history
        for i in range(len(history)):
            if np.array_equal(previous_history[i], -np.ones(1)):
                np.save(f, history[i])
            elif np.array_equal(history[i], -np.ones(1)):
                np.save(f, previous_history[i])
            else:
                np.save(f, np.concat((previous_history[i], history[i]), axis=0))

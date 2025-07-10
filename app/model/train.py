import os
from glob import glob

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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
    num_batches = len(dataloader)
    losses = []

    # turn into train mode
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
            print(f"[{batch + 1:>3d}/{num_batches:>3d}] loss: {loss.item():>7f}")

    return np.concat(losses)


def test_loop(
    benign_dataloader: DataLoader,
    anomalous_dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str = "cpu",
):
    benign_loss = []
    anomalous_loss = []

    # turn into test mode
    model.eval()
    with torch.no_grad():  # temporarily disable auto gradient
        for batch, x in enumerate(benign_dataloader):
            x = x.to(device)

            # test
            pred = model(x)
            loss = loss_fn(pred, x)  # shape: (batch_size, input_dim)
            loss = loss.mean(dim=1)  # shape: (batch_size)

            benign_loss.append(loss.cpu().numpy())

            if (batch + 1) % 100 == len(benign_dataloader) % 100:
                print(
                    f"[{batch + 1:>3d}/{len(benign_dataloader):>3d}] loss: {loss.mean().item():>7f}"
                )

        print()

        for batch, x in enumerate(anomalous_dataloader):
            x = x.to(device)

            # test
            pred = model(x)
            loss = loss_fn(pred, x)  # shape: (batch_size, input_dim)
            loss = loss.mean(dim=1)  # shape: (batch_size)

            anomalous_loss.append(loss.cpu().numpy())
            if (batch + 1) % 100 == len(anomalous_dataloader) % 100:
                print(
                    f"[{batch + 1:>3d}/{len(anomalous_dataloader):>3d}] loss: {loss.mean().item():>7f}"
                )

    benign_loss = np.concat(benign_loss)
    anomalous_loss = np.concat(anomalous_loss)

    y_score = np.concat(
        (benign_loss, anomalous_loss),
        axis=0,
    )
    y_true = np.concat(
        (np.zeros_like(benign_loss), np.ones_like(anomalous_loss)),
        axis=0,
    )
    auc = roc_auc_score(y_true, y_score)

    return benign_loss, anomalous_loss, auc


def train(
    model_name: str = None,
    model_type="ae",
    loss_type="mae",
    batch_size=64,
    learning_rate=1e-3,
    epochs=5,
):
    if model_name is not None:
        model_dir = "model"
        model_path = f"{model_dir}/{model_name}_model.pth"
        history_path = f"{model_dir}/{model_name}_history.npy"

        if not os.path.exists(model_path):
            for file in glob(f"{model_dir}/{model_name}*"):
                print(f"Delete {file}...")
                os.remove(file)
            print()

    # Get supported compute device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    print("Loading data...\n")
    train_loader, benign_test_loader, anomalous_test_loader = load_data(batch_size)

    match model_type:
        case "ae":
            model = Autoencoder().to(device)

    match loss_type:
        case "mae":
            loss_fn = nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0
    if model_name is not None and os.path.exists(model_path):
        print("Loading checkpoint...\n")
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

    train_losses = np.zeros(epochs)
    benign_test_losses = np.zeros(epochs)
    anomalous_test_losses = np.zeros(epochs)
    auc = np.zeros(epochs)

    print("Training model...")
    for i in range(epochs):
        print(f"Epoch: {epoch + i + 1}")
        train_loss = train_loop(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        print()
        benign_test_loss, anomalous_test_loss, auc[i] = test_loop(
            benign_dataloader=benign_test_loader,
            anomalous_dataloader=anomalous_test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )
        print()

        train_losses[i] = np.mean(train_loss)
        benign_test_losses[i] = np.mean(benign_test_loss)
        anomalous_test_losses[i] = np.mean(anomalous_test_loss)

        print(f"Train Loss: {train_losses[i]:>8f}")
        print(f"Benign Test Loss: {benign_test_losses[i]:>8f}")
        print(f"Anomalous Test Loss: {anomalous_test_losses[i]:>8f}")
        print(f"AUC: {auc[i]:>8f}")
        print()

    if model_name is not None:
        print("Saving model...\n")
        torch.save(
            {
                "epoch": epoch + epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_path,
        )

        train_history = None
        benign_test_history = None
        anomalous_test_history = None
        auc_history = None
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                # Retrieve history
                train_history = np.load(f)
                benign_test_history = np.load(f)
                anomalous_test_history = np.load(f)
                auc_history = np.load(f)

        with open(history_path, "wb") as f:
            # Append results to the history
            np.save(
                f,
                train_losses
                if train_history is None
                else np.concat((train_history, train_losses), axis=0),
            )
            np.save(
                f,
                benign_test_losses
                if benign_test_history is None
                else np.concat((benign_test_history, benign_test_losses), axis=0),
            )
            np.save(
                f,
                anomalous_test_losses
                if anomalous_test_history is None
                else np.concat((anomalous_test_history, anomalous_test_losses), axis=0),
            )
            np.save(
                f,
                auc if auc_history is None else np.concat((auc_history, auc), axis=0),
            )
            # Save the most updated losses
            np.save(f, train_loss)
            np.save(f, benign_test_loss)
            np.save(f, anomalous_test_loss)

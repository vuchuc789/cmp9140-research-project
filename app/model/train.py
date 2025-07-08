import os
from glob import glob

import numpy as np
import torch
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


def train(model_name: str):
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 50
    regularization_rate = 1e-5  # not used yet

    model_dir = "model"
    params_path = f"{model_dir}/{model_name}_params.pth"
    history_path = f"{model_dir}/{model_name}_history.npy"

    for file in glob(f"{model_dir}/{model_name}*"):
        print(f"Delete {file}...")
        os.remove(file)

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

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    print("Training model...")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss[epoch] = train_loop(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss[epoch] = test_loop(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )
        print()

    print("Saving model...\n")
    torch.save(model.state_dict(), params_path)
    with open(history_path, "wb") as f:
        np.save(f, train_loss)
        np.save(f, test_loss)

    print(f"Avg training loss: {train_loss[-1]:>7f}")
    print(f"Avg testing loss: {test_loss[-1]:>7f}\n")

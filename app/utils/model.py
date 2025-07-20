import os
from collections import OrderedDict
from typing import List

import numpy as np
import torch

from app.model.model import Autoencoder


def get_model(model_type="ae"):
    match model_type:
        case _:  # ae
            return Autoencoder()


def get_optimizer(
    optimizer_type="sgd",
):
    match optimizer_type:
        case "adam":
            return torch.optim.Adam
        case _:  # sgd
            return torch.optim.SGD


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


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

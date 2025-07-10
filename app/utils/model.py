from collections import OrderedDict
from typing import List

import numpy as np
import torch


def generate_model_name(
    model_type: str,
    loss_type: str,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    regularization_rate: float,
):
    return f"model_{model_type}_{loss_type}_{batch_size}_{learning_rate:.0e}_{epochs}_{regularization_rate}"


def parse_model_name(model_name: str):
    split = model_name.split("_")[1:]

    return (
        str(split[0]),
        str(split[1]),
        int(split[2]),
        float(split[3]),
        int(split[4]),
        float(split[5]),
    )


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

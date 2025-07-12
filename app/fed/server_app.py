from typing import Optional, Union

import numpy as np
import torch
from flwr.common import (
    Context,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app.model.train import get_model
from app.utils.model import get_parameters, set_parameters


class Strategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            model_dir = "model"
            model_path = f"{model_dir}/round_{server_round}_model.pth"

            net = get_model()
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )
            set_parameters(net, aggregated_ndarrays)

            print(f"Saving round {server_round} model...")
            torch.save({"model_state_dict": net.state_dict()}, model_path)

        return aggregated_parameters, aggregated_metrics


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    net = get_model()
    ndarrays = get_parameters(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = Strategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

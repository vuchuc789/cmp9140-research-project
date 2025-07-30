import os
from typing import Optional, Union

import numpy as np
import torch
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx

from app.model.train import init_model
from app.utils.model import get_model, get_parameters, save_history, set_parameters


def aggregate_fit_metrics(
    client_metrics: list[tuple[int, dict[str, Metrics]]],
) -> Metrics:
    train_loss = 0
    num_train = 0

    for num, metrics in client_metrics:
        train_loss += metrics["train_loss"] * num
        num_train += num

    train_loss /= num_train

    return {"train_loss": train_loss}


def aggregate_evaluate_metrics(
    client_metrics: list[tuple[int, dict[str, Metrics]]],
) -> Metrics:
    benign_test_loss = 0
    anomalous_test_loss = 0
    auc = 0
    num_benign_test = 0
    num_anomalous_test = 0

    for num, metrics in client_metrics:
        benign_test_loss += metrics["benign_test_loss"] * metrics["num_benign_test"]
        anomalous_test_loss += (
            metrics["anomalous_test_loss"] * metrics["num_anomalous_test"]
        )
        auc += metrics["auc"] * (
            metrics["num_benign_test"] + metrics["num_anomalous_test"]
        )
        num_benign_test += metrics["num_benign_test"]
        num_anomalous_test += metrics["num_anomalous_test"]

    benign_test_loss /= num_benign_test
    anomalous_test_loss /= num_anomalous_test
    auc /= num_benign_test + num_anomalous_test

    return {
        "benign_test_loss": benign_test_loss,
        "anomalous_test_loss": anomalous_test_loss,
        "auc": auc,
    }


class Strategy(FedProx):
    model_dir = "model"
    model_path = f"{model_dir}/distributed_model.pth"
    history_path = f"{model_dir}/distributed_history.npy"
    optimizer_path = f"{model_dir}/distributed_optimizer.npy"

    def __init__(
        self,
        last_round: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
            proximal_mu=1e-3,  # prox
            # proximal_mu=1e-3,  # adam + prox
            # proximal_mu=0,  # avg
        )

        self.eta = 1e-2

        self.server_learning_rate = 1e-1
        self.server_momentum = 0.9
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )

        if os.path.exists(self.optimizer_path) and (
            (hasattr(self, "m_t") and hasattr(self, "v_t"))
            or hasattr(self, "momentum_vector")
        ):
            ts = []

            with open(self.optimizer_path, "rb") as f:
                while True:
                    try:
                        ts.append(np.load(f))
                    except EOFError:
                        break

            if hasattr(self, "m_t") and hasattr(self, "v_t"):
                self.m_t = ts[: len(ts) // 2]
                self.v_t = ts[len(ts) // 2 :]
            elif hasattr(self, "momentum_vector"):
                self.momentum_vector = ts

        self.last_round = last_round

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"CustomFed(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        round_path = (
            f"{self.model_dir}/distributed_model_{self.last_round + server_round}.pth"
        )

        if aggregated_parameters is not None:
            net = get_model()
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )
            set_parameters(net, aggregated_ndarrays)

            torch.save(
                {
                    "epoch": self.last_round + server_round,
                    "model_state_dict": net.state_dict(),
                },
                self.model_path,
            )
            torch.save(
                {
                    "epoch": self.last_round + server_round,
                    "model_state_dict": net.state_dict(),
                },
                round_path,
            )

            if (
                hasattr(self, "m_t")
                and hasattr(self, "v_t")
                and self.m_t is not None
                and self.v_t is not None
            ):
                with open(self.optimizer_path, "wb") as f:
                    for t in self.m_t:
                        np.save(f, t)
                    for t in self.v_t:
                        np.save(f, t)

            if hasattr(self, "momentum_vector") and self.momentum_vector is not None:
                with open(self.optimizer_path, "wb") as f:
                    for t in self.momentum_vector:
                        np.save(f, t)

        train_loss = np.array([aggregated_metrics["train_loss"]])
        save_history(
            history_path=self.history_path,
            train_loss=train_loss,
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        model_dir = "model"
        history_path = f"{model_dir}/distributed_history.npy"

        benign_test_loss = np.array([aggregated_metrics["benign_test_loss"]])
        anomalous_test_loss = np.array([aggregated_metrics["anomalous_test_loss"]])
        auc = np.array([aggregated_metrics["auc"]])
        save_history(
            history_path=history_path,
            benign_test_loss=benign_test_loss,
            anomalous_test_loss=anomalous_test_loss,
            auc=auc,
        )

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, aggregated_metrics


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    (
        net,
        _,
        _,
        last_round,
        *_,
    ) = init_model(model_name="distributed")
    ndarrays = get_parameters(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = Strategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        last_round=last_round if last_round != -1 else 0,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

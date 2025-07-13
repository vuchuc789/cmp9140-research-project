from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch import nn, optim
from torch.utils.data import DataLoader

from app.model.train import fit_model, init_model
from app.utils.model import get_parameters, set_parameters


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        train_loader: DataLoader,
        benign_test_loader: DataLoader,
        anomalous_test_loader: DataLoader,
        local_epochs: int,
    ):
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.benign_test_loader = benign_test_loader
        self.anomalous_test_loader = anomalous_test_loader
        self.local_epochs = local_epochs

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_loss, *_ = fit_model(
            model=self.net,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            epochs=self.local_epochs,
            device=self.device,
            train_loader=self.train_loader,
        )
        return (
            get_parameters(self.net),
            len(self.train_loader.dataset),
            {"train_loss": float(train_loss)},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        _, benign_loss, anomalous_loss, auc = fit_model(
            model=self.net,
            loss_fn=self.loss_fn,
            device=self.device,
            benign_test_loader=self.benign_test_loader,
            anomalous_test_loader=self.anomalous_test_loader,
        )
        return (
            benign_loss,
            len(self.benign_test_loader.dataset)
            + len(self.anomalous_test_loader.dataset),
            {
                "benign_test_loss": float(benign_loss),
                "anomalous_test_loss": float(anomalous_loss),
                "auc": float(auc),
                "num_benign_test": len(self.benign_test_loader.dataset),
                "num_anomalous_test": len(self.anomalous_test_loader.dataset),
            },
        )


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    (
        net,
        loss_fn,
        optimizer,
        _,
        device,
        train_loader,
        benign_test_loader,
        anomalous_test_loader,
    ) = init_model(partition=f"iid_{num_partitions}", partition_id=partition_id)

    # Return Client instance
    return FlowerClient(
        net,
        loss_fn,
        optimizer,
        device,
        train_loader,
        benign_test_loader,
        anomalous_test_loader,
        local_epochs,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

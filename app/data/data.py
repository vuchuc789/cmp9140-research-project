import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, Dataset


class DDoSDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        split="all",
        save_normalization=False,
        transform=None,
        target_transform=None,
    ) -> None:
        df = pd.read_parquet(data_file)

        # Use source IP as partition id for Flow partioners
        self.partition_ids = df["Source IP"].values

        # Drop unused columns
        df = df.drop(
            columns=[
                "Source IP",
                "Source Port",
                "Destination IP",
                "Destination Port",
                "Timestamp",
            ]
        )

        # Do one-hot encoding on the protocol column
        protocol = df["Protocol"].values.reshape(-1, 1)

        if save_normalization:
            enc = OneHotEncoder(handle_unknown="ignore").fit(protocol)
            joblib.dump(enc, "model/Protocol_OneHotEncoder.gz")
        else:
            enc = joblib.load("model/Protocol_OneHotEncoder.gz")

        protocol = enc.transform(protocol).toarray()
        df = df.drop(columns=["Protocol"])

        # Normalizing numeric columns
        num_columns = df.select_dtypes(include="number").columns
        df[num_columns] = np.log1p(df[num_columns])
        scaler = MinMaxScaler().fit(df[num_columns])

        if save_normalization:
            scaler = MinMaxScaler().fit(df[num_columns])
            joblib.dump(scaler, "model/MinMaxScaler.gz")
        else:
            scaler = joblib.load("model/MinMaxScaler.gz")

        df[num_columns] = scaler.transform(df[num_columns])

        self.data = np.hstack((protocol, df.values)).astype(np.float32)

        # Split train/test
        if split in ["train", "test"]:
            data_train, data_test, partition_ids_train, partition_ids_test = (
                train_test_split(
                    self.data,
                    self.partition_ids,
                    test_size=0.2,
                    random_state=42,
                )
            )

            if split == "train":
                self.data = data_train
                self.partition_ids = partition_ids_train
            else:
                self.data = data_test
                self.partition_ids = partition_ids_test

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.partition_ids)

    def __getitem__(self, idx) -> None:
        data = self.data[idx]
        partition_id = self.partition_ids[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            partition_id = self.target_transform(partition_id)

        return data, partition_id


def load_data(batch_size: int):
    benign_data_train = DDoSDataset(
        "data/Benign.parquet.zst",
        split="train",
        save_normalization=True,
        transform=torch.from_numpy,
    )
    benign_data_test = DDoSDataset(
        "data/Benign.parquet.zst",
        split="test",
        save_normalization=False,
        transform=torch.from_numpy,
    )
    anomalous_data = DDoSDataset(
        "data/Anomalous.parquet.zst",
        split="all",
        save_normalization=False,
        transform=torch.from_numpy,
    )

    return (
        DataLoader(
            benign_data_train,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            benign_data_test,
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            anomalous_data,
            batch_size=batch_size,
            shuffle=False,
        ),
    )

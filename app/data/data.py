import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as PADatatset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    GroupedNaturalIdPartitioner,
    IidPartitioner,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, Dataset


class DDoSDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        split="all",
        partition="none",
        partition_id=0,
        save_normalization=False,
        transform=None,
    ) -> None:
        df = pd.read_parquet(data_file)

        # Drop unused columns
        df.drop(
            columns=[
                # "Source IP",  # used as partition id for Flower partioners
                "Source Port",
                "Destination IP",
                "Destination Port",
                "Timestamp",
            ],
            inplace=True,
        )

        # Do one-hot encoding on the protocol column
        protocol = df["Protocol"].values.reshape(-1, 1)

        if save_normalization:
            enc = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ).set_output(transform="pandas")

            enc.fit(protocol)
            joblib.dump(enc, "data/Protocol_OneHotEncoder.gz")
        else:
            enc: OneHotEncoder = joblib.load("data/Protocol_OneHotEncoder.gz")

        protocol: pd.DataFrame = enc.transform(protocol)
        protocol_idx = df.columns.get_loc("Protocol")  # for futher inserts
        df.drop(columns=["Protocol"], inplace=True)

        # Normalizing other numeric columns
        num_columns = df.select_dtypes(include="number").columns
        df[num_columns] = np.log1p(df[num_columns])
        scaler = MinMaxScaler().fit(df[num_columns])

        if save_normalization:
            scaler = MinMaxScaler().fit(df[num_columns])
            joblib.dump(scaler, "data/MinMaxScaler.gz")
        else:
            scaler = joblib.load("data/MinMaxScaler.gz")

        df[num_columns] = scaler.transform(df[num_columns])

        # Insert one-hot columns to the previous Protocol column place
        for i, col in enumerate(protocol.columns):
            df.insert(protocol_idx + i, col, protocol[col])

        # Simulate non-iid data
        self.partitioner = None
        if partition != "none":
            partition = partition.split("_")

            match partition[0]:
                case "iid":
                    self.partitioner = IidPartitioner(num_partitions=int(partition[1]))
                case "dirichlet":
                    self.partitioner = DirichletPartitioner(
                        num_partitions=int(partition[1]),
                        partition_by="Source IP",
                        alpha=float(partition[2]),
                        min_partition_size=64,  # 1 batch
                    )
                case "natural":
                    self.partitioner = GroupedNaturalIdPartitioner(
                        partition_by="Source IP",
                        group_size=int(partition[1]),
                    )

            original_columns = df.columns
            # Flower partitioners use pyarrow datasets
            dataset = PADatatset.from_pandas(df)
            self.partitioner.dataset = dataset

            # Convert back to pandas dataframe
            df = self.partitioner.load_partition(partition_id).to_pandas()
            # Drop surprising columns
            df.drop(
                columns=[col for col in df.columns if col not in original_columns],
                inplace=True,
            )

        # No longer need partition ids
        df.drop(columns=["Source IP"], inplace=True)

        # Split train/test
        if split in ["train", "test"]:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            df = train_df if split == "train" else test_df

        self.data = df.values.astype(np.float32, casting="same_kind")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> None:
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data


def load_data(batch_size: int, partition="none", partition_id=0):
    benign_data_train = DDoSDataset(
        "data/Benign.parquet.zst",
        split="train",
        partition=partition,
        partition_id=partition_id,
        save_normalization=True,
        transform=torch.from_numpy,
    )
    benign_data_test = DDoSDataset(
        "data/Benign.parquet.zst",
        split="test",
        partition=partition,
        partition_id=partition_id,
        save_normalization=False,
        transform=torch.from_numpy,
    )
    anomalous_data = DDoSDataset(
        "data/Anomalous.parquet.zst",
        split="all",
        partition=partition,
        partition_id=partition_id,
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

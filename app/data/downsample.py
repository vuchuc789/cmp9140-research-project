import os
from glob import glob

import pandas as pd

from app.utils.file import convert_size


def downsample():
    data_dir = "data"
    downsampled_file_prefix = "Reduced_"

    benign_len = len(pd.read_parquet(f"{data_dir}/Benign.parquet.zst"))

    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        parquet_name = parquet_path.split("/")[-1]
        if (
            parquet_name.startswith(downsampled_file_prefix)
            or parquet_name == "Benign.parquet.zst"
        ):
            continue

        downsampled_parquet_path = f"{data_dir}/{downsampled_file_prefix}{parquet_name}"

        print(f"Downsampling {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        if len(df) > benign_len:
            df = df.sample(n=benign_len, random_state=1)

        print(
            f"Writing {benign_len if len(df) == benign_len else len(df)} rows to {downsampled_parquet_path}..."
        )
        df.to_parquet(
            downsampled_parquet_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        print(
            f"Size of {downsampled_parquet_path}: {convert_size(os.stat(downsampled_parquet_path).st_size)}"
        )

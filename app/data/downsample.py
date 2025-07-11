import os
from glob import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.file import convert_size


def downsample(ratio=0.2):
    data_dir = "data"
    downsampled_file = "Anomalous.parquet.zst"
    downsampled_path = f"{data_dir}/{downsampled_file}"

    if os.path.exists(downsampled_path):
        print(f"Deleting {downsampled_path}...")
        os.remove(downsampled_path)

    file_size: dict[str, int] = {}
    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        print(f"Reading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        file_size[parquet_path] = len(df)

    benign_size = file_size[f"{data_dir}/Benign.parquet.zst"]
    anomalous_size = 0
    for file in file_size:
        if not file.endswith("Benign.parquet.zst"):
            anomalous_size += file_size[file]

    pq_writer: pq.ParquetWriter = None

    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        if parquet_path.endswith("Benign.parquet.zst"):
            continue

        print(f"Downsampling {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        sample_num = round(
            file_size[parquet_path] * benign_size / anomalous_size * ratio
        )

        if sample_num <= 0:
            continue
        elif sample_num < file_size[parquet_path]:
            df = df.sample(n=sample_num, random_state=1)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if pq_writer is None:
            pq_writer = pq.ParquetWriter(
                downsampled_path, table.schema, compression="ZSTD"
            )

        print(f"Writing {len(df)} rows to {downsampled_path}...")
        pq_writer.write_table(table)
        print(
            f"Size of {downsampled_path}: {convert_size(os.stat(downsampled_path).st_size)}"
        )

    pq_writer.close()

    print(f"Verifying {downsampled_path}...")
    df = pd.read_parquet(downsampled_path)
    df.sort_values("Timestamp", inplace=True)
    df.to_parquet(downsampled_path, index=False, engine="pyarrow", compression="zstd")
    print(
        f"Size of {downsampled_path}: {convert_size(os.stat(downsampled_path).st_size)}"
    )

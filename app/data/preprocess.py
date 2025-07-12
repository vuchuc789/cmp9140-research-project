import os
import threading
from glob import glob
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.utils.file import convert_size
from app.utils.pandas import col_name_mapping, data_types, dropped_columns, label_map


def handle_label_group(
    label: str,
    group: pd.DataFrame,
    data_dir: str,
    pq_writers: dict[str, pq.ParquetWriter],
):
    # Clean data
    group.drop(
        dropped_columns,
        axis=1,
        inplace=True,
        errors="ignore",
    )
    numeric_cols = group.select_dtypes(include="number").columns
    group[numeric_cols] = group[numeric_cols].clip(lower=0)
    group.replace(np.inf, np.nan, inplace=True)
    group.dropna(inplace=True)
    group = group.astype(data_types)

    # Write to files using pyarrow
    group_parquet_path = f"{data_dir}/{label}.parquet.zst"
    table = pa.Table.from_pandas(group, preserve_index=False)

    if group_parquet_path not in pq_writers:
        pq_writers[group_parquet_path] = pq.ParquetWriter(
            group_parquet_path, table.schema, compression="ZSTD"
        )

    print(f"Writing {len(group)} rows to {group_parquet_path}...")
    pq_writers[group_parquet_path].write_table(table)
    print(
        f"Size of {group_parquet_path}: {convert_size(os.stat(group_parquet_path).st_size)}"
    )


def preprocess():
    data_dir = "data"
    pq_writers: dict[str, pq.ParquetWriter] = {}

    # Delete all existing files
    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        print(f"Deleting {parquet_path}...")
        os.remove(parquet_path)

    # Read zip files
    zip_paths = glob(f"{data_dir}/*.zip")
    for zip_path in zip_paths:
        with ZipFile(zip_path) as z:
            for zip_info in z.filelist:
                # Process csv files only
                if not zip_info.filename.endswith(".csv"):
                    continue

                print("=========================================================")
                print(
                    f"Size of {data_dir}/{zip_info.filename}: {convert_size(zip_info.file_size)}"
                )
                print("=========================================================")

                # Open csv files without extracting
                csv_file = z.open(zip_info.filename)

                # Use pandas to read csv file in chunks of 500,000 rows
                with pd.read_csv(csv_file, chunksize=5e5) as chunks:
                    for chunk in chunks:
                        # Normalize column names and labels
                        chunk.rename(
                            lambda col: col_name_mapping[col.strip()]
                            if col.strip() in col_name_mapping
                            else col.strip(),
                            axis="columns",
                            inplace=True,
                        )
                        chunk["Label"] = chunk["Label"].map(label_map)

                        # List contains threads processing label groups
                        group_threads: list[threading.Thread] = []

                        # Group the chunk by label and process groups simultaneously
                        for label, group in chunk.groupby(by="Label"):
                            t = threading.Thread(
                                target=handle_label_group,
                                args=(label, group, data_dir, pq_writers),
                            )
                            group_threads.append(t)
                            t.start()

                        # Wait for all threads to be done
                        for t in group_threads:
                            t.join()

    # Close all openning writers
    for writer in pq_writers.values():
        writer.close()

    # Delete duplicate rows (which is complex if handle in chunks)
    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        print(f"Verifying {parquet_path}...")

        df = pd.read_parquet(parquet_path)

        duplicate = df.duplicated()
        duplicate_sum = duplicate.sum()

        if duplicate_sum > 0:
            print(f"Deleting {duplicate_sum} duplicate rows from {parquet_path}...")
            df = df[~duplicate.values]
            df.to_parquet(
                parquet_path, index=False, engine="pyarrow", compression="zstd"
            )

        print(f"Size of {parquet_path}: {convert_size(os.stat(parquet_path).st_size)}")

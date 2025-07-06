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
    # Drop columns specified in `dropped_columns` list.
    # `errors='ignore'` prevents an error if a column is not found.
    group.drop(
        dropped_columns,
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # Identify numeric columns to apply numerical cleaning operations.
    numeric_cols = group.select_dtypes(include="number").columns
    # Replace any negative values in numeric columns with 0 to ensure non-negativity.
    group[numeric_cols] = group[numeric_cols].clip(lower=0)

    # Replace infinite values (e.g., from division by zero) with NaN.
    group.replace(np.inf, np.nan, inplace=True)

    # Drop rows that contain any NaN values after cleaning.
    group.dropna(inplace=True)

    # Convert data types of columns as defined in the `data_types` dictionary for consistency.
    group = group.astype(data_types)

    # Construct the full path for the output Parquet file, including ZSTD compression extension.
    group_parquet_path = f"{data_dir}/{label}.parquet.zst"

    # The 'Label' column is now implicitly represented by the filename, so it's dropped from the DataFrame.
    group.drop(["Label"], axis=1, inplace=True)

    # Convert the processed pandas DataFrame group into a PyArrow Table, which is efficient for Parquet.
    table = pa.Table.from_pandas(group)

    # Check if a ParquetWriter instance already exists for this specific label's file.
    # If not, create a new ParquetWriter, specifying the schema and ZSTD compression.
    if group_parquet_path not in pq_writers:
        pq_writers[group_parquet_path] = pq.ParquetWriter(
            group_parquet_path, table.schema, compression="ZSTD"
        )

    # Write the PyArrow Table (representing the current data group) to the Parquet file.
    print(f"Writing {len(group)} rows to {group_parquet_path}...")
    pq_writers[group_parquet_path].write_table(table)
    # After writing, print the current size of the Parquet file for monitoring.
    print(
        f"Size of {group_parquet_path}: {convert_size(os.stat(group_parquet_path).st_size)}"
    )


def preprocess():
    data_dir = "data"
    # Initialize a dictionary to hold ParquetWriter objects. This allows appending data
    # to the same Parquet file across different chunks and threads for each label.
    pq_writers: dict[str, pq.ParquetWriter] = {}

    # Clean up: Delete any existing ZSTD compressed Parquet files in the data directory
    # to ensure that new data is written to fresh files without conflicts.
    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        print(f"Deleting {parquet_path}...")
        os.remove(parquet_path)

    # Discover and iterate through all ZIP archive files in the specified data directory.
    zip_paths = glob(f"{data_dir}/*.zip")
    for zip_path in zip_paths:
        # Open each ZIP file in read mode.
        with ZipFile(zip_path) as z:
            # Iterate through each file entry within the opened ZIP archive.
            for zip_info in z.filelist:
                # Skip any files in the ZIP archive that are not CSV files.
                if not zip_info.filename.endswith(".csv"):
                    continue

                # Print the size of the current CSV file being processed from within the ZIP.
                print("=========================================================")
                print(
                    f"Size of {data_dir}/{zip_info.filename}: {convert_size(zip_info.file_size)}"
                )
                print("=========================================================")
                # Open the CSV file directly from the ZIP archive as a file-like object.
                csv_file = z.open(zip_info.filename)

                # Read the CSV file in chunks to efficiently handle large datasets that might
                # not fit entirely into memory. `chunksize=5e5` means 500,000 rows per chunk.
                with pd.read_csv(csv_file, chunksize=5e5) as chunks:
                    for chunk in chunks:
                        # Rename columns in the current chunk based on the `col_name_mapping` dictionary.
                        # It first strips whitespace from column names to ensure accurate mapping.
                        chunk.rename(
                            lambda col: col_name_mapping[col.strip()]
                            if col.strip() in col_name_mapping
                            else col.strip(),
                            axis="columns",
                            inplace=True,
                        )
                        # Apply the label mapping function to the 'Label' column to standardize values.
                        chunk["Label"] = chunk["Label"].map(label_map)

                        # Initialize a list to hold thread objects for concurrent processing of groups.
                        group_threads: list[threading.Thread] = []

                        # Group the processed chunk by the 'Label' column. Each group represents data
                        # for a specific traffic class (e.g., "Benign", "DDoS").
                        for label, group in chunk.groupby(by="Label"):
                            # Create a new thread for each label group to process it in parallel.
                            # `handle_label_group` function is the target, with necessary arguments.
                            t = threading.Thread(
                                target=handle_label_group,
                                args=(label, group, data_dir, pq_writers),
                            )
                            group_threads.append(t)
                            # Start the thread.
                            t.start()

                        # Wait for all threads to complete their execution before moving to the next chunk.
                        for t in group_threads:
                            t.join()

    for writer in pq_writers.values():
        writer.close()

    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        print(f"Verifying {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        duplicated = df.duplicated().sum()

        if duplicated > 0:
            print(f"Deleting {duplicated} duplicated rows...")
            df = df.drop_duplicates()
            df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")

        print(f"Size of {parquet_path}: {convert_size(os.stat(parquet_path).st_size)}")

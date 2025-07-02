import os
from glob import glob
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.utils.file import convert_size
from app.utils.pandas import col_name_mapping, data_types, dropped_columns, label_map


def preprocess():
    """
    Preprocesses raw data files.

    This function performs the following steps:
    1. Deletes any existing Parquet files in the 'data' directory to ensure a clean slate.
    2. Extracts CSV files from ZIP archives in the 'data' directory.
    3. Reads each extracted CSV file in chunks to handle large datasets efficiently.
    4. Renames columns based on a predefined mapping.
    5. Drops specified unnecessary columns.
    6. Converts data types of columns as specified.
    7. Maps labels to standardized names.
    8. Groups the data by 'Label' and writes each group to a separate ZSTD compressed Parquet file.
    """
    data_dir = "data"
    # Dictionary to hold Parquet writers for each label, to efficiently append data
    pq_writers: dict[str, pq.ParquetWriter] = {}

    # Delete any existing ZSTD compressed Parquet files to ensure a clean run
    for parquet_path in glob(f"{data_dir}/*.parquet.zst"):
        print(f"Deleting {parquet_path}...")
        os.remove(parquet_path)

    # Process each ZIP file found in the data directory
    zip_paths = glob(f"{data_dir}/*.zip")
    for zip_path in zip_paths:
        # Open the ZIP file
        with ZipFile(zip_path) as z:
            # Iterate through each file within the zip archive
            for zip_info in z.filelist:
                # Only process files that end with .csv
                if not zip_info.filename.endswith(".csv"):
                    continue

                # Print the size of the CSV file within the zip
                print(
                    f"Size of {data_dir}/{zip_info.filename}: {convert_size(zip_info.file_size)}"
                )
                # Open the CSV file from within the zip archive
                csv_file = z.open(zip_info.filename)

                # Read the CSV file in chunks to handle large datasets efficiently
                # chunksize=5e5 means 500,000 rows per chunk
                with pd.read_csv(csv_file, chunksize=5e5) as chunks:
                    for chunk in chunks:
                        # Rename columns based on the col_name_mapping dictionary
                        # It strips whitespace from column names before mapping
                        chunk.rename(
                            lambda col: col_name_mapping[col.strip()]
                            if col.strip() in col_name_mapping
                            else col.strip(),
                            axis="columns",
                            inplace=True,
                        )
                        # Drop columns specified in dropped_columns list
                        # errors='ignore' ensures that no error is raised if a column is not found
                        chunk.drop(
                            dropped_columns,
                            axis=1,
                            inplace=True,
                            errors="ignore",
                        )

                        # Convert data types of columns as defined in data_types
                        chunk = chunk.astype(data_types)

                        # Map the 'Label' column values using the label_map function
                        chunk["Label"] = chunk["Label"].map(label_map)

                        # Group the processed chunk by the 'Label' column
                        for label, group in chunk.groupby(by="Label"):
                            # Define the output Parquet file path for the current label group
                            group_parquet_path = f"{data_dir}/{label}.parquet.zst"

                            # Drop the 'Label' column from the group as it's now part of the filename
                            group.drop(["Label"], axis=1, inplace=True)

                            # Convert the pandas DataFrame group to a PyArrow Table
                            table = pa.Table.from_pandas(group)

                            # If a ParquetWriter for this label doesn't exist, create one
                            if group_parquet_path not in pq_writers:
                                pq_writers[group_parquet_path] = pq.ParquetWriter(
                                    group_parquet_path, table.schema, compression="ZSTD"
                                )

                            # Write the PyArrow Table to the corresponding Parquet file
                            print(f"Writing to {group_parquet_path}...")
                            pq_writers[group_parquet_path].write_table(table)
                            # Print the size of the updated Parquet file
                            print(
                                f"Size of {group_parquet_path}: {convert_size(os.stat(group_parquet_path).st_size)}"
                            )

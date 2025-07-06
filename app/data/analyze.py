import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze():
    df = pd.read_parquet("data/Benign.parquet.zst")
    print(df.info(verbose=True))

    df["Source IP"].value_counts().head(20).plot(kind="bar")
    plt.title("Top 20 Source IPs")
    plt.xlabel("IP Address")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    df["Source Port"].plot.hist(bins=64)
    plt.title("Source Ports")
    plt.xlabel("Port")
    plt.ylabel("Count")
    plt.show()

    df["Destination IP"].value_counts().head(20).plot(kind="bar")
    plt.title("Top 20 Destination IPs")
    plt.xlabel("IP Address")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    df["Destination Port"].plot.hist(bins=64)
    plt.title("Destination Ports")
    plt.xlabel("Port")
    plt.ylabel("Count")
    plt.show()

    df = df.select_dtypes(include="number")
    df = np.log(df + 1).astype(np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey="row")

    for i, col in enumerate(df.columns):
        df[col].plot.hist(bins=50, alpha=0.5, ax=axes[(i // 4) % 2, i % 4])
        axes[(i // 4) % 2, i % 4].set_title(col)
        stats = (
            f"max = {np.max(df[col]):.2e}\n"
            f"min = {np.min(df[col]):.2e}\n"
            f"$\\mu$ = {np.mean(df[col]):.2e}\n"
            f"$\\sigma$ = {np.std(df[col]):.2e}\n"
        )
        axes[(i // 4) % 2, i % 4].text(
            0.95,
            0.65,
            stats,
            fontsize=9,
            transform=axes[(i // 4) % 2, i % 4].transAxes,
            horizontalalignment="right",
        )

        if i % 8 == 7 or i == len(df.columns) - 1:
            print(f"Showing page {i // 8 + 1}...")

            plt.suptitle("Benign Data Statistics")
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey="row")

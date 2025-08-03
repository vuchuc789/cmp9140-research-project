import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze(filepath="data/Benign.parquet.zst"):
    df = pd.read_parquet(filepath)
    print(df.info(verbose=True))
    print("Duplicate Count:", df.duplicated().sum())
    print("Source IP Count:", len(pd.unique(df["Source IP"])))

    # Seaborn style
    sns.set(style="whitegrid")

    # ───── IP Frequency ─────
    def plot_top_ips(column, title):
        top_ips = df[column].value_counts().head(20)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_ips.index, y=top_ips.values)
        plt.title(f"{title} ({filepath})", fontsize=14)
        plt.xlabel("IP Address", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    plot_top_ips("Source IP", "Top 20 Source IPs")
    plot_top_ips("Destination IP", "Top 20 Destination IPs")

    # ───── Port Histograms ─────
    def plot_port_dist(column, title):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], bins=64, kde=False, color="steelblue")
        plt.title(f"{title} ({filepath})", fontsize=14)
        plt.xlabel("Port", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.show()

    plot_port_dist("Source Port", "Source Port Distribution")
    plot_port_dist("Destination Port", "Destination Port Distribution")

    # ───── Numeric Feature Stats ─────
    df_numeric = df.select_dtypes(include="number")
    df_numeric = np.log1p(df_numeric).astype(np.float32)

    num_cols = df_numeric.shape[1]
    cols_per_page = 8

    for page_start in range(0, num_cols, cols_per_page):
        page_cols = df_numeric.columns[page_start : page_start + cols_per_page]
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey="row")

        for i, col in enumerate(page_cols):
            ax = axes[i // 4, i % 4]
            sns.histplot(df_numeric[col], bins=50, ax=ax, color="darkorange", alpha=0.7)
            ax.set_title(col, fontsize=11)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))

            stats = (
                f"max = {np.max(df_numeric[col]):.2e}\n"
                f"min = {np.min(df_numeric[col]):.2e}\n"
                f"μ = {np.mean(df_numeric[col]):.2e}\n"
                f"σ = {np.std(df_numeric[col]):.2e}"
            )
            ax.text(
                0.95,
                0.65,
                stats,
                fontsize=9,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
            )

        fig.suptitle(f"Log-Scaled Feature Distributions ({filepath})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
